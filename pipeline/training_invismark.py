"""
InvisMark Training Pipeline Implementation

This module provides a TrainingPipeline implementation for the InvisMark
image watermarking architecture, adapting the existing training code to work
with the standardized pipeline interface.
"""

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml

# Add the invismark training directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "pool" / "invismark" / "training"))
from training_base import TrainingPipeline, TrainingConfig

# Import InvisMark modules using direct file imports to avoid naming conflicts
import importlib.util


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_path = Path(__file__).parent / "pool" / "invismark" / "training"
invismark_model = import_module_from_path("invismark_model", base_path / "model.py")
invismark_metrics = import_module_from_path("invismark_metrics", base_path / "metrics.py")
invismark_utils = import_module_from_path("invismark_utils", base_path / "utils.py")
invismark_noise = import_module_from_path("invismark_noise", base_path / "noise.py")
loss_module = import_module_from_path("invismark_loss", base_path / "loss.py")
WatermarkLoss = loss_module.WatermarkLoss

logger = logging.getLogger(__name__)


@dataclass
class InvisMarkTrainingConfig(TrainingConfig):
    """Extended configuration for InvisMark training"""
    # InvisMark specific parameters
    config_file: str = "pipeline/pool/invismark/training/configs/config.yaml"
    dataset_path: Optional[str] = None
    img_train_path: str = "dalle/train"
    img_test_path: str = "dalle/test"
    train_mode: str = "image"
    # Image parameters
    image_size: Tuple[int, int] = (256, 256)
    num_channels: int = 3
    # Watermark parameters
    num_bits: int = 256
    ecc_mode: str = "ecc"
    ecc_t: int = 16
    ecc_m: int = 8
    # Training parameters (override defaults)
    num_epochs: int = 100
    beta_min: float = 1e-4
    beta_max: float = 40.0
    beta_start_epoch: int = 10
    beta_epochs: int = 50
    warmup_epochs: int = 5
    noise_start_epoch: int = 50
    # Distributed training
    distributed: bool = False
    local_rank: int = 0


class SyntheticImageDataset(Dataset):
    """Synthetic image dataset for testing InvisMark training"""
    def __init__(
        self,
        size: int = 1000,
        image_size: Tuple[int, int] = (256, 256),
        num_channels: int = 3
    ):
        self.size = size
        self.image_size = image_size
        self.num_channels = num_channels

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random image in range [-1, 1]
        image = torch.randn(self.num_channels, *self.image_size)
        image = torch.clamp(image, -1.0, 1.0)
        return (image,)


class InvisMarkTrainingPipeline(TrainingPipeline):
    """Training pipeline for InvisMark image watermarking"""
    def __init__(self, config: InvisMarkTrainingConfig):
        super().__init__(config)
        self.config: InvisMarkTrainingConfig = config
        # Load InvisMark configuration
        self.invismark_config = self._load_invismark_config()
        # Initialize components
        self.loss_fn = None
        self.noiser = None
        self.bchecc = None

    def _load_invismark_config(self) -> dict:
        """Load and merge InvisMark YAML configuration"""
        config_path = Path(self.config.config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            logger.info(f"Loaded InvisMark config from {config_path}")
        else:
            # Default configuration
            yaml_config = {
                'IMAGE': {'SIZE': list(self.config.image_size)},
                'WATERMARK': {
                    'NUM_BITS': self.config.num_bits,
                    'ECC_MODE': self.config.ecc_mode,
                    'ECC_T': self.config.ecc_t,
                    'ECC_M': self.config.ecc_m
                },
                'ENCODER': {
                    'NUM_DOWN_LEVELS': 4,
                    'NUM_INITIAL_CHANNELS': 32,
                    'HIDDEN_DIM': 16,
                    'NUM_FRAMES': 1
                },
                'DECODER': {'NAME': 'IMAGENET1K_V1'},
                'DISCRIMINATOR': {
                    'NAME': 'ResNet18_Weights.IMAGENET1K_V1',
                    'NUM_CLASSES': 1
                }
            }
            logger.warning(f"Config file not found at {config_path}, using defaults")
        # Merge with training config
        yaml_config['num_epochs'] = self.config.num_epochs
        yaml_config['batch_size'] = self.config.batch_size
        yaml_config['lr'] = self.config.learning_rate
        yaml_config['warmup_epochs'] = self.config.warmup_epochs
        yaml_config['train_mode'] = self.config.train_mode
        yaml_config['beta_min'] = self.config.beta_min
        yaml_config['beta_max'] = self.config.beta_max
        yaml_config['beta_start_epoch'] = self.config.beta_start_epoch
        yaml_config['beta_epochs'] = self.config.beta_epochs
        yaml_config['noise_start_epoch'] = self.config.noise_start_epoch
        return yaml_config

    def load_model(self) -> nn.Module:
        """Load InvisMark model"""
        try:
            model = invismark_model.InvisMark(self.invismark_config)
            logger.info("Created InvisMark model")
            # Initialize loss function and utilities
            self.loss_fn = WatermarkLoss(self.invismark_config)
            self.noiser = invismark_noise.Noiser(num_transforms=1)
            # Initialize ECC if needed
            if self.invismark_config['WATERMARK']['ECC_MODE'] == 'ecc':
                self.bchecc = invismark_model.BCHECC(
                    t=self.invismark_config['WATERMARK']['ECC_T'],
                    m=self.invismark_config['WATERMARK']['ECC_M']
                )
                logger.info(
                    f"Initialized BCH ECC: tot_bits={self.invismark_config['WATERMARK']['NUM_BITS']}, "
                    f"data_bytes={self.bchecc.data_bytes}"
                )
            return model
        except Exception as e:
            logger.error(f"Failed to load InvisMark model: {e}")
            raise

    def create_dataset(self) -> Dataset:
        """Create synthetic image dataset"""
        return SyntheticImageDataset(
            size=self.config.dataset_size,
            image_size=tuple(self.invismark_config['IMAGE']['SIZE']),
            num_channels=self.config.num_channels
        )

    def _generate_watermark(self, batch_size: int) -> torch.Tensor:
        """Generate random watermark bits"""
        if self.invismark_config['WATERMARK']['ECC_MODE'] == 'uuid':
            bits, _ = invismark_utils.uuid_to_bits(batch_size)
        elif self.invismark_config['WATERMARK']['ECC_MODE'] == 'ecc':
            if self.bchecc is None:
                raise ValueError("BCH ECC not initialized")
            bits = self.bchecc.batch_encode(batch_size)
        else:
            raise ValueError(
                f"Unsupported ECC mode: {self.invismark_config['WATERMARK']['ECC_MODE']}"
            )
        return bits[:, :self.invismark_config['WATERMARK']['NUM_BITS']].to(self.device)

    def train_model(self, model: nn.Module, train_loader: DataLoader) -> List[Tuple[int, float]]:
        """Train the InvisMark model"""
        model = model.to(self.device)
        model.train()
        # Setup optimizer
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        loss_history = []
        step = 0
        # Get a fixed batch for warmup if needed
        fixed_batch = next(iter(train_loader))
        logger.info(f"Starting InvisMark training for {self.config.num_epochs} epochs")
        for epoch in range(self.config.num_epochs):
            epoch_losses = []
            # Use fixed batch during warmup
            if epoch < self.invismark_config['warmup_epochs']:
                batches = [fixed_batch]
                logger.info(f"Epoch {epoch}: Using fixed batch for warmup")
            else:
                batches = train_loader
            for batch_idx, batch in enumerate(batches):
                optimizer.zero_grad()
                # Get images from batch
                images = batch[0].to(self.device)
                # Convert to video format [B, F, C, H, W] if needed
                if images.ndim == 4:
                    images = images.unsqueeze(1).repeat(
                        1, self.invismark_config['ENCODER']['NUM_FRAMES'], 1, 1, 1
                    )
                # Generate watermark
                watermark = self._generate_watermark(images.shape[0])
                # Forward pass
                outputs = model(images, watermark)
                # Calculate loss
                loss = self.loss_fn(outputs, watermark, epoch)
                # Backward pass
                loss.backward()
                optimizer.step()
                # Track loss
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                if step % 10 == 0:
                    logger.info(f"Epoch {epoch}, Step {step}: Loss = {loss_val:.4f}")
                    loss_history.append((step, loss_val))
                step += 1
            # Log epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch} completed: Average Loss = {avg_loss:.4f}")
        return loss_history

    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate InvisMark model"""
        model = model.to(self.device)
        model.eval()
        # Create evaluation dataset
        eval_dataset = SyntheticImageDataset(
            size=100,
            image_size=tuple(self.invismark_config['IMAGE']['SIZE']),
            num_channels=self.config.num_channels
        )
        eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        avg_metrics = defaultdict(float)
        num_batches = 0
        logger.info("Starting InvisMark evaluation")
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                if batch_idx >= 50:  # Limit evaluation batches
                    break
                # Get images
                images = batch[0].to(self.device)
                # Convert to video format if needed
                if images.ndim == 4:
                    images = images.unsqueeze(1).repeat(
                        1, self.invismark_config['ENCODER']['NUM_FRAMES'], 1, 1, 1
                    )
                # Generate watermark
                watermark = self._generate_watermark(images.shape[0])
                # Forward pass
                outputs = model(images, watermark)
                # Flatten images for metrics
                orig_imgs = images.view(-1, 3, *images.shape[-2:])
                wm_imgs = outputs['final_outputs']
                # Calculate metrics
                batch_metrics = self._calculate_metrics(orig_imgs, wm_imgs, watermark, model)
                for key, value in batch_metrics.items():
                    avg_metrics[key] += value
                num_batches += 1
        # Average metrics
        for key in avg_metrics:
            avg_metrics[key] = avg_metrics[key] / num_batches
        logger.info("Evaluation completed")
        for task, score in avg_metrics.items():
            logger.info(f"  {task}: {score:.4f}")
        return dict(avg_metrics)

    def _calculate_metrics(self, orig_imgs: torch.Tensor, wm_imgs: torch.Tensor, watermark: torch.Tensor, model: nn.Module) -> Dict[str, float]:
        """Calculate image quality and watermark extraction metrics"""
        metrics = {}
        # Repeat watermark if needed
        if orig_imgs.shape[0] != watermark.shape[0]:
            watermark = watermark.repeat(orig_imgs.shape[0] // watermark.shape[0], 1)
        # Image quality metrics
        metrics['psnr'] = invismark_metrics.image_psnr(
            wm_imgs.cpu(), orig_imgs.cpu()
        ).item()
        metrics['ssim'] = invismark_metrics.image_ssim(
            wm_imgs.cpu(), orig_imgs.cpu()
        ).item()
        # Watermark extraction accuracy
        dec_wm = model.decode(wm_imgs)
        metrics['bit_accuracy'] = invismark_metrics.bit_accuracy(
            watermark, dec_wm
        ).item()
        # Test robustness to transformations
        supported_transforms = invismark_noise.supported_transforms(
            tuple(self.invismark_config['IMAGE']['SIZE'])
        )
        for transform_name in list(supported_transforms.keys())[:3]:  # Test first 3
            noised_imgs = self.noiser(wm_imgs, [transform_name])
            dec_wm_noised = model.decode(noised_imgs)
            bit_acc = invismark_metrics.bit_accuracy(watermark, dec_wm_noised).item()
            metrics[f'bit_acc_{transform_name}'] = bit_acc
        return metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train InvisMark image watermarking model')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for results')
    parser.add_argument('--model_file', type=str, default='pipeline/pool/invismark/invismark_base.py', help='Path to model file (for compatibility)')
    parser.add_argument('--config_file', type=str, default='pipeline/pool/invismark/training/configs/config.yaml', help='InvisMark config file path')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='Warmup epochs')
    parser.add_argument('--dataset_size', type=int, default=1000, help='Dataset size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--output_dir', type=str, default='files/analysis', help='Output directory')
    parser.add_argument('--sanity_test', type=str, default='False', help='Run quick sanity test')
    args = parser.parse_args()
    # Sanity test overrides
    if args.sanity_test.lower() == 'true':
        logger.info("Sanity test mode: Using minimal parameters")
        args.num_epochs = 2
        args.batch_size = 2
        args.warmup_epochs = 1
        args.dataset_size = 10
    return args


def main():
    """Main training function"""
    args = parse_arguments()
    logger.info(f"Starting InvisMark training with arguments: {args}")
    # Create configuration
    config = InvisMarkTrainingConfig(
        model_file=args.model_file,
        model_name=args.model_name,
        config_file=args.config_file,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        dataset_size=args.dataset_size,
        device=args.device,
        output_dir=args.output_dir
    )
    # Create and run pipeline
    try:
        pipeline = InvisMarkTrainingPipeline(config)
        results = pipeline.run()
        if results.success:
            logger.info("InvisMark training completed successfully!")
            return 0
        else:
            logger.error(f"Training failed: {results.error_message}")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    exit(main())
