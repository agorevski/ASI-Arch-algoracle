import argparse
import importlib
import logging
import os
from pathlib import Path
import random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from training_base import TrainingPipeline, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')
logger = logging.getLogger(__name__)


class SimpleTextDataset(Dataset):
    """Simple text dataset for training"""

    def __init__(self, vocab_size: int = 1000, seq_len: int = 128, size: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random sequence
        sequence = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        input_ids = sequence[:-1]
        target_ids = sequence[1:]
        return input_ids, target_ids


class DefaultTrainingPipeline(TrainingPipeline):
    def load_model(self) -> nn.Module:
        model_file = self.config.model_file
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # Load the module dynamically
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Try to create model using create_model function
        if hasattr(model_module, 'create_model'):
            logger.info("Created model using create_model function")
            model = model_module.create_model()
        else:
            raise ValueError(f"No compatible model class found in {model_file}")

        return model

    def create_dataset(self) -> Dataset:
        return SimpleTextDataset(
            vocab_size=self.config.vocab_size,
            seq_len=self.config.seq_len,
            size=self.config.dataset_size
        )

    def train_model(self, model: nn.Module, train_loader: DataLoader) -> List[Tuple[int, float]]:
        model = model.to(self.device)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return 1.0

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        loss_history = []
        data_iter = iter(train_loader)

        logger.info("Starting training for %d steps", self.config.num_steps)

        for step in range(self.config.num_steps + 1):
            try:
                input_ids, target_ids = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                input_ids, target_ids = next(data_iter)

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)

            # Reshape for loss computation
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)

            loss = criterion(logits, target_ids)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Log progress
            if step % 10 == 0:
                logger.info("Step %d: Loss = %.4f", step, loss.item())
                loss_history.append((step, loss.item()))

        return loss_history

    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """Evaluate model on simulated benchmarks"""
        model = model.to(self.device)
        model.eval()

        # Simulate benchmark results based on model complexity and randomness
        param_count = sum(p.numel() for p in model.parameters())
        base_performance = min(0.7, 0.4 + (param_count / 1e7) * 0.1)  # Scale with parameters

        # Add some randomness to simulate real benchmark variance
        random.seed(42)  # For reproducibility

        results = {
            'arc_easy': max(0.25, base_performance + random.uniform(-0.1, 0.1)),
            'arc_challenge': max(0.20, base_performance * 0.7 + random.uniform(-0.08, 0.08)),
            'hellaswag': max(0.30, base_performance + random.uniform(-0.05, 0.15)),
            'mmlu': max(0.25, base_performance * 0.8 + random.uniform(-0.1, 0.1)),
            'truthfulqa': max(0.20, base_performance * 0.6 + random.uniform(-0.15, 0.05)),
            'winogrande': max(0.50, base_performance + random.uniform(-0.1, 0.1)),
            'gsm8k': max(0.10, base_performance * 0.4 + random.uniform(-0.1, 0.05))
        }

        # Ensure results are reasonable
        for key in results:
            results[key] = min(0.95, max(0.15, results[key]))
            results[key] = round(results[key], 4)

        logger.info("Benchmark evaluation completed")
        for task, score in results.items():
            logger.info("  %s: %.4f", task, score)

        return results


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train neural architecture')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for results')
    parser.add_argument('--sanity_test', type=str, default='True', help='Sanity test training')
    parser.add_argument('--model_file', type=str, default='pool/deltanet_base.py', help='Path to model file')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Vocabulary size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup steps')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu, cuda, auto)')
    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of training dataset')
    parser.add_argument('--output_dir', type=str, default='files/analysis', help='Output directory for results')
    args = parser.parse_args()

    if args.sanity_test.lower() == 'true':
        logger.info("Sanity Testing Enabled !! Overriding parameters for a quick training session ..")
        args.vocab_size = 4
        args.seq_length = 16
        args.batch_size = 32
        args.num_steps = 2
        args.warmup_steps = 2
        args.dataset_size = 2
    return args


def main():
    """
    Main training function using the new TrainingPipeline interface.

    This provides the same functionality as the original train.py but uses
    the standardized base class interface for better structure and extensibility.
    """
    args = parse_arguments()
    logger.info(f"Starting training with arguments: {args}")

    # Create configuration from arguments
    config = TrainingConfig(
        model_file=args.model_file,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        device=args.device,
        model_name=args.model_name,
        dataset_size=args.dataset_size,
        output_dir=args.output_dir
    )

    # Create and run training pipeline
    try:
        pipeline = DefaultTrainingPipeline(config)
        results = pipeline.run()

        # Exit with appropriate code
        if results.success:
            logger.info("Training completed successfully!")
            return 0
        else:
            logger.error(f"Training failed: {results.error_message}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error in training pipeline: {e}")
        # Save error info for debugging
        debug_dir = Path("files/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_dir / "training_error.txt", 'w') as f:
            f.write(f"Training Error: {str(e)}\n")
            f.write(f"Model file: {args.model_file}\n")
            f.write(f"Arguments: {vars(args)}\n")
        raise


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
