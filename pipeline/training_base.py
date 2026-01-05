from abc import ABC, abstractmethod
from dataclasses import dataclass
import csv
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import traceback

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config_loader import Config

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    # Model configuration
    model_file: str
    vocab_size: int = 1000
    model_name: Optional[str] = None

    # Training configuration
    num_steps: int = 500
    learning_rate: float = 5e-4
    warmup_steps: int = 100
    batch_size: int = 32

    # Dataset configuration
    seq_len: int = 128
    dataset_size: int = 10000

    # System configuration
    device: str = 'auto'

    # Output configuration
    save_loss_history: bool = True
    save_benchmarks: bool = True
    output_dir: str = "files/analysis"

    def __post_init__(self):
        """Post-initialization processing.

        Sets model_name from model_file if not provided, creates debug directory
        path based on current architecture, and ensures output directories exist.
        """
        if self.model_name is None:
            self.model_name = Path(self.model_file).stem

        self.debug_dir = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'settings', 'architecture', Config.get('ARCHITECTURE'), 'debug'))

        # Ensure output directories
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.debug_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingResults:
    """Results container for training pipeline outputs"""
    # Training metrics
    loss_history: List[Tuple[int, float]]
    final_loss: float
    training_time: float

    # Model information
    model_name: str
    parameter_count: int

    # Evaluation results
    benchmark_results: Dict[str, float]
    average_benchmark: float

    # Output files
    loss_file: Optional[str] = None
    benchmark_file: Optional[str] = None

    # Metadata
    config: Optional[TrainingConfig] = None
    success: bool = True
    error_message: Optional[str] = None


class TrainingPipeline(ABC):
    """
    Abstract base class for training pipelines.

    Provides standardized interface for:
    - Input/output handling
    - Model loading and training
    - Evaluation and result saving
    - Error handling and logging
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize training pipeline with configuration.

        Args:
            config: TrainingConfig containing all training parameters
        """
        self.config = config
        self.device = self._determine_device()
        self.model: Optional[nn.Module] = None
        self.dataset: Optional[Dataset] = None
        self.train_loader: Optional[DataLoader] = None
        logger.info(f"Initialized {self.__class__.__name__} with device: {self.device}")

    def _determine_device(self) -> str:
        """Determine the computation device to use.

        Returns:
            str: The device string ('cuda' or 'cpu'). If config.device is 'auto',
                returns 'cuda' if available, otherwise 'cpu'.
        """
        if self.config.device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.config.device

    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Load model from configuration.

        Returns:
            nn.Module: Loaded model

        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model cannot be loaded
        """
        pass

    @abstractmethod
    def create_dataset(self) -> Dataset:
        """
        Create dataset for training.

        Returns:
            Dataset: Training dataset
        """
        pass

    def create_dataloader(self, dataset: Dataset) -> DataLoader:
        """
        Create data loader from dataset.

        Args:
            dataset: Dataset to create loader for

        Returns:
            DataLoader: Configured data loader
        """
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    @abstractmethod
    def train_model(self, model: nn.Module, train_loader: DataLoader) -> List[Tuple[int, float]]:
        """
        Train the model.

        Args:
            model: Model to train
            train_loader: Data loader for training

        Returns:
            List[Tuple[int, float]]: Loss history as (step, loss) tuples
        """
        pass

    @abstractmethod
    def evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate trained model.

        Args:
            model: Trained model to evaluate

        Returns:
            Dict[str, float]: Benchmark results
        """
        pass

    def save_results(self, results: TrainingResults) -> Tuple[Optional[str], Optional[str]]:
        """
        Save training results to files.

        Args:
            results: TrainingResults object containing all results

        Returns:
            Tuple[Optional[str], Optional[str]]: (loss_file_path, benchmark_file_path)
        """
        loss_file = None
        benchmark_file = None

        try:
            if self.config.save_loss_history and results.loss_history:
                loss_file = self._save_loss_history(results.loss_history)

            if self.config.save_benchmarks and results.benchmark_results:
                benchmark_file = self._save_benchmark_results(
                    results.benchmark_results, results.model_name
                )

            return loss_file, benchmark_file

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None, None

    def _save_loss_history(self, loss_history: List[Tuple[int, float]]) -> str:
        """Save training loss history to CSV file.

        Args:
            loss_history: List of (step, loss) tuples representing
                the training loss at each recorded step.

        Returns:
            str: Path to the saved loss history CSV file.
        """
        loss_file = Path(self.config.output_dir) / "loss.csv"
        with open(loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
            writer.writerows(loss_history)

        logger.info(f"Loss history saved to {loss_file}")
        return str(loss_file)

    def _save_benchmark_results(self, benchmark_results: Dict[str, float], model_name: str) -> str:
        """Save benchmark results to CSV file.

        Args:
            benchmark_results: Dictionary mapping benchmark names to their scores.
            model_name: Name of the model being evaluated.

        Returns:
            str: Path to the saved benchmark results CSV file.
        """
        benchmark_file = Path(self.config.output_dir) / "benchmark.csv"
        with open(benchmark_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['model'] + list(benchmark_results.keys())
            writer.writerow(header)
            row = [model_name] + list(benchmark_results.values())
            writer.writerow(row)

        logger.info(f"Benchmark results saved to {benchmark_file}")
        return str(benchmark_file)

    def run(self) -> TrainingResults:
        """
        Execute the complete training pipeline.

        Returns:
            TrainingResults: Complete results of training pipeline
        """
        start_time = time.time()
        try:
            logger.info(f"Starting training pipeline for {self.config.model_name}")
            # Load model
            logger.info(f"Loading model from {self.config.model_file}")
            self.model = self.load_model()
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model loaded with {param_count:,} parameters")
            # Create dataset and dataloader
            logger.info("Creating dataset and dataloader")
            self.dataset = self.create_dataset()
            self.train_loader = self.create_dataloader(self.dataset)
            # Train model
            logger.info("Starting training")
            loss_history = self.train_model(self.model, self.train_loader)
            training_time = time.time() - start_time
            final_loss = loss_history[-1][1] if loss_history else 0.0
            # Evaluate model
            logger.info("Starting evaluation")
            benchmark_results = self.evaluate_model(self.model)
            avg_benchmark = sum(benchmark_results.values()) / len(benchmark_results)
            # Create results object
            results = TrainingResults(
                loss_history=loss_history,
                final_loss=final_loss,
                training_time=training_time,
                model_name=self.config.model_name,
                parameter_count=param_count,
                benchmark_results=benchmark_results,
                average_benchmark=avg_benchmark,
                config=self.config,
                success=True
            )
            # Save results
            logger.info("Saving results")
            loss_file, benchmark_file = self.save_results(results)
            results.loss_file = loss_file
            results.benchmark_file = benchmark_file
            logger.info("Training pipeline completed successfully")
            self._print_summary(results)
            return results
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            exception_stack = traceback.format_exc()
            logger.error(f"Traceback: {exception_stack}")
            # Save error information
            self._save_error_info(e, exception_stack)
            # Return failed results
            results = TrainingResults(
                loss_history=[],
                final_loss=0.0,
                training_time=time.time() - start_time,
                model_name=self.config.model_name,
                parameter_count=0,
                benchmark_results={},
                average_benchmark=0.0,
                config=self.config,
                success=False,
                error_message=str(e)
            )
            return results

    def _save_error_info(self, error: Exception, exception_stack: str):
        """Save error information for debugging.

        Creates a timestamped error file in the debug directory containing
        the error message, stack trace, model file path, and configuration.

        Args:
            error: The exception that occurred during training.
            exception_stack: The formatted stack trace string.
        """
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_file = Path(self.config.debug_dir) / f"training_error_{timestamp}.txt"
        with open(error_file, 'w') as f:
            f.write(f"Training Error: {str(error)}\n")
            f.write(f"Stack Trace: {exception_stack}\n")
            f.write(f"Model file: {self.config.model_file}\n")
            f.write(f"Configuration: {self.config}\n")
        logger.info(f"Error information saved to {error_file}")

    def _print_summary(self, results: TrainingResults):
        """Print training summary.

        Logs a formatted summary of training results including model name,
        parameter count, training time, final loss, average benchmark score,
        and output file paths.

        Args:
            results: TrainingResults object containing all training metrics.
        """
        logger.info("\n" + "="*50)
        logger.info("TRAINING SUMMARY")
        logger.info("="*50)
        logger.info(f"Model: {results.model_name}")
        logger.info(f"Parameters: {results.parameter_count:,}")
        logger.info(f"Training time: {results.training_time:.2f}s")
        logger.info(f"Final loss: {results.final_loss:.4f}")
        logger.info(f"Average benchmark: {results.average_benchmark:.4f}")
        if results.loss_file:
            logger.info(f"Loss file: {results.loss_file}")
        if results.benchmark_file:
            logger.info(f"Benchmark file: {results.benchmark_file}")
        logger.info("="*50)

    def get_config(self) -> TrainingConfig:
        """Get current configuration.

        Returns:
            TrainingConfig: The configuration object used to initialize
                this training pipeline.
        """
        return self.config

    def get_model(self) -> Optional[nn.Module]:
        """Get loaded model.

        Returns:
            Optional[nn.Module]: The loaded model if load_model has been called,
                otherwise None.
        """
        return self.model

    def get_dataset(self) -> Optional[Dataset]:
        """Get created dataset.

        Returns:
            Optional[Dataset]: The created dataset if create_dataset has been
                called, otherwise None.
        """
        return self.dataset
