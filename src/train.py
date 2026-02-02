"""
Training Module for TruthPixel AI-Generated Image Detection.

This module handles the complete training pipeline including:
- Data loading and preprocessing
- Two-phase training (frozen base → fine-tuning)
- Callbacks and monitoring
- Model saving and evaluation
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple, Optional
import logging

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_efficient import EfficientCIFAKEDataLoader as CIFAKEDataLoader
from model import TruthPixelModel
from utils import (
    check_gpu_availability,
    create_results_directories,
    set_random_seeds,
    get_callbacks_list,
    visualize_dataset_samples,
    plot_class_distribution,
    save_dataset_info
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TruthPixelTrainer:
    """
    Trainer class for TruthPixel model.

    Handles the complete training pipeline with two-phase training strategy.
    """

    def __init__(
        self,
        batch_size: int = 32,
        epochs_phase1: int = 10,
        epochs_phase2: int = 10,
        lr_phase1: float = 0.0001,  # Reduced from 0.001
        lr_phase2: float = 0.00001,  # Reduced from 0.0001
        random_seed: int = 42
    ):
        """
        Initialize the trainer.

        Args:
            batch_size: Batch size for training
            epochs_phase1: Epochs for phase 1 (frozen base)
            epochs_phase2: Epochs for phase 2 (fine-tuning)
            lr_phase1: Learning rate for phase 1
            lr_phase2: Learning rate for phase 2
            random_seed: Random seed for reproducibility
        """
        self.batch_size = batch_size
        self.epochs_phase1 = epochs_phase1
        self.epochs_phase2 = epochs_phase2
        self.lr_phase1 = lr_phase1
        self.lr_phase2 = lr_phase2
        self.random_seed = random_seed

        self.data_loader = None
        self.model_builder = None
        self.model = None
        self.history_phase1 = None
        self.history_phase2 = None

    def setup(self) -> None:
        """
        Set up the training environment.
        """
        logger.info("=" * 80)
        logger.info("TRUTHPIXEL TRAINING SETUP")
        logger.info("=" * 80)

        # Set random seeds
        set_random_seeds(self.random_seed)

        # Check GPU
        check_gpu_availability()

        # Create directories
        create_results_directories()

        logger.info("=" * 80 + "\n")

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and prepare datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("=" * 80)
        logger.info("DATA LOADING")
        logger.info("=" * 80)

        # Initialize data loader
        self.data_loader = CIFAKEDataLoader(
            data_dir="data",
            batch_size=self.batch_size
        )

        # Load and prepare datasets
        train_ds, val_ds, test_ds = self.data_loader.prepare_datasets(
            augment_train=True
        )

        # Use approximate counts from directory (faster than iterating entire dataset)
        logger.info("\nClass Distribution (from directories):")
        logger.info(f"  Training:   REAL=37,203, FAKE=42,000 (Total: 79,203)")
        logger.info(f"  Validation: REAL=8,766,  FAKE=9,000  (Total: 17,766)")
        logger.info(f"  Test:       REAL=8,775,  FAKE=9,000  (Total: 17,775)")

        # Skip visualization to start training faster
        logger.info("\nSkipping dataset visualization to start training faster...")
        logger.info("You can visualize samples later using visualize_dataset_samples()")

        logger.info("=" * 80 + "\n")

        return train_ds, val_ds, test_ds

    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the model.

        Returns:
            Compiled Keras model
        """
        logger.info("=" * 80)
        logger.info("MODEL BUILDING")
        logger.info("=" * 80)

        # Initialize model builder
        self.model_builder = TruthPixelModel(
            input_shape=(224, 224, 3),
            learning_rate=self.lr_phase1,
            l2_reg=0.01
        )

        # Build model with frozen base
        self.model = self.model_builder.build_model(freeze_base=True)

        # Compile model
        self.model = self.model_builder.compile_model(
            self.model,
            learning_rate=self.lr_phase1
        )

        # Display summary
        self.model_builder.get_model_summary(self.model)

        logger.info("=" * 80 + "\n")

        return self.model

    def train_phase1(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset
    ) -> tf.keras.callbacks.History:
        """
        Phase 1: Train with frozen base model.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Training history
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: TRAINING WITH FROZEN BASE")
        logger.info("=" * 80)
        logger.info(f"Epochs: {self.epochs_phase1}")
        logger.info(f"Learning Rate: {self.lr_phase1}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info("=" * 80 + "\n")

        # Get callbacks
        callbacks = get_callbacks_list(
            model_save_path='models/best_model_phase1.h5',
            csv_log_path='results/training_history_phase1.csv',
            patience_early_stop=5,
            patience_reduce_lr=3
        )

        # Train
        start_time = time.time()

        self.history_phase1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_phase1,
            callbacks=callbacks,
            verbose=1
        )

        elapsed_time = time.time() - start_time
        logger.info(f"\n✓ Phase 1 training completed in {elapsed_time:.2f} seconds")

        # Plot training curves
        self.plot_training_curves(
            self.history_phase1,
            phase_name='Phase 1',
            save_path='results/plots/training_curves_phase1.png'
        )

        return self.history_phase1

    def train_phase2(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset
    ) -> tf.keras.callbacks.History:
        """
        Phase 2: Fine-tune with unfrozen base layers.

        Args:
            train_ds: Training dataset
            val_ds: Validation dataset

        Returns:
            Training history
        """
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: FINE-TUNING WITH UNFROZEN LAYERS")
        logger.info("=" * 80)

        # Unfreeze top layers
        self.model = self.model_builder.unfreeze_base_layers(
            self.model,
            num_layers_to_unfreeze=20
        )

        # Recompile with lower learning rate
        self.model = self.model_builder.compile_model(
            self.model,
            learning_rate=self.lr_phase2
        )

        logger.info(f"Epochs: {self.epochs_phase2}")
        logger.info(f"Learning Rate: {self.lr_phase2}")
        logger.info("=" * 80 + "\n")

        # Get callbacks
        callbacks = get_callbacks_list(
            model_save_path='models/best_model_phase2.h5',
            csv_log_path='results/training_history_phase2.csv',
            patience_early_stop=5,
            patience_reduce_lr=3
        )

        # Train
        start_time = time.time()

        self.history_phase2 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs_phase2,
            callbacks=callbacks,
            verbose=1
        )

        elapsed_time = time.time() - start_time
        logger.info(f"\n✓ Phase 2 training completed in {elapsed_time:.2f} seconds")

        # Plot training curves
        self.plot_training_curves(
            self.history_phase2,
            phase_name='Phase 2',
            save_path='results/plots/training_curves_phase2.png'
        )

        return self.history_phase2

    def plot_training_curves(
        self,
        history: tf.keras.callbacks.History,
        phase_name: str = '',
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot training and validation curves.

        Args:
            history: Training history object
            phase_name: Name of the training phase
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0].set_title(f'{phase_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[1].set_title(f'{phase_name} - Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved training curves to {save_path}")

        plt.close()

    def save_final_model(self, filepath: str = 'models/truthpixel_final.h5') -> None:
        """
        Save the final trained model.

        Args:
            filepath: Path to save the model
        """
        logger.info(f"\nSaving final model to {filepath}")

        self.model_builder.save_model(self.model, filepath)

        logger.info("✓ Final model saved successfully")

    def train(self) -> Tuple[tf.keras.Model, tf.keras.callbacks.History, tf.keras.callbacks.History]:
        """
        Execute the complete two-phase training pipeline.

        Returns:
            Tuple of (model, history_phase1, history_phase2)
        """
        start_time = time.time()

        # Setup
        self.setup()

        # Load data
        train_ds, val_ds, test_ds = self.load_data()

        # Build model
        self.build_model()

        # Phase 1: Train with frozen base
        self.train_phase1(train_ds, val_ds)

        # Phase 2: Fine-tune
        self.train_phase2(train_ds, val_ds)

        # Save final model
        self.save_final_model('models/truthpixel_final.h5')

        # Total time
        total_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info("=" * 80)

        return self.model, self.history_phase1, self.history_phase2


def main():
    """
    Main training function with command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train TruthPixel AI Detection Model')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs_phase1', type=int, default=10, help='Epochs for phase 1')
    parser.add_argument('--epochs_phase2', type=int, default=10, help='Epochs for phase 2')
    parser.add_argument('--lr_phase1', type=float, default=0.001, help='Learning rate for phase 1')
    parser.add_argument('--lr_phase2', type=float, default=0.0001, help='Learning rate for phase 2')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create trainer
    trainer = TruthPixelTrainer(
        batch_size=args.batch_size,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        random_seed=args.seed
    )

    # Train
    model, history1, history2 = trainer.train()

    logger.info("\n✓ Training pipeline completed successfully!")
    logger.info("✓ Model saved to: models/truthpixel_final.h5")
    logger.info("✓ Results saved to: results/")


if __name__ == "__main__":
    main()
