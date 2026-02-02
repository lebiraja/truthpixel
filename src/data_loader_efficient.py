"""
Memory-Efficient Data Loading for TruthPixel.

Uses TensorFlow's image_dataset_from_directory for lazy loading.
Only loads images into memory when needed (not all at once).
"""

import os
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)


class EfficientCIFAKEDataLoader:
    """
    Memory-efficient data loader using TensorFlow's lazy loading.

    Loads images on-demand instead of loading everything into memory.
    Perfect for systems with limited RAM/VRAM.
    """

    def __init__(
        self,
        data_dir: str = "data",
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 16  # Smaller default for 6GB VRAM
    ):
        """
        Initialize the efficient data loader.

        Args:
            data_dir: Root directory containing train/val/test folders
            img_size: Target image dimensions
            batch_size: Batch size for training (use 8-16 for 6GB VRAM)
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def get_augmentation_layers(self) -> tf.keras.Sequential:
        """
        Create data augmentation pipeline.

        Returns:
            Sequential model with augmentation layers
        """
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(
                factor=0.15,
                name='random_rotation'
            ),
            tf.keras.layers.RandomFlip(
                mode='horizontal',
                name='random_flip'
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.2, 0.2),
                width_factor=(-0.2, 0.2),
                name='random_zoom'
            ),
            tf.keras.layers.RandomBrightness(
                factor=0.2,
                name='random_brightness'
            ),
        ], name='data_augmentation')

        return augmentation

    def prepare_datasets(
        self,
        augment_train: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare train, validation, and test datasets using lazy loading.

        Args:
            augment_train: Whether to apply augmentation to training data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("=" * 80)
        logger.info("LOADING DATASETS (MEMORY-EFFICIENT MODE)")
        logger.info("=" * 80)

        # Verify directories exist
        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        for split_dir in [train_dir, val_dir, test_dir]:
            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Directory not found: {split_dir}\n"
                    f"Please run: python src/download_dataset.py --organize"
                )

        # Create datasets using TensorFlow's efficient loader
        logger.info(f"Loading from: {self.data_dir}")

        # Training set
        logger.info(f"Loading training data from {train_dir}")
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            str(train_dir),
            labels='inferred',
            label_mode='binary',
            class_names=['FAKE', 'REAL'],  # 0=FAKE, 1=REAL
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=True,
            seed=RANDOM_SEED,
            interpolation='bilinear'
        )

        # Validation set
        logger.info(f"Loading validation data from {val_dir}")
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            str(val_dir),
            labels='inferred',
            label_mode='binary',
            class_names=['FAKE', 'REAL'],
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=False,
            seed=RANDOM_SEED,
            interpolation='bilinear'
        )

        # Test set
        logger.info(f"Loading test data from {test_dir}")
        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            str(test_dir),
            labels='inferred',
            label_mode='binary',
            class_names=['FAKE', 'REAL'],
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,
            shuffle=False,
            seed=RANDOM_SEED,
            interpolation='bilinear'
        )

        # Keep images in [0, 255] range for EfficientNet
        # EfficientNet was trained on this range and doesn't need [0,1] normalization
        logger.info("Keeping images in [0, 255] range (EfficientNet requirement)")

        # No normalization needed - EfficientNet handles it internally

        # Apply augmentation to training data only
        if augment_train:
            logger.info("Applying data augmentation to training set")
            augmentation = self.get_augmentation_layers()

            def augment_and_clip(image, label):
                """Apply augmentation and clip values to [0, 255] range."""
                augmented = augmentation(image, training=True)
                clipped = tf.clip_by_value(augmented, 0.0, 255.0)
                return clipped, label

            self.train_ds = self.train_ds.map(
                augment_and_clip,
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Performance optimizations
        self.train_ds = self.train_ds.prefetch(tf.data.AUTOTUNE)
        self.val_ds = self.val_ds.prefetch(tf.data.AUTOTUNE)
        self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)

        # Keep original labels: FAKE=0, REAL=1 (natural ordering)
        logger.info("Labels: FAKE=0, REAL=1")

        logger.info("✓ Datasets prepared successfully (lazy loading enabled)")
        logger.info("  Images are loaded on-demand during training")
        logger.info("  Memory usage: LOW (only current batch in memory)")
        logger.info("=" * 80 + "\n")

        return self.train_ds, self.val_ds, self.test_ds

    def get_class_distribution(self, dataset: tf.data.Dataset) -> Dict[int, int]:
        """
        Calculate class distribution in a dataset.

        Args:
            dataset: TensorFlow dataset to analyze

        Returns:
            Dictionary mapping class labels to counts
        """
        class_counts = {0: 0, 1: 0}  # 0=Real, 1=Fake

        logger.info("Counting classes (this may take a moment)...")

        for _, labels_batch in dataset:
            # labels_batch shape is (batch_size, 1)
            for label in labels_batch.numpy().flatten():
                class_counts[int(label)] += 1

        return class_counts


def main():
    """
    Test the efficient data loader.
    """
    logger.info("Testing memory-efficient data loader...")

    data_loader = EfficientCIFAKEDataLoader(
        data_dir="data",
        batch_size=16
    )

    try:
        # Load datasets
        train_ds, val_ds, test_ds = data_loader.prepare_datasets(augment_train=True)

        # Test one batch
        logger.info("\nTesting batch loading...")
        for images, labels in train_ds.take(1):
            logger.info(f"✓ Batch shape: {images.shape}")
            logger.info(f"✓ Labels shape: {labels.shape}")
            logger.info(f"✓ Image range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
            logger.info(f"✓ Label range: [{tf.reduce_min(labels):.0f}, {tf.reduce_max(labels):.0f}]")

        logger.info("\n✓ Memory-efficient loading works!")
        logger.info("✓ Ready for training with low memory usage")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
