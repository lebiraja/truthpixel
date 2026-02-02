"""
Local Data Loading Module for TruthPixel AI-Generated Image Detection.

This module handles loading CIFAKE dataset from local directories.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple, Dict
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class LocalCIFAKEDataLoader:
    """
    Data loader for CIFAKE dataset from local directories.

    Expects structure:
        data/train/REAL/*.png
        data/train/FAKE/*.png
        data/val/REAL/*.png
        data/val/FAKE/*.png
        data/test/REAL/*.png
        data/test/FAKE/*.png
    """

    def __init__(
        self,
        data_dir: str = "data",
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32
    ):
        """
        Initialize the local data loader.

        Args:
            data_dir: Root directory containing train/val/test folders
            img_size: Target image dimensions
            batch_size: Batch size for training
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_images_from_directory(
        self,
        directory: Path,
        label: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images from a directory with a specific label.

        Args:
            directory: Path to image directory
            label: Class label (0 or 1)

        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []

        # Find all image files
        image_files = list(directory.glob("*.png")) + list(directory.glob("*.jpg")) + list(directory.glob("*.jpeg"))

        logger.info(f"Loading {len(image_files)} images from {directory}")

        for img_path in image_files:
            try:
                # Load image
                img = Image.open(img_path)

                # Convert to RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize
                img = img.resize(self.img_size)

                # Convert to array and normalize
                img_array = np.array(img) / 255.0

                images.append(img_array)
                labels.append(label)

            except Exception as e:
                logger.warning(f"Failed to load {img_path}: {e}")
                continue

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a complete dataset split (train/val/test).

        Args:
            split: Split name ('train', 'val', or 'test')

        Returns:
            Tuple of (all_images, all_labels)
        """
        split_dir = self.data_dir / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Please download and organize the dataset first.\n"
                f"Run: python src/download_dataset.py"
            )

        # Load REAL images (label = 0)
        real_dir = split_dir / "REAL"
        real_images, real_labels = self.load_images_from_directory(real_dir, label=0)

        # Load FAKE images (label = 1)
        fake_dir = split_dir / "FAKE"
        fake_images, fake_labels = self.load_images_from_directory(fake_dir, label=1)

        # Combine
        all_images = np.concatenate([real_images, fake_images], axis=0)
        all_labels = np.concatenate([real_labels, fake_labels], axis=0)

        # Shuffle
        indices = np.random.permutation(len(all_images))
        all_images = all_images[indices]
        all_labels = all_labels[indices]

        logger.info(f"{split} split: {len(all_images)} images (REAL: {len(real_images)}, FAKE: {len(fake_images)})")

        return all_images, all_labels

    def get_augmentation_layers(self) -> tf.keras.Sequential:
        """
        Create data augmentation pipeline using Keras preprocessing layers.

        Returns:
            Sequential model with augmentation layers
        """
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(
                factor=0.15,  # ±15 degrees
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
                factor=0.2,  # ±20% brightness
                name='random_brightness'
            ),
        ], name='data_augmentation')

        return augmentation

    def prepare_datasets(
        self,
        augment_train: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare train, validation, and test datasets.

        Args:
            augment_train: Whether to apply augmentation to training data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        logger.info("=" * 80)
        logger.info("LOADING DATASETS FROM LOCAL DIRECTORIES")
        logger.info("=" * 80)

        # Load all splits
        train_images, train_labels = self.load_split('train')
        val_images, val_labels = self.load_split('val')
        test_images, test_labels = self.load_split('test')

        # Create TensorFlow datasets
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        self.test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

        # Apply augmentation to training data only
        if augment_train:
            logger.info("Applying data augmentation to training set")
            augmentation = self.get_augmentation_layers()
            self.train_ds = self.train_ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )

        # Batch and prefetch for performance
        self.train_ds = (
            self.train_ds
            .shuffle(buffer_size=1000, seed=RANDOM_SEED)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.val_ds = (
            self.val_ds
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        self.test_ds = (
            self.test_ds
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        logger.info("✓ Datasets prepared successfully")
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
        class_counts = {0: 0, 1: 0}  # 0=Real, 1=AI-Generated

        for _, labels in dataset.unbatch():
            label = int(labels.numpy())
            class_counts[label] += 1

        return class_counts


def main():
    """
    Main function to test data loading.
    """
    # Initialize data loader
    data_loader = LocalCIFAKEDataLoader(
        data_dir="data",
        batch_size=32
    )

    try:
        # Load and prepare datasets
        train_ds, val_ds, test_ds = data_loader.prepare_datasets(augment_train=True)

        # Display dataset information
        logger.info("\nDataset Information:")
        logger.info(f"Train batches: {len(list(train_ds))}")
        logger.info(f"Validation batches: {len(list(val_ds))}")
        logger.info(f"Test batches: {len(list(test_ds))}")

        # Check class distribution
        logger.info("\nClass Distribution:")
        train_dist = data_loader.get_class_distribution(train_ds)
        logger.info(f"Training: REAL={train_dist.get(0, 0):,}, FAKE={train_dist.get(1, 0):,}")

        # Display sample batch
        for images, labels in train_ds.take(1):
            logger.info(f"\nSample batch shape: {images.shape}")
            logger.info(f"Labels shape: {labels.shape}")
            logger.info(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")

        logger.info("\n✓ Data loading test successful!")

    except FileNotFoundError as e:
        logger.error(f"\n{e}")
        logger.info("\nPlease download the dataset first:")
        logger.info("Run: python src/download_dataset.py")


if __name__ == "__main__":
    main()
