"""
Data Loading Module for TruthPixel AI-Generated Image Detection.

This module handles loading the CIFAKE dataset from HuggingFace,
preprocessing images, and creating TensorFlow data pipelines.
"""

import os
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from typing import Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class CIFAKEDataLoader:
    """
    Data loader for CIFAKE dataset with preprocessing and augmentation.

    Attributes:
        dataset_name: HuggingFace dataset identifier
        img_size: Target image dimensions (height, width)
        batch_size: Number of images per batch
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
    """

    def __init__(
        self,
        dataset_name: str = "yanbax/CIFAKE_autotrain_compatible",
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize the data loader.

        Args:
            dataset_name: HuggingFace dataset identifier
            img_size: Target image dimensions
            batch_size: Batch size for training
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
        """
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Verify ratios sum to 1
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Train, val, and test ratios must sum to 1.0")

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def load_dataset(self) -> Dict:
        """
        Load CIFAKE dataset from HuggingFace.

        Returns:
            Dictionary containing the loaded dataset

        Raises:
            Exception: If dataset loading fails
        """
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            self.dataset = load_dataset(self.dataset_name)
            logger.info("Dataset loaded successfully")
            logger.info(f"Dataset structure: {self.dataset}")
            return self.dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a single image.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image normalized to [0, 1]
        """
        # Convert PIL Image to numpy array if needed
        if hasattr(image, 'convert'):
            image = image.convert('RGB')
            image = np.array(image)

        # Ensure RGB format
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # RGBA
            image = image[..., :3]

        # Resize to target size
        image = tf.image.resize(image, self.img_size)

        # Normalize to [0, 1]
        image = image / 255.0

        return image

    def create_tf_dataset(self, data: Dict) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from HuggingFace dataset.

        Args:
            data: HuggingFace dataset split

        Returns:
            TensorFlow dataset
        """
        # Extract images and labels
        images = []
        labels = []

        logger.info(f"Processing {len(data)} samples...")

        for i, item in enumerate(data):
            try:
                image = self.preprocess_image(item['image'])
                label = item['label']  # 0 = Real, 1 = AI-Generated

                images.append(image.numpy())
                labels.append(label)

                if (i + 1) % 10000 == 0:
                    logger.info(f"Processed {i + 1}/{len(data)} samples")
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        logger.info(f"Created dataset with {len(images)} samples")
        logger.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        return dataset

    def split_dataset(self) -> Tuple[Dict, Dict, Dict]:
        """
        Split dataset into train, validation, and test sets.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if self.dataset is None:
            self.load_dataset()

        # Get the train split from HuggingFace dataset
        # CIFAKE usually has 'train' split
        if 'train' in self.dataset:
            full_data = self.dataset['train']
        else:
            # If no train split, use the first available split
            full_data = list(self.dataset.values())[0]

        total_samples = len(full_data)
        logger.info(f"Total samples: {total_samples}")

        # Calculate split indices
        train_size = int(total_samples * self.train_ratio)
        val_size = int(total_samples * self.val_ratio)

        # Shuffle indices for random split
        indices = np.random.permutation(total_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        logger.info(f"Train samples: {len(train_indices)}")
        logger.info(f"Validation samples: {len(val_indices)}")
        logger.info(f"Test samples: {len(test_indices)}")

        # Create splits
        train_data = full_data.select(train_indices.tolist())
        val_data = full_data.select(val_indices.tolist())
        test_data = full_data.select(test_indices.tolist())

        return train_data, val_data, test_data

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
        Prepare train, validation, and test datasets with preprocessing.

        Args:
            augment_train: Whether to apply augmentation to training data

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Split dataset
        train_data, val_data, test_data = self.split_dataset()

        # Create TensorFlow datasets
        logger.info("Creating TensorFlow datasets...")
        self.train_ds = self.create_tf_dataset(train_data)
        self.val_ds = self.create_tf_dataset(val_data)
        self.test_ds = self.create_tf_dataset(test_data)

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

        logger.info("Datasets prepared successfully")

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
    data_loader = CIFAKEDataLoader(
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

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
    logger.info(f"Training: {train_dist}")

    # Display sample batch
    for images, labels in train_ds.take(1):
        logger.info(f"\nSample batch shape: {images.shape}")
        logger.info(f"Labels shape: {labels.shape}")
        logger.info(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")


if __name__ == "__main__":
    main()
