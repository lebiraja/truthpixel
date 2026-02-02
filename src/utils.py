"""
Utility functions for TruthPixel AI-Generated Image Detection.

This module provides helper functions for visualization, data validation,
and general utilities used across the project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def visualize_dataset_samples(
    dataset: tf.data.Dataset,
    num_samples: int = 16,
    class_names: List[str] = ['Real', 'AI-Generated'],
    save_path: Optional[str] = None
) -> None:
    """
    Visualize sample images from the dataset.

    Args:
        dataset: TensorFlow dataset to visualize
        num_samples: Number of samples to display
        class_names: Names of the classes
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 15))

    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # Get samples (take enough batches to get num_samples)
    images_shown = 0
    for images_batch, labels_batch in dataset.unbatch().take(num_samples):
        plt.subplot(grid_size, grid_size, images_shown + 1)

        # Clip values to [0, 1] for display (augmentation can push outside this range)
        display_image = tf.clip_by_value(images_batch, 0, 1)
        plt.imshow(display_image)

        # Extract scalar label (handle both scalar and array labels)
        if hasattr(labels_batch, 'numpy'):
            label_val = labels_batch.numpy()
        else:
            label_val = labels_batch

        if isinstance(label_val, (np.ndarray, list)):
            label_val = label_val.flatten()[0] if len(label_val) > 0 else 0

        plt.title(f"{class_names[int(label_val)]}")
        plt.axis('off')

        images_shown += 1

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
        plt.close()  # Close to save memory
    else:
        plt.close()  # Don't show interactively during training


def visualize_augmented_images(
    dataset: tf.data.Dataset,
    augmentation_layer: tf.keras.Sequential,
    num_augmentations: int = 5,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize the effect of data augmentation on sample images.

    Args:
        dataset: TensorFlow dataset
        augmentation_layer: Augmentation sequential model
        num_augmentations: Number of augmented versions per image
        save_path: Path to save the figure (optional)
    """
    # Get a single image
    for images, labels in dataset.take(1):
        original_image = images[0]
        label = labels[0]
        break

    plt.figure(figsize=(15, 3))

    # Show original image
    plt.subplot(1, num_augmentations + 1, 1)
    plt.imshow(original_image)
    plt.title('Original')
    plt.axis('off')

    # Show augmented versions
    for i in range(num_augmentations):
        augmented_image = augmentation_layer(
            tf.expand_dims(original_image, 0),
            training=True
        )[0]

        plt.subplot(1, num_augmentations + 1, i + 2)
        plt.imshow(augmented_image)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved augmentation visualization to {save_path}")

    plt.show()


def plot_class_distribution(
    class_counts: Dict[int, int],
    class_names: List[str] = ['Real', 'AI-Generated'],
    title: str = 'Class Distribution',
    save_path: Optional[str] = None
) -> None:
    """
    Plot class distribution as a bar chart.

    Args:
        class_counts: Dictionary mapping class indices to counts
        class_names: Names of the classes
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(8, 6))

    labels = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    colors = ['#2ecc71', '#e74c3c']  # Green for Real, Red for AI
    bars = plt.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved class distribution to {save_path}")

    plt.show()


def verify_dataset_balance(
    class_counts: Dict[int, int],
    threshold: float = 0.05
) -> bool:
    """
    Verify if dataset is balanced within a threshold.

    Args:
        class_counts: Dictionary mapping class indices to counts
        threshold: Maximum allowed imbalance ratio (default 5%)

    Returns:
        True if balanced, False otherwise
    """
    counts = list(class_counts.values())
    total = sum(counts)

    imbalances = []
    for count in counts:
        expected_ratio = 1.0 / len(counts)
        actual_ratio = count / total
        imbalance = abs(actual_ratio - expected_ratio)
        imbalances.append(imbalance)

    max_imbalance = max(imbalances)

    is_balanced = max_imbalance <= threshold

    if is_balanced:
        logger.info(f"✓ Dataset is balanced (max imbalance: {max_imbalance:.2%})")
    else:
        logger.warning(f"✗ Dataset is imbalanced (max imbalance: {max_imbalance:.2%})")

    return is_balanced


def calculate_dataset_statistics(
    dataset: tf.data.Dataset
) -> Dict[str, float]:
    """
    Calculate dataset statistics (mean, std, min, max).

    Args:
        dataset: TensorFlow dataset

    Returns:
        Dictionary with statistics
    """
    all_images = []

    logger.info("Calculating dataset statistics...")

    for images, _ in dataset:
        all_images.append(images.numpy())

    all_images = np.concatenate(all_images, axis=0)

    stats = {
        'mean': np.mean(all_images),
        'std': np.std(all_images),
        'min': np.min(all_images),
        'max': np.max(all_images),
        'shape': all_images.shape
    }

    logger.info(f"Dataset Statistics:")
    logger.info(f"  Mean: {stats['mean']:.4f}")
    logger.info(f"  Std:  {stats['std']:.4f}")
    logger.info(f"  Min:  {stats['min']:.4f}")
    logger.info(f"  Max:  {stats['max']:.4f}")
    logger.info(f"  Shape: {stats['shape']}")

    return stats


def check_gpu_availability() -> None:
    """
    Check and display GPU availability for TensorFlow.
    """
    logger.info("=" * 60)
    logger.info("GPU Availability Check")
    logger.info("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        logger.info(f"✓ {len(gpus)} GPU(s) available:")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")

        # Display GPU details
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("✓ Memory growth enabled for GPUs")
        except RuntimeError as e:
            logger.warning(f"Could not set memory growth: {e}")
    else:
        logger.warning("✗ No GPU available - using CPU")
        logger.warning("  Training will be significantly slower on CPU")

    logger.info(f"\nTensorFlow version: {tf.__version__}")
    logger.info("=" * 60)


def create_results_directories() -> None:
    """
    Create necessary directories for saving results.
    """
    directories = [
        'results/plots',
        'results/gradcam_visualizations',
        'models',
        'data/train',
        'data/val',
        'data/test'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def save_dataset_info(
    train_counts: Dict[int, int],
    val_counts: Dict[int, int],
    test_counts: Dict[int, int],
    save_path: str = 'results/dataset_info.txt'
) -> None:
    """
    Save dataset information to a text file.

    Args:
        train_counts: Training set class counts
        val_counts: Validation set class counts
        test_counts: Test set class counts
        save_path: Path to save the info file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TruthPixel Dataset Information\n")
        f.write("=" * 60 + "\n\n")

        f.write("Training Set:\n")
        f.write(f"  Real: {train_counts.get(0, 0):,}\n")
        f.write(f"  AI-Generated: {train_counts.get(1, 0):,}\n")
        f.write(f"  Total: {sum(train_counts.values()):,}\n\n")

        f.write("Validation Set:\n")
        f.write(f"  Real: {val_counts.get(0, 0):,}\n")
        f.write(f"  AI-Generated: {val_counts.get(1, 0):,}\n")
        f.write(f"  Total: {sum(val_counts.values()):,}\n\n")

        f.write("Test Set:\n")
        f.write(f"  Real: {test_counts.get(0, 0):,}\n")
        f.write(f"  AI-Generated: {test_counts.get(1, 0):,}\n")
        f.write(f"  Total: {sum(test_counts.values()):,}\n\n")

        total_samples = sum(train_counts.values()) + sum(val_counts.values()) + sum(test_counts.values())
        f.write(f"Total Samples: {total_samples:,}\n")
        f.write("=" * 60 + "\n")

    logger.info(f"Saved dataset info to {save_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"✓ Random seeds set to {seed}")


def get_callbacks_list(
    model_save_path: str = 'models/best_model.h5',
    csv_log_path: str = 'results/training_history.csv',
    patience_early_stop: int = 5,
    patience_reduce_lr: int = 3
) -> List[tf.keras.callbacks.Callback]:
    """
    Get list of training callbacks.

    Args:
        model_save_path: Path to save best model
        csv_log_path: Path to save training log
        patience_early_stop: Patience for early stopping
        patience_reduce_lr: Patience for learning rate reduction

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),

        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=1e-7,
            verbose=1
        ),

        # CSV logger
        tf.keras.callbacks.CSVLogger(
            filename=csv_log_path,
            separator=',',
            append=False
        )
    ]

    return callbacks


if __name__ == "__main__":
    # Test GPU availability
    check_gpu_availability()

    # Create directories
    create_results_directories()

    # Set random seeds
    set_random_seeds(42)
