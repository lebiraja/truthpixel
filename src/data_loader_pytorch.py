"""
PyTorch Data Loading for TruthPixel.
Cleaner, faster, and more control than TensorFlow.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CIFAKEDataLoaderPyTorch:
    """
    PyTorch data loader for CIFAKE dataset.

    Uses ImageFolder for automatic class detection.
    Expects structure:
        data/train/FAKE/*.png
        data/train/REAL/*.png
        (same for val and test)
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 16,
        num_workers: int = 4,
        img_size: int = 224
    ):
        """
        Initialize PyTorch data loader.

        Args:
            data_dir: Root directory with train/val/test folders
            batch_size: Batch size
            num_workers: Number of workers for data loading
            img_size: Image size (EfficientNet uses 224)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        # EfficientNet ImageNet preprocessing
        # These are the EXACT values EfficientNet was trained with
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def get_transforms(self, augment: bool = False):
        """
        Get image transforms.

        Args:
            augment: Whether to apply data augmentation

        Returns:
            torchvision transforms
        """
        if augment:
            # Training transforms with augmentation
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            # Validation/test transforms (no augmentation)
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])

    def prepare_loaders(self):
        """
        Prepare train, val, and test data loaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("=" * 80)
        logger.info("LOADING DATASETS (PYTORCH)")
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

        # Create datasets
        logger.info(f"Loading from: {self.data_dir}")

        train_dataset = datasets.ImageFolder(
            root=str(train_dir),
            transform=self.get_transforms(augment=True)
        )

        val_dataset = datasets.ImageFolder(
            root=str(val_dir),
            transform=self.get_transforms(augment=False)
        )

        test_dataset = datasets.ImageFolder(
            root=str(test_dir),
            transform=self.get_transforms(augment=False)
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        logger.info(f"Classes: {train_dataset.classes}")
        logger.info(f"Class to index: {train_dataset.class_to_idx}")

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # Faster GPU transfer
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        logger.info("✓ Data loaders created successfully")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        logger.info("=" * 80 + "\n")

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    logger.info("Testing PyTorch data loader...")

    data_loader = CIFAKEDataLoaderPyTorch(
        data_dir="data",
        batch_size=16,
        num_workers=2
    )

    train_loader, val_loader, test_loader = data_loader.prepare_loaders()

    # Test one batch
    images, labels = next(iter(train_loader))
    logger.info(f"\n✓ Batch loaded:")
    logger.info(f"  Images shape: {images.shape}")
    logger.info(f"  Labels shape: {labels.shape}")
    logger.info(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
    logger.info(f"  Unique labels: {labels.unique().tolist()}")

    logger.info("\n✓ PyTorch data loading works!")
