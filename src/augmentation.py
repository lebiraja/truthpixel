"""
Advanced Data Augmentation Pipeline for AI-Generated Image Detection

Implements progressive augmentation strategies for different training phases:
- Phase 1: Basic augmentation (CIFAKE baseline)
- Phase 2+: Enhanced augmentation (multi-dataset training)

Includes JPEG compression simulation, Gaussian blur, and other techniques
to improve model robustness.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter
import random
import io
import numpy as np


class JPEGCompression:
    """Simulate JPEG compression artifacts."""

    def __init__(self, quality_range=(75, 95)):
        """
        Args:
            quality_range: Tuple of (min_quality, max_quality) for JPEG compression
        """
        self.quality_range = quality_range

    def __call__(self, img):
        """
        Apply random JPEG compression to image.

        Args:
            img: PIL Image

        Returns:
            Compressed PIL Image
        """
        # Random quality in range
        quality = random.randint(self.quality_range[0], self.quality_range[1])

        # Compress to bytes and decompress
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')


class AdaptiveGaussianBlur:
    """
    Gaussian blur with adaptive kernel size and sigma.
    More realistic than fixed blur.
    """

    def __init__(self, kernel_size=3, sigma_range=(0.1, 2.0), p=0.5):
        """
        Args:
            kernel_size: Kernel size for Gaussian blur
            sigma_range: Range for random sigma selection
            p: Probability of applying blur
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, img):
        """Apply random Gaussian blur."""
        if random.random() < self.p:
            sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


def get_augmentation_transforms(phase: int = 1, img_size: int = 224):
    """
    Get augmentation transforms for specific training phase.

    Args:
        phase: Training phase (1, 2, or 3)
        img_size: Target image size (224 for EfficientNet)

    Returns:
        torchvision.transforms.Compose
    """
    # ImageNet normalization (EfficientNet standard)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 1:
        # Phase 1: Basic augmentation for CIFAKE baseline
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    elif phase >= 2:
        # Phase 2 & 3: Enhanced augmentation for multi-dataset training
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),

            # Basic geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),

            # Color augmentations
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            ),

            # Cropping
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),

            # Advanced augmentations
            AdaptiveGaussianBlur(kernel_size=3, sigma_range=(0.1, 2.0), p=0.3),
            JPEGCompression(quality_range=(75, 95)),

            # Convert to tensor
            transforms.ToTensor(),

            # Random erasing (after tensor conversion)
            transforms.RandomErasing(
                p=0.1,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3)
            ),

            # Normalization
            transforms.Normalize(mean=mean, std=std)
        ])


def get_validation_transforms(img_size: int = 224):
    """
    Get transforms for validation/test (no augmentation).

    Args:
        img_size: Target image size

    Returns:
        torchvision.transforms.Compose
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


class MixUp:
    """
    MixUp data augmentation (optional for Phase 3).

    Reference: https://arxiv.org/abs/1710.09412
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter for mixup
        """
        self.alpha = alpha

    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Apply MixUp to a batch.

        Args:
            images: Batch of images [batch_size, 3, H, W]
            labels: Batch of labels [batch_size]

        Returns:
            Mixed images and labels
        """
        batch_size = images.size(0)

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Random permutation
        index = torch.randperm(batch_size).to(images.device)

        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]

        # Mix labels
        if labels.dim() == 1:
            # Binary labels
            labels_a = labels
            labels_b = labels[index]
            mixed_labels = lam * labels_a.float() + (1 - lam) * labels_b.float()
        else:
            # One-hot labels
            mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_images, mixed_labels, lam


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing Augmentation Pipeline...")
    print("=" * 80)

    # Create dummy image
    dummy_img = Image.new('RGB', (256, 256), color='red')

    # Test Phase 1 augmentation
    print("\nPhase 1 Augmentation:")
    phase1_transform = get_augmentation_transforms(phase=1, img_size=224)
    aug_tensor = phase1_transform(dummy_img)
    print(f"  Output shape: {aug_tensor.shape}")
    print(f"  Output range: [{aug_tensor.min():.3f}, {aug_tensor.max():.3f}]")

    # Test Phase 2 augmentation
    print("\nPhase 2 Augmentation:")
    phase2_transform = get_augmentation_transforms(phase=2, img_size=224)
    aug_tensor = phase2_transform(dummy_img)
    print(f"  Output shape: {aug_tensor.shape}")
    print(f"  Output range: [{aug_tensor.min():.3f}, {aug_tensor.max():.3f}]")

    # Test validation transforms
    print("\nValidation Transforms (no augmentation):")
    val_transform = get_validation_transforms(img_size=224)
    val_tensor = val_transform(dummy_img)
    print(f"  Output shape: {val_tensor.shape}")
    print(f"  Output range: [{val_tensor.min():.3f}, {val_tensor.max():.3f}]")

    # Test JPEG compression
    print("\nJPEG Compression Test:")
    jpeg_aug = JPEGCompression(quality_range=(75, 95))
    for i in range(3):
        compressed = jpeg_aug(dummy_img)
        print(f"  Iteration {i+1}: Size={compressed.size}, Mode={compressed.mode}")

    # Test MixUp
    print("\nMixUp Test:")
    mixup = MixUp(alpha=0.2)
    batch_images = torch.randn(4, 3, 224, 224)
    batch_labels = torch.tensor([0, 1, 0, 1])

    mixed_images, mixed_labels, lam = mixup(batch_images, batch_labels)
    print(f"  Original images shape: {batch_images.shape}")
    print(f"  Mixed images shape: {mixed_images.shape}")
    print(f"  Original labels: {batch_labels.tolist()}")
    print(f"  Mixed labels: {mixed_labels.tolist()}")
    print(f"  Lambda: {lam:.3f}")

    print("\n" + "=" * 80)
    print("✓ All augmentation tests passed!")
