"""
PyTorch Model for TruthPixel AI-Generated Image Detection.

Uses EfficientNet-B0 from torchvision with transfer learning.
"""

import torch
import torch.nn as nn
from torchvision import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TruthPixelModelPyTorch(nn.Module):
    """
    EfficientNet-B0 based model for binary classification.

    Architecture:
        EfficientNet-B0 (ImageNet pretrained)
        ↓
        Adaptive Average Pooling
        ↓
        Dropout(0.3)
        ↓
        Linear(1280 → 256) + ReLU + Dropout(0.5)
        ↓
        Linear(256 → 128) + ReLU + Dropout(0.3)
        ↓
        Linear(128 → 1) + Sigmoid
    """

    def __init__(self, freeze_base: bool = True, dropout_rate: float = 0.3):
        """
        Initialize model.

        Args:
            freeze_base: Whether to freeze EfficientNet base layers
            dropout_rate: Dropout rate for regularization
        """
        super(TruthPixelModelPyTorch, self).__init__()

        # Load pretrained EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            logger.info("✓ Base model frozen (transfer learning mode)")
        else:
            logger.info("✓ Base model unfrozen (fine-tuning mode)")

        # Get the number of features from EfficientNet
        # EfficientNet-B0 outputs 1280 features
        num_features = self.base_model.classifier[1].in_features

        # Replace the classifier
        self.base_model.classifier = nn.Identity()  # Remove original classifier

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),

            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        logger.info("✓ Model architecture built successfully")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, 3, 224, 224)

        Returns:
            Output tensor (batch_size, 1) with values in [0, 1]
        """
        # Extract features with EfficientNet
        features = self.base_model(x)

        # Pass through classifier
        output = self.classifier(features)

        return output

    def unfreeze_base_layers(self, num_layers: int = 20):
        """
        Unfreeze the last N layers of EfficientNet for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the end
        """
        # Get all layers
        all_layers = list(self.base_model.children())

        # Unfreeze last num_layers
        for layer in all_layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        logger.info(f"✓ Unfrozen last {num_layers} layers for fine-tuning")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        logger.info(f"  Trainable params: {trainable_params:,}")
        logger.info(f"  Total params: {total_params:,}")


def create_model(freeze_base: bool = True, device: str = 'cuda'):
    """
    Create and initialize the model.

    Args:
        freeze_base: Whether to freeze base layers
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Model instance
    """
    model = TruthPixelModelPyTorch(freeze_base=freeze_base)
    model = model.to(device)

    logger.info(f"✓ Model moved to {device}")

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info("\n" + "=" * 80)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("=" * 80 + "\n")

    return model


if __name__ == "__main__":
    # Test model creation
    logger.info("Testing PyTorch model creation...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create model
    model = create_model(freeze_base=True, device=device)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)

    logger.info(f"\n✓ Forward pass test:")
    logger.info(f"  Input shape: {dummy_input.shape}")
    logger.info(f"  Output shape: {output.shape}")
    logger.info(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

    logger.info("\n✓ Model creation successful!")
