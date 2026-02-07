"""
Enhanced EfficientNet-B0 Model for AI-Generated Image Detection

Architecture:
- EfficientNet-B0 backbone (ImageNet pretrained)
- Deeper classification head with BatchNorm
- Optional domain classifier for Phase 4 training
- Uncertainty quantification support (MC Dropout)
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class EnhancedEfficientNet(nn.Module):
    """
    Enhanced EfficientNet-B0 for binary classification (Real vs AI-generated).

    Features:
    - Deeper classification head (1280 → 512 → 256 → 128 → 1)
    - BatchNorm layers for training stability
    - Increased dropout for regularization
    - Optional domain classifier branch
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        use_domain_classifier: bool = False,
        num_domains: int = 4
    ):
        """
        Args:
            num_classes: Number of output classes (1 for binary with BCELoss)
            pretrained: Whether to use ImageNet pretrained weights
            dropout_rate: Base dropout rate (will vary by layer)
            use_domain_classifier: Whether to include domain classifier
            num_domains: Number of datasets for domain classification
        """
        super(EnhancedEfficientNet, self).__init__()

        self.use_domain_classifier = use_domain_classifier

        # Load EfficientNet-B0 backbone from timm
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling (we'll add our own)
        )

        # EfficientNet-B0 outputs 1280 features
        backbone_out_features = 1280

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Enhanced classification head
        self.classifier = nn.Sequential(
            # Layer 1: 1280 → 512
            nn.Dropout(p=dropout_rate),
            nn.Linear(backbone_out_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # Layer 2: 512 → 256
            nn.Dropout(p=dropout_rate * 0.8),  # 0.4 if dropout_rate=0.5
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # Layer 3: 256 → 128
            nn.Dropout(p=dropout_rate * 0.6),  # 0.3 if dropout_rate=0.5
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            # Output layer: 128 → 1
            nn.Linear(128, num_classes)
        )

        # Optional domain classifier for domain-adversarial training (Phase 4)
        if use_domain_classifier:
            self.domain_classifier = nn.Sequential(
                nn.Linear(backbone_out_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_domains)
            )
        else:
            self.domain_classifier = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_domain: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input images [batch_size, 3, 224, 224]
            return_features: Whether to return backbone features
            return_domain: Whether to return domain predictions

        Returns:
            - If return_features=False, return_domain=False: class_logits
            - If return_features=True: (class_logits, features)
            - If return_domain=True: (class_logits, domain_logits)
            - If both: (class_logits, domain_logits, features)
        """
        # Extract features from backbone
        features = self.backbone(x)  # [batch_size, 1280, 7, 7]

        # Global average pooling
        pooled = self.global_pool(features)  # [batch_size, 1280, 1, 1]
        pooled = pooled.flatten(1)  # [batch_size, 1280]

        # Classification head
        class_logits = self.classifier(pooled)  # [batch_size, 1]

        # Prepare outputs based on flags
        outputs = [class_logits]

        if return_domain and self.domain_classifier is not None:
            domain_logits = self.domain_classifier(pooled)
            outputs.append(domain_logits)

        if return_features:
            outputs.append(pooled)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def freeze_backbone(self):
        """Freeze all backbone parameters (for Phase 1 & 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for Phase 3)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen")

    def unfreeze_last_n_layers(self, n: int = 20):
        """
        Unfreeze only the last n layers of the backbone (for Phase 3).

        Args:
            n: Number of layers to unfreeze from the end
        """
        # First freeze everything
        self.freeze_backbone()

        # Get all named parameters in backbone
        backbone_params = list(self.backbone.named_parameters())

        # Unfreeze last n layers
        for name, param in backbone_params[-n:]:
            param.requires_grad = True

        print(f"✓ Unfroze last {n} layers of backbone")

    def enable_mc_dropout(self):
        """
        Enable Monte Carlo Dropout for uncertainty quantification.
        Keeps dropout active during inference.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        print("✓ MC Dropout enabled")

    def get_num_params(self) -> dict:
        """Get number of parameters in model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def create_model(
    pretrained: bool = True,
    dropout_rate: float = 0.5,
    use_domain_classifier: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> EnhancedEfficientNet:
    """
    Create and initialize the enhanced EfficientNet-B0 model.

    Args:
        pretrained: Whether to use ImageNet pretrained weights
        dropout_rate: Dropout rate for regularization
        use_domain_classifier: Whether to include domain classifier
        device: Device to move model to

    Returns:
        Initialized model
    """
    model = EnhancedEfficientNet(
        num_classes=1,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        use_domain_classifier=use_domain_classifier,
        num_domains=4  # CIFAKE, Faces, Tristan, Shoes
    )

    model = model.to(device)

    # Print model info
    params = model.get_num_params()
    print("\n" + "=" * 80)
    print("MODEL CREATED")
    print("=" * 80)
    print(f"Architecture: Enhanced EfficientNet-B0")
    print(f"Device: {device}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    print(f"Domain classifier: {'Yes' if use_domain_classifier else 'No'}")

    # Estimate model size
    model_size_mb = params['total'] * 4 / (1024 ** 2)  # 4 bytes per float32
    print(f"Estimated model size: {model_size_mb:.1f} MB")
    print("=" * 80 + "\n")

    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing Enhanced EfficientNet-B0...")

    # Create model
    model = create_model(pretrained=True, use_domain_classifier=False)

    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    if torch.cuda.is_available():
        dummy_input = dummy_input.cuda()

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    # Test with domain classifier
    print("\n" + "=" * 80)
    print("Testing with domain classifier...")
    print("=" * 80)

    model_with_domain = create_model(
        pretrained=True,
        use_domain_classifier=True
    )

    with torch.no_grad():
        class_out, domain_out = model_with_domain(dummy_input, return_domain=True)

    print(f"Class output shape: {class_out.shape}")
    print(f"Domain output shape: {domain_out.shape}")

    # Test freezing/unfreezing
    print("\n" + "=" * 80)
    print("Testing freeze/unfreeze...")
    print("=" * 80)

    print("\nInitial trainable params:", model.get_num_params()['trainable'])

    model.freeze_backbone()
    print("After freeze:", model.get_num_params()['trainable'])

    model.unfreeze_last_n_layers(20)
    print("After unfreezing last 20 layers:", model.get_num_params()['trainable'])

    model.unfreeze_backbone()
    print("After full unfreeze:", model.get_num_params()['trainable'])

    print("\n✓ All model tests passed!")
