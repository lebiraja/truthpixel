"""
Baseline Model Training Script - Phase 1

Trains separate models for each dataset (GenImage, CIFAKE, Faces).
This establishes baseline performance for each dataset individually.

Usage:
    python src/train_baseline.py --dataset genimage
    python src/train_baseline.py --dataset cifake
    python src/train_baseline.py --dataset faces
"""

import argparse
import sys
import json
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import timm
from tqdm import tqdm

from data_loader_multi import MultiDatasetLoader


class DeepfakeDetector(nn.Module):
    """
    EfficientNetB0-based deepfake detector.

    Architecture:
    - EfficientNetB0 backbone (pretrained on ImageNet)
    - GlobalAveragePooling
    - Dense(512, relu, L2 regularization)
    - Dropout(0.5)
    - Dense(256, relu, L2 regularization)
    - Dropout(0.3)
    - Dense(1, sigmoid)
    """

    def __init__(self, pretrained=True, dropout=0.5):
        super(DeepfakeDetector, self).__init__()

        # EfficientNetB0 backbone
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

        # Custom classification head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.pool(features)
        features = self.flatten(features)

        # Classify
        output = self.classifier(features)
        return output


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels, _) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })

        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, epoch=0, writer=None, prefix="Val"):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{prefix}]")

        for images, labels, _ in pbar:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'loss': f"{running_loss/len(loader):.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def train_single_dataset_model(dataset_name: str, config: dict):
    """
    Train a model on a single dataset (baseline).

    Args:
        dataset_name: 'genimage', 'cifake', or 'faces'
        config: Configuration dictionary

    Returns:
        Best validation accuracy
    """
    print("="*80)
    print(f"TRAINING BASELINE MODEL: {dataset_name.upper()}")
    print("="*80)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create data loaders
    print("\nLoading data...")
    data_loader = MultiDatasetLoader(
        data_dir=config['data']['base_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )

    train_loader, val_loader, test_loader = data_loader.create_single_dataset_loader(
        dataset_name=dataset_name,
        phase=1
    )

    # Create model
    print("\nCreating model...")
    model = DeepfakeDetector(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    ).to(device)

    # Freeze backbone initially
    if config['training']['phase1_baseline']['freeze_backbone']:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen (transfer learning)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase1_baseline']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['callbacks']['reduce_lr']['factor'],
        patience=config['callbacks']['reduce_lr']['patience'],
        verbose=True
    )

    # Setup directories
    model_dir = Path(config['paths']['models']) / 'baseline'
    checkpoint_dir = Path(config['paths']['checkpoints']) / 'baseline' / dataset_name
    log_dir = Path(config['logging']['tensorboard_dir']) / 'baseline' / dataset_name

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    num_epochs = config['training']['phase1_baseline']['epochs']
    save_freq = config['callbacks']['model_checkpoint']['save_freq']
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = config['callbacks']['early_stopping']['patience']

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer, "Val"
        )

        # Update scheduler
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Log to tensorboard
        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            best_model_path = model_dir / f"{dataset_name}_baseline_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, best_model_path)

            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Save checkpoint every N epochs
        if epoch % save_freq == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history,
                'config': config
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved (epoch {epoch})")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered (patience={early_stop_patience})")
            break

    # Save final model
    final_model_path = model_dir / f"{dataset_name}_baseline.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'history': history,
        'config': config
    }, final_model_path)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {final_model_path}")

    # Test on test set if available
    if test_loader:
        print("\n" + "="*80)
        print("TEST SET EVALUATION")
        print("="*80)

        # Load best model
        checkpoint = torch.load(model_dir / f"{dataset_name}_baseline_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc = validate(
            model, test_loader, criterion, device, prefix="Test"
        )

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")

        # Save test results
        results_dir = Path(config['paths']['results']) / 'metrics'
        results_dir.mkdir(parents=True, exist_ok=True)

        test_results = {
            'dataset': dataset_name,
            'best_val_acc': float(best_val_acc),
            'best_val_loss': float(best_val_loss),
            'test_acc': float(test_acc),
            'test_loss': float(test_loss),
            'total_epochs': epoch,
            'history': history
        }

        with open(results_dir / f"{dataset_name}_baseline_results.json", 'w') as f:
            json.dump(test_results, f, indent=2)

    writer.close()

    return best_val_acc


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train baseline model on single dataset")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['genimage', 'cifake', 'faces'],
        help='Dataset to train on'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train model
    try:
        best_acc = train_single_dataset_model(args.dataset, config)
        print(f"\n✓ Training complete! Best accuracy: {best_acc:.2f}%")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
