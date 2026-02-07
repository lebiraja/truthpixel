"""
Combined Model Training Script - Phase 2

Trains a unified model on all three datasets (GenImage, CIFAKE, Faces).
Uses weighted sampling for balanced multi-dataset training.

Usage:
    python src/train_combined.py
"""

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
    """EfficientNetB0-based deepfake detector (same as baseline)."""

    def __init__(self, pretrained=True, dropout=0.5):
        super(DeepfakeDetector, self).__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]

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
        features = self.backbone(x)
        features = self.pool(features)
        features = self.flatten(features)
        output = self.classifier(features)
        return output


def unfreeze_layers(model, num_layers=20):
    """Unfreeze last N layers of the backbone."""
    # Get all backbone parameters
    backbone_params = list(model.backbone.parameters())

    # Unfreeze last num_layers
    for param in backbone_params[-num_layers:]:
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfroze last {num_layers} layers. Trainable params: {trainable:,}")


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, (images, labels, dataset_ids) in enumerate(pbar):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({
            'loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })

        if writer and batch_idx % 10 == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), step)

    return running_loss / len(loader), 100. * correct / total


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

    return running_loss / len(loader), 100. * correct / total


def train_combined_model(config: dict):
    """Train unified model on all three datasets."""
    print("="*80)
    print("TRAINING COMBINED MODEL - PHASE 2")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load all datasets with weighted sampling
    print("\nLoading data...")
    data_loader = MultiDatasetLoader(
        data_dir=config['data']['base_dir'],
        datasets_to_use=['genimage', 'cifake', 'faces'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )

    train_loader, val_loader, test_loader = data_loader.create_loaders(
        phase=2,
        balanced_sampling=True
    )

    # Create model
    print("\nCreating model...")
    model = DeepfakeDetector(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    ).to(device)

    # Freeze backbone initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("Backbone frozen initially")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase2_combined']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['callbacks']['reduce_lr']['factor'],
        patience=config['callbacks']['reduce_lr']['patience'],
        verbose=True
    )

    # Setup directories
    model_dir = Path(config['paths']['models']) / 'combined'
    checkpoint_dir = Path(config['paths']['checkpoints']) / 'combined'
    log_dir = Path(config['logging']['tensorboard_dir']) / 'combined'

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # Training loop
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)

    num_epochs = config['training']['phase2_combined']['epochs']
    unfreeze_epoch = config['training']['phase2_combined']['unfreeze_after_epoch']
    unfreeze_layers_count = config['training']['phase2_combined']['unfreeze_layer_count']
    save_freq = config['callbacks']['model_checkpoint']['save_freq']
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = config['training']['phase2_combined']['early_stopping_patience']

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)

        # Unfreeze backbone layers after N epochs
        if epoch == unfreeze_epoch + 1:
            print(f"\n🔓 Unfreezing last {unfreeze_layers_count} layers...")
            unfreeze_layers(model, unfreeze_layers_count)

            # Re-create optimizer with all trainable parameters
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config['training']['phase2_combined']['learning_rate'] * 0.1,  # Lower LR
                weight_decay=config['training']['weight_decay']
            )

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer, "Val"
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, model_dir / "combined_model_best.pt")

            print(f"  ✓ Best model saved (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Save checkpoint
        if epoch % save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'history': history,
                'config': config
            }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
            print(f"  ✓ Checkpoint saved (epoch {epoch})")

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config
    }, model_dir / "combined_model.pt")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    # Test evaluation
    if test_loader:
        checkpoint = torch.load(model_dir / "combined_model_best.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_acc = validate(
            model, test_loader, criterion, device, prefix="Test"
        )

        print(f"\nTest Accuracy: {test_acc:.2f}%")

        results = {
            'model': 'combined',
            'datasets': ['genimage', 'cifake', 'faces'],
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'total_epochs': epoch,
            'history': history
        }

        results_dir = Path(config['paths']['results']) / 'metrics'
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "combined_model_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    writer.close()
    return best_val_acc


def main():
    """Main function."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        best_acc = train_combined_model(config)
        print(f"\n✓ Training complete! Best accuracy: {best_acc:.2f}%")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
