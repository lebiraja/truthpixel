"""
4-Phase Progressive Training Script for Multi-Dataset AI Image Detection

Implements the complete training pipeline:
- Phase 1: CIFAKE Foundation (frozen backbone, single dataset)
- Phase 2: Multi-Dataset Joint Training (frozen backbone, balanced sampling)
- Phase 3: Fine-Tuning (unfreeze last 20 layers, low LR)
- Phase 4: Domain Adversarial Training (optional, gradient reversal)

Features:
- TensorBoard logging
- Automatic checkpointing
- Early stopping
- Learning rate scheduling
- Per-dataset validation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
import time
import json
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from model_enhanced import create_model
from data_loader_multi import MultiDatasetLoader


class Trainer:
    """Multi-phase trainer for AI-generated image detection."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        save_dir: str = "models",
        log_dir: str = "logs"
    ):
        """
        Args:
            model: PyTorch model to train
            device: Device to train on (cuda/cpu)
            save_dir: Directory to save model checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.log_dir = Path(log_dir)

        self.save_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        self.writer = None
        self.global_step = 0

    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        phase: int,
        use_domain_loss: bool = False,
        domain_weight: float = 0.1
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            phase: Training phase (1, 2, or 3)
            use_domain_loss: Whether to use domain adversarial loss
            domain_weight: Weight for domain loss

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_class_loss = 0.0
        total_domain_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Training Phase {phase}")

        for batch_idx, (images, labels, dataset_ids) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            dataset_ids = dataset_ids.to(self.device)

            # Forward pass
            optimizer.zero_grad()

            if use_domain_loss:
                class_logits, domain_logits = self.model(images, return_domain=True)

                # Classification loss
                class_loss = criterion(class_logits.squeeze(), labels)

                # Domain loss (predict which dataset)
                domain_criterion = nn.CrossEntropyLoss()
                domain_loss = domain_criterion(domain_logits, dataset_ids)

                # Combined loss
                loss = class_loss + domain_weight * domain_loss

                total_domain_loss += domain_loss.item()
            else:
                class_logits = self.model(images)
                loss = criterion(class_logits.squeeze(), labels)
                class_loss = loss

            total_class_loss += class_loss.item()

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate accuracy
            predictions = (torch.sigmoid(class_logits) > 0.5).float()
            correct += (predictions.squeeze() == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

            # TensorBoard logging
            if self.writer and batch_idx % 10 == 0:
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/BatchAcc', 100.0 * correct / total, self.global_step)

            self.global_step += 1

        metrics = {
            'loss': total_loss / len(train_loader),
            'class_loss': total_class_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }

        if use_domain_loss:
            metrics['domain_loss'] = total_domain_loss / len(train_loader)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader,
        criterion,
        dataset_name: str = "validation"
    ) -> Dict[str, float]:
        """
        Validate model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # For calculating precision, recall, F1
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for images, labels, _ in tqdm(val_loader, desc=f"Validating {dataset_name}"):
            images = images.to(self.device)
            labels = labels.float().to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = criterion(logits.squeeze(), labels)

            total_loss += loss.item()

            # Calculate accuracy
            predictions = (torch.sigmoid(logits) > 0.5).float().squeeze()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # For precision/recall
            true_positives += ((predictions == 1) & (labels == 1)).sum().item()
            false_positives += ((predictions == 1) & (labels == 0)).sum().item()
            false_negatives += ((predictions == 0) & (labels == 1)).sum().item()

        # Calculate metrics
        accuracy = 100.0 * correct / total
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_checkpoint(
        self,
        epoch: int,
        phase: int,
        metrics: Dict[str, float],
        optimizer: optim.Optimizer,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step
        }

        # Save regular checkpoint
        filename = f"phase{phase}_epoch{epoch:03d}.pth"
        torch.save(checkpoint, self.save_dir / filename)

        # Save best checkpoint
        if is_best:
            best_filename = f"phase{phase}_best.pth"
            torch.save(checkpoint, self.save_dir / best_filename)
            print(f"✓ Saved best checkpoint: {best_filename}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, int]:
        """
        Load checkpoint.

        Returns:
            Tuple of (start_epoch, phase)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)

        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}, Phase: {checkpoint['phase']}")
        print(f"  Metrics: {checkpoint['metrics']}")

        return checkpoint['epoch'], checkpoint['phase']


def train_phase(
    phase: int,
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    config: Dict,
    trainer: Trainer,
    checkpoint_path: Optional[str] = None
):
    """
    Train a single phase.

    Args:
        phase: Phase number (1, 2, 3, or 4)
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use
        config: Configuration dictionary
        trainer: Trainer instance
        checkpoint_path: Optional checkpoint to resume from
    """
    print("\n" + "=" * 80)
    print(f"PHASE {phase} TRAINING")
    print("=" * 80)

    # Phase-specific configuration
    if phase == 1:
        print("Strategy: CIFAKE Foundation")
        print("  - Frozen backbone")
        print("  - Single dataset training")
        learning_rate = config.get('lr_phase1', 1e-3)
        epochs = config.get('epochs_phase1', 15)
        model.freeze_backbone()

    elif phase == 2:
        print("Strategy: Multi-Dataset Joint Training")
        print("  - Frozen backbone")
        print("  - Balanced sampling across all datasets")
        learning_rate = config.get('lr_phase2', 5e-4)
        epochs = config.get('epochs_phase2', 20)
        model.freeze_backbone()

    elif phase == 3:
        print("Strategy: Fine-Tuning")
        print("  - Unfreeze last 20 layers")
        print("  - Very low learning rate")
        learning_rate = config.get('lr_phase3', 1e-5)
        epochs = config.get('epochs_phase3', 15)
        model.unfreeze_last_n_layers(20)

    else:  # phase == 4
        print("Strategy: Domain Adversarial Training")
        print("  - Domain classifier enabled")
        print("  - Gradient reversal for domain-invariant features")
        learning_rate = config.get('lr_phase4', 1e-5)
        epochs = config.get('epochs_phase4', 10)
        model.unfreeze_last_n_layers(20)

    print(f"\nHyperparameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {config['batch_size']}")
    print("")

    # Setup optimizer and loss
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=config.get('weight_decay', 1e-4)
    )

    criterion = nn.BCEWithLogitsLoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # TensorBoard writer
    trainer.writer = SummaryWriter(log_dir=trainer.log_dir / f"phase{phase}")

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path:
        start_epoch, _ = trainer.load_checkpoint(checkpoint_path)
        start_epoch += 1

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = config.get('early_stop_patience', 5)

    for epoch in range(start_epoch, epochs):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 80}")

        # Train
        train_metrics = trainer.train_epoch(
            train_loader,
            optimizer,
            criterion,
            phase,
            use_domain_loss=(phase == 4),
            domain_weight=config.get('domain_weight', 0.1)
        )

        print(f"\nTraining Results:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")

        # Validate
        val_metrics = trainer.validate(val_loader, criterion)

        print(f"\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.2f}%")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1-Score: {val_metrics['f1']:.4f}")

        # TensorBoard logging
        trainer.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
        trainer.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
        trainer.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        trainer.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
        trainer.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)

        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
        else:
            patience_counter += 1

        trainer.save_checkpoint(
            epoch,
            phase,
            {**train_metrics, **{'val_' + k: v for k, v in val_metrics.items()}},
            optimizer,
            is_best=is_best
        )

        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠ Early stopping triggered (patience={early_stop_patience})")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break

    trainer.writer.close()

    print(f"\n{'=' * 80}")
    print(f"PHASE {phase} COMPLETE")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'=' * 80}\n")

    return best_val_acc


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train AI-Generated Image Detector")

    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Training phase (1=CIFAKE baseline, 2=multi-dataset, 3=fine-tune, 4=domain-adversarial)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config defaults)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides phase defaults)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Configuration
    config = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'weight_decay': 1e-4,
        'early_stop_patience': 5,
        'domain_weight': 0.2,

        # Phase 1
        'lr_phase1': 1e-3,
        'epochs_phase1': 15,

        # Phase 2
        'lr_phase2': 5e-4,
        'epochs_phase2': 20,

        # Phase 3
        'lr_phase3': 1e-5,
        'epochs_phase3': 15,

        # Phase 4
        'lr_phase4': 1e-5,
        'epochs_phase4': 10
    }

    # Override with command line args
    if args.epochs:
        config[f'epochs_phase{args.phase}'] = args.epochs
    if args.lr:
        config[f'lr_phase{args.phase}'] = args.lr

    print("=" * 80)
    print("AI-GENERATED IMAGE DETECTION TRAINING")
    print("=" * 80)
    print(f"Phase: {args.phase}")
    print(f"Device: {args.device}")
    print(f"Batch size: {config['batch_size']}")
    print("")

    # Create model
    use_domain_classifier = (args.phase == 4)
    model = create_model(
        pretrained=True,
        dropout_rate=0.5,
        use_domain_classifier=use_domain_classifier,
        device=args.device
    )

    # Load checkpoint if continuing from previous phase
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint: {args.checkpoint}\n")

    # Create data loaders
    if args.phase == 1:
        # Phase 1: Single dataset (CIFAKE only)
        datasets_to_use = ['cifake']
        balanced_sampling = False
    else:
        # Phase 2+: All available datasets with balanced sampling
        # Note: Using 3 datasets (cifake, faces, tristan) - shoes skipped due to storage
        datasets_to_use = ['cifake', 'faces', 'tristan']
        balanced_sampling = True

    data_loader = MultiDatasetLoader(
        data_dir=args.data_dir,
        datasets_to_use=datasets_to_use,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    train_loader, val_loader, _ = data_loader.create_loaders(
        phase=args.phase,
        balanced_sampling=balanced_sampling
    )

    if val_loader is None:
        raise RuntimeError("No validation data found!")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=args.device,
        save_dir='models',
        log_dir='logs'
    )

    # Train
    best_acc = train_phase(
        phase=args.phase,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        config=config,
        trainer=trainer,
        checkpoint_path=None  # We already loaded it above
    )

    # Save final results
    results = {
        'phase': args.phase,
        'best_val_accuracy': best_acc,
        'config': config
    }

    results_file = Path('results') / f'phase{args.phase}_results.json'
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("\nNext steps:")

    if args.phase == 1:
        print("  python src/train_multi_dataset.py --phase 2 --checkpoint models/phase1_best.pth")
    elif args.phase == 2:
        print("  python src/train_multi_dataset.py --phase 3 --checkpoint models/phase2_best.pth")
    elif args.phase == 3:
        print("  python src/evaluate_comprehensive.py --checkpoint models/phase3_best.pth")
        print("  Or continue with Phase 4 (optional):")
        print("  python src/train_multi_dataset.py --phase 4 --checkpoint models/phase3_best.pth")

    print("")


if __name__ == "__main__":
    main()
