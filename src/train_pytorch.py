"""
PyTorch Training Script for TruthPixel.

Clean, simple, and actually works!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import time
from pathlib import Path
import logging
from tqdm import tqdm

from data_loader_pytorch import CIFAKEDataLoaderPyTorch
from model_pytorch import create_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class TruthPixelTrainer:
    """
    PyTorch trainer with full control.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate: float = 0.001,
        save_dir: str = "models"
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            learning_rate: Initial learning rate
            save_dir: Directory to save models
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Loss and optimizer
        self.criterion = nn.BCELoss()  # Binary Cross Entropy
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3
        )

        # Metrics
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training", ncols=100)

        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)  # (batch_size, 1)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            current_acc = 100.0 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%'
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self):
        """
        Validate the model.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(self.val_loader, desc="Validation", ncols=100)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                # Move to device
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item()
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                # Update progress bar
                current_acc = 100.0 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_acc:.2f}%'
                })

        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int = 10, phase_name: str = "Phase 1"):
        """
        Train for multiple epochs.

        Args:
            num_epochs: Number of epochs
            phase_name: Name of training phase

        Returns:
            Training history
        """
        logger.info("=" * 80)
        logger.info(f"{phase_name.upper()}: TRAINING")
        logger.info("=" * 80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        logger.info(f"Batch Size: {self.train_loader.batch_size}")
        logger.info("=" * 80 + "\n")

        best_epoch = 0
        patience = 5
        patience_counter = 0

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Update learning rate
            self.scheduler.step(val_acc)

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            epoch_time = time.time() - epoch_start

            # Log epoch results
            print("\n" + "─" * 100)
            logger.info(
                f"📊 EPOCH [{epoch + 1}/{num_epochs}] COMPLETE ({epoch_time:.1f}s)\n"
                f"   📈 Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}%\n"
                f"   🎯 Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%"
            )

            # Save best model
            if val_acc > self.best_val_acc:
                prev_best = self.best_val_acc
                self.best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                save_path = self.save_dir / f"{phase_name.lower().replace(' ', '_')}_best.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)

                if prev_best > 0:
                    logger.info(f"   💾 NEW BEST MODEL! Val Acc: {val_acc:.2f}% (improved by {val_acc - prev_best:.2f}%)")
                else:
                    logger.info(f"   💾 BEST MODEL SAVED! Val Acc: {val_acc:.2f}%")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break

        total_time = time.time() - start_time

        logger.info("\n" + "=" * 80)
        logger.info(f"{phase_name.upper()} COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
        logger.info(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {best_epoch})")
        logger.info("=" * 80 + "\n")

        return self.history


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train TruthPixel with PyTorch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs_phase1', type=int, default=10, help='Phase 1 epochs')
    parser.add_argument('--epochs_phase2', type=int, default=10, help='Phase 2 epochs')
    parser.add_argument('--lr_phase1', type=float, default=0.001, help='Phase 1 learning rate')
    parser.add_argument('--lr_phase2', type=float, default=0.0001, help='Phase 2 learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRUTHPIXEL TRAINING (PYTORCH)")
    logger.info("=" * 80)

    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    logger.info("=" * 80 + "\n")

    # Load data
    logger.info("LOADING DATA...")
    data_loader = CIFAKEDataLoaderPyTorch(
        data_dir="data",
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader, val_loader, test_loader = data_loader.prepare_loaders()

    # Phase 1: Train with frozen base
    logger.info("PHASE 1: FROZEN BASE TRAINING\n")
    model = create_model(freeze_base=True, device=device)

    trainer = TruthPixelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr_phase1
    )

    trainer.train(num_epochs=args.epochs_phase1, phase_name="Phase 1")

    # Phase 2: Fine-tune with unfrozen layers
    logger.info("PHASE 2: FINE-TUNING\n")
    model.unfreeze_base_layers(num_layers=20)

    trainer = TruthPixelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr_phase2
    )

    trainer.train(num_epochs=args.epochs_phase2, phase_name="Phase 2")

    # Save final model
    final_path = Path("models") / "truthpixel_final_pytorch.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"✓ Final model saved to {final_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
