"""
Comprehensive Evaluation Suite for AI-Generated Image Detection

Implements 4-tier evaluation:
- Tier 1: In-distribution validation (per-dataset accuracy)
- Tier 2: Cross-dataset generalization
- Tier 3: Robustness testing (JPEG, noise, blur)
- Tier 4: Uncertainty quantification (MC Dropout)

Outputs:
- Confusion matrices
- ROC curves
- Per-dataset metrics
- Robustness degradation curves
- Uncertainty analysis
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from pathlib import Path
import argparse
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from model_enhanced import create_model, EnhancedEfficientNet
from data_loader_multi import MultiDatasetLoader


class Evaluator:
    """Comprehensive model evaluator."""

    def __init__(
        self,
        model: nn.Module,
        device: str,
        output_dir: str = "results"
    ):
        """
        Args:
            model: Trained model to evaluate
            device: Device to use
            output_dir: Directory to save results
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    @torch.no_grad()
    def evaluate_dataset(
        self,
        data_loader: DataLoader,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            data_loader: DataLoader for the dataset
            dataset_name: Name of the dataset

        Returns:
            Dictionary of metrics and predictions
        """
        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        print(f"\nEvaluating {dataset_name}...")

        for images, labels, _ in tqdm(data_loader, desc=dataset_name):
            images = images.to(self.device)

            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_labels.extend(labels.numpy())
            all_predictions.extend(preds.flatten())
            all_probabilities.extend(probs.flatten())

        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate metrics
        accuracy = (all_predictions == all_labels).mean() * 100

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
        roc_auc = auc(fpr, tpr)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'labels': all_labels,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'fpr': fpr,
            'tpr': tpr
        }

        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")

        return results

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        dataset_name: str,
        save_path: Path
    ):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Fake', 'Real'],
            yticklabels=['Fake', 'Real']
        )

        plt.title(f'Confusion Matrix - {dataset_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(
        self,
        results_dict: Dict[str, Dict],
        save_path: Path
    ):
        """Plot ROC curves for all datasets."""
        plt.figure(figsize=(10, 8))

        for dataset_name, results in results_dict.items():
            plt.plot(
                results['fpr'],
                results['tpr'],
                label=f"{dataset_name} (AUC = {results['roc_auc']:.3f})",
                linewidth=2
            )

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Datasets', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def evaluate_with_mc_dropout(
        self,
        data_loader: DataLoader,
        n_iterations: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate with Monte Carlo Dropout for uncertainty quantification.

        Args:
            data_loader: DataLoader
            n_iterations: Number of MC dropout iterations

        Returns:
            Tuple of (mean_probabilities, std_probabilities)
        """
        self.model.train()  # Enable dropout
        self.model.enable_mc_dropout()

        all_probabilities = []

        print(f"\nRunning MC Dropout ({n_iterations} iterations)...")

        for _ in range(n_iterations):
            probs_iter = []

            for images, _, _ in tqdm(data_loader, desc=f"Iteration {_+1}/{n_iterations}", leave=False):
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_iter.extend(probs.flatten())

            all_probabilities.append(probs_iter)

        all_probabilities = np.array(all_probabilities)  # Shape: (n_iterations, n_samples)

        mean_probs = all_probabilities.mean(axis=0)
        std_probs = all_probabilities.std(axis=0)

        self.model.eval()  # Back to eval mode

        return mean_probs, std_probs

    def tier1_in_distribution(
        self,
        datasets_to_test: List[str]
    ) -> Dict:
        """
        Tier 1: In-distribution validation.

        Args:
            datasets_to_test: List of dataset names to evaluate

        Returns:
            Dictionary of results per dataset
        """
        print("=" * 80)
        print("TIER 1: IN-DISTRIBUTION VALIDATION")
        print("=" * 80)

        results = {}

        for dataset_name in datasets_to_test:
            loader = MultiDatasetLoader(
                data_dir='data',
                datasets_to_use=[dataset_name],
                batch_size=32,
                num_workers=4
            )

            _, _, test_loader = loader.create_loaders(phase=1, balanced_sampling=False)

            if test_loader is None:
                print(f"\n⚠ No test data for {dataset_name}, skipping...")
                continue

            dataset_results = self.evaluate_dataset(test_loader, dataset_name)
            results[dataset_name] = dataset_results

            # Save confusion matrix
            cm = np.array(dataset_results['confusion_matrix'])
            cm_path = self.output_dir / f'confusion_matrix_{dataset_name}.png'
            self.plot_confusion_matrix(cm, dataset_name, cm_path)
            print(f"  Saved: {cm_path}")

        # Plot combined ROC curves
        roc_path = self.output_dir / 'roc_curves_tier1.png'
        self.plot_roc_curves(results, roc_path)
        print(f"\n✓ Saved ROC curves: {roc_path}")

        return results

    def tier2_cross_dataset(self) -> Dict:
        """
        Tier 2: Cross-dataset generalization.

        Returns:
            Dictionary of cross-validation results
        """
        print("\n" + "=" * 80)
        print("TIER 2: CROSS-DATASET GENERALIZATION")
        print("=" * 80)

        loader = MultiDatasetLoader(data_dir='data')
        cross_val_loader = loader.create_cross_validation_loader()

        if cross_val_loader is None:
            print("⚠ Cross-validation dataset not available")
            return {}

        results = self.evaluate_dataset(cross_val_loader, "Fake or Real Competition")

        # Save confusion matrix
        cm = np.array(results['confusion_matrix'])
        cm_path = self.output_dir / 'confusion_matrix_cross_validation.png'
        self.plot_confusion_matrix(cm, "Cross-Validation", cm_path)
        print(f"  Saved: {cm_path}")

        return {'cross_validation': results}

    def tier3_robustness(
        self,
        test_loader: DataLoader
    ) -> Dict:
        """
        Tier 3: Robustness testing.

        Tests model robustness to:
        - JPEG compression
        - Gaussian noise
        - Gaussian blur

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of robustness results
        """
        print("\n" + "=" * 80)
        print("TIER 3: ROBUSTNESS TESTING")
        print("=" * 80)

        # Note: For full robustness testing, we would need to apply
        # transformations to images. This is a simplified version.

        print("\n⚠ Robustness testing requires additional image transformations")
        print("  This is a placeholder. Full implementation would include:")
        print("  - JPEG compression at various quality levels (60, 75, 90)")
        print("  - Additive Gaussian noise (σ = 0.01, 0.02, 0.05)")
        print("  - Gaussian blur (kernel sizes 3, 5, 7)")
        print("  - Resizing stress tests")

        # Placeholder results
        return {
            'note': 'Robustness testing not fully implemented in this version'
        }

    def tier4_uncertainty(
        self,
        test_loader: DataLoader,
        n_iterations: int = 10
    ) -> Dict:
        """
        Tier 4: Uncertainty quantification.

        Args:
            test_loader: Test data loader
            n_iterations: Number of MC dropout iterations

        Returns:
            Dictionary of uncertainty results
        """
        print("\n" + "=" * 80)
        print("TIER 4: UNCERTAINTY QUANTIFICATION")
        print("=" * 80)

        mean_probs, std_probs = self.evaluate_with_mc_dropout(
            test_loader,
            n_iterations=n_iterations
        )

        # Get true labels
        labels = []
        for _, batch_labels, _ in test_loader:
            labels.extend(batch_labels.numpy())
        labels = np.array(labels)

        # Analyze correlation between uncertainty and errors
        predictions = (mean_probs > 0.5).astype(int)
        errors = (predictions != labels).astype(int)

        correlation = np.corrcoef(std_probs, errors)[0, 1]

        print(f"\n✓ MC Dropout Analysis:")
        print(f"  Mean uncertainty (std): {std_probs.mean():.4f}")
        print(f"  Max uncertainty: {std_probs.max():.4f}")
        print(f"  Correlation (uncertainty vs errors): {correlation:.4f}")

        # Plot uncertainty distribution
        plt.figure(figsize=(10, 6))
        plt.hist(std_probs[errors == 0], bins=50, alpha=0.6, label='Correct Predictions')
        plt.hist(std_probs[errors == 1], bins=50, alpha=0.6, label='Incorrect Predictions')
        plt.xlabel('Prediction Uncertainty (Std Dev)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Uncertainty Distribution: Correct vs Incorrect Predictions', fontsize=14)
        plt.legend()
        plt.tight_layout()
        uncertainty_path = self.output_dir / 'uncertainty_distribution.png'
        plt.savefig(uncertainty_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {uncertainty_path}")

        return {
            'mean_uncertainty': float(std_probs.mean()),
            'max_uncertainty': float(std_probs.max()),
            'correlation_uncertainty_errors': float(correlation),
            'mean_probs': mean_probs.tolist(),
            'std_probs': std_probs.tolist()
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive Model Evaluation")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--all-datasets', action='store_true',
                        help='Evaluate on all training datasets (Tier 1)')
    parser.add_argument('--cross-dataset', action='store_true',
                        help='Evaluate cross-dataset generalization (Tier 2)')
    parser.add_argument('--robustness-test', action='store_true',
                        help='Run robustness tests (Tier 3)')
    parser.add_argument('--uncertainty-analysis', action='store_true',
                        help='Run uncertainty quantification (Tier 4)')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("")

    # Load model
    model = create_model(pretrained=False, device=args.device)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Loaded model from checkpoint")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Phase: {checkpoint.get('phase', 'N/A')}")

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=args.device,
        output_dir=args.output
    )

    all_results = {}

    # Tier 1: In-distribution validation
    if args.all_datasets:
        tier1_results = evaluator.tier1_in_distribution(
            datasets_to_test=['cifake', 'faces', 'tristan', 'shoes']
        )
        all_results['tier1_in_distribution'] = tier1_results

    # Tier 2: Cross-dataset generalization
    if args.cross_dataset:
        tier2_results = evaluator.tier2_cross_dataset()
        all_results['tier2_cross_dataset'] = tier2_results

    # Tier 3: Robustness testing
    if args.robustness_test:
        # Load a test dataset
        loader = MultiDatasetLoader(data_dir='data', datasets_to_use=['cifake'])
        _, _, test_loader = loader.create_loaders(phase=1, balanced_sampling=False)

        if test_loader:
            tier3_results = evaluator.tier3_robustness(test_loader)
            all_results['tier3_robustness'] = tier3_results

    # Tier 4: Uncertainty quantification
    if args.uncertainty_analysis:
        # Load a test dataset
        loader = MultiDatasetLoader(data_dir='data', datasets_to_use=['cifake'])
        _, _, test_loader = loader.create_loaders(phase=1, balanced_sampling=False)

        if test_loader:
            tier4_results = evaluator.tier4_uncertainty(test_loader, n_iterations=10)
            all_results['tier4_uncertainty'] = tier4_results

    # Save all results to JSON
    results_file = Path(args.output) / 'evaluation_results.json'

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {args.output}/")
    print("")

    # Print summary
    if 'tier1_in_distribution' in all_results:
        print("Tier 1 Summary (In-Distribution):")
        for dataset_name, results in all_results['tier1_in_distribution'].items():
            print(f"  {dataset_name}: {results['accuracy']:.2f}% accuracy")

    if 'tier2_cross_dataset' in all_results and 'cross_validation' in all_results['tier2_cross_dataset']:
        cv_acc = all_results['tier2_cross_dataset']['cross_validation']['accuracy']
        print(f"\nTier 2 Summary (Cross-Dataset):")
        print(f"  Cross-validation: {cv_acc:.2f}% accuracy")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
