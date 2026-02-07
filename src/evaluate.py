"""
Comprehensive Evaluation Script

Evaluates all trained models and generates comprehensive metrics.

Usage:
    python src/evaluate.py
"""

import sys
import json
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve
)

import torch
from tqdm import tqdm

from data_loader_multi import MultiDatasetLoader
from train_baseline import DeepfakeDetector


def evaluate_model(model, loader, device):
    """Evaluate model and return comprehensive metrics."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.float()

            outputs = model(images).cpu().numpy().flatten()
            predictions = (outputs > 0.5).astype(int)

            all_probs.extend(outputs)
            all_preds.extend(predictions)
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def plot_confusion_matrix(cm, title, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'],
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(results_dict, save_path):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))

    for model_name, metrics in results_dict.items():
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
        auc = metrics['auc']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_all_models(config):
    """Evaluate all trained models."""
    print("="*80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    data_loader = MultiDatasetLoader(
        data_dir=config['data']['base_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )

    models_dir = Path(config['paths']['models'])
    results_dir = Path(config['paths']['results'])

    plots_dir = results_dir / 'plots'
    metrics_dir = results_dir / 'metrics'
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Evaluate baseline models
    for dataset in ['genimage', 'cifake', 'faces']:
        print(f"\n{'='*80}")
        print(f"EVALUATING: {dataset.upper()} Baseline")
        print(f"{'='*80}")

        model_path = models_dir / 'baseline' / f'{dataset}_baseline_best.pt'

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        # Load model
        model = DeepfakeDetector().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get test loader
        _, _, test_loader = data_loader.create_single_dataset_loader(dataset, phase=1)

        if not test_loader:
            print("No test data available")
            continue

        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        all_results[f'{dataset}_baseline'] = metrics

        # Print metrics
        print(f"\nMetrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
        print(f"  Precision: {metrics['precision']:.2f}%")
        print(f"  Recall:    {metrics['recall']:.2f}%")
        print(f"  F1 Score:  {metrics['f1']:.2f}%")
        print(f"  AUC:       {metrics['auc']:.3f}")

        # Plot confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            f'Confusion Matrix - {dataset.capitalize()} Baseline',
            plots_dir / f'{dataset}_baseline_confusion_matrix.png'
        )

    # Evaluate combined model
    print(f"\n{'='*80}")
    print("EVALUATING: Combined Model")
    print(f"{'='*80}")

    model_path = models_dir / 'combined' / 'combined_model_best.pt'

    if model_path.exists():
        model = DeepfakeDetector().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get test loader (all datasets)
        _, _, test_loader = data_loader.create_loaders(phase=2, balanced_sampling=False)

        if test_loader:
            metrics = evaluate_model(model, test_loader, device)
            all_results['combined'] = metrics

            print(f"\nMetrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.2f}%")
            print(f"  Precision: {metrics['precision']:.2f}%")
            print(f"  Recall:    {metrics['recall']:.2f}%")
            print(f"  F1 Score:  {metrics['f1']:.2f}%")
            print(f"  AUC:       {metrics['auc']:.3f}")

            cm = np.array(metrics['confusion_matrix'])
            plot_confusion_matrix(
                cm,
                'Confusion Matrix - Combined Model',
                plots_dir / 'combined_confusion_matrix.png'
            )

    # Plot ROC curves
    if all_results:
        plot_roc_curve(all_results, plots_dir / 'roc_curves_all_models.png')

    # Save summary
    summary = {}
    for model_name, metrics in all_results.items():
        summary[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'confusion_matrix': metrics['confusion_matrix']
        }

    with open(metrics_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-"*80)

    for model_name, metrics in summary.items():
        print(f"{model_name:<20} {metrics['accuracy']:>8.2f}% {metrics['precision']:>8.2f}% "
              f"{metrics['recall']:>8.2f}% {metrics['f1']:>8.2f}% {metrics['auc']:>9.3f}")

    print("\n✓ Evaluation complete!")
    print(f"\nResults saved to:")
    print(f"  - Metrics: {metrics_dir}/")
    print(f"  - Plots: {plots_dir}/")

    return summary


def main():
    """Main function."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        summary = evaluate_all_models(config)
        return 0
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
