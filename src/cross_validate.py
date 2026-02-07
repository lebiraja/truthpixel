"""
Cross-Dataset Validation Script - Phase 3

Tests models trained on one dataset against other datasets to measure generalization.

Usage:
    python src/cross_validate.py
"""

import sys
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader_multi import MultiDatasetLoader
from train_baseline import DeepfakeDetector


def evaluate_model_on_dataset(model, dataset_name, data_loader, device):
    """Evaluate a model on a specific dataset."""
    _, _, test_loader = data_loader.create_single_dataset_loader(
        dataset_name=dataset_name,
        phase=1
    )

    if not test_loader:
        print(f"No test data for {dataset_name}")
        return None

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc=f"Evaluating on {dataset_name}"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            predicted = (outputs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = 100. * correct / total

    # Calculate precision, recall, F1
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'auc': auc
    }


def cross_validate_all_models(config):
    """Perform cross-validation for all models."""
    print("="*80)
    print("CROSS-DATASET VALIDATION - PHASE 3")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    data_loader = MultiDatasetLoader(
        data_dir=config['data']['base_dir'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        img_size=config['data']['img_size']
    )

    datasets = ['genimage', 'cifake', 'faces']
    models_dir = Path(config['paths']['models'])

    # Results matrix: rows=trained_on, cols=tested_on
    results_matrix = np.zeros((len(datasets) + 1, len(datasets)))  # +1 for combined
    model_names = datasets + ['combined']

    # Test each model on each dataset
    for model_idx, train_dataset in enumerate(model_names):
        print(f"\n{'='*80}")
        print(f"MODEL: {train_dataset.upper()}")
        print(f"{'='*80}")

        # Load model
        if train_dataset == 'combined':
            model_path = models_dir / 'combined' / 'combined_model_best.pt'
        else:
            model_path = models_dir / 'baseline' / f'{train_dataset}_baseline_best.pt'

        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        model = DeepfakeDetector().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Test on each dataset
        for test_idx, test_dataset in enumerate(datasets):
            print(f"\nTesting on {test_dataset}...")

            metrics = evaluate_model_on_dataset(
                model, test_dataset, data_loader, device
            )

            if metrics:
                results_matrix[model_idx, test_idx] = metrics['accuracy']

                print(f"  Accuracy: {metrics['accuracy']:.2f}%")
                print(f"  Precision: {metrics['precision']:.2f}%")
                print(f"  Recall: {metrics['recall']:.2f}%")
                print(f"  F1 Score: {metrics['f1']:.2f}%")

    # Create DataFrame for better visualization
    df = pd.DataFrame(
        results_matrix,
        index=[f"{m} (trained)" for m in model_names],
        columns=[f"{d} (test)" for d in datasets]
    )

    print("\n" + "="*80)
    print("CROSS-VALIDATION MATRIX (Accuracy %)")
    print("="*80)
    print(df.to_string())

    # Save results
    results_dir = Path(config['paths']['results']) / 'cross_validation'
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / 'cross_validation_matrix.csv')

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', vmin=50, vmax=100,
                cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Cross-Dataset Validation Results', fontsize=16, fontweight='bold')
    plt.xlabel('Test Dataset', fontsize=12)
    plt.ylabel('Training Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / 'cross_validation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Heatmap saved to: {results_dir / 'cross_validation_heatmap.png'}")

    # Calculate generalization metrics
    print("\n" + "="*80)
    print("GENERALIZATION ANALYSIS")
    print("="*80)

    for i, model_name in enumerate(model_names):
        same_dataset_acc = results_matrix[i, datasets.index(model_name)] if model_name != 'combined' else np.mean(results_matrix[i])
        cross_dataset_acc = np.mean([results_matrix[i, j] for j in range(len(datasets)) if datasets[j] != model_name]) if model_name != 'combined' else np.mean(results_matrix[i])

        print(f"\n{model_name.upper()}:")
        if model_name != 'combined':
            print(f"  Same dataset: {same_dataset_acc:.2f}%")
            print(f"  Cross dataset: {cross_dataset_acc:.2f}%")
            print(f"  Generalization gap: {same_dataset_acc - cross_dataset_acc:.2f}%")
        else:
            print(f"  Average accuracy: {cross_dataset_acc:.2f}%")

    return df


def main():
    """Main function."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    try:
        df = cross_validate_all_models(config)
        print("\n✓ Cross-validation complete!")
        return 0
    except Exception as e:
        print(f"\n✗ Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
