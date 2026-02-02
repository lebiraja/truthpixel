"""
Evaluation Module for TruthPixel AI-Generated Image Detection.

This module handles comprehensive model evaluation including:
- Metrics calculation (accuracy, precision, recall, F1, AUC)
- Confusion matrix generation
- ROC curve plotting
- Classification report
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from typing import Dict, Tuple, Optional
import logging

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_efficient import EfficientCIFAKEDataLoader as CIFAKEDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for TruthPixel.

    Handles metrics calculation and visualization generation.
    """

    def __init__(
        self,
        model_path: str,
        class_names: list = ['Real', 'AI-Generated']
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the trained model
            class_names: Names of the classes
        """
        self.model_path = model_path
        self.class_names = class_names
        self.model = None

    def load_model(self) -> tf.keras.Model:
        """
        Load the trained model.

        Returns:
            Loaded Keras model
        """
        logger.info(f"Loading model from {self.model_path}")

        self.model = tf.keras.models.load_model(self.model_path)

        logger.info("✓ Model loaded successfully")

        return self.model

    def get_predictions(
        self,
        dataset: tf.data.Dataset
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model predictions on dataset.

        Args:
            dataset: TensorFlow dataset

        Returns:
            Tuple of (y_true, y_pred, y_scores)
        """
        logger.info("Generating predictions...")

        y_true = []
        y_scores = []

        for images, labels in dataset:
            # Get predictions
            predictions = self.model.predict(images, verbose=0)

            y_true.extend(labels.numpy())
            y_scores.extend(predictions.flatten())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores > 0.5).astype(int)  # Binary threshold at 0.5

        logger.info(f"✓ Generated predictions for {len(y_true)} samples")

        return y_true, y_pred, y_scores

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores

        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating metrics...")

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_true, y_scores)
        }

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)

        metrics['precision_real'] = precision_per_class[0]
        metrics['precision_ai'] = precision_per_class[1]
        metrics['recall_real'] = recall_per_class[0]
        metrics['recall_ai'] = recall_per_class[1]
        metrics['f1_real'] = f1_per_class[0]
        metrics['f1_ai'] = f1_per_class[1]

        logger.info("✓ Metrics calculated")

        return metrics

    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print metrics in a formatted table.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 80)
        print("EVALUATION METRICS")
        print("=" * 80)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        print(f"\nPer-Class Metrics:")
        print(f"  Real Images:")
        print(f"    Precision: {metrics['precision_real']:.4f}")
        print(f"    Recall:    {metrics['recall_real']:.4f}")
        print(f"    F1-Score:  {metrics['f1_real']:.4f}")

        print(f"  AI-Generated Images:")
        print(f"    Precision: {metrics['precision_ai']:.4f}")
        print(f"    Recall:    {metrics['recall_ai']:.4f}")
        print(f"    F1-Score:  {metrics['f1_ai']:.4f}")

        print("=" * 80 + "\n")

    def save_metrics(
        self,
        metrics: Dict[str, float],
        save_path: str = 'results/metrics.txt'
    ) -> None:
        """
        Save metrics to a text file.

        Args:
            metrics: Dictionary of metrics
            save_path: Path to save the metrics
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRUTHPIXEL EVALUATION METRICS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")

            f.write("Per-Class Metrics:\n")
            f.write("  Real Images:\n")
            f.write(f"    Precision: {metrics['precision_real']:.4f}\n")
            f.write(f"    Recall:    {metrics['recall_real']:.4f}\n")
            f.write(f"    F1-Score:  {metrics['f1_real']:.4f}\n\n")

            f.write("  AI-Generated Images:\n")
            f.write(f"    Precision: {metrics['precision_ai']:.4f}\n")
            f.write(f"    Recall:    {metrics['recall_ai']:.4f}\n")
            f.write(f"    F1-Score:  {metrics['f1_ai']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"✓ Saved metrics to {save_path}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )

        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        # Add accuracy per class
        for i in range(len(self.class_names)):
            accuracy = cm[i, i] / cm[i].sum()
            plt.text(
                i + 0.5,
                i - 0.3,
                f'{accuracy:.2%}',
                ha='center',
                va='center',
                color='red',
                fontweight='bold',
                fontsize=10
            )

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved confusion matrix to {save_path}")

        plt.close()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot ROC curve.

        Args:
            y_true: True labels
            y_scores: Prediction scores
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)

        plt.figure(figsize=(8, 6))

        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label=f'ROC Curve (AUC = {auc_score:.4f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✓ Saved ROC curve to {save_path}")

        plt.close()

    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Print classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80 + "\n")

        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            digits=4
        )

        print(report)
        print("=" * 80 + "\n")

    def evaluate(
        self,
        test_dataset: tf.data.Dataset
    ) -> Dict[str, float]:
        """
        Perform complete evaluation on test dataset.

        Args:
            test_dataset: Test dataset

        Returns:
            Dictionary of metrics
        """
        logger.info("=" * 80)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 80)

        # Load model
        if self.model is None:
            self.load_model()

        # Get predictions
        y_true, y_pred, y_scores = self.get_predictions(test_dataset)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_scores)

        # Print metrics
        self.print_metrics(metrics)

        # Print classification report
        self.print_classification_report(y_true, y_pred)

        # Save metrics
        self.save_metrics(metrics)

        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true,
            y_pred,
            save_path='results/plots/confusion_matrix.png'
        )

        # Plot ROC curve
        self.plot_roc_curve(
            y_true,
            y_scores,
            save_path='results/plots/roc_curve.png'
        )

        logger.info("✓ Evaluation completed successfully")
        logger.info("=" * 80)

        return metrics


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description='Evaluate TruthPixel Model')

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/truthpixel_final.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )

    args = parser.parse_args()

    # Load test dataset
    logger.info("Loading test dataset...")
    data_loader = CIFAKEDataLoader(data_dir="data", batch_size=args.batch_size)
    _, _, test_ds = data_loader.prepare_datasets(augment_train=False)

    # Create evaluator
    evaluator = ModelEvaluator(model_path=args.model_path)

    # Evaluate
    metrics = evaluator.evaluate(test_ds)

    # Check if target metrics achieved
    logger.info("\n" + "=" * 80)
    logger.info("TARGET METRICS CHECK")
    logger.info("=" * 80)

    target_accuracy = 0.92
    target_f1 = 0.92

    if metrics['accuracy'] >= target_accuracy:
        logger.info(f"✓ Accuracy target achieved: {metrics['accuracy']:.4f} >= {target_accuracy}")
    else:
        logger.warning(f"✗ Accuracy target not met: {metrics['accuracy']:.4f} < {target_accuracy}")

    if metrics['f1_score'] >= target_f1:
        logger.info(f"✓ F1-Score target achieved: {metrics['f1_score']:.4f} >= {target_f1}")
    else:
        logger.warning(f"✗ F1-Score target not met: {metrics['f1_score']:.4f} < {target_f1}")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
