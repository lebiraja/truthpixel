"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Module for TruthPixel.

This module implements Grad-CAM for visual explanation of model predictions,
showing which parts of the image influenced the classification decision.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import cv2
from typing import Tuple, Optional, List
import logging

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader_efficient import EfficientCIFAKEDataLoader as CIFAKEDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for visual explanations.

    Generates heatmaps showing which regions of an image
    influenced the model's prediction.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        last_conv_layer_name: Optional[str] = None,
        class_names: List[str] = ['Real', 'AI-Generated']
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of last convolutional layer
            class_names: Names of the classes
        """
        self.model = model
        self.class_names = class_names

        # Find last convolutional layer if not specified
        if last_conv_layer_name is None:
            last_conv_layer_name = self._find_last_conv_layer()

        self.last_conv_layer_name = last_conv_layer_name
        logger.info(f"Using convolutional layer: {last_conv_layer_name}")

    def _find_last_conv_layer(self) -> str:
        """
        Find the last convolutional layer in the model.

        Returns:
            Name of the last convolutional layer
        """
        # For EfficientNetB0, the last conv layer is typically 'top_conv'
        # or within the base model

        # Try to find EfficientNet base model
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
                # Look for last conv layer in base model
                for base_layer in reversed(layer.layers):
                    if isinstance(base_layer, tf.keras.layers.Conv2D):
                        return base_layer.name

        # Fallback: search entire model
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

        raise ValueError("Could not find convolutional layer in model")

    def get_gradcam_heatmap(
        self,
        img_array: np.ndarray,
        pred_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            img_array: Preprocessed image array (1, H, W, 3)
            pred_index: Class index to generate heatmap for (None = predicted class)

        Returns:
            Heatmap as numpy array
        """
        # Create a model that maps the input to the activations of the last conv layer
        # and the output predictions
        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )

        # Compute the gradient of the top predicted class for the input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            # For binary classification with sigmoid, we use the raw output
            class_channel = predictions[:, 0]

        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)

        # Compute the guided gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the channels by the corresponding gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to the heatmap (only positive influences)
        heatmap = tf.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap /= tf.reduce_max(heatmap) + 1e-10

        return heatmap.numpy()

    def overlay_heatmap_on_image(
        self,
        img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay Grad-CAM heatmap on original image.

        Args:
            img: Original image (H, W, 3) in [0, 1] range
            heatmap: Grad-CAM heatmap
            alpha: Transparency of heatmap overlay
            colormap: OpenCV colormap to use

        Returns:
            Image with heatmap overlay
        """
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = heatmap.astype(np.float32) / 255.0

        # Convert image to [0, 255] range if needed
        if img.max() <= 1.0:
            img = img.copy()
        else:
            img = img.astype(np.float32) / 255.0

        # Overlay heatmap on image
        overlayed = heatmap * alpha + img * (1 - alpha)
        overlayed = np.clip(overlayed, 0, 1)

        return overlayed

    def generate_visualization(
        self,
        img_array: np.ndarray,
        original_img: np.ndarray,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """
        Generate complete Grad-CAM visualization.

        Args:
            img_array: Preprocessed image for model (1, 224, 224, 3)
            original_img: Original image for visualization
            save_path: Path to save visualization

        Returns:
            Tuple of (heatmap, overlayed_image, prediction_label, confidence)
        """
        # Get prediction
        predictions = self.model.predict(img_array, verbose=0)
        confidence = float(predictions[0][0])

        # Determine predicted class
        pred_class = 1 if confidence > 0.5 else 0
        pred_label = self.class_names[pred_class]

        # Generate heatmap
        heatmap = self.get_gradcam_heatmap(img_array)

        # Overlay on original image
        overlayed = self.overlay_heatmap_on_image(original_img, heatmap)

        # Save visualization if path provided
        if save_path:
            self.save_visualization(
                original_img,
                heatmap,
                overlayed,
                pred_label,
                confidence,
                save_path
            )

        return heatmap, overlayed, pred_label, confidence

    def save_visualization(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        overlayed: np.ndarray,
        pred_label: str,
        confidence: float,
        save_path: str
    ) -> None:
        """
        Save Grad-CAM visualization as image file.

        Args:
            original_img: Original image
            heatmap: Grad-CAM heatmap
            overlayed: Overlayed image
            pred_label: Predicted label
            confidence: Prediction confidence
            save_path: Path to save the visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        # Overlayed
        axes[2].imshow(overlayed)

        # Color-code title based on prediction
        title_color = 'green' if pred_label == 'Real' else 'red'
        axes[2].set_title(
            f'Prediction: {pred_label}\nConfidence: {confidence:.2%}',
            fontsize=12,
            fontweight='bold',
            color=title_color
        )
        axes[2].axis('off')

        plt.tight_layout()

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save with high DPI
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✓ Saved Grad-CAM visualization to {save_path}")

    def batch_generate_examples(
        self,
        test_dataset: tf.data.Dataset,
        num_examples: int = 10,
        output_dir: str = 'results/gradcam_visualizations/'
    ) -> None:
        """
        Generate Grad-CAM visualizations for multiple test samples.

        Args:
            test_dataset: Test dataset
            num_examples: Number of examples to generate
            output_dir: Directory to save visualizations
        """
        logger.info(f"Generating {num_examples} Grad-CAM examples...")

        os.makedirs(output_dir, exist_ok=True)

        examples_generated = 0
        sample_idx = 0

        # Try to get diverse examples
        correct_real = []
        correct_ai = []
        incorrect_real = []
        incorrect_ai = []

        for images, labels in test_dataset:
            for i in range(images.shape[0]):
                if examples_generated >= num_examples * 2:  # Get extra to filter
                    break

                img = images[i:i+1]
                true_label = int(labels[i])
                original_img = images[i].numpy()

                # Get prediction
                predictions = self.model.predict(img, verbose=0)
                confidence = float(predictions[0][0])
                pred_class = 1 if confidence > 0.5 else 0

                # Categorize
                is_correct = (pred_class == true_label)

                if is_correct and true_label == 0 and len(correct_real) < 3:
                    correct_real.append((img, original_img, true_label, sample_idx))
                elif is_correct and true_label == 1 and len(correct_ai) < 3:
                    correct_ai.append((img, original_img, true_label, sample_idx))
                elif not is_correct and true_label == 0 and len(incorrect_real) < 2:
                    incorrect_real.append((img, original_img, true_label, sample_idx))
                elif not is_correct and true_label == 1 and len(incorrect_ai) < 2:
                    incorrect_ai.append((img, original_img, true_label, sample_idx))

                sample_idx += 1
                examples_generated += 1

            if examples_generated >= num_examples * 2:
                break

        # Combine diverse examples
        all_examples = (
            correct_real[:3] +
            correct_ai[:3] +
            incorrect_real[:2] +
            incorrect_ai[:2]
        )

        # Generate visualizations
        for idx, (img, original_img, true_label, sample_idx) in enumerate(all_examples[:num_examples]):
            save_path = os.path.join(
                output_dir,
                f'gradcam_example_{idx+1}_true_{self.class_names[true_label]}.png'
            )

            self.generate_visualization(img, original_img, save_path)

        logger.info(f"✓ Generated {min(len(all_examples), num_examples)} Grad-CAM examples")


def main():
    """
    Main function for Grad-CAM generation.
    """
    parser = argparse.ArgumentParser(description='Generate Grad-CAM Visualizations')

    parser.add_argument(
        '--model_path',
        type=str,
        default='models/truthpixel_final.h5',
        help='Path to trained model'
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=10,
        help='Number of examples to generate'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/gradcam_visualizations/',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size'
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # Load test dataset
    logger.info("Loading test dataset...")
    data_loader = CIFAKEDataLoader(data_dir="data", batch_size=args.batch_size)
    _, _, test_ds = data_loader.prepare_datasets(augment_train=False)

    # Create Grad-CAM generator
    gradcam = GradCAM(model=model)

    # Generate examples
    gradcam.batch_generate_examples(
        test_dataset=test_ds,
        num_examples=args.num_examples,
        output_dir=args.output_dir
    )

    logger.info("\n✓ Grad-CAM generation completed successfully!")
    logger.info(f"✓ Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
