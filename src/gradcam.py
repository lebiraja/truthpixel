"""
Grad-CAM Explainability Script - Phase 4

Generates Grad-CAM visualizations for model interpretability.

Usage:
    python src/gradcam.py --model all --samples 20
    python src/gradcam.py --model genimage --samples 10
"""

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from data_loader_multi import MultiDatasetLoader
from train_baseline import DeepfakeDetector


class GradCAMVisualizer:
    """Grad-CAM visualization for deepfake detection."""

    def __init__(self, model, target_layer, device):
        """
        Initialize Grad-CAM.

        Args:
            model: Trained model
            target_layer: Layer to compute gradients (e.g., model.backbone.blocks[-1])
            device: torch device
        """
        self.model = model
        self.device = device

        # Use last convolutional block
        self.cam = GradCAM(
            model=model,
            target_layers=[target_layer],
            use_cuda=(device.type == 'cuda')
        )

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, 3, H, W)
            target_class: Target class (1 for fake, 0 for real)

        Returns:
            Heatmap as numpy array
        """
        targets = [BinaryClassifierOutputTarget(target_class)]

        # Generate CAM
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)

        # Get first image from batch
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam

    def visualize(self, image_path, save_path=None):
        """
        Create Grad-CAM visualization for an image.

        Args:
            image_path: Path to input image
            save_path: Path to save visualization

        Returns:
            Visualization as numpy array
        """
        # Load and preprocess image
        from PIL import Image

        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))

        # To tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_tensor = transform(img).unsqueeze(0).to(self.device)

        # Get prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.item()
            predicted_class = int(prediction > 0.5)

        # Generate heatmap
        grayscale_cam = self.generate_heatmap(input_tensor, predicted_class)

        # Convert image to RGB array
        rgb_img = np.array(img).astype(np.float32) / 255.0

        # Overlay heatmap
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(visualization)
        prediction_text = f"{'FAKE' if predicted_class == 1 else 'REAL'} ({prediction:.2%})"
        axes[2].set_title(f'Overlay - Pred: {prediction_text}', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return visualization


def batch_generate_gradcam(model_name, dataset_name, config, num_samples=20):
    """Generate Grad-CAM visualizations for multiple samples."""
    print(f"\nGenerating Grad-CAM for {model_name} on {dataset_name}...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    models_dir = Path(config['paths']['models'])

    if model_name == 'combined':
        model_path = models_dir / 'combined' / 'combined_model_best.pt'
    else:
        model_path = models_dir / 'baseline' / f'{model_name}_baseline_best.pt'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = DeepfakeDetector().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get target layer (last conv block in EfficientNet)
    target_layer = model.backbone.blocks[-1]

    # Create visualizer
    visualizer = GradCAMVisualizer(model, target_layer, device)

    # Get sample images
    dataset_path = Path(config['data']['base_dir']) / dataset_name / 'test'

    fake_images = list((dataset_path / 'FAKE').glob('*.jpg'))[:num_samples//2]
    real_images = list((dataset_path / 'REAL').glob('*.jpg'))[:num_samples//2]

    if not fake_images:
        fake_images = list((dataset_path / 'FAKE').glob('*.png'))[:num_samples//2]
    if not real_images:
        real_images = list((dataset_path / 'REAL').glob('*.png'))[:num_samples//2]

    all_images = fake_images + real_images

    # Create output directory
    output_dir = Path(config['paths']['results']) / 'gradcam' / model_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    for idx, img_path in enumerate(all_images):
        save_path = output_dir / f'gradcam_{idx:03d}_{img_path.stem}.png'

        try:
            visualizer.visualize(img_path, save_path)
        except Exception as e:
            print(f"  Failed on {img_path.name}: {e}")

    print(f"  ✓ Saved {len(all_images)} visualizations to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument(
        '--model',
        type=str,
        default='all',
        choices=['genimage', 'cifake', 'faces', 'combined', 'all'],
        help='Model to visualize'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=20,
        help='Number of samples per model/dataset'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("="*80)
    print("GRAD-CAM GENERATION - PHASE 4")
    print("="*80)

    datasets = ['genimage', 'cifake', 'faces']

    if args.model == 'all':
        models = ['genimage', 'cifake', 'faces', 'combined']
    else:
        models = [args.model]

    try:
        for model_name in models:
            for dataset_name in datasets:
                batch_generate_gradcam(
                    model_name,
                    dataset_name,
                    config,
                    num_samples=args.samples
                )

        print("\n" + "="*80)
        print("✓ Grad-CAM generation complete!")
        print("="*80)
        print(f"\nVisualizations saved to: results/gradcam/")
        return 0

    except Exception as e:
        print(f"\n✗ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
