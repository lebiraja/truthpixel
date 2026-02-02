"""
TruthPixel Inference Module
Load model and make predictions on images
"""

import torch
from torchvision import transforms
from PIL import Image
import sys
sys.path.append('../src')
from model_pytorch import TruthPixelModelPyTorch


class TruthPixelDetector:
    """
    AI-generated image detector
    """

    def __init__(self, model_path='../models/phase_2_best.pth'):
        """
        Initialize detector with trained model

        Args:
            model_path: Path to trained model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on {self.device}...")

        # Load model
        self.model = TruthPixelModelPyTorch(freeze_base=False)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded! Validation accuracy: {checkpoint['val_acc']:.2f}%")

        # ImageNet preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        """
        Predict if image is AI-generated or real

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = output.item()

        # Interpret results
        # Class 0 = FAKE, Class 1 = REAL
        # Output close to 0 = FAKE, close to 1 = REAL

        if probability < 0.5:
            prediction = "AI-Generated (FAKE)"
            confidence = (1 - probability) * 100
        else:
            prediction = "Real (HUMAN)"
            confidence = probability * 100

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probability': probability,
            'is_fake': probability < 0.5
        }


if __name__ == "__main__":
    # Test the detector
    detector = TruthPixelDetector()

    # Example prediction
    result = detector.predict('../data/test/FAKE/0.png')
    print(f"\nPrediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
