# AuthentiScan API Reference

This document provides reference documentation for the core modules and functions in AuthentiScan.

## Core Modules

### `src.data_loader_multi`

Handles loading and processing of multiple datasets (GenImage, CIFAKE, Faces).

#### `class MultiDatasetLoader`

```python
def __init__(self, data_dir="data", datasets_to_use=None, batch_size=32, num_workers=4, img_size=224):
    """
    Initialize the multi-dataset loader.

    Args:
        data_dir (str): Root directory containing datasets.
        datasets_to_use (list): List of dataset names to include (default: all).
        batch_size (int): Batch size for training.
        num_workers (int): Number of worker threads.
        img_size (int): Input image size (default: 224).
    """

def create_loaders(self, phase=1, balanced_sampling=True):
    """
    Create train, validation, and test loaders.

    Args:
        phase (int): Training phase (1-4). Affects augmentation intensity.
        balanced_sampling (bool): Whether to use WeightedRandomSampler.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """

def create_single_dataset_loader(self, dataset_name, phase=1):
    """
    Create loaders for a single dataset (Phase 1).

    Args:
        dataset_name (str): Name of dataset ('genimage', 'cifake', 'faces').
        phase (int): Training phase.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
```

---

### `src.train_baseline`

Implements Phase 1 training (baseline models).

#### `class DeepfakeDetector(nn.Module)`

EfficientNetB0-based binary classifier.

```python
def __init__(self, pretrained=True, dropout=0.5):
    """
    Initialize the model.

    Args:
        pretrained (bool): Use ImageNet weights.
        dropout (float): Dropout rate for classification head.
    """

def forward(self, x):
    """
    Forward pass.

    Args:
        x (tensor): Input tensor (B, 3, 224, 224).

    Returns:
        tensor: Probability score (0-1).
    """
```

#### Training Functions

```python
def train_single_dataset_model(dataset_name, config):
    """
    Train a baseline model for a specific dataset.

    Args:
        dataset_name (str): Target dataset.
        config (dict): Configuration dictionary.

    Returns:
        float: Best validation accuracy.
    """
```

---

### `src.train_combined`

Implements Phase 2 training (combined model).

```python
def train_combined_model(config):
    """
    Train a unified model on all datasets.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        float: Best validation accuracy.
    """
```

---

### `src.cross_validate`

Implements Phase 3 (cross-dataset validation).

```python
def evaluate_model_on_dataset(model, dataset_name, config):
    """
    Evaluate a specific model on a specific dataset.

    Args:
        model (nn.Module): Loaded model.
        dataset_name (str): Target dataset for evaluation.
        config (dict): Configuration.

    Returns:
        dict: Metrics (accuracy, loss, etc.).
    """

def generate_cross_validation_matrix(models_dir, output_dir):
    """
    Generate and save cross-validation performance matrix.

    Args:
        models_dir (str): Directory containing trained models.
        output_dir (str): Output directory for results.
    """
```

---

### `src.gradcam`

Implements Phase 4 (explainability).

#### `class GradCAM`

```python
def __init__(self, model, target_layer=None):
    """
    Initialize Grad-CAM.

    Args:
        model (nn.Module): Trained model.
        target_layer (nn.Module): Target convolutional layer (default: last conv).
    """

def generate_heatmap(self, input_tensor, target_class=None):
    """
    Generate heatmap for input image.

    Args:
        input_tensor (tensor): Input image tensor.
        target_class (int): Target class index.

    Returns:
        numpy.ndarray: Heatmap (0-1).
    """
```

---

### `src.evaluate`

Comprehensive evaluation script.

```python
def evaluate_all_models(config):
    """
    Run full evaluation pipeline on all trained models.

    Generates:
    - JSON metrics
    - ROC curves
    - Confusion matrices
    """
```

---

## Helper Scripts

### `src.download_multi_datasets.py`
Downloads datasets from Kaggle and Google Drive.

### `src.organize_datasets.py`
Organizes raw downloaded data into standardized directory structure.

### `scripts/verify_datasets.py`
Verifies dataset integrity and structure.
