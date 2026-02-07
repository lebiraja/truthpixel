"""
Multi-Dataset PyTorch Data Loading with Balanced Sampling

Supports training on 3 datasets simultaneously:
- GenImage (8 AI generators, 400K images)
- CIFAKE (120K images, Stable Diffusion v1.4)
- 140k Real and Fake Faces (StyleGAN)

Total: ~660K images from 9+ AI generators

Key Features:
- Balanced sampling with WeightedRandomSampler
- Dataset ID tracking for domain-aware training
- Progressive augmentation based on training phase
"""

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
from augmentation import get_augmentation_transforms, get_validation_transforms

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class DatasetWrapper(Dataset):
    """Wraps a dataset to add dataset ID to each sample."""

    def __init__(self, dataset: Dataset, dataset_id: int, dataset_name: str):
        """
        Args:
            dataset: Original PyTorch ImageFolder dataset
            dataset_id: Unique ID for this dataset
            dataset_name: Name of the dataset
        """
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Return (image, label, dataset_id)
        return image, label, self.dataset_id


class MultiDatasetLoader:
    """
    Multi-dataset loader with balanced sampling for training.

    Expected directory structure:
        data/
        ├── genimage/
        │   ├── train/{FAKE, REAL}/
        │   ├── val/{FAKE, REAL}/
        │   └── test/{FAKE, REAL}/
        ├── cifake/
        │   ├── train/{FAKE, REAL}/
        │   ├── val/{FAKE, REAL}/
        │   └── test/{FAKE, REAL}/
        └── faces/
            ├── train/{FAKE, REAL}/
            ├── val/{FAKE, REAL}/
            └── test/{FAKE, REAL}/
    """

    # Dataset configuration (NEW: 3 datasets with balanced weights)
    DATASETS = {
        'genimage': {'id': 0, 'weight': 0.44},  # 400K images, 44%
        'cifake': {'id': 1, 'weight': 0.18},    # 120K images, 18%
        'faces': {'id': 2, 'weight': 0.38}      # 140K images, 38%
    }

    def __init__(
        self,
        data_dir: str = "data",
        datasets_to_use: Optional[List[str]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
        pin_memory: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing all datasets
            datasets_to_use: List of dataset names (if None, uses all)
            custom_weights: Custom sampling weights (overrides defaults)
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            img_size: Image size (224 for EfficientNet)
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.data_dir = Path(data_dir)
        self.datasets_to_use = datasets_to_use or list(self.DATASETS.keys())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.pin_memory = pin_memory

        # Setup sampling weights
        self.weights = {}
        for ds_name in self.datasets_to_use:
            if custom_weights and ds_name in custom_weights:
                self.weights[ds_name] = custom_weights[ds_name]
            else:
                self.weights[ds_name] = self.DATASETS[ds_name]['weight']

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _load_dataset(
        self,
        dataset_name: str,
        split: str,
        transform
    ) -> Optional[DatasetWrapper]:
        """
        Load a single dataset split.

        Args:
            dataset_name: Name of dataset (cifake, faces, etc.)
            split: Split name (train, val, test)
            transform: Transform to apply

        Returns:
            DatasetWrapper or None if not found
        """
        dataset_path = self.data_dir / dataset_name / split

        if not dataset_path.exists():
            return None

        try:
            base_dataset = datasets.ImageFolder(
                root=str(dataset_path),
                transform=transform
            )

            dataset_id = self.DATASETS[dataset_name]['id']

            wrapped_dataset = DatasetWrapper(
                dataset=base_dataset,
                dataset_id=dataset_id,
                dataset_name=dataset_name
            )

            return wrapped_dataset

        except Exception as e:
            logger.warning(f"Failed to load {dataset_name}/{split}: {e}")
            return None

    def _create_balanced_sampler(
        self,
        datasets_list: List[DatasetWrapper]
    ) -> WeightedRandomSampler:
        """
        Create weighted sampler for balanced multi-dataset training.

        Args:
            datasets_list: List of dataset wrappers

        Returns:
            WeightedRandomSampler
        """
        sample_weights = []

        for dataset_wrapper in datasets_list:
            dataset_name = dataset_wrapper.dataset_name
            dataset_size = len(dataset_wrapper)
            dataset_weight = self.weights[dataset_name]

            # Weight per sample = dataset_weight / dataset_size
            weight_per_sample = dataset_weight / dataset_size

            sample_weights.extend([weight_per_sample] * dataset_size)

        sample_weights_tensor = torch.DoubleTensor(sample_weights)

        # Total samples = sum of all dataset sizes
        total_samples = sum(len(ds) for ds in datasets_list)

        return WeightedRandomSampler(
            weights=sample_weights_tensor,
            num_samples=total_samples,
            replacement=True
        )

    def create_loaders(
        self,
        phase: int = 1,
        balanced_sampling: bool = True
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create train, val, and test data loaders.

        Args:
            phase: Training phase (1, 2, or 3) - affects augmentation
            balanced_sampling: Whether to use weighted sampling for training

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        logger.info("=" * 80)
        logger.info("MULTI-DATASET LOADER")
        logger.info("=" * 80)
        logger.info(f"Phase: {phase}")
        logger.info(f"Datasets: {self.datasets_to_use}")
        logger.info(f"Balanced sampling: {balanced_sampling}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info("")

        # Get transforms
        train_transform = get_augmentation_transforms(phase=phase, img_size=self.img_size)
        val_transform = get_validation_transforms(img_size=self.img_size)

        # Load datasets
        train_datasets = []
        val_datasets = []
        test_datasets = []

        for dataset_name in self.datasets_to_use:
            logger.info(f"Loading {dataset_name}...")

            # Train
            train_ds = self._load_dataset(dataset_name, 'train', train_transform)
            if train_ds:
                train_datasets.append(train_ds)
                logger.info(f"  Train: {len(train_ds):,} samples")

            # Val
            val_ds = self._load_dataset(dataset_name, 'val', val_transform)
            if val_ds:
                val_datasets.append(val_ds)
                logger.info(f"  Val: {len(val_ds):,} samples")

            # Test
            test_ds = self._load_dataset(dataset_name, 'test', val_transform)
            if test_ds:
                test_datasets.append(test_ds)
                logger.info(f"  Test: {len(test_ds):,} samples")

        if not train_datasets:
            raise RuntimeError(
                f"No training datasets found in {self.data_dir}\n"
                f"Please run: python src/download_multi_datasets.py"
            )

        # Combine datasets
        combined_train = ConcatDataset(train_datasets)
        combined_val = ConcatDataset(val_datasets) if val_datasets else None
        combined_test = ConcatDataset(test_datasets) if test_datasets else None

        logger.info("")
        logger.info("-" * 80)
        logger.info(f"Total train samples: {len(combined_train):,}")
        if combined_val:
            logger.info(f"Total val samples: {len(combined_val):,}")
        if combined_test:
            logger.info(f"Total test samples: {len(combined_test):,}")

        # Create sampler and loader for training
        sampler = None
        shuffle = True

        if balanced_sampling and len(train_datasets) > 1:
            logger.info("")
            logger.info("Creating balanced sampler...")
            sampler = self._create_balanced_sampler(train_datasets)
            shuffle = False  # Sampler handles sampling

            # Log expected distribution
            logger.info("Expected sampling distribution per epoch:")
            for ds_name in self.datasets_to_use:
                weight = self.weights[ds_name]
                expected = int(len(combined_train) * weight)
                logger.info(f"  {ds_name}: ~{expected:,} samples ({weight*100:.1f}%)")

        train_loader = DataLoader(
            combined_train,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True  # Drop last incomplete batch
        )

        val_loader = None
        if combined_val:
            val_loader = DataLoader(
                combined_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        test_loader = None
        if combined_test:
            test_loader = DataLoader(
                combined_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )

        logger.info("")
        logger.info("✓ Data loaders created")
        logger.info(f"  Train batches: {len(train_loader):,}")
        if val_loader:
            logger.info(f"  Val batches: {len(val_loader):,}")
        if test_loader:
            logger.info(f"  Test batches: {len(test_loader):,}")
        logger.info("=" * 80)
        logger.info("")

        return train_loader, val_loader, test_loader

    def create_single_dataset_loader(
        self,
        dataset_name: str,
        phase: int = 1
    ) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Create loaders for a single dataset (used for baseline training).

        Args:
            dataset_name: Name of dataset ('genimage', 'cifake', or 'faces')
            phase: Training phase (affects augmentation)

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info("=" * 80)
        logger.info(f"SINGLE DATASET LOADER: {dataset_name.upper()}")
        logger.info("=" * 80)

        # Get transforms
        train_transform = get_augmentation_transforms(phase=phase, img_size=self.img_size)
        val_transform = get_validation_transforms(img_size=self.img_size)

        # Load dataset
        train_ds = self._load_dataset(dataset_name, 'train', train_transform)
        val_ds = self._load_dataset(dataset_name, 'val', val_transform)
        test_ds = self._load_dataset(dataset_name, 'test', val_transform)

        if not train_ds:
            raise RuntimeError(f"Training data not found for {dataset_name}")

        logger.info(f"Train: {len(train_ds):,} samples")
        if val_ds:
            logger.info(f"Val: {len(val_ds):,} samples")
        if test_ds:
            logger.info(f"Test: {len(test_ds):,} samples")

        # Create loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if val_ds else None

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        ) if test_ds else None

        logger.info("=" * 80)
        logger.info("")

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test multi-dataset loader
    logger.info("TESTING MULTI-DATASET LOADER - AuthentiScan")
    logger.info("=" * 80)
    logger.info("")

    # Test single dataset (for baseline training)
    logger.info("TEST 1: Single Dataset Loader (GenImage baseline)")
    logger.info("-" * 80)

    try:
        loader_single = MultiDatasetLoader(
            data_dir="data",
            batch_size=8,
            num_workers=2
        )

        train_loader, val_loader, test_loader = loader_single.create_single_dataset_loader(
            dataset_name='genimage',
            phase=1
        )

        # Test batch
        images, labels, dataset_ids = next(iter(train_loader))
        logger.info(f"✓ Batch shape: {images.shape}")
        logger.info(f"  Labels: {labels.shape}, unique: {labels.unique().tolist()}")
        logger.info(f"  Dataset IDs: {dataset_ids.unique().tolist()}")
        logger.info("")

    except Exception as e:
        logger.error(f"✗ Test 1 failed: {e}")
        logger.info("")

    # Test multi-dataset with balanced sampling (for combined model)
    logger.info("TEST 2: Multi-Dataset with Balanced Sampling")
    logger.info("-" * 80)

    try:
        loader_multi = MultiDatasetLoader(
            data_dir="data",
            datasets_to_use=['genimage', 'cifake', 'faces'],
            batch_size=16,
            num_workers=2
        )

        train_loader, val_loader, test_loader = loader_multi.create_loaders(
            phase=2,
            balanced_sampling=True
        )

        # Test batch
        images, labels, dataset_ids = next(iter(train_loader))
        logger.info(f"✓ Batch shape: {images.shape}")
        logger.info(f"  Dataset distribution in batch:")
        for ds_id in range(3):
            count = (dataset_ids == ds_id).sum().item()
            if count > 0:
                ds_name = [k for k, v in loader_multi.DATASETS.items() if v['id'] == ds_id][0]
                logger.info(f"    {ds_name} (ID={ds_id}): {count} samples")
        logger.info("")

    except Exception as e:
        logger.error(f"✗ Test 2 failed: {e}")
        logger.info("")

    logger.info("=" * 80)
    logger.info("✓ All tests completed!")
