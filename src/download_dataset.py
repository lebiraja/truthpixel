"""
Manual Dataset Downloader for CIFAKE Dataset.

Downloads and prepares the CIFAKE dataset from Kaggle or direct sources.
"""

import os
import zipfile
import requests
from tqdm import tqdm
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url: str, destination: str) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Local file path to save to
    """
    logger.info(f"Downloading from {url}")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    os.makedirs(os.path.dirname(destination), exist_ok=True)

    with open(destination, 'wb') as f, tqdm(
        desc=destination,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"✓ Downloaded to {destination}")


def download_cifake_from_kaggle():
    """
    Download CIFAKE dataset from Kaggle.

    Note: Requires Kaggle API credentials.
    Setup: https://www.kaggle.com/docs/api
    """
    logger.info("Downloading CIFAKE from Kaggle...")

    try:
        import kaggle

        # Download dataset
        kaggle.api.dataset_download_files(
            'birdy654/cifake-real-and-ai-generated-synthetic-images',
            path='data/raw',
            unzip=True
        )

        logger.info("✓ Downloaded CIFAKE from Kaggle")
        return True

    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Failed to download from Kaggle: {e}")
        logger.info("Please download manually from:")
        logger.info("https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
        return False


def download_cifake_manual():
    """
    Instructions for manual download.
    """
    print("\n" + "=" * 80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 80)
    print("\nOption 1: Kaggle (Recommended)")
    print("-" * 80)
    print("1. Go to: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images")
    print("2. Click 'Download' button")
    print("3. Extract the zip file")
    print("4. Move the extracted folders to: data/raw/")
    print("   Expected structure:")
    print("   data/raw/")
    print("     ├── train/")
    print("     │   ├── REAL/")
    print("     │   └── FAKE/")
    print("     └── test/")
    print("         ├── REAL/")
    print("         └── FAKE/")

    print("\nOption 2: Direct Download from HuggingFace")
    print("-" * 80)
    print("1. Download train.zip from:")
    print("   https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible/resolve/main/train.zip")
    print("2. Extract to data/raw/")

    print("\nOption 3: Use Kaggle API")
    print("-" * 80)
    print("1. Install: pip install kaggle")
    print("2. Setup credentials: https://www.kaggle.com/docs/api")
    print("3. Run: python src/download_dataset.py --use-kaggle")

    print("\n" + "=" * 80)
    print("\nAfter downloading, run: python src/download_dataset.py --organize")
    print("=" * 80 + "\n")


def organize_dataset():
    """
    Organize downloaded dataset into train/val/test splits.
    """
    import numpy as np
    from PIL import Image

    logger.info("Organizing dataset...")

    raw_dir = Path("data/raw")

    # Find all images
    real_images = []
    fake_images = []

    # Look for REAL and FAKE folders
    for folder in raw_dir.rglob("*"):
        if folder.is_dir():
            folder_name = folder.name.upper()
            if "REAL" in folder_name:
                real_images.extend(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))
            elif "FAKE" in folder_name or "AI" in folder_name:
                fake_images.extend(list(folder.glob("*.png")) + list(folder.glob("*.jpg")))

    logger.info(f"Found {len(real_images)} REAL images")
    logger.info(f"Found {len(fake_images)} FAKE images")

    if len(real_images) == 0 or len(fake_images) == 0:
        logger.error("No images found! Please download dataset first.")
        download_cifake_manual()
        return False

    # Shuffle
    np.random.seed(42)
    np.random.shuffle(real_images)
    np.random.shuffle(fake_images)

    # Calculate splits (70/15/15)
    n_real = len(real_images)
    n_fake = len(fake_images)

    real_train_end = int(n_real * 0.7)
    real_val_end = int(n_real * 0.85)

    fake_train_end = int(n_fake * 0.7)
    fake_val_end = int(n_fake * 0.85)

    # Split datasets
    splits = {
        'train': {
            'REAL': real_images[:real_train_end],
            'FAKE': fake_images[:fake_train_end]
        },
        'val': {
            'REAL': real_images[real_train_end:real_val_end],
            'FAKE': fake_images[fake_train_end:fake_val_end]
        },
        'test': {
            'REAL': real_images[real_val_end:],
            'FAKE': fake_images[fake_val_end:]
        }
    }

    # Copy files to organized structure
    for split_name, split_data in splits.items():
        for class_name, image_list in split_data.items():
            output_dir = Path(f"data/{split_name}/{class_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Copying {len(image_list)} {class_name} images to {split_name}...")

            for img_path in tqdm(image_list, desc=f"{split_name}/{class_name}"):
                try:
                    # Copy file
                    dest = output_dir / img_path.name
                    shutil.copy2(img_path, dest)
                except Exception as e:
                    logger.warning(f"Failed to copy {img_path}: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("DATASET ORGANIZATION COMPLETE")
    print("=" * 80)
    for split_name in ['train', 'val', 'test']:
        real_count = len(list(Path(f"data/{split_name}/REAL").glob("*")))
        fake_count = len(list(Path(f"data/{split_name}/FAKE").glob("*")))
        total = real_count + fake_count
        print(f"\n{split_name.upper()}:")
        print(f"  REAL: {real_count:,}")
        print(f"  FAKE: {fake_count:,}")
        print(f"  Total: {total:,}")
    print("=" * 80 + "\n")

    logger.info("✓ Dataset organized successfully!")
    logger.info("You can now run: python src/train.py")

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Download and organize CIFAKE dataset')
    parser.add_argument('--use-kaggle', action='store_true', help='Download using Kaggle API')
    parser.add_argument('--organize', action='store_true', help='Organize downloaded dataset')
    parser.add_argument('--manual', action='store_true', help='Show manual download instructions')

    args = parser.parse_args()

    if args.use_kaggle:
        success = download_cifake_from_kaggle()
        if success:
            organize_dataset()
    elif args.organize:
        organize_dataset()
    else:
        # Default: show manual instructions
        download_cifake_manual()


if __name__ == "__main__":
    main()
