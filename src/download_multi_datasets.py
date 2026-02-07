"""
Dataset Download Script for AuthentiScan/TruthPixel

Downloads 3 datasets for multi-dataset training:
1. GenImage (Manual Google Drive download)
2. CIFAKE (Kaggle)
3. 140k Real and Fake Faces (Kaggle)

Total: ~18GB, 660K images, 9+ AI generators
"""

import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from typing import Optional

# Try importing HuggingFace datasets
try:
    from datasets import load_dataset
    from PIL import Image
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def create_directory_structure(base_dir: str = "data"):
    """Create the required directory structure."""
    print_header("Creating Directory Structure")

    directories = [
        # GenImage
        f"{base_dir}/genimage/train/FAKE",
        f"{base_dir}/genimage/train/REAL",
        f"{base_dir}/genimage/val/FAKE",
        f"{base_dir}/genimage/val/REAL",
        f"{base_dir}/genimage/test/FAKE",
        f"{base_dir}/genimage/test/REAL",

        # CIFAKE
        f"{base_dir}/cifake/train/FAKE",
        f"{base_dir}/cifake/train/REAL",
        f"{base_dir}/cifake/val/FAKE",
        f"{base_dir}/cifake/val/REAL",
        f"{base_dir}/cifake/test/FAKE",
        f"{base_dir}/cifake/test/REAL",

        # Faces
        f"{base_dir}/faces/train/FAKE",
        f"{base_dir}/faces/train/REAL",
        f"{base_dir}/faces/val/FAKE",
        f"{base_dir}/faces/val/REAL",
        f"{base_dir}/faces/test/FAKE",
        f"{base_dir}/faces/test/REAL",

        # Downloads directory
        f"{base_dir}/downloads",
        f"{base_dir}/downloads/genimage",
        f"{base_dir}/downloads/cifake",
        f"{base_dir}/downloads/faces"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"✓ Created {len(directories)} directories")


def check_kaggle_setup():
    """Check if Kaggle CLI is properly set up."""
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Kaggle CLI found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("\n✗ Kaggle CLI not found!")
    print("\nSetup instructions:")
    print("  1. Install Kaggle: pip install kaggle")
    print("  2. Get API credentials:")
    print("     - Go to https://www.kaggle.com/settings")
    print("     - Scroll to 'API' and click 'Create New Token'")
    print("     - Save kaggle.json to ~/.kaggle/kaggle.json")
    print("  3. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    print("")
    return False


def check_dataset_exists(output_dir: Path) -> bool:
    """Check if dataset already exists and has files."""
    if not output_dir.exists():
        return False

    # Check if directory has image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    for ext in image_extensions:
        if list(output_dir.rglob(f'*{ext}')) or list(output_dir.rglob(f'*{ext.upper()}')):
            return True

    return False


def download_kaggle_dataset(slug: str, output_dir: str) -> bool:
    """Download a Kaggle dataset."""
    print(f"\nDownloading: {slug}")

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", output_dir],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"✗ Download failed: {result.stderr}")
            return False

        print(f"✓ Downloaded successfully")

        # Find and extract zip file
        zip_files = list(Path(output_dir).glob("*.zip"))
        if zip_files:
            zip_file = zip_files[0]
            print(f"  Extracting {zip_file.name}...")

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(output_dir)

            zip_file.unlink()  # Remove zip file
            print(f"✓ Extracted and cleaned up")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def download_huggingface_dataset(name: str, output_dir: str) -> bool:
    """Download a HuggingFace dataset."""
    if not HF_AVAILABLE:
        print("✗ HuggingFace datasets library not available")
        print("  Install with: pip install datasets pillow")
        return False

    print(f"\nDownloading: {name}")

    try:
        dataset = load_dataset(name, split="train")
        print(f"✓ Loaded {len(dataset)} samples")

        fake_dir = Path(output_dir) / "FAKE"
        real_dir = Path(output_dir) / "REAL"
        fake_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)

        print("  Saving images...")

        for idx, sample in enumerate(dataset):
            try:
                # Adjust based on actual dataset structure
                if 'image' in sample and 'label' in sample:
                    img = sample['image']
                    label = sample['label']

                    # Determine directory based on label
                    # (adjust logic based on actual label encoding)
                    if label == 0 or str(label).lower() in ['fake', '0']:
                        save_dir = fake_dir
                    else:
                        save_dir = real_dir

                    img.save(save_dir / f"image_{idx:06d}.png")

                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1} images...")

            except Exception as e:
                print(f"  Warning: Failed to process sample {idx}: {e}")
                continue

        print(f"✓ Saved {len(dataset)} images")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Main download function."""
    print_header("Multi-Dataset Download Script")

    base_dir = Path(__file__).parent.parent / "data"
    downloads_dir = base_dir / "downloads"

    print(f"\nBase directory: {base_dir}")

    # Create directory structure
    create_directory_structure(str(base_dir))

    # Check Kaggle setup
    print_header("Checking Kaggle Setup")
    kaggle_available = check_kaggle_setup()

    if not kaggle_available:
        print("\n⚠ Kaggle CLI not available. Kaggle datasets will be skipped.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Dataset configurations
    datasets = [
        {
            'name': 'GenImage',
            'source': 'manual',
            'url': 'https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS',
            'output': downloads_dir / 'genimage',
            'size': '~8GB',
            'images': '400K',
            'instructions': (
                "GenImage requires manual download from Google Drive:\n"
                "  1. Visit: https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS\n"
                "  2. Download all folders (real/ and fake/ with 8 generators)\n"
                "  3. Extract to: data/downloads/genimage/\n"
                "  4. Final structure should be:\n"
                "     data/downloads/genimage/\n"
                "       ├── real/\n"
                "       └── fake/\n"
                "           ├── stable_diffusion_v1.4/\n"
                "           ├── stable_diffusion_v1.5/\n"
                "           ├── midjourney/\n"
                "           ├── glide/\n"
                "           ├── adm/\n"
                "           ├── vqdm/\n"
                "           ├── biggan/\n"
                "           └── wukong/\n"
                "  5. Then run: python src/organize_datasets.py --datasets genimage"
            )
        },
        {
            'name': 'CIFAKE',
            'source': 'kaggle',
            'slug': 'birdy654/cifake-real-and-ai-generated-synthetic-images',
            'output': downloads_dir / 'cifake',
            'size': '~3GB',
            'images': '120K'
        },
        {
            'name': '140k Real and Fake Faces',
            'source': 'kaggle',
            'slug': 'xhlulu/140k-real-and-fake-faces',
            'output': downloads_dir / 'faces',
            'size': '~7GB',
            'images': '140K'
        }
    ]

    # Check existing datasets
    print_header("Checking Existing Datasets")

    existing = []
    missing = []

    for ds in datasets:
        if check_dataset_exists(ds['output']):
            existing.append(ds['name'])
            print(f"✓ {ds['name']} - Already downloaded")
        else:
            missing.append(ds['name'])
            print(f"⊗ {ds['name']} - Not found")

    if existing:
        print(f"\n{len(existing)} dataset(s) already exist and will be skipped.")

    if not missing:
        print("\n✓ All datasets already downloaded!")
        print("\nNext steps:")
        print("  python src/organize_datasets.py")
        return

    print(f"\n{len(missing)} dataset(s) need to be downloaded.")

    # Download datasets
    print_header("Downloading Missing Datasets")

    success_count = 0
    skipped_count = len(existing)

    for ds in datasets:
        print(f"\n{'-' * 80}")
        print(f"Dataset: {ds['name']}")
        print(f"Source: {ds['source']}")
        print(f"Size: {ds.get('size', 'N/A')}")
        print(f"Images: {ds.get('images', 'N/A')}")

        # Check if already exists
        if check_dataset_exists(ds['output']):
            print(f"✓ Already exists, skipping download")
            continue

        ds['output'].mkdir(parents=True, exist_ok=True)

        if ds['source'] == 'manual':
            print("\n⚠ Manual download required:")
            print(ds.get('instructions', 'No instructions available'))
            print("\nAfter downloading, re-run this script to verify.")
        elif ds['source'] == 'kaggle' and kaggle_available:
            if download_kaggle_dataset(ds['slug'], str(ds['output'])):
                success_count += 1
        else:
            print(f"⊗ Skipped (Kaggle not available)")

    # Summary
    print_header("Download Summary")
    print(f"Already existed: {skipped_count}/{len(datasets)} datasets")
    print(f"Successfully downloaded: {success_count}/{len(datasets) - skipped_count} datasets")
    print(f"Total available: {success_count + skipped_count}/{len(datasets)} datasets")
    print(f"\nData directory: {base_dir}")

    if success_count > 0 or any(ds['source'] == 'manual' for ds in datasets):
        print("\n⚠ IMPORTANT: Post-download organization required!")
        print("\nAfter all datasets are downloaded, organize them with:")
        print("  python src/organize_datasets.py --datasets genimage cifake faces")
        print("\nOr organize individually:")
        print("  python src/organize_datasets.py --datasets genimage")
        print("  python src/organize_datasets.py --datasets cifake")
        print("  python src/organize_datasets.py --datasets faces")
        print("\nExpected splits:")
        print("  - GenImage: 70% train, 15% val, 15% test (~400K images)")
        print("  - CIFAKE: 70% train, 15% val, 15% test (~120K images)")
        print("  - Faces: 60% train, 20% val, 20% test (~140K images)")
    else:
        print("\n✗ No datasets downloaded successfully.")
        print("Please check error messages above and resolve issues.")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
