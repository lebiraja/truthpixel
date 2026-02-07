"""
Dataset Organization Script (No Subsampling)

Organizes downloaded datasets into train/val/test splits.
Does NOT subsample - keeps all images.
"""

import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import argparse
from tqdm import tqdm


def format_size(bytes: int) -> str:
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def get_dir_size(path: Path) -> int:
    """Get directory size in bytes."""
    total = 0
    try:
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception as e:
        print(f"Warning: {e}")
    return total


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG'}
    files = []

    if not directory.exists():
        return files

    for f in directory.rglob('*'):
        if f.is_file() and f.suffix in extensions:
            files.append(f)

    return files


def copy_files(files: List[Path], dest_dir: Path, desc: str = "Copying"):
    """Copy files to destination directory with progress bar."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(files, desc=desc, leave=False):
        try:
            dest_file = dest_dir / file.name
            # Skip if already exists
            if not dest_file.exists():
                shutil.copy2(file, dest_file)
        except Exception as e:
            print(f"Warning: Could not copy {file.name}: {e}")


def organize_cifake(downloads_dir: Path, output_dir: Path):
    """Organize CIFAKE dataset."""
    print("\n" + "="*70)
    print("ORGANIZING CIFAKE")
    print("="*70)

    cifake_src = downloads_dir / "cifake"

    if not cifake_src.exists():
        print("✗ CIFAKE not found")
        return False

    # Try to find train/test structure
    train_fake = None
    train_real = None
    test_fake = None
    test_real = None

    # Search for the actual directories
    for root, dirs, files in os.walk(cifake_src):
        root_path = Path(root)

        if 'train' in root.lower():
            if 'FAKE' in dirs or 'fake' in dirs:
                train_fake = root_path / ('FAKE' if 'FAKE' in dirs else 'fake')
            if 'REAL' in dirs or 'real' in dirs:
                train_real = root_path / ('REAL' if 'REAL' in dirs else 'real')

        if 'test' in root.lower():
            if 'FAKE' in dirs or 'fake' in dirs:
                test_fake = root_path / ('FAKE' if 'FAKE' in dirs else 'fake')
            if 'REAL' in dirs or 'real' in dirs:
                test_real = root_path / ('REAL' if 'REAL' in dirs else 'real')

    if not all([train_fake, train_real, test_fake, test_real]):
        print("✗ Could not find train/test FAKE/REAL structure")
        print("  Please manually check:", cifake_src)
        return False

    # Get all images
    fake_train_imgs = get_image_files(train_fake)
    real_train_imgs = get_image_files(train_real)
    fake_test_imgs = get_image_files(test_fake)
    real_test_imgs = get_image_files(test_real)

    print(f"Found:")
    print(f"  Train: {len(fake_train_imgs)} FAKE, {len(real_train_imgs)} REAL")
    print(f"  Test:  {len(fake_test_imgs)} FAKE, {len(real_test_imgs)} REAL")

    # Create 80/20 split from train for train/val
    random.seed(42)
    random.shuffle(fake_train_imgs)
    random.shuffle(real_train_imgs)

    fake_val_idx = int(len(fake_train_imgs) * 0.8)
    real_val_idx = int(len(real_train_imgs) * 0.8)

    fake_train = fake_train_imgs[:fake_val_idx]
    fake_val = fake_train_imgs[fake_val_idx:]
    real_train = real_train_imgs[:real_val_idx]
    real_val = real_train_imgs[real_val_idx:]

    print(f"\nSplitting:")
    print(f"  Train: {len(fake_train)} FAKE, {len(real_train)} REAL")
    print(f"  Val:   {len(fake_val)} FAKE, {len(real_val)} REAL")
    print(f"  Test:  {len(fake_test_imgs)} FAKE, {len(real_test_imgs)} REAL")

    # Copy files
    cifake_out = output_dir / "cifake"
    copy_files(fake_train, cifake_out / "train" / "FAKE", "Train FAKE")
    copy_files(real_train, cifake_out / "train" / "REAL", "Train REAL")
    copy_files(fake_val, cifake_out / "val" / "FAKE", "Val FAKE")
    copy_files(real_val, cifake_out / "val" / "REAL", "Val REAL")
    copy_files(fake_test_imgs, cifake_out / "test" / "FAKE", "Test FAKE")
    copy_files(real_test_imgs, cifake_out / "test" / "REAL", "Test REAL")

    size = get_dir_size(cifake_out)
    print(f"\n✓ CIFAKE organized: {format_size(size)}")
    return True


def organize_faces(downloads_dir: Path, output_dir: Path):
    """Organize Faces dataset."""
    print("\n" + "="*70)
    print("ORGANIZING FACES")
    print("="*70)

    faces_src = downloads_dir / "faces"

    if not faces_src.exists():
        print("✗ Faces not found")
        return False

    # Find real and fake directories
    real_imgs = []
    fake_imgs = []

    # Common patterns
    for root, dirs, files in os.walk(faces_src):
        root_lower = root.lower()

        if 'real' in root_lower and 'fake' not in root_lower:
            real_imgs.extend(get_image_files(Path(root)))
        elif 'fake' in root_lower or 'ai' in root_lower or 'generated' in root_lower:
            fake_imgs.extend(get_image_files(Path(root)))

    if not real_imgs or not fake_imgs:
        print("✗ Could not find real/fake images")
        print("  Please check:", faces_src)
        return False

    print(f"Found:")
    print(f"  REAL: {len(real_imgs)} images")
    print(f"  FAKE: {len(fake_imgs)} images")

    # Balance classes
    min_count = min(len(real_imgs), len(fake_imgs))
    real_imgs = real_imgs[:min_count]
    fake_imgs = fake_imgs[:min_count]

    print(f"\nBalanced to {min_count} images per class")

    # Split: 60% train, 20% val, 20% test
    random.seed(42)
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)

    split1 = int(len(real_imgs) * 0.6)
    split2 = int(len(real_imgs) * 0.8)

    real_train = real_imgs[:split1]
    real_val = real_imgs[split1:split2]
    real_test = real_imgs[split2:]

    fake_train = fake_imgs[:split1]
    fake_val = fake_imgs[split1:split2]
    fake_test = fake_imgs[split2:]

    print(f"\nSplits:")
    print(f"  Train: {len(fake_train)} FAKE, {len(real_train)} REAL")
    print(f"  Val:   {len(fake_val)} FAKE, {len(real_val)} REAL")
    print(f"  Test:  {len(fake_test)} FAKE, {len(real_test)} REAL")

    # Copy files
    faces_out = output_dir / "faces"
    copy_files(fake_train, faces_out / "train" / "FAKE", "Train FAKE")
    copy_files(real_train, faces_out / "train" / "REAL", "Train REAL")
    copy_files(fake_val, faces_out / "val" / "FAKE", "Val FAKE")
    copy_files(real_val, faces_out / "val" / "REAL", "Val REAL")
    copy_files(fake_test, faces_out / "test" / "FAKE", "Test FAKE")
    copy_files(real_test, faces_out / "test" / "REAL", "Test REAL")

    size = get_dir_size(faces_out)
    print(f"\n✓ Faces organized: {format_size(size)}")
    return True


def organize_genimage(downloads_dir: Path, output_dir: Path):
    """
    Organize GenImage dataset into train/val/test splits.

    GenImage structure (after manual download):
    data/downloads/genimage/
        ├── real/
        │   └── [image files]
        └── fake/
            ├── stable_diffusion_v1.4/
            ├── stable_diffusion_v1.5/
            ├── midjourney/
            ├── glide/
            ├── adm/
            ├── vqdm/
            ├── biggan/
            └── wukong/

    Target structure:
    data/genimage/
        ├── train/{FAKE, REAL}/
        ├── val/{FAKE, REAL}/
        └── test/{FAKE, REAL}/
    """
    print("\n" + "="*70)
    print("ORGANIZING GENIMAGE")
    print("="*70)

    genimage_src = downloads_dir / "genimage"

    if not genimage_src.exists():
        print("✗ GenImage not found")
        print("  Please download manually from Google Drive:")
        print("  https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS")
        return False

    # Find real images
    real_dir = genimage_src / "real"
    if not real_dir.exists():
        print("✗ real/ directory not found")
        return False

    real_imgs = get_image_files(real_dir)
    print(f"Found {len(real_imgs)} REAL images")

    # Find fake images from all generators
    fake_dir = genimage_src / "fake"
    if not fake_dir.exists():
        print("✗ fake/ directory not found")
        return False

    fake_imgs = []
    generators = ['stable_diffusion_v1.4', 'stable_diffusion_v1.5', 'midjourney',
                  'glide', 'adm', 'vqdm', 'biggan', 'wukong']

    generator_stats = {}
    for gen in generators:
        gen_dir = fake_dir / gen
        if gen_dir.exists():
            gen_imgs = get_image_files(gen_dir)
            fake_imgs.extend(gen_imgs)
            generator_stats[gen] = len(gen_imgs)
            print(f"  {gen}: {len(gen_imgs)} images")

    print(f"\nTotal FAKE images: {len(fake_imgs)} from {len(generator_stats)} generators")

    if not real_imgs or not fake_imgs:
        print("✗ Could not find real/fake images")
        print("  Please check:", genimage_src)
        return False

    # Balance classes (use minimum count)
    min_count = min(len(real_imgs), len(fake_imgs))
    print(f"\nBalancing to {min_count} images per class")

    random.seed(42)
    random.shuffle(real_imgs)
    random.shuffle(fake_imgs)

    real_imgs = real_imgs[:min_count]
    fake_imgs = fake_imgs[:min_count]

    # Split: 70% train, 15% val, 15% test
    split1 = int(len(real_imgs) * 0.7)
    split2 = int(len(real_imgs) * 0.85)

    real_train = real_imgs[:split1]
    real_val = real_imgs[split1:split2]
    real_test = real_imgs[split2:]

    fake_train = fake_imgs[:split1]
    fake_val = fake_imgs[split1:split2]
    fake_test = fake_imgs[split2:]

    print(f"\nSplits:")
    print(f"  Train: {len(fake_train)} FAKE, {len(real_train)} REAL")
    print(f"  Val:   {len(fake_val)} FAKE, {len(real_val)} REAL")
    print(f"  Test:  {len(fake_test)} FAKE, {len(real_test)} REAL")

    # Copy files
    genimage_out = output_dir / "genimage"
    copy_files(fake_train, genimage_out / "train" / "FAKE", "Train FAKE")
    copy_files(real_train, genimage_out / "train" / "REAL", "Train REAL")
    copy_files(fake_val, genimage_out / "val" / "FAKE", "Val FAKE")
    copy_files(real_val, genimage_out / "val" / "REAL", "Val REAL")
    copy_files(fake_test, genimage_out / "test" / "FAKE", "Test FAKE")
    copy_files(real_test, genimage_out / "test" / "REAL", "Test REAL")

    size = get_dir_size(genimage_out)
    print(f"\n✓ GenImage organized: {format_size(size)}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Organize datasets into train/val/test")
    parser.add_argument('--downloads-dir', type=str, default='data/downloads',
                        help='Directory containing downloaded datasets')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory for organized datasets')
    parser.add_argument('--datasets', nargs='+',
                        choices=['genimage', 'cifake', 'faces', 'all'],
                        default=['all'],
                        help='Which datasets to organize')

    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    output_dir = Path(args.output_dir)

    print("="*70)
    print("DATASET ORGANIZATION - AuthentiScan")
    print("="*70)
    print(f"Source: {downloads_dir}")
    print(f"Output: {output_dir}")
    print("")

    # Find available datasets
    available = []
    if (downloads_dir / "genimage").exists():
        available.append("genimage")
    if (downloads_dir / "cifake").exists():
        available.append("cifake")
    if (downloads_dir / "faces").exists():
        available.append("faces")

    if not available:
        print("✗ No datasets found in downloads directory!")
        print("\nExpected structure:")
        print("  data/downloads/genimage/")
        print("  data/downloads/cifake/")
        print("  data/downloads/faces/")
        print("\nRun first: python src/download_multi_datasets.py")
        return

    print(f"Available: {', '.join(available)}")

    datasets_to_process = available if 'all' in args.datasets else args.datasets

    # Organize each dataset
    success = []
    for ds in datasets_to_process:
        if ds not in available:
            continue

        if ds == 'genimage':
            if organize_genimage(downloads_dir, output_dir):
                success.append('genimage')
        elif ds == 'cifake':
            if organize_cifake(downloads_dir, output_dir):
                success.append('cifake')
        elif ds == 'faces':
            if organize_faces(downloads_dir, output_dir):
                success.append('faces')

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for ds in success:
        ds_dir = output_dir / ds
        size = get_dir_size(ds_dir)
        print(f"\n✓ {ds.upper()}: {format_size(size)}")

        for split in ['train', 'val', 'test']:
            split_dir = ds_dir / split
            if split_dir.exists():
                fake_count = len(list((split_dir / "FAKE").glob("*")))
                real_count = len(list((split_dir / "REAL").glob("*")))
                print(f"    {split:5s}: {fake_count:6,d} FAKE, {real_count:6,d} REAL")

    total_size = sum(get_dir_size(output_dir / ds) for ds in success)
    print(f"\n{'='*70}")
    print(f"Total organized: {format_size(total_size)}")
    print(f"{'='*70}")

    print("\n✓ Organization complete!")
    print("\nNext steps:")
    print("  1. Verify datasets: python scripts/verify_datasets.py")
    print("  2. Test data loader: python src/data_loader_multi.py")
    print("  3. Start training:")
    print("     - Baseline models: python src/train_baseline.py")
    print("     - Combined model:  python src/train_combined.py")
    print("  4. Or run full pipeline: bash scripts/train_all.sh")
    print("")


if __name__ == "__main__":
    main()
