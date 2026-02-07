#!/usr/bin/env python3
"""
Dataset Verification Script for AuthentiScan

Verifies that all datasets are properly organized with correct structure.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


def count_images(directory: Path) -> int:
    """Count image files in a directory."""
    if not directory.exists():
        return 0

    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.JPG', '.JPEG', '.PNG'}
    count = 0

    for file in directory.rglob('*'):
        if file.is_file() and file.suffix in extensions:
            count += 1

    return count


def verify_dataset(dataset_name: str, base_path: Path) -> Dict:
    """
    Verify a dataset has correct structure.

    Returns:
        dict with verification results
    """
    dataset_path = base_path / dataset_name
    results = {
        'name': dataset_name,
        'exists': dataset_path.exists(),
        'splits': {},
        'total_images': 0,
        'status': 'unknown'
    }

    if not dataset_path.exists():
        results['status'] = 'missing'
        return results

    # Check each split
    expected_splits = ['train', 'val', 'test']
    all_good = True

    for split in expected_splits:
        split_path = dataset_path / split
        fake_path = split_path / 'FAKE'
        real_path = split_path / 'REAL'

        fake_count = count_images(fake_path)
        real_count = count_images(real_path)

        results['splits'][split] = {
            'FAKE': fake_count,
            'REAL': real_count
        }

        results['total_images'] += fake_count + real_count

        if fake_count == 0 or real_count == 0:
            all_good = False

    if all_good and results['total_images'] > 0:
        results['status'] = 'ok'
    else:
        results['status'] = 'incomplete'

    return results


def print_dataset_results(results: Dict):
    """Print verification results for a dataset."""
    name = results['name'].upper()
    status = results['status']

    # Header with status
    if status == 'ok':
        status_icon = f"{Colors.GREEN}✓{Colors.NC}"
        print(f"\n{status_icon} {Colors.GREEN}{name}{Colors.NC}")
    elif status == 'missing':
        status_icon = f"{Colors.RED}✗{Colors.NC}"
        print(f"\n{status_icon} {Colors.RED}{name} - NOT FOUND{Colors.NC}")
        return
    else:
        status_icon = f"{Colors.YELLOW}⚠{Colors.NC}"
        print(f"\n{status_icon} {Colors.YELLOW}{name} - INCOMPLETE{Colors.NC}")

    print("="*70)

    # Print split information
    for split in ['train', 'val', 'test']:
        if split in results['splits']:
            fake = results['splits'][split]['FAKE']
            real = results['splits'][split]['REAL']

            if fake > 0 and real > 0:
                icon = f"{Colors.GREEN}✓{Colors.NC}"
            else:
                icon = f"{Colors.RED}✗{Colors.NC}"

            print(f"  {icon} {split:5s}: {fake:7,d} FAKE, {real:7,d} REAL")

    print(f"\n  Total: {results['total_images']:,} images")
    print("="*70)


def main():
    """Main verification function."""
    print("="*70)
    print("  AuthentiScan Dataset Verification")
    print("="*70)

    base_path = Path("data")

    if not base_path.exists():
        print(f"\n{Colors.RED}✗ data/ directory not found!{Colors.NC}")
        print("\nPlease download and organize datasets first:")
        print("  1. bash scripts/download_datasets.sh")
        print("  2. python src/organize_datasets.py")
        return 1

    # Verify each dataset
    datasets = ['genimage', 'cifake', 'faces']
    results = {}

    for dataset in datasets:
        results[dataset] = verify_dataset(dataset, base_path)
        print_dataset_results(results[dataset])

    # Overall summary
    print("\n" + "="*70)
    print("  Summary")
    print("="*70)

    total_images = 0
    all_ok = True

    for dataset, result in results.items():
        status = result['status']
        total_images += result['total_images']

        if status == 'ok':
            status_str = f"{Colors.GREEN}OK{Colors.NC}"
        elif status == 'missing':
            status_str = f"{Colors.RED}MISSING{Colors.NC}"
            all_ok = False
        else:
            status_str = f"{Colors.YELLOW}INCOMPLETE{Colors.NC}"
            all_ok = False

        images = result['total_images']
        print(f"  {dataset:10s}: {status_str} ({images:,} images)")

    print(f"\n  Total images: {total_images:,}")
    print("="*70)

    if all_ok:
        print(f"\n{Colors.GREEN}🎉 All datasets verified successfully!{Colors.NC}")
        print("\nNext steps:")
        print("  1. Test data loader: python src/data_loader_multi.py")
        print("  2. Start training:")
        print("     - Baseline models: python src/train_baseline.py")
        print("     - Combined model:  python src/train_combined.py")
        print("  3. Or run full pipeline: bash scripts/train_all.sh")
        return 0
    else:
        print(f"\n{Colors.YELLOW}⚠  Some datasets have issues{Colors.NC}")
        print("\nTo fix:")
        print("  1. Download missing datasets: bash scripts/download_datasets.sh")
        print("  2. Organize datasets: python src/organize_datasets.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
