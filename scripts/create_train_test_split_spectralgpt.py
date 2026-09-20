
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

import os
from pathlib import Path

def create_geographic_train_test_split(train_dir, test_dir, output_dir):
    """
    Create train.txt and test.txt with geographic split.
    Uses simple file names (e.g., CentIA_chip_0_0) assuming all chips 
    will be aggregated into the final ImageSets directory.
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    
    print("="*70)
    print("GEOGRAPHIC SPLIT")
    print("="*70)
    
    # Get Train chips
    print(f"\nProcessing TRAIN region: {train_dir.name}")
    train_image_files = sorted(train_dir.glob('*chip_*.tif'))
    train_image_files = [f for f in train_image_files if '_mask' not in f.name]
    train_chip_names = [f.stem for f in train_image_files]
    print(f"  Found {len(train_chip_names)} chips")
    
    # Get Test chips
    print(f"\nProcessing TEST/VAL region: {test_dir.name}")
    test_image_files = sorted(test_dir.glob('*chip_*.tif'))
    test_image_files = [f for f in test_image_files if '_mask' not in f.name]
    test_chip_names = [f.stem for f in test_image_files]
    print(f"  Found {len(test_chip_names)} chips")
    
    # Create ImageSets folder
    imagesets_dir = output_dir / 'ImageSets'
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # Write train.txt with simple names
    train_txt = imagesets_dir / 'train.txt'
    with open(train_txt, 'w') as f:
        for name in train_chip_names:
            f.write(f"{name}\n")
    
    print(f"\nSaved: {train_txt}")
    print(f"  Train chips: {len(train_chip_names)} (from {train_dir.name})")
    print(f"  Format: {train_chip_names[0] if train_chip_names else 'chip_X'}")
    
    # Write test.txt with simple names
    test_txt = imagesets_dir / 'test.txt'
    with open(test_txt, 'w') as f:
        for name in test_chip_names:
            f.write(f"{name}\n")
    
    print(f"\nSaved: {test_txt}")
    print(f"  Test chips: {len(test_chip_names)} (from {test_dir.name})")
    print(f"  Format: {test_chip_names[0] if test_chip_names else 'chip_X'}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Train: {len(train_chip_names)} chips from {train_dir.name}")
    print(f"Test:  {len(test_chip_names)} chips from {test_dir.name}")
    print(f"\nOutput: {imagesets_dir}")
    print("="*70)
    
    return train_chip_names, test_chip_names


if __name__ == "__main__":
    
    TRAIN_DIR = f"{DATA_ROOT}/SpectralGPT_chips_multitemporal/WCIL"
    TEST_DIR = f"{DATA_ROOT}/SpectralGPT_chips_multitemporal/ECIL"
    OUTPUT_DIR = f"{DATA_ROOT}/SpectralGPT_chips_multitemporal/WCECIL"
    
    print("="*70)
    print("CREATING GEOGRAPHIC TRAIN/TEST SPLIT")
    print("="*70)
    
    create_geographic_train_test_split(TRAIN_DIR, TEST_DIR, OUTPUT_DIR)
    
    print("\nDONE! Ready to train SpectralGPT!")