"""
Create train.txt and val.txt for MMSegmentation
Format: relative_image_path relative_mask_path
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import json
from pathlib import Path


def create_mmseg_file_lists(chips_dir):
    """Create train/val/test txt files for MMSegmentation"""
    chips_dir = Path(chips_dir)
    
    # Check if split.json exists
    split_file = chips_dir / 'split.json'
    if not split_file.exists():
        print(f" ERROR: {split_file} not found!")
        print("Please create split.json first using the split creation script.")
        return
    
    # Load split
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    print(f"Loaded split.json:")
    print(f"  Train: {len(split['train'])} chips")
    print(f"  Val: {len(split['val'])} chips")
    print(f"  Test: {len(split['test'])} chips")
    
    # Create train.txt
    train_txt = chips_dir / 'train.txt'
    with open(train_txt, 'w') as f:
        for chip_name in split['train']:
            img_path = chip_name
            mask_path = chip_name.replace('.tif', '_mask.tif')
            f.write(f"{img_path} {mask_path}\n")
    print(f" Created {train_txt}")
    
    # Create val.txt
    val_txt = chips_dir / 'val.txt'
    with open(val_txt, 'w') as f:
        for chip_name in split['val']:
            img_path = chip_name
            mask_path = chip_name.replace('.tif', '_mask.tif')
            f.write(f"{img_path} {mask_path}\n")
    print(f" Created {val_txt}")
    
    # Create test.txt
    test_txt = chips_dir / 'test.txt'
    with open(test_txt, 'w') as f:
        for chip_name in split['test']:
            img_path = chip_name
            mask_path = chip_name.replace('.tif', '_mask.tif')
            f.write(f"{img_path} {mask_path}\n")
    print(f" Created {test_txt}")
    
    print("\n Done! Files created:")
    print(f"   {train_txt}")
    print(f"   {val_txt}")
    print(f"   {test_txt}")


if __name__ == '__main__':
    CHIPS_DIR = f'{DATA_ROOT}/SatMAE_chips_multitemporal/CentIA'
    create_mmseg_file_lists(CHIPS_DIR)