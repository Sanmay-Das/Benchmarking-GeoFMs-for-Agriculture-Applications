
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

import numpy as np
import rasterio
import glob
import os
import json
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# CONFIGURATION -- change these
# ============================================================================

LOCATIONS = {
    'NWIA': {
        'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/NWIA',
        'prefix': 'chip_'
    },
    'EastNC': {
        'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/EastNC',
        'prefix': 'chip_'
    },
    'SouthMN': {
        'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/SouthMN',
        'prefix': 'chip_'
    },
    'SouthCA': {
        'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/SouthCA',
        'prefix': 'chip_'
    }
    # 'SCIL': {
    #     'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/SCIL',
    #     'prefix': 'chip_'
    # },
    # 'WCIL': {
    #     'chip_dir': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/WCIL',
    #     'prefix': 'chip_'
    
}

CLASS_NAMES = [
    'No Data',            # 0
    'Natural Vegetation', # 1
    'Forest',             # 2
    'Corn',               # 3
    'Soybeans',           # 4
    'Wetlands',           # 5
    'Developed/Barren',   # 6
    'Open Water',         # 7
    'Winter Wheat',       # 8
    'Alfalfa',            # 9
    'Fallow/Idle',        # 10
    'Cotton',             # 11
    'Sorghum',            # 12
    'Other'               # 13
]

# ============================================================================

def compute_stats(chip_dir, prefix):
    """Compute class distribution statistics from mask chips"""

    mask_files = sorted(glob.glob(os.path.join(chip_dir, f'{prefix}*.mask.tif')))
    print(f"  Found {len(mask_files)} mask chips")

    if len(mask_files) == 0:
        print("  ERROR: No mask files found!")
        return None

    # Pixel counts per class (0-13)
    class_counts = np.zeros(14, dtype=np.int64)
    total_pixels = 0

    for mask_path in tqdm(mask_files, desc="  Processing"):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # values 0-13, 0=NoData

        for class_id in range(14):
            class_counts[class_id] += np.sum(mask == class_id)

        total_pixels += mask.size

    return class_counts, total_pixels, len(mask_files)


def print_stats(location, class_counts, total_pixels, num_chips):
    """Print statistics in tabular format"""

    valid_pixels = total_pixels - class_counts[0]  # exclude NoData

    print(f"\n{'='*65}")
    print(f"  {location} -- {num_chips} chips | {total_pixels:,} total px | {valid_pixels:,} valid px")
    print(f"{'='*65}")
    print(f"  {'Class':<22} {'Pixels':>12} {'% of Valid':>12} {'Present':>8}")
    print(f"  {'-'*58}")

    for class_id in range(14):
        name = CLASS_NAMES[class_id]
        count = class_counts[class_id]
        pct = count / valid_pixels * 100 if valid_pixels > 0 and class_id > 0 else 0.0
        present = "yes" if count > 0 else "no"

        if class_id == 0:
            print(f"  {name:<22} {count:>12,} {'(NoData)':>12} {present:>8}")
        else:
            print(f"  {name:<22} {count:>12,} {pct:>11.2f}% {present:>8}")

    print(f"  {'-'*58}")
    print(f"  {'Total Valid':<22} {valid_pixels:>12,} {'100.00%':>12}")
    print(f"{'='*65}\n")


def main():
    all_stats = {}

    for location, config in LOCATIONS.items():
        print(f"\n{'#'*65}")
        print(f"# Processing: {location}")
        print(f"{'#'*65}")

        result = compute_stats(config['chip_dir'], config['prefix'])
        if result is None:
            continue

        class_counts, total_pixels, num_chips = result
        print_stats(location, class_counts, total_pixels, num_chips)

        # Store for JSON
        all_stats[location] = {
            'num_chips': num_chips,
            'total_pixels': int(total_pixels),
            'class_distribution': {
                CLASS_NAMES[i]: int(class_counts[i]) for i in range(14)
            }
        }

    # Save JSON
    out_path = f'{DATA_ROOT}/dataset_statistics.json'
    with open(out_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"Statistics saved to: {out_path}")


if __name__ == '__main__':
    main()