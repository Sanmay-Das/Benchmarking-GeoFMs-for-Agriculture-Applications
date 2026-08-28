"""
Class distribution analysis across all available test regions.
Scans mask TIFs in each region directory and computes per-class pixel counts/percentages.

Available on disk (train/val dirs were removed to free space):
  Iowa test:  NWIA/          (4627 chips)
  MN test:    SouthMN/       (33595 chips)
  NC test:    EastNC/        (16447 chips)
  NC train:   NENC/          (32060 chips)  <- only NC has train masks
  NC val:     ECNC/          (21306 chips)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

BASE = Path(f'{DATA_ROOT}/SatMAE_chips_multitemporal')

CLASS_NAMES = [
    'NoData',
    'Natural Veg', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
    'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
    'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]

# Regions to analyze: (label, directory, split role)
REGIONS = [
    ('Iowa -- Test (NWIA)',     BASE / 'NWIA',    'Iowa',     'Test'),
    ('MN   -- Test (SouthMN)', BASE / 'SouthMN', 'Minnesota', 'Test'),
    ('NC   -- Train (NENC)',   BASE / 'NENC',    'N.Carolina','Train'),
    ('NC   -- Val   (ECNC)',   BASE / 'ECNC',    'N.Carolina','Val'),
    ('NC   -- Test  (EastNC)', BASE / 'EastNC',  'N.Carolina','Test'),
]


def compute_distribution(mask_dir: Path):
    counts = np.zeros(14, dtype=np.int64)  # labels 0-13
    mask_files = list(mask_dir.glob('*_mask.tif'))
    for f in tqdm(mask_files, desc=str(mask_dir.name), leave=False):
        with rasterio.open(f) as src:
            data = src.read(1).astype(np.int64)
        data = np.clip(data, 0, 13)
        for c in range(14):
            counts[c] += (data == c).sum()
    return counts


def print_table(label, counts):
    total = counts[1:].sum()  # exclude NoData for percentage
    total_all = counts.sum()
    print(f"\n{'-'*55}")
    print(f"  {label}")
    print(f"  Total valid pixels: {total:,}  |  NoData pixels: {counts[0]:,}")
    print(f"{'-'*55}")
    print(f"  {'Class':<22s}  {'Pixels':>10s}  {'%':>7s}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*7}")
    for i in range(1, 14):
        pct = 100.0 * counts[i] / total if total > 0 else 0
        if counts[i] > 0:
            print(f"  {CLASS_NAMES[i]:<22s}  {counts[i]:>10,}  {pct:>6.2f}%")
        else:
            print(f"  {CLASS_NAMES[i]:<22s}  {'--':>10s}  {'--':>7s}")
    print(f"{'-'*55}")


def write_csv(all_results, out_path):
    rows = []
    header = ['State', 'Split', 'Region'] + [CLASS_NAMES[i] for i in range(1, 14)] + ['Total_valid']
    rows.append(','.join(header))
    for (label, dirpath, state, split), counts in all_results:
        total = counts[1:].sum()
        pcts = [f"{100.0 * counts[i] / total:.4f}" if total > 0 else "0" for i in range(1, 14)]
        rows.append(','.join([state, split, dirpath.name] + pcts + [str(total)]))
    with open(out_path, 'w') as f:
        f.write('\n'.join(rows))
    print(f"\nCSV saved: {out_path}")


def main():
    all_results = []
    for (label, dirpath, state, split) in REGIONS:
        if not dirpath.exists():
            print(f"  MISSING: {dirpath}")
            continue
        counts = compute_distribution(dirpath)
        print_table(label, counts)
        all_results.append(((label, dirpath, state, split), counts))

    out_csv = Path(f'{PREDICTIONS}/class_distribution.csv')
    out_csv.parent.mkdir(exist_ok=True)
    write_csv(all_results, out_csv)

    # Print compact comparison table: crop classes only, Iowa vs MN
    print("\n" + "="*70)
    print("COMPACT COMPARISON: Iowa test vs MN test (% of valid pixels)")
    print("="*70)
    ia_counts = next(c for (l, d, s, sp), c in all_results if s == 'Iowa')
    mn_counts = next(c for (l, d, s, sp), c in all_results if s == 'Minnesota')
    ia_total = ia_counts[1:].sum()
    mn_total = mn_counts[1:].sum()
    print(f"  {'Class':<22s}  {'Iowa %':>8s}  {'MN %':>8s}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    for i in range(1, 14):
        ia_pct = 100.0 * ia_counts[i] / ia_total if ia_total > 0 else 0
        mn_pct = 100.0 * mn_counts[i] / mn_total if mn_total > 0 else 0
        if ia_counts[i] > 0 or mn_counts[i] > 0:
            print(f"  {CLASS_NAMES[i]:<22s}  {ia_pct:>7.2f}%  {mn_pct:>7.2f}%")
    print("="*70)


if __name__ == '__main__':
    main()
