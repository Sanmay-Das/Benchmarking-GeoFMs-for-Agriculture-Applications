"""
Generate change detection CSV files for SatMAE CA regions.

Scans change_detection_chips/satmae/<REGION>/images/ and labels/,
writes CSV files matching the format used by other regions.

Output CSVs (written to change_detection_chips/satmae/):
    NorthCA_chips.csv, CentCA_chips.csv, SouthCA_chips.csv
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS, save_chips_csv


import os
import glob
import numpy as np
import pandas as pd
import rasterio

BASE    = f'{DATA_ROOT}/change_detection_chips/satmae'
REGIONS = ['NorthCA', 'CentCA', 'SouthCA']


def generate_csv(region):
    img_dir = os.path.join(BASE, region, 'images')
    lbl_dir = os.path.join(BASE, region, 'labels')
    out_csv = os.path.join(BASE, f'{region}_chips.csv')

    t1_files = sorted(glob.glob(os.path.join(img_dir, f'{region}_*_t1.tif')))
    if not t1_files:
        print(f"  WARNING: no T1 files found in {img_dir}")
        return

    rows_out = []
    missing  = 0

    for t1_path in t1_files:
        fname   = os.path.basename(t1_path)
        parts   = fname.replace('.tif', '').split('_')
        col_str = parts[-2]
        row_str = parts[-3]
        row     = int(row_str)
        col     = int(col_str)

        t2_path   = t1_path.replace('_t1.tif', '_t2.tif')
        mask_name = fname.replace('_t1.tif', '_change.tif')
        mask_path = os.path.join(lbl_dir, mask_name)

        if not os.path.exists(t2_path) or not os.path.exists(mask_path):
            missing += 1
            continue

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        valid      = mask != 255
        change_pct = round(float(mask[valid].sum()) / float(valid.sum() + 1e-8) * 100, 4)

        rows_out.append({
            'location':   region,
            'row':        row,
            'col':        col,
            't1':         t1_path,
            't2':         t2_path,
            'mask':       mask_path,
            'change_pct': change_pct,
        })

    df = pd.DataFrame(rows_out)
    save_chips_csv(df, out_csv)
    print(f"  {region}: {len(df)} chips -> {out_csv}  (skipped {missing})")


if __name__ == '__main__':
    print("Generating SatMAE CSVs for CA regions...\n")
    for region in REGIONS:
        generate_csv(region)
    print("\nDone.")
