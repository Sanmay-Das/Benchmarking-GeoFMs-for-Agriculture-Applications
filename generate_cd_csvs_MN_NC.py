"""
Generate change detection CSV files for MN and NC regions.

Scans each region's images/ and labels/ directories, parses row/col
from filenames, computes change_pct from mask, and writes CSV files
in the same format as CentIA_chips.csv / NWIA_chips.csv.

Usage:
    python generate_cd_csvs_MN_NC.py

Output CSVs (written to change_detection_chips/prithvi/):
    NorthMN_chips.csv, CentMN_chips.csv, SouthMN_chips.csv
    NENC_chips.csv,    ECNC_chips.csv,   EastNC_chips.csv
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import glob
import numpy as np
import pandas as pd
import rasterio

BASE     = f'{DATA_ROOT}/change_detection_chips/prithvi'
REGIONS  = ['NorthMN', 'CentMN', 'SouthMN', 'NENC', 'ECNC', 'EastNC']


def generate_csv(region):
    img_dir  = os.path.join(BASE, region, 'images')
    lbl_dir  = os.path.join(BASE, region, 'labels')
    out_csv  = os.path.join(BASE, f'{region}_chips.csv')

    # Find all T1 files
    t1_files = sorted(glob.glob(os.path.join(img_dir, f'{region}_*_t1.tif')))
    if not t1_files:
        print(f"  WARNING: no T1 files found in {img_dir}")
        return

    rows_out = []
    missing  = 0

    for t1_path in t1_files:
        fname  = os.path.basename(t1_path)                  # e.g. SouthMN_00112_07504_t1.tif
        parts  = fname.replace('.tif', '').split('_')

        # Parse row and col -- last 3 tokens before _t1 are: region parts + row + col
        # Filename pattern: REGION_RRRRR_CCCCC_t1.tif
        # Region may have underscores (e.g. NorthMN has none, but ECNC, NENC, EastNC do)
        # So take the last two numeric parts before _t1
        col_str = parts[-2]   # CCCCC
        row_str = parts[-3]   # RRRRR
        row = int(row_str)
        col = int(col_str)

        t2_path   = t1_path.replace('_t1.tif', '_t2.tif')
        mask_name = fname.replace('_t1.tif', '_change.tif')
        mask_path = os.path.join(lbl_dir, mask_name)

        if not os.path.exists(t2_path):
            missing += 1
            continue
        if not os.path.exists(mask_path):
            missing += 1
            continue

        # Compute change percentage
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        valid        = mask != 255
        change_pct   = float(mask[valid].sum()) / float(valid.sum() + 1e-8)

        rows_out.append({
            'location':   region,
            'row':        row,
            'col':        col,
            't1':         t1_path,
            't2':         t2_path,
            'mask':       mask_path,
            'change_pct': round(change_pct * 100, 4),
        })

    df = pd.DataFrame(rows_out)
    df.to_csv(out_csv, index=False)
    print(f"  {region}: {len(df)} chips -> {out_csv}  (skipped {missing})")


if __name__ == '__main__':
    print("Generating CSVs for MN and NC regions...\n")
    for region in REGIONS:
        generate_csv(region)
    print("\nDone.")
