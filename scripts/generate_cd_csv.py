"""
Generate a change-detection chip manifest for any model/region.

Replaces the per-region generate_cd_csvs_*.py scripts. The manifest is fully
derived from the chips themselves, so it never needs to be distributed --
regenerate it locally after downloading the chip archives.

Usage:
    python generate_cd_csv.py --base /path/to/change_detection_chips/satmae --region SouthMN
    python generate_cd_csv.py --base /path/to/change_detection_chips/prithvi --region all

Writes <BASE>/<REGION>_chips.csv with columns:
    location, row, col, t1, t2, mask, change_pct
"""

import os
import glob
import argparse
import pandas as pd
import rasterio


def generate_csv(base, region):
    img_dir = os.path.join(base, region, 'images')
    lbl_dir = os.path.join(base, region, 'labels')
    out_csv = os.path.join(base, f'{region}_chips.csv')

    t1_files = sorted(glob.glob(os.path.join(img_dir, f'{region}_*_t1.tif')))
    if not t1_files:
        print(f"  WARNING: no T1 files found in {img_dir}")
        return

    rows_out = []
    missing = 0

    for t1_path in t1_files:
        fname = os.path.basename(t1_path)
        parts = fname.replace('.tif', '').split('_')
        row = int(parts[-3])
        col = int(parts[-2])

        t2_path = t1_path.replace('_t1.tif', '_t2.tif')
        mask_path = os.path.join(lbl_dir, fname.replace('_t1.tif', '_change.tif'))

        if not os.path.exists(t2_path) or not os.path.exists(mask_path):
            missing += 1
            continue

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        valid = mask != 255
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
    df.to_csv(out_csv, index=False)
    print(f"  {region}: {len(df)} chips -> {out_csv}  (skipped {missing})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', required=True,
                   help='model chip dir, e.g. .../change_detection_chips/satmae')
    p.add_argument('--region', required=True,
                   help='region name, or "all" to scan every subfolder')
    args = p.parse_args()

    if args.region == 'all':
        regions = sorted(d for d in os.listdir(args.base)
                         if os.path.isdir(os.path.join(args.base, d)))
    else:
        regions = [args.region]

    print(f"Generating manifests in {args.base}\n")
    for region in regions:
        generate_csv(args.base, region)
    print("\nDone.")


if __name__ == '__main__':
    main()
