"""
Plain class-color visualization for SatMAE segmentation predictions (NWIA region).

Produces two stitched RGB GeoTIFFs per decoder:
  1. Prediction map  — plain class color, NoData (pred=255) → black
  2. Ground Truth map — plain class color, NoData (GT=0)    → black

No brightness modulation; colors match the Prithvi / SpectralGPT style maps.

Usage:
  python vismap_satmae_plain_NWIA.py --decoder fpn
  python vismap_satmae_plain_NWIA.py --decoder psanet
  python vismap_satmae_plain_NWIA.py --decoder fcn
"""

import os
import argparse
import numpy as np
import rasterio
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
CHIP_DIR  = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/NWIA'
TEST_TXT  = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/Iowa/test.txt'
CHIP_SIZE = 96

PRED_PATHS = {
    'psanet': '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_psanet_NWIA/NWIA_SatMAE_PSANet_Prediction.tif',
    'fcn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fcn_NWIA/NWIA_SatMAE_FCN_Prediction.tif',
    'fpn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fpn_NWIA/NWIA_SatMAE_FPN_Prediction.tif',
}

OUTPUT_DIRS = {
    'psanet': '/bigdata/eldawylab/sdas050/MS_Research/visualizations_geotiff/satmae_psanet_NWIA',
    'fcn':    '/bigdata/eldawylab/sdas050/MS_Research/visualizations_geotiff/satmae_fcn_NWIA',
    'fpn':    '/bigdata/eldawylab/sdas050/MS_Research/visualizations_geotiff/satmae_fpn_NWIA',
}

# Class colors for classes 1-13 (index 0 = class 1)
# Black (0,0,0) is reserved for NoData
COLORS = np.array([
    [ 34, 139,  34],   #  1: Natural Vegetation
    [  0, 100,   0],   #  2: Forest
    [255, 215,   0],   #  3: Corn
    [139,  69,  19],   #  4: Soybeans
    [  0, 191, 255],   #  5: Wetlands
    [128, 128, 128],   #  6: Developed/Barren
    [  0,   0, 255],   #  7: Open Water
    [255, 165,   0],   #  8: Winter Wheat
    [147, 112, 219],   #  9: Alfalfa
    [210, 180, 140],   # 10: Fallow/Idle
    [255, 192, 203],   # 11: Cotton
    [165,  42,  42],   # 12: Sorghum
    [192, 192, 192],   # 13: Other
], dtype=np.uint8)

CLASS_NAMES = [
    'Natural Vegetation', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
    'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
    'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]


def class_to_rgb(label_map, nodata_val):
    """
    Convert integer label map to uint8 RGB.
    label_map : (H, W) int32, values 1-13 = class, nodata_val = NoData
    Returns   : (H, W, 3) uint8
    """
    H, W = label_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)   # default black = NoData

    for cls in range(1, 14):                     # classes 1-13
        mask = label_map == cls
        rgb[mask] = COLORS[cls - 1]

    return rgb


def save_rgb_geotiff(rgb, profile, path, description):
    """Write (H, W, 3) uint8 array as a 3-band RGB GeoTIFF."""
    p = profile.copy()
    p.update({
        'count':       3,
        'dtype':       'uint8',
        'compress':    'lzw',
        'photometric': 'RGB',
        'nodata':      None,
    })
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
        dst.update_tags(description=description)
    print(f"Saved: {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', choices=['psanet', 'fcn', 'fpn'], default='fpn')
    args = parser.parse_args()

    pred_path  = PRED_PATHS[args.decoder]
    output_dir = OUTPUT_DIRS[args.decoder]
    os.makedirs(output_dir, exist_ok=True)

    dec = args.decoder.upper()
    print(f"\nDecoder:    {dec}")
    print(f"Prediction: {pred_path}")
    print(f"Output:     {output_dir}\n")

    # ── Load stitched prediction ──────────────────────────────────────────────
    with rasterio.open(pred_path) as src:
        pred_full = src.read(1).astype(np.int32)   # 0-13, 255=NoData
        profile   = src.profile.copy()
        H, W      = src.height, src.width

    print(f"Prediction map: {H} × {W}")

    # ── Assemble stitched GT from chip masks ──────────────────────────────────
    print("Assembling ground truth from chip masks...")
    gt_full = np.zeros((H, W), dtype=np.int32)   # 0 = NoData

    with open(TEST_TXT) as f:
        chip_names = [l.strip() for l in f if l.strip()]

    missing = 0
    for chip_name in tqdm(chip_names, desc="Loading GT masks"):
        parts = chip_name.split('_')
        row, col = int(parts[1]), int(parts[2])
        mask_path = os.path.join(CHIP_DIR, f'{chip_name}_mask.tif')
        if not os.path.exists(mask_path):
            missing += 1
            continue
        with rasterio.open(mask_path) as src:
            gt_chip = src.read(1).astype(np.int32)
        gt_full[row:row+CHIP_SIZE, col:col+CHIP_SIZE] = gt_chip

    if missing:
        print(f"WARNING: {missing} masks not found")
    print(f"GT assembled. Unique GT values: {np.unique(gt_full)}\n")

    # ── Build prediction RGB ──────────────────────────────────────────────────
    print("Generating prediction RGB map...")
    # Replace NoData sentinel (255) with 0 so class_to_rgb leaves it black
    pred_vis = pred_full.copy()
    pred_vis[pred_vis == 255] = 0

    pred_rgb = class_to_rgb(pred_vis, nodata_val=0)

    pred_out = os.path.join(output_dir,
                            f'NWIA_SatMAE_{dec}_Prediction_Color.tif')
    save_rgb_geotiff(pred_rgb, profile, pred_out,
                     f'SatMAE {dec} plain-color prediction map. NoData=black.')

    # ── Build GT RGB ─────────────────────────────────────────────────────────
    print("Generating GT RGB map...")
    gt_rgb = class_to_rgb(gt_full, nodata_val=0)

    gt_out = os.path.join(output_dir,
                          f'NWIA_GT_Color.tif')
    save_rgb_geotiff(gt_rgb, profile, gt_out,
                     'Ground Truth plain-color map. NoData=black.')

    # ── Color legend ──────────────────────────────────────────────────────────
    print(f"\nColor Legend (NoData → black):")
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = COLORS[i]
        print(f"  Class {i+1:2d}  {name:<24}  RGB({r:3d},{g:3d},{b:3d})")
    print()


if __name__ == '__main__':
    main()
