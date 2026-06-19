"""
Visualize SatMAE segmentation predictions for NWIA test region.

Produces a full stitched RGB GeoTIFF (same CRS/extent as the prediction):
  - GT color = class color (from COLORS table)
  - Correct pixels  → full brightness
  - Incorrect pixels → 30% brightness (dimmed)
  - NoData pixels   → black (0, 0, 0)

Usage:
  python vismap_satmae_NWIA.py --decoder fpn
  python vismap_satmae_NWIA.py --decoder psanet
  python vismap_satmae_NWIA.py --decoder fcn
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

# Class colors (1-13), index 0 = class 1 (Natural Vegetation)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', choices=['psanet', 'fcn', 'fpn'], default='fpn')
    args = parser.parse_args()

    pred_path  = PRED_PATHS[args.decoder]
    output_dir = OUTPUT_DIRS[args.decoder]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDecoder:    {args.decoder.upper()}")
    print(f"Prediction: {pred_path}")
    print(f"Output:     {output_dir}\n")

    # ── Load stitched prediction ──────────────────────────────────────────────
    with rasterio.open(pred_path) as src:
        pred_full = src.read(1).astype(np.int32)   # (H, W) values 0-13, 255=NoData
        profile   = src.profile.copy()
        H, W      = src.height, src.width

    print(f"Prediction map: {H}×{W}")

    # ── Assemble stitched GT from chip masks ──────────────────────────────────
    print("Assembling ground truth from chip masks...")
    gt_full = np.zeros((H, W), dtype=np.int32)   # 0=NoData by default

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

    # ── Build RGB visualization ───────────────────────────────────────────────
    print("Generating RGB visualization...")
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    # Color from GT class (1-13 → COLORS index 0-12)
    for cls in range(1, 14):
        mask = gt_full == cls
        rgb[mask] = COLORS[cls - 1].astype(np.float32)

    # Brightness modulation
    # valid pixels: GT != 0 AND pred != 255
    valid   = (gt_full != 0) & (pred_full != 255)

    # Shift both to 0-indexed for comparison
    gt_0idx   = gt_full   - 1   # 0-12 for valid, -1 for NoData
    pred_0idx = pred_full - 1   # 0-12 for valid, -1 for NoData pred, 254 for masked

    correct = valid & (gt_0idx == pred_0idx)

    # Correct → 1.0, incorrect valid → 0.3, NoData → 0.0
    brightness          = np.zeros((H, W), dtype=np.float32)
    brightness[correct] = 1.0
    brightness[valid & ~correct] = 0.3

    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] * brightness).clip(0, 255)

    rgb_out = rgb.astype(np.uint8)

    # ── Compute summary accuracy ──────────────────────────────────────────────
    total_valid   = valid.sum()
    total_correct = correct.sum()
    oa = total_correct / total_valid * 100 if total_valid > 0 else 0.0
    print(f"Overall Accuracy (on valid pixels): {oa:.2f}%")
    print(f"Valid pixels: {total_valid:,}  |  Correct: {total_correct:,}")

    # ── Save stitched RGB GeoTIFF ─────────────────────────────────────────────
    out_file = os.path.join(output_dir,
                            f'NWIA_SatMAE_{args.decoder.upper()}_Visualization.tif')
    profile.update({
        'count':       3,
        'dtype':       'uint8',
        'compress':    'lzw',
        'photometric': 'RGB',
        'nodata':      None,
    })
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(rgb_out[:, :, 0], 1)
        dst.write(rgb_out[:, :, 1], 2)
        dst.write(rgb_out[:, :, 2], 3)
        dst.update_tags(
            decoder=args.decoder.upper(),
            overall_accuracy=f'{oa:.2f}%',
            description='SatMAE segmentation visualization. Correct=bright, Incorrect=dim, NoData=black'
        )

    print(f"\nSaved: {out_file}")
    print(f"Size:  {os.path.getsize(out_file)/1e6:.1f} MB")

    # ── Color legend ──────────────────────────────────────────────────────────
    print(f"\nColor Legend:")
    for i, name in enumerate(CLASS_NAMES):
        r, g, b = COLORS[i]
        print(f"  Class {i+1:2d} {name:<24} RGB({r:3d},{g:3d},{b:3d})")
    print()


if __name__ == '__main__':
    main()
