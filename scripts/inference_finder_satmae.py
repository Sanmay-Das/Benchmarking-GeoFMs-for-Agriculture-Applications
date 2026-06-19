"""
Compute test mIoU for SatMAE (PSANet / FCN / FPN) from stitched prediction GeoTIFFs.

The stitched prediction covers the full NWIA region (6043×2304).
Chip names encode pixel offsets: chip_ROW_COL → pred[ROW:ROW+96, COL:COL+96].

Labels:
  Prediction: 0=NoData class, 1-13=crops, 255=NoData pixels (masked)
  GT mask:    0=NoData (ignore), 1-13=crop classes

Usage:
  python inference_finder_satmae.py --decoder fpn
  python inference_finder_satmae.py --decoder psanet
  python inference_finder_satmae.py --decoder fcn
"""

import os
import sys
import argparse
import numpy as np
import rasterio
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
CHIP_DIR   = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/NWIA'
TEST_TXT   = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/Iowa/test.txt'
CHIP_SIZE  = 96
NUM_CLASSES = 13  # classes 1-13

PRED_PATHS = {
    'psanet': '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_psanet_NWIA/NWIA_SatMAE_PSANet_Prediction.tif',
    'fcn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fcn_NWIA/NWIA_SatMAE_FCN_Prediction.tif',
    'fpn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fpn_NWIA/NWIA_SatMAE_FPN_Prediction.tif',
}

CLASS_NAMES = [
    'Natural Vegetation', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
    'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
    'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]

IGNORE_INDEX = 0  # NoData


def compute_metrics(confusion):
    """
    Given confusion matrix (num_classes, num_classes),
    compute per-class IoU, mIoU, overall accuracy.
    Rows = GT, Cols = Pred.
    """
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    iou = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), np.nan)
    miou = float(np.nanmean(iou))

    total_correct = tp.sum()
    total_pixels  = confusion.sum()
    oa = total_correct / total_pixels if total_pixels > 0 else 0.0

    return miou, iou, oa


def print_results(miou, iou, oa, decoder):
    print(f"\n{'='*55}")
    print(f"  SatMAE + {decoder.upper():8s} — Test Set Results (NWIA)")
    print(f"{'='*55}")
    print(f"\n  {'Class':<24} {'IoU':>7}")
    print(f"  {'-'*34}")
    for i, (name, v) in enumerate(zip(CLASS_NAMES, iou)):
        if not np.isnan(v):
            print(f"  {i+1:2d} {name:<22} {v*100:>6.2f}%")
        else:
            print(f"  {i+1:2d} {name:<22}    N/A")
    print(f"  {'-'*34}")
    print(f"  {'mIoU':<24} {miou*100:>6.2f}%")
    print(f"  {'Overall Accuracy':<24} {oa*100:>6.2f}%")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', choices=['psanet', 'fcn', 'fpn'], default='fpn',
                        help='Which decoder to evaluate')
    args = parser.parse_args()

    pred_path = PRED_PATHS[args.decoder]
    print(f"\nDecoder:    {args.decoder.upper()}")
    print(f"Prediction: {pred_path}")
    print(f"GT chips:   {CHIP_DIR}")

    # Load chip names from test.txt
    with open(TEST_TXT) as f:
        chip_names = [l.strip() for l in f if l.strip()]
    print(f"Test chips: {len(chip_names)}\n")

    # Load full stitched prediction into memory
    with rasterio.open(pred_path) as src:
        pred_full = src.read(1)   # (H, W) uint8
    print(f"Prediction map: {pred_full.shape}")

    # Confusion matrix — classes 1-13 → indices 0-12
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    missing = 0
    for chip_name in tqdm(chip_names, desc=f"SatMAE {args.decoder.upper()} eval"):
        # Parse pixel offsets from chip name: chip_ROW_COL
        parts = chip_name.split('_')
        row, col = int(parts[1]), int(parts[2])

        mask_path = os.path.join(CHIP_DIR, f'{chip_name}_mask.tif')
        if not os.path.exists(mask_path):
            missing += 1
            continue

        # Extract prediction patch
        pred_patch = pred_full[row:row+CHIP_SIZE, col:col+CHIP_SIZE].astype(np.int32)

        # Load GT mask (0=NoData, 1-13=crops)
        with rasterio.open(mask_path) as src:
            gt_patch = src.read(1).astype(np.int32)

        # Valid pixels: GT != 0 (NoData) AND pred != 255 (masked NoData)
        valid = (gt_patch != IGNORE_INDEX) & (pred_patch != 255)

        gt_valid   = gt_patch[valid]    # 1-13
        pred_valid = pred_patch[valid]  # 0-13 (0=model predicted NoData)

        # Map to 0-indexed: 1-13 → 0-12
        gt_valid   = gt_valid - 1
        pred_valid = pred_valid - 1     # 0 → -1 (will be out of range, handled below)

        # Only keep predictions in valid class range (0-12)
        in_range = (pred_valid >= 0) & (pred_valid < NUM_CLASSES)
        gt_valid   = gt_valid[in_range]
        pred_valid = pred_valid[in_range]

        # Update confusion matrix
        np.add.at(confusion, (gt_valid, pred_valid), 1)

    if missing > 0:
        print(f"WARNING: {missing} chips missing GT masks")

    miou, iou, oa = compute_metrics(confusion)
    print_results(miou, iou, oa, args.decoder)


if __name__ == '__main__':
    main()
