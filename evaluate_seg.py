"""
Evaluate segmentation predictions for any region.

Replaces evaluate_seg_NWIA.py and evaluate_seg_SouthMN.py, which were the same
routine over different regions and a different hardcoded list of models.
Computes mIoU and per-class IoU by comparing prediction GeoTIFFs against the
ground-truth chip masks named in that region's test split.

    python evaluate_seg.py --region NWIA
    python evaluate_seg.py --region SouthMN --models SatMAE_FPN SatMAE_PSANet

Every prediction present on disk is evaluated by default; --models pins an
explicit list.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import (GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS,
                   seg_chips_dir, seg_splits_file)
import registry as R

import os
import argparse
import numpy as np
import rasterio
from tqdm import tqdm

NUM_CLASSES = 13
CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other"
]

# Region-dependent, filled in by configure().
CHIP_DIR = None
TEST_TXT = None
PRED_MAP = {}


def configure(region, wanted=None):
    """Resolve chip/split paths and discover which predictions exist."""
    global CHIP_DIR, TEST_TXT, PRED_MAP

    R.region(region)

    # Ground truth comes from the chip masks named in the test split, so these
    # must be the same chips and the same split file infer_seg.py used. Both
    # resolvers prefer the published layout and fall back to the ad hoc
    # directories the original runs wrote.
    CHIP_DIR = str(seg_chips_dir("satmae", region))
    TEST_TXT = str(seg_splits_file("satmae", region))

    found = {}
    for model, head, reg in R.seg_pairs():
        if reg != region:
            continue
        label = R.SEG_MODELS[model]["label"]
        name = "{}_{}".format(label, head.upper()) if head else label
        suffix = "_{}".format(head) if head else ""
        path = (PREDICTIONS / "{}{}_{}".format(model, suffix, region)
                / "{}_{}{}_Prediction.tif".format(
                    region, label, "_" + head.upper() if head else ""))
        if os.path.exists(path):
            found[name] = (str(path), model)

    if wanted:
        missing = set(wanted) - set(found)
        if missing:
            raise SystemExit("No prediction for: {}. Available: {}".format(
                ", ".join(sorted(missing)), ", ".join(sorted(found)) or "none"))
        found = {k: v for k, v in found.items() if k in wanted}

    if not found:
        raise SystemExit(
            "No segmentation predictions found for {} under {}.\n"
            "Run infer_seg.py first, e.g.\n"
            "    python infer_seg.py --model satmae --head fpn --region {}".format(
                region, PREDICTIONS, region))

    PRED_MAP = found
    print("Evaluating: {}".format(", ".join(sorted(PRED_MAP))))


def compute_iou(conf_matrix):
    iou_per_class = []
    for c in range(NUM_CLASSES):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        denom = tp + fp + fn
        iou_per_class.append(tp / denom if denom > 0 else float('nan'))
    return np.array(iou_per_class)


def evaluate(pred_path, enc, chip_names, chip_coords, min_row, min_col):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    with rasterio.open(pred_path) as pred_src:
        pred_full = pred_src.read(1)

    for name, (r, c) in tqdm(zip(chip_names, chip_coords), total=len(chip_names), desc=os.path.basename(pred_path)):
        mask_path = os.path.join(CHIP_DIR, name + '_mask.tif')
        if not os.path.exists(mask_path):
            continue
        with rasterio.open(mask_path) as src:
            gt = src.read(1).astype(np.int16)   # 1-13=class, 0=nodata

        # chip coords map directly into prediction TIF (no offset subtraction)
        pred_chip = pred_full[r:r+96, c:c+96].astype(np.int16)

        # Convert to 0-12 class index
        # GT: 1-13 -> 0-12; 0 -> nodata
        gt_cls = gt - 1          # 0-12, nodata -> -1

        if enc == 'satmae':
            # pred values 1-13=class, 0 or 255=nodata
            pred_cls = pred_chip - 1     # 0-12, nodata -> -1 or 254
            pred_cls = np.where(pred_chip == 255, -1, pred_cls)
            pred_cls = np.where(pred_chip == 0,   -1, pred_cls)
        else:
            # Prithvi/SpectralGPT: 0-12=class, 255=nodata
            pred_cls = pred_chip.copy()
            pred_cls = np.where(pred_chip == 255, -1, pred_cls)

        valid = (gt_cls >= 0) & (gt_cls < NUM_CLASSES) & \
                (pred_cls >= 0) & (pred_cls < NUM_CLASSES)

        g = gt_cls[valid].astype(np.int32)
        p = pred_cls[valid].astype(np.int32)
        np.add.at(conf, (g, p), 1)

    return conf


def main():
    with open(TEST_TXT) as f:
        chip_names = [l.strip() for l in f if l.strip()]

    coords = [(int(n.split('_')[1]), int(n.split('_')[2])) for n in chip_names]
    rows = [r for r, c in coords]
    cols = [c for r, c in coords]
    min_row, min_col = min(rows), min(cols)

    print(f"Test chips: {len(chip_names)}")
    print(f"Canvas origin: row={min_row}, col={min_col}\n")

    results = {}
    for model, (path, enc) in PRED_MAP.items():
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        print(f"\n{'='*50}")
        print(f"Evaluating: {model}")
        conf = evaluate(path, enc, chip_names, coords, min_row, min_col)
        iou = compute_iou(conf)
        miou = np.nanmean(iou)
        total_px = conf.sum()
        correct_px = np.diag(conf).sum()
        oa = correct_px / total_px if total_px > 0 else 0
        results[model] = {'miou': miou, 'oa': oa, 'per_class_iou': iou}

        print(f"  mIoU: {miou*100:.2f}%  |  OA: {oa*100:.2f}%")
        print(f"  Per-class IoU:")
        for i, (name, v) in enumerate(zip(CLASS_NAMES, iou)):
            print(f"    {i:2d} {name:<20s}: {v*100:.2f}%" if not np.isnan(v) else f"    {i:2d} {name:<20s}: N/A")

    print(f"\n{'='*50}")
    print("SUMMARY -- NWIA Iowa Test Set")
    print(f"{'Model':<20s} {'mIoU':>8s} {'OA':>8s}")
    print('-' * 38)
    for model, r in results.items():
        print(f"{model:<20s} {r['miou']*100:>7.2f}% {r['oa']*100:>7.2f}%")




def cli():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--region', required=True, choices=sorted(R.REGIONS))
    ap.add_argument('--models', nargs='+', default=None,
                    help='restrict to these predictions (default: all found)')
    args = ap.parse_args()
    configure(args.region, args.models)
    main()


if __name__ == '__main__':
    cli()
