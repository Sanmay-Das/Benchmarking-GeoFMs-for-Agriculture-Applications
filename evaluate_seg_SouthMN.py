"""
Evaluate SatMAE PSANet segmentation on SouthMN test set.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import numpy as np
import rasterio
from tqdm import tqdm

NUM_CLASSES = 13
CLASS_NAMES = [
    "Natural Veg", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other"
]

BASE     = str(MSR_ROOT)
CHIP_DIR = os.path.join(BASE, 'SatMAE_chips_MN/SouthMN')
TEST_TXT = os.path.join(BASE, 'SatMAE_chips_multitemporal/MN/test.txt')

PREDICTIONS = {
    'SatMAE_FPN': (os.path.join(BASE, 'predictions/satmae_fpn_SouthMN/SouthMN_SatMAE_FPN_Prediction.tif'), 'satmae'),
    'SatMAE_FCN': (os.path.join(BASE, 'predictions/satmae_fcn_SouthMN/SouthMN_SatMAE_FCN_Prediction.tif'), 'satmae'),
    'SatMAE_PSANet': (os.path.join(BASE, 'predictions/satmae_psanet_SouthMN/SouthMN_SatMAE_PSANet_Prediction.tif'), 'satmae'),
}


def compute_iou(conf):
    iou = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom = tp + fp + fn
        iou.append(tp / denom if denom > 0 else float('nan'))
    return np.array(iou)


def evaluate(pred_path, enc, chip_names, coords):
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    with rasterio.open(pred_path) as src:
        pred_full = src.read(1)

    for name, (r, c) in tqdm(zip(chip_names, coords), total=len(chip_names), desc='Evaluating'):
        mask_path = os.path.join(CHIP_DIR, name + '_mask.tif')
        if not os.path.exists(mask_path):
            continue
        with rasterio.open(mask_path) as src:
            gt = src.read(1).astype(np.int16)  # 1-13=class, 0=nodata

        pred_chip = pred_full[r:r+96, c:c+96].astype(np.int16)
        if pred_chip.shape != (96, 96):
            continue

        # GT: 1-13 -> 0-12, 0 -> -1 (nodata)
        gt_cls = gt - 1

        if enc == 'satmae':
            # values 0-13: 0=NoData, 1-13=class -> shift to 0-12
            pred_cls = pred_chip - 1
            pred_cls = np.where(pred_chip == 0, -1, pred_cls)
            pred_cls = np.where(pred_chip == 255, -1, pred_cls)
        else:
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
    print(f"Test chips: {len(chip_names)}")

    for model, (path, enc) in PREDICTIONS.items():
        print(f"\n{'='*50}")
        print(f"Evaluating: {model}")
        conf = evaluate(path, enc, chip_names, coords)
        iou = compute_iou(conf)
        miou = np.nanmean(iou)
        total = conf.sum()
        oa = np.diag(conf).sum() / total if total > 0 else 0

        print(f"  mIoU: {miou*100:.2f}%  |  OA: {oa*100:.2f}%")
        print(f"  Per-class IoU:")
        for i, (name, v) in enumerate(zip(CLASS_NAMES, iou)):
            if np.isnan(v):
                print(f"    {i:2d} {name:<22s}: N/A")
            else:
                print(f"    {i:2d} {name:<22s}: {v*100:.2f}%")


if __name__ == '__main__':
    main()
