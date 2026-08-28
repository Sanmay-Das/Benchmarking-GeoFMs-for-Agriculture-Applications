"""
infer_cd_satmae_EastNC.py
--------------------------
Change detection inference for SatMAE on EastNC test chips.

Outputs:
    predictions/cd_satmae_EastNC/
        EastNC_SatMAE_CD_pred.tif   -- binary change map (0=unchanged,1=changed,255=nodata)
        EastNC_SatMAE_CD_gt.tif     -- GT change map
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS, load_chips_csv


import os
import sys
import numpy as np
import pandas as pd
import torch
import rasterio
from tqdm import tqdm

BASE       = str(MSR_ROOT)
CHIPS_CSV  = f'{DATA_ROOT}/change_detection_chips/satmae/EastNC_chips.csv'
CHECKPOINT = f'{BASE}/SatMAE/ChangeDetection/cd_train_satmae_NC/best_F1_model.pth'
OUTPUT_DIR = f'{PREDICTIONS}/cd_satmae_EastNC'

CHIP_SIZE  = 96

sys.path.insert(0, f'{BASE}/SatMAE/ChangeDetection')

# SatMAE z-score normalization stats (same for T1 and T2)
SATMAE_MEAN = np.array([
    1184.3824625,
    1120.77120066,
    1136.26026392,
    1972.62420416,
    1732.16362238,
    1247.91870117,
], dtype=np.float32)

SATMAE_STD = np.array([
    650.2842772,
    712.12507725,
    965.23119807,
    1364.38688993,
    1310.36996126,
    1087.6020813,
], dtype=np.float32)


def normalize(img):
    """Per-band z-score normalization with SatMAE stats. img: (C,H,W) float32."""
    img = img.copy()
    for b in range(img.shape[0]):
        img[b] = np.where(img[b] == -9999, SATMAE_MEAN[b], img[b])
    means = SATMAE_MEAN.reshape(-1, 1, 1)
    stds  = SATMAE_STD.reshape(-1, 1, 1)
    return (img - means) / (stds + 1e-8)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from src.model_cd_satmae import build_satmae_cd
    model = build_satmae_cd(pretrain_path=None)
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  best_F1={ckpt['best_f1']*100:.2f}%")

    df = load_chips_csv(CHIPS_CSV)
    print(f"EastNC chips: {len(df)}")

    rows = df['row'].values
    cols = df['col'].values
    min_row, min_col = int(rows.min()), int(cols.min())
    max_row, max_col = int(rows.max()), int(cols.max())
    H = max_row + CHIP_SIZE - min_row
    W = max_col + CHIP_SIZE - min_col
    print(f"Canvas: {H}x{W}")

    with rasterio.open(df['t1'].iloc[0]) as src:
        crs = src.crs
    first_chip_row = df.loc[df['row'] == min_row].iloc[0]
    with rasterio.open(first_chip_row['t1']) as src:
        origin_transform = src.transform
    res_x = origin_transform.a
    res_y = origin_transform.e
    full_transform = rasterio.transform.from_bounds(
        origin_transform.c,
        origin_transform.f + res_y * H,
        origin_transform.c + res_x * W,
        origin_transform.f,
        W, H
    )

    prob_changed = np.zeros((H, W), dtype=np.float32)
    count_map    = np.zeros((H, W), dtype=np.float32)
    gt_canvas    = np.full((H, W), 255, dtype=np.uint8)

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc='SatMAE CD EastNC'):
            r = int(row['row']) - min_row
            c = int(row['col']) - min_col

            with rasterio.open(row['t1']) as src:
                t1 = src.read().astype(np.float32)
            with rasterio.open(row['t2']) as src:
                t2 = src.read().astype(np.float32)

            t1 = normalize(t1)
            t2 = normalize(t2)

            t1_t      = torch.from_numpy(t1).unsqueeze(0).float().to(device)
            t2_t      = torch.from_numpy(t2).unsqueeze(0).float().to(device)
            log_probs = model(t1_t, t2_t)
            probs     = torch.exp(log_probs)[0, 1].cpu().numpy()

            prob_changed[r:r+CHIP_SIZE, c:c+CHIP_SIZE] += probs
            count_map[r:r+CHIP_SIZE,    c:c+CHIP_SIZE] += 1.0

            with rasterio.open(row['mask']) as src:
                gt = src.read(1)
            gt_canvas[r:r+CHIP_SIZE, c:c+CHIP_SIZE] = gt

    count_map[count_map == 0] = 1
    prob_changed /= count_map
    pred = (prob_changed > 0.5).astype(np.uint8)
    pred[gt_canvas == 255] = 255

    profile = {
        'driver': 'GTiff', 'dtype': 'uint8', 'count': 1,
        'height': H, 'width': W, 'crs': crs,
        'transform': full_transform, 'compress': 'lzw', 'nodata': 255,
    }

    pred_path = os.path.join(OUTPUT_DIR, 'EastNC_SatMAE_CD_pred.tif')
    with rasterio.open(pred_path, 'w', **profile) as dst:
        dst.write(pred, 1)
    print(f"Saved prediction: {pred_path}")

    gt_path = os.path.join(OUTPUT_DIR, 'EastNC_SatMAE_CD_gt.tif')
    with rasterio.open(gt_path, 'w', **profile) as dst:
        dst.write(gt_canvas, 1)
    print(f"Saved GT:         {gt_path}")

    valid  = gt_canvas != 255
    pred_v = pred[valid]
    gt_v   = gt_canvas[valid]
    tp = int(((pred_v == 1) & (gt_v == 1)).sum())
    fp = int(((pred_v == 1) & (gt_v == 0)).sum())
    tn = int(((pred_v == 0) & (gt_v == 0)).sum())
    fn = int(((pred_v == 0) & (gt_v == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    oa   = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    print(f"\nEastNC Results:")
    print(f"  OA={oa*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == '__main__':
    main()
