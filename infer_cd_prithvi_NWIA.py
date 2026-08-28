"""
infer_cd_prithvi_NWIA.py
------------------------
Change detection inference for Prithvi on NWIA test chips.

- Reads NWIA_chips.csv (T1/T2 chip pairs + GT masks)
- Runs best_F1_model.pth on each chip pair
- Stitches predictions into full-scene binary change map GeoTIFF
- Also saves full-scene GT change map for comparison

Outputs:
    predictions/cd_prithvi_NWIA/
        NWIA_Prithvi_CD_pred.tif    -- binary change map (0=unchanged,1=changed,255=nodata)
        NWIA_Prithvi_CD_gt.tif      -- GT change map (same encoding)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS, load_chips_csv, cd_checkpoint


import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm

# -- paths --------------------------------------------------------------------
BASE         = str(MSR_ROOT)
CHIPS_CSV    = f'{DATA_ROOT}/change_detection_chips/prithvi/NWIA_chips.csv'
CHECKPOINT = str(cd_checkpoint('prithvi', 'NWIA'))
OUTPUT_DIR   = f'{PREDICTIONS}/cd_prithvi_NWIA'

CHIP_SIZE    = 224
STRIDE       = 112    # 50% overlap used during chip creation

sys.path.insert(0, f'{BASE}/prithvi_finetune/ChangeDetection')

# -- normalization stats (Iowa z-score -- matches dataset_cd_prithvi.py) -------
T1_MEANS = np.array([
    1861.19006065,  # B02
    2033.17032775,  # B03
    2273.37933660,  # B04
    3262.91588412,  # B8A
    4457.44718789,  # B11
    3994.99188433,  # B12
], dtype=np.float32)

T1_STDS = np.array([
    307.48006869,   # B02
    351.67526808,   # B03
    447.86017086,   # B04
    654.81295100,   # B8A
    697.75477528,   # B11
    788.08159599,   # B12
], dtype=np.float32)

T2_MEANS = np.array([
    1704.63697798,  # B02
    1961.37926168,  # B03
    2034.19159947,  # B04
    3929.94252524,  # B8A
    4352.22479367,  # B11
    3695.04113396,  # B12
], dtype=np.float32)

T2_STDS = np.array([
    341.76676414,   # B02
    357.44447699,   # B03
    513.43405597,   # B04
    842.65097823,   # B8A
    920.07998270,   # B11
    1054.49693239,  # B12
], dtype=np.float32)


def normalize(img: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Per-band z-score normalization. img: (C,H,W) float32."""
    img = img.copy()
    # Replace nodata with band mean before normalizing
    for b in range(img.shape[0]):
        img[b] = np.where(img[b] == -9999, means[b], img[b])
    means_ = means.reshape(-1, 1, 1)
    stds_  = stds.reshape(-1, 1, 1)
    return (img - means_) / (stds_ + 1e-8)


# -- main ---------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -- load model -----------------------------------------------------------
    from src.model_cd_prithvi import build_prithvi_cd
    model = build_prithvi_cd(pretrain_path=None)
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  best_F1={ckpt['best_f1']*100:.2f}%")

    # -- read CSV -------------------------------------------------------------
    df = load_chips_csv(CHIPS_CSV)
    print(f"NWIA chips: {len(df)}")

    # -- determine full-scene canvas size from row/col ---------------------
    rows = df['row'].values
    cols = df['col'].values
    min_row, min_col = int(rows.min()), int(cols.min())
    max_row, max_col = int(rows.max()), int(cols.max())
    H = max_row + CHIP_SIZE - min_row
    W = max_col + CHIP_SIZE - min_col
    print(f"Canvas: {H}x{W}  (rows {min_row}-{max_row+CHIP_SIZE}, cols {min_col}-{max_col+CHIP_SIZE})")

    # -- get spatial reference from first chip -----------------------------
    with rasterio.open(df['t1'].iloc[0]) as src:
        crs = src.crs
    first_chip_row = df.loc[df['row'] == min_row].iloc[0]
    with rasterio.open(first_chip_row['t1']) as src:
        origin_transform = src.transform
    res_x = origin_transform.a   # pixel width  (positive)
    res_y = origin_transform.e   # pixel height (negative)
    full_transform = rasterio.transform.from_bounds(
        origin_transform.c,
        origin_transform.f + res_y * H,
        origin_transform.c + res_x * W,
        origin_transform.f,
        W, H
    )

    # -- accumulation buffers ---------------------------------------------
    prob_changed = np.zeros((H, W), dtype=np.float32)
    count_map    = np.zeros((H, W), dtype=np.float32)
    gt_canvas    = np.full((H, W), 255, dtype=np.uint8)   # 255 = nodata

    # -- inference loop ---------------------------------------------------
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc='Prithvi CD Inference'):
            r = int(row['row']) - min_row
            c = int(row['col']) - min_col

            # Load chips
            with rasterio.open(row['t1']) as src:
                t1 = src.read().astype(np.float32)
            with rasterio.open(row['t2']) as src:
                t2 = src.read().astype(np.float32)

            # Z-score normalize (separate stats for T1 and T2)
            t1 = normalize(t1, T1_MEANS, T1_STDS)
            t2 = normalize(t2, T2_MEANS, T2_STDS)

            # Forward pass
            t1_t = torch.from_numpy(t1).unsqueeze(0).float().to(device)  # (1,6,224,224)
            t2_t = torch.from_numpy(t2).unsqueeze(0).float().to(device)
            logits = model(t1_t, t2_t)                # (1,2,224,224) log-probs
            probs  = torch.exp(logits)[0, 1]           # (224,224) prob of changed
            probs  = probs.cpu().numpy()

            # Accumulate
            prob_changed[r:r+CHIP_SIZE, c:c+CHIP_SIZE] += probs
            count_map[r:r+CHIP_SIZE,    c:c+CHIP_SIZE] += 1.0

            # GT
            with rasterio.open(row['mask']) as src:
                gt = src.read(1)   # 0=unchanged, 1=changed, 255=nodata
            gt_canvas[r:r+CHIP_SIZE, c:c+CHIP_SIZE] = gt

    # -- average + threshold ----------------------------------------------
    count_map[count_map == 0] = 1
    prob_changed /= count_map
    pred = (prob_changed > 0.5).astype(np.uint8)

    # Mask GT nodata pixels
    pred[gt_canvas == 255] = 255

    # -- save outputs -----------------------------------------------------
    profile = {
        'driver':    'GTiff',
        'dtype':     'uint8',
        'count':     1,
        'height':    H,
        'width':     W,
        'crs':       crs,
        'transform': full_transform,
        'compress':  'lzw',
        'nodata':    255,
    }

    pred_path = os.path.join(OUTPUT_DIR, 'NWIA_Prithvi_CD_pred.tif')
    with rasterio.open(pred_path, 'w', **profile) as dst:
        dst.write(pred, 1)
    print(f"Saved prediction: {pred_path}")

    gt_path = os.path.join(OUTPUT_DIR, 'NWIA_Prithvi_CD_gt.tif')
    with rasterio.open(gt_path, 'w', **profile) as dst:
        dst.write(gt_canvas, 1)
    print(f"Saved GT:         {gt_path}")

    # -- quick stats ------------------------------------------------------
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
    print(f"\nFull-scene results (NWIA):")
    print(f"  OA={oa*100:.2f}%  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")


if __name__ == '__main__':
    main()
