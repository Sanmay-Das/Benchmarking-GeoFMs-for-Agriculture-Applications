"""
infer_cd_spectralgpt_NWIA.py
-----------------------------
Change detection inference for SpectralGPT on NWIA test chips.

- Reads NWIA_chips.csv (T1/T2 chip pairs + GT masks)
- Runs best_F1_model.pth on each chip pair
- Stitches predictions into full-scene binary change map GeoTIFF
- Also saves full-scene GT change map for comparison

Outputs:
    predictions/cd_spectralgpt_NWIA/
        NWIA_SpectralGPT_CD_pred.tif    — binary change map (0=unchanged,1=changed,255=nodata)
        NWIA_SpectralGPT_CD_gt.tif      — GT change map (same encoding)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_origin
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
BASE         = '/bigdata/eldawylab/sdas050/MS_Research'
CHIPS_CSV    = f'{BASE}/change_detection_chips/spectralgpt/NWIA_chips.csv'
CHECKPOINT   = f'{BASE}/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection/cd_train_spectralgpt/best_F1_model.pth'
OUTPUT_DIR   = f'{BASE}/predictions/cd_spectralgpt_NWIA'

CHIP_SIZE    = 128
STRIDE       = 64      # 50% overlap used during chip creation

sys.path.insert(0, f'{BASE}/IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection')

# ── normalization (per-image min-max to [0,1] — matches dataset_cd.py) ──────
def normalize(img: np.ndarray) -> np.ndarray:
    """Per-band min-max normalization to [0,1]. img: (C,H,W) float32."""
    img = img.copy()
    img[img == -9999] = 0.0
    for c in range(img.shape[0]):
        mn, mx = img[c].min(), img[c].max()
        if mx > mn:
            img[c] = (img[c] - mn) / (mx - mn)
        else:
            img[c] = 0.0
    return np.clip(img, 0.0, 1.0)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── load model ───────────────────────────────────────────────────────────
    from src.model_cd_spectralgpt import build_spectralgpt_cd
    model = build_spectralgpt_cd(pretrain_path=None)
    ckpt  = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}  best_F1={ckpt['best_f1']*100:.2f}%")

    # ── read CSV ─────────────────────────────────────────────────────────────
    df = pd.read_csv(CHIPS_CSV)
    print(f"NWIA chips: {len(df)}")

    # ── determine full-scene canvas size from row/col ─────────────────────
    rows = df['row'].values
    cols = df['col'].values
    min_row, min_col = int(rows.min()), int(cols.min())
    max_row, max_col = int(rows.max()), int(cols.max())
    H = max_row + CHIP_SIZE - min_row
    W = max_col + CHIP_SIZE - min_col
    print(f"Canvas: {H}×{W}  (rows {min_row}–{max_row+CHIP_SIZE}, cols {min_col}–{max_col+CHIP_SIZE})")

    # ── get spatial reference from first chip ─────────────────────────────
    with rasterio.open(df['t1'].iloc[0]) as src:
        chip_profile = src.profile.copy()
        chip_transform = src.transform
        crs = src.crs
    # Shift transform to top-left of full canvas
    res_x = chip_transform.a   # pixel width  (positive)
    res_y = chip_transform.e   # pixel height (negative)
    # The chip at (min_row, min_col) defines the origin
    first_chip_row = df.loc[df['row'] == min_row].iloc[0]
    with rasterio.open(first_chip_row['t1']) as src:
        origin_transform = src.transform
    # Full canvas top-left is the top-left of the minimum-row/col chip
    # but we need to account for the offset within the scene
    # Use chip transform directly — chips are geo-referenced already
    # We'll build a pixel-space canvas and use the first chip's transform as reference
    full_transform = rasterio.transform.from_bounds(
        origin_transform.c,
        origin_transform.f + res_y * H,
        origin_transform.c + res_x * W,
        origin_transform.f,
        W, H
    )

    # ── accumulation buffers ─────────────────────────────────────────────
    prob_changed  = np.zeros((H, W), dtype=np.float32)   # prob of class=1
    count_map     = np.zeros((H, W), dtype=np.float32)
    gt_canvas     = np.full((H, W), 255, dtype=np.uint8)  # 255 = nodata

    # ── inference loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc='SpectralGPT CD Inference'):
            r = int(row['row']) - min_row
            c = int(row['col']) - min_col

            # Load chips
            with rasterio.open(row['t1']) as src:
                t1 = src.read().astype(np.float32)
            with rasterio.open(row['t2']) as src:
                t2 = src.read().astype(np.float32)

            # Normalize
            t1 = normalize(t1)
            t2 = normalize(t2)

            # Forward pass
            t1_t = torch.from_numpy(t1).unsqueeze(0).float().to(device)  # (1,6,128,128)
            t2_t = torch.from_numpy(t2).unsqueeze(0).float().to(device)
            logits = model(t1_t, t2_t)                # (1,2,128,128)
            probs  = F.softmax(logits, dim=1)[0, 1]   # (128,128) prob of changed
            probs  = probs.cpu().numpy()

            # Accumulate
            prob_changed[r:r+CHIP_SIZE, c:c+CHIP_SIZE] += probs
            count_map[r:r+CHIP_SIZE,    c:c+CHIP_SIZE] += 1.0

            # GT
            with rasterio.open(row['mask']) as src:
                gt = src.read(1)   # 0=unchanged, 1=changed, 255=nodata
            gt_canvas[r:r+CHIP_SIZE, c:c+CHIP_SIZE] = gt

    # ── average + threshold ──────────────────────────────────────────────
    count_map[count_map == 0] = 1
    prob_changed /= count_map
    pred = (prob_changed > 0.5).astype(np.uint8)

    # Mask nodata (where count_map was 0 before filling)
    nodata_mask = (count_map == 1) & (prob_changed == 0)
    pred[gt_canvas == 255] = 255

    # ── save outputs ─────────────────────────────────────────────────────
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

    pred_path = os.path.join(OUTPUT_DIR, 'NWIA_SpectralGPT_CD_pred.tif')
    with rasterio.open(pred_path, 'w', **profile) as dst:
        dst.write(pred, 1)
    print(f"Saved prediction: {pred_path}")

    gt_path = os.path.join(OUTPUT_DIR, 'NWIA_SpectralGPT_CD_gt.tif')
    with rasterio.open(gt_path, 'w', **profile) as dst:
        dst.write(gt_canvas, 1)
    print(f"Saved GT:         {gt_path}")

    # ── quick stats ──────────────────────────────────────────────────────
    valid     = gt_canvas != 255
    pred_v    = pred[valid]
    gt_v      = gt_canvas[valid]
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
