"""
visualize_cd_satmae_SouthMN.py
-------------------------------
Adds SatMAE to the existing SouthMN CD visualization outputs.
SpectralGPT and Prithvi maps are already done — this only generates:
    visualizations/cd_SouthMN/
        SouthMN_CD_SatMAE_colored.png
        geotiffs/SouthMN_CD_SatMAE.tif
        crops/highchange/  crop{1,2,3}_SatMAE.png
        crops/informative/ crop{1,2,3}_SatMAE.png
"""

import os
import gc
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from PIL import Image
from tqdm import tqdm

BASE       = '/bigdata/eldawylab/sdas050/MS_Research'
CHIPS_CSV  = f'{BASE}/change_detection_chips/spectralgpt/SouthMN_chips.csv'
CHIP_SIZE  = 128

GT_PATH    = f'{BASE}/predictions/cd_spectralgpt_SouthMN/SouthMN_SpectralGPT_CD_gt.tif'
PRED_PATH  = f'{BASE}/predictions/cd_satmae_SouthMN/SouthMN_SatMAE_CD_pred.tif'
OUTPUT_DIR = f'{BASE}/visualizations/cd_SouthMN'

SCALE      = 4
STRIP_H    = 256
CROP_SIZE  = 512


def to_binary_rgb(arr):
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[arr == 1]   = [255, 255, 255]
    rgb[arr == 0]   = [  0,   0,   0]
    rgb[arr == 255] = [128, 128, 128]
    return rgb


def save_png(rgb, path):
    Image.fromarray(rgb).save(path)
    print(f"    {os.path.basename(path)}")


def save_geotiff_strip(arr, ref_path, out_path):
    with rasterio.open(ref_path) as src:
        profile = src.profile.copy()
    profile.update({'count': 3, 'dtype': 'uint8', 'compress': 'lzw',
                    'photometric': 'RGB', 'nodata': None})
    H, W = arr.shape
    with rasterio.open(out_path, 'w', **profile) as dst:
        for y0 in range(0, H, STRIP_H):
            y1  = min(y0 + STRIP_H, H)
            rgb = to_binary_rgb(arr[y0:y1])
            win = rasterio.windows.Window(0, y0, W, y1 - y0)
            dst.write(rgb[:, :, 0], 1, window=win)
            dst.write(rgb[:, :, 1], 2, window=win)
            dst.write(rgb[:, :, 2], 3, window=win)
            del rgb
    print(f"    {os.path.basename(out_path)}")


def load_and_align(pred_path, H, W):
    with rasterio.open(pred_path) as src:
        pred_raw = src.read(1)
    pred = np.full((H, W), 255, dtype=np.uint8)
    h, w = min(pred_raw.shape[0], H), min(pred_raw.shape[1], W)
    pred[:h, :w] = pred_raw[:h, :w]
    del pred_raw
    return pred


def find_highchange_crops(gt_ds, n=3, crop_ds=128):
    H, W = gt_ds.shape
    step = crop_ds // 2
    best = []
    for r in range(0, H-crop_ds, step):
        for c in range(0, W-crop_ds, step):
            p = gt_ds[r:r+crop_ds, c:c+crop_ds]
            valid = (p != 255).sum()
            if valid < crop_ds*crop_ds*0.5: continue
            pct = (p==1).sum()/(valid+1e-8)
            best.append((-abs(pct-0.40), r, c))
    best.sort()
    return [(r*SCALE, c*SCALE) for _, r, c in best[:n]]


def find_informative_crops(gt_ds, pred_ds, n=3, crop_ds=128):
    """Crops where SatMAE misses the most change (high FN)."""
    H, W = gt_ds.shape
    step = crop_ds // 2
    scores = []
    for r in range(0, H-crop_ds, step):
        for c in range(0, W-crop_ds, step):
            gt_p  = gt_ds[r:r+crop_ds, c:c+crop_ds]
            valid = (gt_p != 255).sum()
            if valid < crop_ds*crop_ds*0.5: continue
            changed = (gt_p == 1)
            fn = ((pred_ds[r:r+crop_ds, c:c+crop_ds]==0) & changed).sum()
            scores.append((-int(fn), r, c))
    scores.sort()
    return [(r*SCALE, c*SCALE) for _, r, c in scores[:n]]


def main():
    # ── Load GT ──────────────────────────────────────────────────────────────
    with rasterio.open(GT_PATH) as src:
        gt = src.read(1)
    H, W = gt.shape
    print(f"GT canvas: {H}×{W}")

    # ── Load SatMAE prediction ────────────────────────────────────────────────
    print("Loading SatMAE prediction...")
    pred = load_and_align(PRED_PATH, H, W)

    # ── Full-scene PNG ────────────────────────────────────────────────────────
    print("\nSatMAE full-scene:")
    save_png(to_binary_rgb(pred[::SCALE, ::SCALE]),
             f'{OUTPUT_DIR}/SouthMN_CD_SatMAE_colored.png')

    # ── Full-scene GeoTIFF ────────────────────────────────────────────────────
    save_geotiff_strip(pred, GT_PATH,
                       f'{OUTPUT_DIR}/geotiffs/SouthMN_CD_SatMAE.tif')

    # ── Find crop locations (same deterministic logic as visualize_cd_SouthMN) ─
    print("\nFinding crop locations...")
    gt_ds   = gt[::SCALE, ::SCALE]
    pred_ds = pred[::SCALE, ::SCALE]

    crop_ds   = CROP_SIZE // SCALE
    hc_crops  = find_highchange_crops(gt_ds, n=3, crop_ds=crop_ds)
    inf_crops = find_informative_crops(gt_ds, pred_ds, n=3, crop_ds=crop_ds)
    print(f"  High-change crops:  {hc_crops}")
    print(f"  Informative crops:  {inf_crops}")
    del gt_ds, pred_ds
    gc.collect()

    # ── SatMAE crop panels ────────────────────────────────────────────────────
    for crop_type, crops in [('highchange', hc_crops), ('informative', inf_crops)]:
        print(f"\n{crop_type} crops:")
        crop_dir = f'{OUTPUT_DIR}/crops/{crop_type}'
        for i, (r, c) in enumerate(crops, 1):
            save_png(to_binary_rgb(pred[r:r+CROP_SIZE, c:c+CROP_SIZE]),
                     f'{crop_dir}/crop{i}_SatMAE.png')

    print(f"\nDone. Outputs added to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
