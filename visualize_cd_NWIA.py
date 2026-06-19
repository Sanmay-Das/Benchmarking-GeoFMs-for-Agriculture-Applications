"""
visualize_cd_NWIA.py
--------------------
Change detection visualization for NWIA following SpectralGPT Fig. 7(d).
Memory-efficient: never holds more than one full-res pred at a time.
For crops: loads only the chips overlapping that crop window.

Binary maps: white=changed, black=unchanged, gray=nodata

Outputs:
    visualizations/cd_NWIA/
        NWIA_CD_GT_colored.png
        NWIA_CD_SpectralGPT_colored.png
        NWIA_CD_Prithvi_colored.png
        NWIA_CD_SatMAE_colored.png
        legend.png
        geotiffs/  NWIA_CD_GT.tif / SpectralGPT.tif / Prithvi.tif / SatMAE.tif
        crops/highchange/   crop1_T1/T2/GT/SpectralGPT/Prithvi/SatMAE.png  x3
        crops/informative/  crop1_T1/T2/GT/SpectralGPT/Prithvi/SatMAE.png  x3
"""

import os
import gc
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
BASE      = '/bigdata/eldawylab/sdas050/MS_Research'
CHIPS_CSV = f'{BASE}/change_detection_chips/spectralgpt/NWIA_chips.csv'
CHIP_SIZE = 128

GT_PATH = f'{BASE}/predictions/cd_spectralgpt_NWIA/NWIA_SpectralGPT_CD_gt.tif'

MODELS = [
    ('SpectralGPT',
     f'{BASE}/predictions/cd_spectralgpt_NWIA/NWIA_SpectralGPT_CD_pred.tif'),
    ('Prithvi',
     f'{BASE}/predictions/cd_prithvi_NWIA/NWIA_Prithvi_CD_pred.tif'),
    ('SatMAE',
     f'{BASE}/predictions/cd_satmae_NWIA/NWIA_SatMAE_CD_pred.tif'),
]

OUTPUT_DIR = f'{BASE}/visualizations/cd_NWIA'
SCALE      = 4      # downsample for full-scene PNGs
STRIP_H    = 256    # strip height for GeoTIFF writing
CROP_SIZE  = 512    # full-res crop size in pixels


# ── helpers ───────────────────────────────────────────────────────────────────
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
    """Load pred TIF and align to (H,W) GT canvas."""
    with rasterio.open(pred_path) as src:
        pred_raw = src.read(1)
    pred = np.full((H, W), 255, dtype=np.uint8)
    h, w = min(pred_raw.shape[0], H), min(pred_raw.shape[1], W)
    pred[:h, :w] = pred_raw[:h, :w]
    del pred_raw
    return pred


# ── downsampled T1/T2 stitching (for full-scene PNG only) ────────────────────
def stitch_rgb_ds(df, chip_size, scale, min_row, min_col, Hd, Wd):
    cs = chip_size // scale
    t1_sum = np.zeros((Hd, Wd, 3), dtype=np.float32)
    t2_sum = np.zeros((Hd, Wd, 3), dtype=np.float32)
    count  = np.zeros((Hd, Wd),    dtype=np.float32)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        r = (int(row['row']) - min_row) // scale
        c = (int(row['col']) - min_col) // scale
        with rasterio.open(row['t1']) as src:
            b = src.read([3, 2, 1]).astype(np.float32)
        b_ds = b[:, ::scale, ::scale]
        h, w = b_ds.shape[1], b_ds.shape[2]
        t1_sum[r:r+h, c:c+w] += b_ds.transpose(1, 2, 0)
        with rasterio.open(row['t2']) as src:
            b = src.read([3, 2, 1]).astype(np.float32)
        b_ds = b[:, ::scale, ::scale]
        t2_sum[r:r+h, c:c+w] += b_ds.transpose(1, 2, 0)
        count[r:r+h, c:c+w] += 1.0
    count[count == 0] = 1
    def to_uint8(img):
        out = np.zeros_like(img, dtype=np.uint8)
        for ch in range(3):
            band = img[..., ch]; valid = band[band > 0]
            if not len(valid): continue
            p2, p98 = np.percentile(valid, (2, 98))
            out[..., ch] = np.clip((band-p2)/(p98-p2+1e-8)*255, 0, 255).astype(np.uint8)
        return out
    return to_uint8(t1_sum / count[..., None]), to_uint8(t2_sum / count[..., None])


# ── targeted T1/T2 stitching for a single crop window ────────────────────────
def get_crop_rgb(df, r0, c0, crop_size, chip_size, min_row, min_col):
    """Load only chips overlapping [r0:r0+crop_size, c0:c0+crop_size]."""
    t1_sum = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    t2_sum = np.zeros((crop_size, crop_size, 3), dtype=np.float32)
    count  = np.zeros((crop_size, crop_size),    dtype=np.float32)

    for _, row in df.iterrows():
        cr = int(row['row']) - min_row
        cc = int(row['col']) - min_col
        # Skip non-overlapping chips
        if cr >= r0+crop_size or cr+chip_size <= r0: continue
        if cc >= c0+crop_size or cc+chip_size <= c0: continue

        # Overlap in crop coords
        yr0 = max(cr, r0) - r0;  yr1 = min(cr+chip_size, r0+crop_size) - r0
        xc0 = max(cc, c0) - c0;  xc1 = min(cc+chip_size, c0+crop_size) - c0
        # Overlap in chip coords
        cy0 = max(cr, r0) - cr;  cy1 = min(cr+chip_size, r0+crop_size) - cr
        cx0 = max(cc, c0) - cc;  cx1 = min(cc+chip_size, c0+crop_size) - cc

        with rasterio.open(row['t1']) as src:
            b = src.read([3, 2, 1]).astype(np.float32)
        t1_sum[yr0:yr1, xc0:xc1] += b[:, cy0:cy1, cx0:cx1].transpose(1, 2, 0)
        with rasterio.open(row['t2']) as src:
            b = src.read([3, 2, 1]).astype(np.float32)
        t2_sum[yr0:yr1, xc0:xc1] += b[:, cy0:cy1, cx0:cx1].transpose(1, 2, 0)
        count[yr0:yr1, xc0:xc1] += 1.0

    count[count == 0] = 1
    def to_uint8(img):
        out = np.zeros_like(img, dtype=np.uint8)
        for ch in range(3):
            band = img[..., ch]; valid = band[band > 0]
            if not len(valid): continue
            p2, p98 = np.percentile(valid, (2, 98))
            out[..., ch] = np.clip((band-p2)/(p98-p2+1e-8)*255, 0, 255).astype(np.uint8)
        return out
    return to_uint8(t1_sum/count[...,None]), to_uint8(t2_sum/count[...,None])


# ── crop finders (operate on downsampled arrays) ──────────────────────────────
def find_highchange_crops(gt_ds, n=3, crop_ds=128):
    H, W, best = gt_ds.shape[0], gt_ds.shape[1], []
    step = crop_ds // 2
    for r in range(0, H-crop_ds, step):
        for c in range(0, W-crop_ds, step):
            p = gt_ds[r:r+crop_ds, c:c+crop_ds]
            valid = (p != 255).sum()
            if valid < crop_ds*crop_ds*0.5: continue
            pct = (p==1).sum()/(valid+1e-8)
            best.append((-abs(pct-0.40), r, c))
    best.sort()
    return [(r*SCALE, c*SCALE) for _, r, c in best[:n]]


def find_informative_crops(gt_ds, preds_ds, n=3, crop_ds=128):
    """SpectralGPT TP high + Prithvi/SatMAE FN high."""
    H, W, score = gt_ds.shape[0], gt_ds.shape[1], None
    step = crop_ds // 2
    scores = []
    for r in range(0, H-crop_ds, step):
        for c in range(0, W-crop_ds, step):
            gt_p   = gt_ds[r:r+crop_ds, c:c+crop_ds]
            valid  = (gt_p != 255).sum()
            if valid < crop_ds*crop_ds*0.5: continue
            changed = (gt_p == 1)
            sgpt_tp  = ((preds_ds['SpectralGPT'][r:r+crop_ds, c:c+crop_ds]==1) & changed).sum()
            others_fn = sum(
                ((preds_ds[m][r:r+crop_ds, c:c+crop_ds]==0) & changed).sum()
                for m in ['Prithvi','SatMAE']
            )
            scores.append((-(int(sgpt_tp)+int(others_fn)), r, c))
    scores.sort()
    return [(r*SCALE, c*SCALE) for _, r, c in scores[:n]]


# ── legend ────────────────────────────────────────────────────────────────────
def save_legend(out_path):
    labels = [('Changed',(255,255,255)),('Unchanged',(0,0,0)),('No Data',(128,128,128))]
    box, pad, tw = 30, 10, 180
    img = Image.new('RGB',(box+pad*2+tw, len(labels)*(box+pad)+pad),(240,240,240))
    draw = ImageDraw.Draw(img)
    for i,(label,color) in enumerate(labels):
        y = pad + i*(box+pad)
        draw.rectangle([pad, y, pad+box, y+box], fill=color, outline=(0,0,0))
        draw.text((pad+box+8, y+6), label, fill=(0,0,0))
    img.save(out_path)
    print(f"    legend.png")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    for d in [OUTPUT_DIR,
              f'{OUTPUT_DIR}/geotiffs',
              f'{OUTPUT_DIR}/crops/highchange',
              f'{OUTPUT_DIR}/crops/informative']:
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(CHIPS_CSV)
    rows, cols = df['row'].values, df['col'].values
    min_row, min_col = int(rows.min()), int(cols.min())

    # Read GT (38MB — stays in memory throughout)
    with rasterio.open(GT_PATH) as src:
        gt = src.read(1)
    H, W = gt.shape
    print(f"GT canvas: {H}×{W}\n")

    # ── Phase 1: full-scene PNGs + GeoTIFFs ──────────────────────────────────
    print("=== Phase 1: Full-scene maps ===")
    Hd, Wd = H//SCALE, W//SCALE

    print("Stitching T1/T2 RGB (downsampled)...")
    t1_ds, t2_ds = stitch_rgb_ds(df, CHIP_SIZE, SCALE, min_row, min_col, Hd, Wd)
    gc.collect()

    print("\nGround Truth:")
    save_png(to_binary_rgb(gt[::SCALE, ::SCALE]),
             f'{OUTPUT_DIR}/NWIA_CD_GT_colored.png')
    save_geotiff_strip(gt, GT_PATH, f'{OUTPUT_DIR}/geotiffs/NWIA_CD_GT.tif')

    preds_ds = {}
    for model_name, pred_path in MODELS:
        print(f"\n{model_name}:")
        pred = load_and_align(pred_path, H, W)
        save_png(to_binary_rgb(pred[::SCALE, ::SCALE]),
                 f'{OUTPUT_DIR}/NWIA_CD_{model_name}_colored.png')
        save_geotiff_strip(pred, GT_PATH,
                           f'{OUTPUT_DIR}/geotiffs/NWIA_CD_{model_name}.tif')
        preds_ds[model_name] = pred[::SCALE, ::SCALE].copy()
        del pred
        gc.collect()

    save_legend(f'{OUTPUT_DIR}/legend.png')

    # ── Phase 2: find crop locations (downsampled — tiny memory) ─────────────
    print("\n=== Phase 2: Finding crop locations ===")
    crop_ds   = CROP_SIZE // SCALE   # 128 at 4x downsample
    gt_ds     = gt[::SCALE, ::SCALE]
    hc_crops  = find_highchange_crops(gt_ds, n=3, crop_ds=crop_ds)
    inf_crops = find_informative_crops(gt_ds, preds_ds, n=3, crop_ds=crop_ds)
    del gt_ds, preds_ds
    gc.collect()
    print(f"  High-change crops:  {hc_crops}")
    print(f"  Informative crops:  {inf_crops}")

    # ── Phase 3: save crops (one pred at a time, targeted chip loading) ───────
    for crop_type, crops in [('highchange', hc_crops), ('informative', inf_crops)]:
        print(f"\n=== Phase 3: {crop_type} crops ===")
        crop_dir = f'{OUTPUT_DIR}/crops/{crop_type}'

        for i, (r, c) in enumerate(crops, 1):
            print(f"  Crop {i} at ({r},{c}):")
            name = f'crop{i}'

            # T1/T2: load only overlapping chips
            t1_crop, t2_crop = get_crop_rgb(
                df, r, c, CROP_SIZE, CHIP_SIZE, min_row, min_col)
            save_png(t1_crop, f'{crop_dir}/{name}_T1.png')
            save_png(t2_crop, f'{crop_dir}/{name}_T2.png')
            del t1_crop, t2_crop

            # GT crop
            save_png(to_binary_rgb(gt[r:r+CROP_SIZE, c:c+CROP_SIZE]),
                     f'{crop_dir}/{name}_GT.png')

            # Pred crops — one at a time
            for model_name, pred_path in MODELS:
                pred = load_and_align(pred_path, H, W)
                save_png(to_binary_rgb(pred[r:r+CROP_SIZE, c:c+CROP_SIZE]),
                         f'{crop_dir}/{name}_{model_name}.png')
                del pred
                gc.collect()

    print(f"\nAll done. Outputs: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
