"""
SatMAE + FCNHead -- chip-based inference for SouthMN segmentation.
Reads chips from SatMAE_chips_MN/SouthMN/, stitches with cosine blending,
computes per-class IoU against mask files, saves prediction GeoTIFF.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from tqdm import tqdm

SATMAE_DIR = f'{MSR_ROOT}/SatMAE'
sys.path.insert(0, SATMAE_DIR)

import models_vit_group_channels
from models_satmae_fcn import SatMAEFCN

# -- config --------------------------------------------------------------------
DATA_DIR    = f'{DATA_ROOT}/SatMAE_chips_MN/SouthMN'
SPLITS_TXT  = f'{DATA_ROOT}/SatMAE_chips_multitemporal/MN/test.txt'
CHECKPOINT  = f'{MSR_ROOT}/SatMAE/output_seg_MN_fcn/checkpoint-best.pth'
OUTPUT_DIR  = f'{PREDICTIONS}/satmae_fcn_SouthMN'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'SouthMN_SatMAE_FCN_Prediction.tif')

CHIP_SIZE   = 96
STRIDE      = 48
DELTA       = 8
NUM_CLASSES = 14
BATCH_SIZE  = 64

_MEAN_6 = np.array([1184.3824625, 1120.77120066, 1136.26026392,
                     1972.62420416, 1732.16362238, 1247.91870117], dtype=np.float32)
_STD_6  = np.array([650.2842772,  712.12507725,  965.23119807,
                     1364.38688993, 1310.36996126, 1087.6020813],  dtype=np.float32)
SATMAE_MEAN = np.tile(_MEAN_6, 3).reshape(18, 1, 1)
SATMAE_STD  = np.tile(_STD_6,  3).reshape(18, 1, 1)

GROUPED_BANDS = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
PATCH_SIZE    = 8
IN_CHANS      = 18
DROP_PATH     = 0.2

CLASS_NAMES = [
    'NoData', 'Natural Veg', 'Forest', 'Corn', 'Soybean', 'Wetlands',
    'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
    'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]


def cosine_blend_mask(chip_size, stride, delta):
    size     = chip_size - 2 * delta
    overlap  = chip_size - stride
    ramp_len = max(min(chip_size // 2, overlap) - delta, 0)
    x = torch.ones(size, dtype=torch.float32)
    if ramp_len > 0:
        ramp = torch.cos(math.pi * (torch.arange(ramp_len, dtype=torch.float32) + 1)
                         / (ramp_len + 1)) / 2 + 0.5
        x[:ramp_len] = ramp.flip(0)
        x[-ramp_len:] = ramp
    return (x[:, None] * x[None, :]) + 1e-6


def compute_iou(pred, gt, num_classes):
    ious = []
    for c in range(1, num_classes):  # skip 0=NoData
        mask = gt != 0
        pred_c = (pred == c) & mask
        gt_c   = (gt   == c) & mask
        inter  = (pred_c & gt_c).sum()
        union  = (pred_c | gt_c).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(100.0 * inter / union)
    return ious


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # -- load chip list --------------------------------------------------------
    with open(SPLITS_TXT) as f:
        chip_names = [l.strip() for l in f if l.strip()]
    print(f"Test chips: {len(chip_names)}")

    # -- parse row/col, compute canvas -----------------------------------------
    coords = []
    for name in chip_names:
        parts = name.split('_')
        coords.append((int(parts[1]), int(parts[2])))
    rows = [r for r, c in coords]
    cols = [c for r, c in coords]
    min_row, min_col = min(rows), min(cols)
    max_row, max_col = max(rows), max(cols)
    H = max_row - min_row + CHIP_SIZE
    W = max_col - min_col + CHIP_SIZE
    print(f"Canvas: {H}x{W}  (min_row={min_row}, min_col={min_col})")

    # -- get reference profile and compute canvas transform -------------------
    ref_chip = os.path.join(DATA_DIR, chip_names[0] + '.tif')
    with rasterio.open(ref_chip) as src:
        ref_profile = src.profile.copy()
        first_transform = src.transform
    first_row, first_col = coords[0]
    row_offset = (first_row - min_row)
    col_offset = (first_col - min_col)
    import rasterio.transform as rt
    res = first_transform.a  # pixel size
    canvas_transform = rasterio.transform.from_origin(
        first_transform.c - col_offset * res,
        first_transform.f + row_offset * res,
        res, res
    )

    # -- build model -----------------------------------------------------------
    print(f"\nBuilding SatMAE + FCNHead...")
    encoder = models_vit_group_channels.vit_large_patch16(
        patch_size=PATCH_SIZE, img_size=CHIP_SIZE, in_chans=IN_CHANS,
        channel_groups=GROUPED_BANDS, num_classes=0, drop_path_rate=DROP_PATH,
        global_pool=False,
    )
    model = SatMAEFCN(encoder=encoder, nb_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded epoch: {ckpt.get('epoch', '?')}")
    model = model.to(device).eval()

    # -- inference with cosine blend -------------------------------------------
    blend_mask = cosine_blend_mask(CHIP_SIZE, STRIDE, DELTA).to(device)
    inner      = CHIP_SIZE - 2 * DELTA

    prob_accum = torch.zeros((NUM_CLASSES, H, W), dtype=torch.float32)
    count_map  = torch.zeros((H, W), dtype=torch.float32)

    # also accumulate GT for IoU
    gt_canvas  = np.zeros((H, W), dtype=np.uint8)
    gt_count   = np.zeros((H, W), dtype=np.float32)

    start = time.time()
    batch_chips, batch_coords, batch_masks = [], [], []

    def flush_batch():
        if not batch_chips:
            return
        chips_t = torch.from_numpy(np.stack(batch_chips)).float().to(device)
        with torch.no_grad():
            logits = model(chips_t, is_train=False)   # (B, 14, 96, 96)
            probs  = F.softmax(logits, dim=1)
        for i, (r, c, mask_arr) in enumerate(zip(
                [bc[0] for bc in batch_coords],
                [bc[1] for bc in batch_coords],
                batch_masks)):
            p_inner = probs[i][:, DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA]
            bm      = blend_mask

            y0, x0 = r - min_row, c - min_col
            y1, x1 = y0 + inner, x0 + inner

            prob_accum[:, y0:y1, x0:x1] += (p_inner * bm).cpu()
            count_map[y0:y1, x0:x1]     += bm.cpu()

            if mask_arr is not None:
                m_inner = mask_arr[DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA].astype(np.float32)
                valid   = (mask_arr[DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA] != 0).astype(np.float32)
                gt_canvas[y0:y1, x0:x1] += (m_inner * valid).astype(np.uint8)
                gt_count[y0:y1, x0:x1]  += valid

        batch_chips.clear(); batch_coords.clear(); batch_masks.clear()

    for name, (r, c) in tqdm(zip(chip_names, coords), total=len(chip_names), desc="FCN Inference"):
        chip_path = os.path.join(DATA_DIR, name + '.tif')
        mask_path = os.path.join(DATA_DIR, name + '_mask.tif')
        if not os.path.exists(chip_path):
            continue
        with rasterio.open(chip_path) as src:
            data = src.read().astype(np.float32)
        data = (data - SATMAE_MEAN) / (SATMAE_STD + 1e-8)
        mask_arr = None
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                mask_arr = src.read(1)
        batch_chips.append(data)
        batch_coords.append((r, c))
        batch_masks.append(mask_arr)
        if len(batch_chips) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    print(f"\nInference done in {time.time()-start:.1f}s")

    # -- stitch prediction -----------------------------------------------------
    count_map[count_map == 0] = 1
    prob_accum /= count_map.unsqueeze(0)
    pred = torch.argmax(prob_accum, dim=0).numpy().astype(np.uint8)

    # -- GT from majority vote -------------------------------------------------
    gt_count[gt_count == 0] = 1
    gt_final = np.round(gt_canvas.astype(np.float32) / gt_count).astype(np.uint8)

    # -- compute IoU -----------------------------------------------------------
    ious = compute_iou(pred, gt_final, NUM_CLASSES)
    valid_ious = [v for v in ious if not math.isnan(v)]
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

    print(f"\nSouthMN FCN Results:")
    print(f"  mIoU: {miou:.2f}%")
    print(f"  Per-class IoU:")
    for i, iou in enumerate(ious):
        name = CLASS_NAMES[i+1]
        print(f"    {name:20s}: {iou:.2f}%" if not math.isnan(iou) else f"    {name:20s}: NaN")

    # -- save GeoTIFF ---------------------------------------------------------
    out_profile = ref_profile.copy()
    out_profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw',
                        'nodata': 255, 'width': W, 'height': H,
                        'transform': canvas_transform})
    with rasterio.open(OUTPUT_FILE, 'w', **out_profile) as dst:
        dst.write(pred, 1)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Output: {pred.shape}")


if __name__ == '__main__':
    main()
