"""
Plain class-color PNG maps for SatMAE + FPN predictions (NWIA region).

Produces:
  NWIA_SatMAE_FPN_colored.png   -- prediction color map
  NWIA_GT_colored.png           -- ground truth color map
  legend.png                    -- class color legend

Output dir: {PREDICTIONS}/colored_maps

Usage:
  python vismap_satmae_fpn_NWIA.py
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
PRED_PATH  = f'{PREDICTIONS}/satmae_fpn_NWIA/NWIA_SatMAE_FPN_Prediction.tif'
CHIP_DIR   = f'{DATA_ROOT}/SatMAE_chips_multitemporal/NWIA'
TEST_TXT   = f'{DATA_ROOT}/SatMAE_chips_multitemporal/Iowa/test.txt'
CHIP_SIZE  = 96
OUTPUT_DIR = f'{PREDICTIONS}/colored_maps'

# ============================================================
# EXACT ESRI RGB COLORS FROM CDL EXCEL  (0-indexed)
# ============================================================
CLASS_COLORS = {
    0:   (233, 255, 190),   # Natural Vegetation
    1:   (149, 206, 147),   # Forest
    2:   (255, 212,   0),   # Corn
    3:   ( 38, 115,   0),   # Soybeans
    4:   (128, 179, 179),   # Wetlands
    5:   (156, 156, 156),   # Developed/Barren
    6:   ( 77, 112, 163),   # Open Water
    7:   (168, 112,   0),   # Winter Wheat
    8:   (255, 168, 227),   # Alfalfa
    9:   (191, 191, 122),   # Fallow/Idle
    10:  (255,  38,  38),   # Cotton
    11:  (255, 158,  15),   # Sorghum
    12:  (204, 191, 163),   # Other
    255: (255, 255, 255),   # NoData -- white
}

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans",
    "Wetlands", "Developed/Barren", "Open Water", "Winter Wheat",
    "Alfalfa", "Fallow/Idle", "Cotton", "Sorghum", "Other"
]


def colorize_map(class_array):
    h, w = class_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[class_array == class_id] = color
    return rgb


def to_0indexed_pred(arr):
    """SatMAE pred: 1-13=class, 0/255=NoData -> 0-12=class, 255=NoData"""
    out = np.full(arr.shape, 255, dtype=np.uint8)
    valid = (arr >= 1) & (arr <= 13)
    out[valid] = (arr[valid] - 1).astype(np.uint8)
    return out


def to_0indexed_gt(arr):
    """GT mask: 1-13=class, 0=NoData -> 0-12=class, 255=NoData"""
    out = np.full(arr.shape, 255, dtype=np.uint8)
    valid = (arr >= 1) & (arr <= 13)
    out[valid] = (arr[valid] - 1).astype(np.uint8)
    return out


def save_legend(output_path):
    n = len(CLASS_NAMES)
    box_size, padding, text_width = 30, 10, 220
    width  = box_size + padding * 2 + text_width
    height = n * (box_size + padding) + padding
    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw   = ImageDraw.Draw(legend)
    for i, name in enumerate(CLASS_NAMES):
        y = padding + i * (box_size + padding)
        draw.rectangle([padding, y, padding + box_size, y + box_size],
                       fill=CLASS_COLORS[i], outline=(0, 0, 0))
        draw.text((padding + box_size + 8, y + 6), f"{i+1}: {name}", fill=(0, 0, 0))
    legend.save(output_path)
    print(f"  Legend saved: {output_path}")


if __name__ == '__main__':
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Prediction ------------------------------------------------------------
    print("\nColorizing FPN prediction...")
    with rasterio.open(PRED_PATH) as src:
        pred = src.read(1).astype(np.int32)
        H, W = src.height, src.width
    print(f"  Shape: {pred.shape}  |  Unique values: {np.unique(pred)}")

    pred_rgb = colorize_map(to_0indexed_pred(pred))
    pred_out = output_dir / 'NWIA_SatMAE_FPN_colored.png'
    Image.fromarray(pred_rgb).save(pred_out)
    print(f"  Saved: {pred_out}")

    # -- Ground Truth ----------------------------------------------------------
    print("\nAssembling ground truth from chip masks...")
    gt_full = np.zeros((H, W), dtype=np.int32)
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
            gt_full[row:row+CHIP_SIZE, col:col+CHIP_SIZE] = src.read(1).astype(np.int32)
    if missing:
        print(f"  WARNING: {missing} masks not found")
    print(f"  Unique GT values: {np.unique(gt_full)}")

    gt_rgb = colorize_map(to_0indexed_gt(gt_full))
    gt_out = output_dir / 'NWIA_GT_colored.png'
    Image.fromarray(gt_rgb).save(gt_out)
    print(f"  Saved: {gt_out}")

    # -- Legend ----------------------------------------------------------------
    save_legend(output_dir / 'legend.png')

    print(f"\nDone! Maps saved to: {OUTPUT_DIR}")
