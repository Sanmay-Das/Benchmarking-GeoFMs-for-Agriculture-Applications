"""
Plain class-color PNG maps for SatMAE segmentation predictions (NWIA region).
Follows the exact same pattern as the Prithvi / SpectralGPT colorize script.

SatMAE stitched prediction encoding:
  1-13  → crop classes  (subtract 1 → 0-12 for CLASS_COLORS)
  0     → model-predicted NoData
  255   → masked / input NoData

Both 0 and 255 are mapped to white (same as Prithvi NoData treatment).

Produces per decoder:
  NWIA_SatMAE_<DECODER>_colored.png   — prediction color map
  NWIA_GT_colored.png                  — ground truth color map (same for all decoders)
  legend.png                           — class color legend

Usage:
  python vismap_satmae_colored_NWIA.py --decoder fpn
  python vismap_satmae_colored_NWIA.py --decoder psanet
  python vismap_satmae_colored_NWIA.py --decoder fcn
"""

import os
import argparse
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
CHIP_DIR  = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/NWIA'
TEST_TXT  = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE_chips_multitemporal/Iowa/test.txt'
CHIP_SIZE = 96

PRED_PATHS = {
    'psanet': '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_psanet_NWIA/NWIA_SatMAE_PSANet_Prediction.tif',
    'fcn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fcn_NWIA/NWIA_SatMAE_FCN_Prediction.tif',
    'fpn':    '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fpn_NWIA/NWIA_SatMAE_FPN_Prediction.tif',
}

OUTPUT_DIR = '/bigdata/eldawylab/sdas050/MS_Research/predictions/colored_maps'

# ============================================================
# EXACT ESRI RGB COLORS FROM CDL EXCEL  (0-indexed, same as Prithvi script)
# ============================================================
CLASS_COLORS = {
    0:   (233, 255, 190),   # Natural Vegetation — CDL 176 Grassland/Pasture
    1:   (149, 206, 147),   # Forest             — CDL 141 Deciduous Forest
    2:   (255, 212,   0),   # Corn               — CDL 1
    3:   ( 38, 115,   0),   # Soybeans           — CDL 5
    4:   (128, 179, 179),   # Wetlands           — CDL 87
    5:   (156, 156, 156),   # Developed/Barren   — CDL 121
    6:   ( 77, 112, 163),   # Open Water         — CDL 111
    7:   (168, 112,   0),   # Winter Wheat       — CDL 24
    8:   (255, 168, 227),   # Alfalfa            — CDL 36
    9:   (191, 191, 122),   # Fallow/Idle        — CDL 61
    10:  (255,  38,  38),   # Cotton             — CDL 2
    11:  (255, 158,  15),   # Sorghum            — CDL 4
    12:  (204, 191, 163),   # Other              — CDL 131 Barren
    255: (255, 255, 255),   # NoData             — white
}

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans",
    "Wetlands", "Developed/Barren", "Open Water", "Winter Wheat",
    "Alfalfa", "Fallow/Idle", "Cotton", "Sorghum", "Other"
]


# ============================================================
# HELPERS
# ============================================================

def colorize_map(class_array):
    """
    class_array : (H, W) uint8/int32 with values 0-12 (class) or 255 (NoData)
    Returns     : (H, W, 3) uint8 RGB image
    """
    h, w = class_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[class_array == class_id] = color
    return rgb


def satmae_pred_to_0indexed(pred_full):
    """
    Convert SatMAE stitched prediction (1-13 classes, 0=NoData, 255=masked)
    to 0-indexed (0-12 classes, 255=NoData) matching CLASS_COLORS keys.
    """
    out = np.full_like(pred_full, 255, dtype=np.uint8)   # default = NoData
    valid = (pred_full >= 1) & (pred_full <= 13)
    out[valid] = (pred_full[valid] - 1).astype(np.uint8)  # 1-13 → 0-12
    return out


def gt_to_0indexed(gt_full):
    """
    Convert GT mask (1-13 classes, 0=NoData)
    to 0-indexed (0-12 classes, 255=NoData) matching CLASS_COLORS keys.
    """
    out = np.full_like(gt_full, 255, dtype=np.uint8)      # default = NoData
    valid = (gt_full >= 1) & (gt_full <= 13)
    out[valid] = (gt_full[valid] - 1).astype(np.uint8)    # 1-13 → 0-12
    return out


def save_legend(output_path):
    n = len(CLASS_NAMES)
    box_size   = 30
    padding    = 10
    text_width = 220
    width  = box_size + padding * 2 + text_width
    height = n * (box_size + padding) + padding

    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw   = ImageDraw.Draw(legend)

    for i, name in enumerate(CLASS_NAMES):
        y     = padding + i * (box_size + padding)
        color = CLASS_COLORS.get(i, (128, 128, 128))
        draw.rectangle([padding, y, padding + box_size, y + box_size],
                       fill=color, outline=(0, 0, 0))
        draw.text((padding + box_size + 8, y + 6), f"{i+1}: {name}", fill=(0, 0, 0))

    legend.save(output_path)
    print(f"  Legend saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', choices=['psanet', 'fcn', 'fpn'], default='fpn')
    args = parser.parse_args()

    dec        = args.decoder.upper()
    pred_path  = PRED_PATHS[args.decoder]
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDecoder:    {dec}")
    print(f"Prediction: {pred_path}")
    print(f"Output:     {output_dir}\n")

    # ── Load stitched prediction ──────────────────────────────────────────────
    with rasterio.open(pred_path) as src:
        pred_full = src.read(1).astype(np.int32)
        H, W      = src.height, src.width

    print(f"Prediction map: {H} × {W}")
    print(f"Unique pred values: {np.unique(pred_full)}")

    # ── Colorize prediction ───────────────────────────────────────────────────
    pred_0idx = satmae_pred_to_0indexed(pred_full)
    pred_rgb  = colorize_map(pred_0idx)

    pred_out = output_dir / f'NWIA_SatMAE_{dec}_colored.png'
    Image.fromarray(pred_rgb).save(pred_out)
    print(f"  Prediction PNG saved: {pred_out}")

    # ── Assemble stitched GT ──────────────────────────────────────────────────
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
            gt_chip = src.read(1).astype(np.int32)
        gt_full[row:row+CHIP_SIZE, col:col+CHIP_SIZE] = gt_chip

    if missing:
        print(f"WARNING: {missing} masks not found")
    print(f"Unique GT values: {np.unique(gt_full)}")

    # ── Colorize GT ───────────────────────────────────────────────────────────
    gt_0idx = gt_to_0indexed(gt_full)
    gt_rgb  = colorize_map(gt_0idx)

    gt_out = output_dir / 'NWIA_GT_colored.png'
    Image.fromarray(gt_rgb).save(gt_out)
    print(f"  GT PNG saved: {gt_out}")

    # ── Legend ────────────────────────────────────────────────────────────────
    save_legend(output_dir / 'legend.png')

    print(f"\nAll done! Maps saved to: {output_dir}")


if __name__ == '__main__':
    main()
