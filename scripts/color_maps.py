
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

# GT_PATH          = f"{PREDICTIONS}/gt_maps/SouthMN_gt_13class.tif"
PRITHVI_PATH     = f"{PREDICTIONS}/prithvi_NWIA_terratorch/NWIA_Prithvi_Prediction_Stitched_HLSweights.tif"
# SPECTRALGPT_PATH = f"{PREDICTIONS}/spectralgpt_SouthMN_terratorch/SouthMN_SpectralGPT_Prediction_Stitched.tif"
OUTPUT_DIR       = f"{PREDICTIONS}/colored_maps"

# ============================================================
# EXACT ESRI RGB COLORS FROM CDL EXCEL
# ============================================================

CLASS_COLORS = {
    0:   (233, 255, 190),  # Natural Vegetation -- CDL 176 Grassland/Pasture
    1:   (149, 206, 147),  # Forest             -- CDL 141 Deciduous Forest
    2:   (255, 212, 0),    # Corn               -- CDL 1
    3:   (38,  115, 0),    # Soybeans           -- CDL 5
    4:   (128, 179, 179),  # Wetlands           -- CDL 87
    5:   (156, 156, 156),  # Developed/Barren   -- CDL 121
    6:   (77,  112, 163),  # Open Water         -- CDL 111
    7:   (168, 112, 0),    # Winter Wheat       -- CDL 24
    8:   (255, 168, 227),  # Alfalfa            -- CDL 36
    9:   (191, 191, 122),  # Fallow/Idle        -- CDL 61
    10:  (255, 38,  38),   # Cotton             -- CDL 2
    11:  (255, 158, 15),   # Sorghum            -- CDL 4
    12:  (204, 191, 163),  # Other              -- CDL 131 Barren
    255: (255,   255,   255),    # NoData             -- white
}

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans",
    "Wetlands", "Developed/Barren", "Open Water", "Winter Wheat",
    "Alfalfa", "Fallow/Idle", "Cotton", "Sorghum", "Other"
]


# ============================================================
# COLORIZE
# ============================================================

def colorize_map(class_array):
    h, w = class_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[class_array == class_id] = color
    return rgb


def save_legend(output_path):
    n = len(CLASS_NAMES)
    box_size = 30
    padding = 10
    text_width = 220
    width = box_size + padding * 2 + text_width
    height = n * (box_size + padding) + padding

    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    for i, name in enumerate(CLASS_NAMES):
        y = padding + i * (box_size + padding)
        color = CLASS_COLORS.get(i, (128, 128, 128))
        draw.rectangle([padding, y, padding + box_size, y + box_size],
                       fill=color, outline=(0, 0, 0))
        draw.text((padding + box_size + 8, y + 6), f"{i}: {name}", fill=(0, 0, 0))

    legend.save(output_path)
    print(f"  [OK] Legend saved: {output_path}")


def colorize_tif(tif_path, output_png_path, label):
    print(f"\nColorizing {label}...")
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        print(f"  Shape: {data.shape}")
        print(f"  Unique values: {np.unique(data)}")

    rgb = colorize_map(data)
    Image.fromarray(rgb).save(output_png_path)
    print(f"  [OK] Saved: {output_png_path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # colorize_tif(GT_PATH,          output_dir / "SouthMN_GT_colored.png",          "Ground Truth")
    colorize_tif(PRITHVI_PATH,     output_dir / "NWIA_Prithvi_colored_HLS.png",     "Prithvi")
    # colorize_tif(SPECTRALGPT_PATH, output_dir / "SouthMN_SpectralGPT_colored.png", "SpectralGPT")

    save_legend(output_dir / "legend.png")

    print(f"\n[OK] All done! Colored maps saved to: {OUTPUT_DIR}")