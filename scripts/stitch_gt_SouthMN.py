"""
Stitch GT chip masks into a single GeoTIF for SouthMN.
Output values: 0-12=class, 255=NoData (same encoding as prediction TIFs).
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import numpy as np
import rasterio
import rasterio.transform
from tqdm import tqdm

BASE      = str(MSR_ROOT)
CHIP_DIR  = os.path.join(str(DATA_ROOT), 'SatMAE_chips_MN/SouthMN')
TEST_TXT  = os.path.join(str(DATA_ROOT), 'SatMAE_chips_multitemporal/MN/test.txt')
OUT_PATH  = os.path.join(str(PREDICTIONS), 'GT_SouthMN.tif')

CHIP_SIZE = 96

with open(TEST_TXT) as f:
    chip_names = [l.strip() for l in f if l.strip()]

coords = [(int(n.split('_')[1]), int(n.split('_')[2])) for n in chip_names]
rows = [r for r, c in coords]
cols = [c for r, c in coords]
min_row, min_col = min(rows), min(cols)
max_row, max_col = max(rows), max(cols)
H = max_row - min_row + CHIP_SIZE
W = max_col - min_col + CHIP_SIZE

print(f"Canvas: {H}x{W}  chips: {len(chip_names)}")

gt = np.full((H, W), 255, dtype=np.uint8)
ref_profile = None
ref_transform = None
first_r, first_c = coords[0]

for name, (r, c) in tqdm(zip(chip_names, coords), total=len(chip_names), desc='Stitching GT'):
    mask_path = os.path.join(CHIP_DIR, name + '_mask.tif')
    if not os.path.exists(mask_path):
        continue
    with rasterio.open(mask_path) as src:
        if ref_profile is None:
            ref_profile = src.profile.copy()
            ref_transform = src.transform
            first_r, first_c = r, c
        chip = src.read(1).astype(np.int16)  # 1-13=class, 0=nodata

    y0 = r - min_row
    x0 = c - min_col

    # shift 1-13 -> 0-12, nodata(0) -> 255
    chip_out = np.where(chip == 0, 255, chip - 1).astype(np.uint8)
    gt[y0:y0+CHIP_SIZE, x0:x0+CHIP_SIZE] = chip_out

# Build canvas geotransform
res = ref_transform.a
canvas_transform = rasterio.transform.from_origin(
    ref_transform.c - (first_c - min_col) * res,
    ref_transform.f + (first_r - min_row) * res,
    res, res,
)

ref_profile.update({
    'width': W, 'height': H, 'count': 1,
    'dtype': 'uint8', 'transform': canvas_transform,
    'compress': 'lzw', 'nodata': 255,
})

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with rasterio.open(OUT_PATH, 'w', **ref_profile) as dst:
    dst.write(gt, 1)

print(f"Saved: {OUT_PATH}")
print(f"Unique values: {np.unique(gt)}")
