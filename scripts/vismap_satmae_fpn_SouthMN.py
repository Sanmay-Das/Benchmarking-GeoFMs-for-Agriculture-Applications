"""
Convert SouthMN SatMAE FPN colored GeoTIFFs to PNG.
Reads from visualizations/seg_SouthMN/ (already colorized RGB GeoTIFFs).
Saves PNGs to predictions/colored_maps/.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import numpy as np
import rasterio
from PIL import Image
from pathlib import Path

BASE_DIR  = Path(str(MSR_ROOT))
VIZ_DIR   = BASE_DIR / 'visualizations/seg_SouthMN'
OUT_DIR   = BASE_DIR / 'predictions/colored_maps'

FILES = {
    'SouthMN_SatMAE_FPN_colored.png': VIZ_DIR / 'SouthMN_SatMAE_FPN_pred.tif',
    'SouthMN_GT_colored.png':         VIZ_DIR / 'SouthMN_GT.tif',
}


def tif_to_png(tif_path, png_path):
    print(f"  Reading: {tif_path}")
    with rasterio.open(tif_path) as src:
        r = src.read(1)
        g = src.read(2)
        b = src.read(3)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    print(f"  Shape: {rgb.shape}  dtype: {rgb.dtype}")
    img = Image.fromarray(rgb, mode='RGB')
    img.save(png_path)
    print(f"  Saved: {png_path}")


if __name__ == '__main__':
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, tif_path in FILES.items():
        if not tif_path.exists():
            print(f"  MISSING: {tif_path}")
            continue
        tif_to_png(tif_path, OUT_DIR / out_name)

    print(f"\nDone. PNGs in: {OUT_DIR}")
