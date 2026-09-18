"""
Segmentation visualization for SouthMN -- SatMAE (FCN/FPN/PSANet) vs Prithvi vs SpectralGPT vs GT.
Saves colorized GeoTIFFs to visualizations/seg_SouthMN/.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
import glob
import numpy as np
import rasterio
import rasterio.transform
from tqdm import tqdm

# -- Color palette (class index 0-12 -> RGB) ------------------------------------
# Class 0=Natural Veg, 1=Forest, 2=Corn, 3=Soybeans, 4=Wetlands,
#       5=Developed/Barren, 6=Open Water, 7=Winter Wheat, 8=Alfalfa,
#       9=Fallow/Idle, 10=Cotton, 11=Sorghum, 12=Other
COLORS = np.array([
    [204, 190, 162],   # 0: Natural Vegetation
    [104, 171, 102],   # 1: Forest
    [255, 212,   0],   # 2: Corn
    [ 38, 115,   0],   # 3: Soybeans
    [ 75, 144, 226],   # 4: Wetlands
    [204, 190, 162],   # 5: Developed/Barren
    [  0, 168, 226],   # 6: Open Water
    [168, 112,   0],   # 7: Winter Wheat
    [255, 168, 227],   # 8: Alfalfa
    [204, 190, 162],   # 9: Fallow/Idle
    [255,  38,  38],   # 10: Cotton
    [255, 158,  15],   # 11: Sorghum
    [204, 204, 204],   # 12: Other
], dtype=np.uint8)

NODATA_COLOR = np.array([0, 0, 0], dtype=np.uint8)  # black for NoData

BASE_DIR  = str(MSR_ROOT)
PRED_DIR  = str(PREDICTIONS)
CHIP_DIR  = os.path.join(str(DATA_ROOT), 'SatMAE_chips_MN/SouthMN')
SPLITS_TXT = os.path.join(str(DATA_ROOT), 'SatMAE_chips_multitemporal/MN/test.txt')
OUT_DIR   = os.path.join(str(OUTPUT_ROOT), 'visualizations/seg_SouthMN')

PREDICTIONS = {
    'SatMAE_FCN':    os.path.join(PRED_DIR, 'satmae_fcn_SouthMN/SouthMN_SatMAE_FCN_Prediction.tif'),
    'SatMAE_FPN':    os.path.join(PRED_DIR, 'satmae_fpn_SouthMN/SouthMN_SatMAE_FPN_Prediction.tif'),
    'SatMAE_PSANet': os.path.join(PRED_DIR, 'satmae_psanet_SouthMN/SouthMN_SatMAE_PSANet_Prediction.tif'),
    'Prithvi':       os.path.join(PRED_DIR, 'prithvi_SouthMN_terratorch/SouthMN_Prithvi_Prediction_Stitched.tif'),
    'SpectralGPT':   os.path.join(PRED_DIR, 'spectralgpt_SouthMN_terratorch/SouthMN_SpectralGPT_Prediction_Stitched.tif'),
}


def labels_to_rgb(arr, nodata_val=255):
    """arr: 2D uint8, values 0-12 (class) or nodata_val."""
    H, W = arr.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(13):
        mask = arr == c
        rgb[mask] = COLORS[c]
    rgb[arr == nodata_val] = NODATA_COLOR
    return rgb


def save_rgb_tif(rgb, output_path, profile):
    out_profile = profile.copy()
    out_profile.update({
        'count': 3, 'dtype': 'uint8',
        'compress': 'lzw', 'photometric': 'RGB', 'nodata': None,
    })
    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)
    print(f"  Saved: {output_path}")


def stitch_gt():
    """Stitch GT from SouthMN chip mask files. Values 0=NoData, 1-13=class."""
    print("Stitching GT from chip masks...")
    with open(SPLITS_TXT) as f:
        chip_names = [l.strip() for l in f if l.strip()]

    coords = [(int(n.split('_')[1]), int(n.split('_')[2])) for n in chip_names]
    rows = [r for r, c in coords]
    cols = [c for r, c in coords]
    min_row, min_col = min(rows), min(cols)
    H = max(rows) - min_row + 96
    W = max(cols) - min_col + 96

    gt = np.zeros((H, W), dtype=np.uint8)
    ref_profile = None

    for name, (r, c) in tqdm(zip(chip_names, coords), total=len(chip_names), desc="GT stitch"):
        mask_path = os.path.join(CHIP_DIR, name + '_mask.tif')
        if not os.path.exists(mask_path):
            continue
        with rasterio.open(mask_path) as src:
            if ref_profile is None:
                ref_profile = src.profile.copy()
            chip = src.read(1)
        y0, x0 = r - min_row, c - min_col
        gt[y0:y0+96, x0:x0+96] = chip

    # shift 1-13 -> 0-12, NoData(0) -> 255
    gt_shifted = gt.astype(np.int16) - 1          # 0-12, NoData -> -1
    gt_out = np.where(gt_shifted < 0, 255, gt_shifted).astype(np.uint8)

    # Build geotransform for the stitched canvas
    first_chip = os.path.join(CHIP_DIR, chip_names[0] + '.tif')
    with rasterio.open(first_chip) as src:
        ft = src.transform
        canvas_profile = src.profile.copy()
    first_r, first_c = coords[0]
    res = ft.a
    canvas_transform = rasterio.transform.from_origin(
        ft.c - (first_c - min_col) * res,
        ft.f + (first_r - min_row) * res,
        res, res,
    )
    canvas_profile.update({'width': W, 'height': H, 'count': 1,
                           'dtype': 'uint8', 'transform': canvas_transform})
    return gt_out, canvas_profile


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -- 1. Stitch and save GT -------------------------------------------------
    gt, gt_profile = stitch_gt()
    rgb = labels_to_rgb(gt, nodata_val=255)
    save_rgb_tif(rgb, os.path.join(OUT_DIR, 'SouthMN_GT.tif'), gt_profile)

    # -- 2. Colorize each prediction -------------------------------------------
    for name, path in PREDICTIONS.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        print(f"\nColorizing {name}...")
        with rasterio.open(path) as src:
            pred = src.read(1)
            profile = src.profile.copy()

        if name.startswith('SatMAE'):
            # Values 0-13: 0=NoData, 1-13=class -> shift to 0-12, nodata=255
            shifted = pred.astype(np.int16) - 1
            pred_vis = np.where(shifted < 0, 255, shifted).astype(np.uint8)
            nodata_val = 255
        else:
            # Prithvi/SpectralGPT: already 0-12, nodata=255
            pred_vis = pred.astype(np.uint8)
            nodata_val = 255

        rgb = labels_to_rgb(pred_vis, nodata_val=nodata_val)
        out_path = os.path.join(OUT_DIR, f'SouthMN_{name}_pred.tif')
        save_rgb_tif(rgb, out_path, profile)

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Load all GeoTIFFs in QGIS for side-by-side comparison.")


if __name__ == '__main__':
    main()
