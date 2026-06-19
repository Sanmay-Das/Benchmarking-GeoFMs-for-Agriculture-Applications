import os
import numpy as np
import rasterio
import glob
from tqdm import tqdm

# Official CDL colors mapped to your 13 classes (0-12)
# Class mapping: 0=Natural Veg, 1=Forest, 2=Corn, 3=Soybeans, 4=Wetlands,
#                5=Developed/Barren, 6=Open Water, 7=Winter Wheat, 8=Alfalfa,
#                9=Fallow/Idle, 10=Cotton, 11=Sorghum, 12=Other
COLORS = np.array([
    [204, 190, 162],   # 0: Natural Vegetation (tan/hay color from CDL)
    [104, 171, 102],   # 1: Forest (forest green from CDL #141)
    [255, 212,   0],   # 2: Corn (CDL gold #ffd400)
    [ 38, 115,   0],   # 3: Soybeans (CDL dark green #267300)
    [ 75, 144, 226],   # 4: Wetlands (CDL wetland blue)
    [204, 190, 162],   # 5: Developed/Barren (grey-tan from CDL #131)
    [  0, 168, 226],   # 6: Open Water (CDL blue #00a9e6)
    [168, 112,   0],   # 7: Winter Wheat (CDL medium brown #a87000)
    [255, 168, 227],   # 8: Alfalfa (CDL pink #ffa8e3)
    [204, 190, 162],   # 9: Fallow/Idle (CDL tan #61)
    [255,  38,  38],   # 10: Cotton (CDL red #ff2626)
    [255, 158,  15],   # 11: Sorghum (CDL orange #ff9e0f)
    [204, 204, 204],   # 12: Other (grey)
], dtype=np.uint8)

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other"
]

def labels_to_rgb(array, colors):
    H, W = array.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for class_id in range(len(colors)):
        mask = array == class_id
        rgb[mask] = colors[class_id]
    return rgb

def stitch_gt(gt_chip_dir, H, W, chip_size=224):
    print(f"Stitching GT masks from: {gt_chip_dir}")
    mask_files = sorted(glob.glob(os.path.join(gt_chip_dir, 'chip_*.mask.tif')))
    print(f"Found {len(mask_files)} GT chips\n")

    gt_full = np.zeros((H, W), dtype=np.int32)

    for mask_path in tqdm(mask_files, desc="Stitching GT"):
        fname = os.path.basename(mask_path)
        parts = fname.replace('.mask.tif', '').split('_')
        row, col = int(parts[1]), int(parts[2])

        with rasterio.open(mask_path) as src:
            chip = src.read(1).astype(np.int32)  # values 1-13, 0=NoData

        gt_full[row:row+chip_size, col:col+chip_size] = chip

    # Shift 1-13 → 0-12, NoData(0) stays as 0 after clip
    gt_shifted = gt_full - 1          # 0-12, NoData → -1
    gt_clipped = np.clip(gt_shifted, 0, 12).astype(np.uint8)

    return gt_clipped

def colorize_and_save(label_array, output_path, profile, title=""):
    rgb = labels_to_rgb(label_array, COLORS)

    out_profile = profile.copy()
    out_profile.update({
        'count': 3,
        'dtype': 'uint8',
        'compress': 'lzw',
        'photometric': 'RGB',
        'nodata': None
    })

    with rasterio.open(output_path, 'w', **out_profile) as dst:
        dst.write(rgb[:, :, 0], 1)
        dst.write(rgb[:, :, 1], 2)
        dst.write(rgb[:, :, 2], 3)

    print(f"Saved {title}: {output_path}")

def main():
    base_dir    = '/bigdata/eldawylab/sdas050/MS_Research'
    gt_chip_dir = os.path.join(base_dir, 'data/multi_temporal_crop_segmentation/NWIA')
    stack_path  = os.path.join(base_dir, 'scripts/processed_stacks/NWIA/NWIA_multitemporal_stack.tif')

    prithvi_pred_path  = os.path.join(base_dir, 'predictions/prithvi_NWIA_corrected/NWIA_Prithvi_Prediction_Stitched.tif')
    spectral_pred_path = os.path.join(base_dir, 'predictions/spectralgpt_NWIA_stitched/NWIA_SpectralGPT_Prediction_Stitched.tif')

    out_dir = os.path.join(base_dir, 'visualizations_geotiff/NWIA_comparison')
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(stack_path) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
    print(f"Full map size: {H} x {W}\n")

    # ── 1. Stitch and colorize GT ─────────────────────────────────────────
    gt_full = stitch_gt(gt_chip_dir, H, W, chip_size=224)
    colorize_and_save(
        gt_full,
        os.path.join(out_dir, 'NWIA_GT.tif'),
        profile,
        title="GT"
    )

    # ── 2. Colorize Prithvi prediction (0-12) ────────────────────────────
    with rasterio.open(prithvi_pred_path) as src:
        prithvi_pred = src.read(1).astype(np.uint8)
    colorize_and_save(
        prithvi_pred,
        os.path.join(out_dir, 'NWIA_Prithvi_pred.tif'),
        profile,
        title="Prithvi Prediction"
    )

    # ── 3. Colorize SpectralGPT prediction (0-12) ────────────────────────
    with rasterio.open(spectral_pred_path) as src:
        spectral_pred = src.read(1).astype(np.uint8)
    colorize_and_save(
        spectral_pred,
        os.path.join(out_dir, 'NWIA_SpectralGPT_pred.tif'),
        profile,
        title="SpectralGPT Prediction"
    )

    print(f"\nAll outputs saved to: {out_dir}")
    print("Load all 3 GeoTIFFs in QGIS for side-by-side comparison!")

if __name__ == '__main__':
    main()