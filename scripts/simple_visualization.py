# SpectralGPT
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import rasterio
# import glob

# COLORS = np.array([
#     [34, 139, 34],    # 0: Natural Vegetation
#     [0, 100, 0],      # 1: Forest
#     [255, 215, 0],    # 2: Corn
#     [139, 69, 19],    # 3: Soybeans
#     [0, 191, 255],    # 4: Wetlands
#     [128, 128, 128],  # 5: Developed/Barren
#     [0, 0, 255],      # 6: Open Water
#     [255, 165, 0],    # 7: Winter Wheat
#     [147, 112, 219],  # 8: Alfalfa
#     [210, 180, 140],  # 9: Fallow/Idle
#     [255, 192, 203],  # 10: Cotton
#     [165, 42, 42],    # 11: Sorghum
#     [192, 192, 192]   # 12: Other
# ]) / 255.0

# CLASS_NAMES = [
#     "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
#     "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
#     "Fallow/Idle", "Cotton", "Sorghum", "Other"
# ]

# def class_to_rgb(array, colors):
#     H, W = array.shape
#     rgb = np.zeros((H, W, 3))
#     for class_id in range(len(colors)):
#         mask = array == class_id
#         rgb[mask] = colors[class_id]
#     return rgb

# def main():
#     base_dir   = '/bigdata/eldawylab/sdas050/MS_Research'
#     gt_dir     = os.path.join(base_dir, 'SpectralGPT_chips_multitemporal/NWIA/ImageSets')
#     sgpt_dir   = os.path.join(base_dir, 'predictions/spectralgpt_NWIA')

#     # Two separate output dirs — one for GT tifs, one for prediction tifs
#     gt_output_dir   = os.path.join(base_dir, 'visualizations_geotiff/spectralgpt_NWIA_GT')
#     pred_output_dir = os.path.join(base_dir, 'visualizations_geotiff/spectralgpt_NWIA_pred')
#     os.makedirs(gt_output_dir, exist_ok=True)
#     os.makedirs(pred_output_dir, exist_ok=True)

#     pred_files = sorted(glob.glob(os.path.join(sgpt_dir, 'NWIA_chip_*.npy')))
#     print(f"Found {len(pred_files)} SpectralGPT prediction files\n")

#     for pred_path in pred_files:
#         chip_name = os.path.basename(pred_path).replace('.npy', '')
#         mask_path = os.path.join(gt_dir, f'{chip_name}_mask.tif')
#         chip_path = os.path.join(gt_dir, f'{chip_name}.tif')

#         if not os.path.exists(mask_path):
#             print(f"⚠️  Mask not found: {chip_name}")
#             continue

#         # Load GT (1-13, 0=NoData) → shift to 0-12
#         with rasterio.open(mask_path) as src:
#             gt_raw  = src.read(1)
#             profile = src.profile.copy()

#         gt = gt_raw.astype(np.int32) - 1   # 0-12, NoData → -1

#         # Load SpectralGPT prediction (0-12)
#         sgpt_pred = np.load(pred_path)

#         # Convert to RGB
#         gt_rgb   = class_to_rgb(np.clip(gt, 0, 12), COLORS)
#         sgpt_rgb = class_to_rgb(sgpt_pred, COLORS)

#         # NoData pixels → black
#         nodata = gt < 0
#         gt_rgb[nodata] = 0

#         # Convert to uint8
#         gt_uint8   = (gt_rgb   * 255).astype(np.uint8)
#         sgpt_uint8 = (sgpt_rgb * 255).astype(np.uint8)

#         # Update profile for RGB GeoTIFF
#         profile.update({
#             'driver': 'GTiff',
#             'dtype': 'uint8',
#             'count': 3,
#             'compress': 'lzw',
#             'photometric': 'RGB',
#             'nodata': None
#         })

#         # Save GT GeoTIFF
#         gt_out = os.path.join(gt_output_dir, f'{chip_name}_GT.tif')
#         with rasterio.open(gt_out, 'w', **profile) as dst:
#             dst.write(gt_uint8[:, :, 0], 1)
#             dst.write(gt_uint8[:, :, 1], 2)
#             dst.write(gt_uint8[:, :, 2], 3)

#         # Save Prediction GeoTIFF
#         pred_out = os.path.join(pred_output_dir, f'{chip_name}_pred.tif')
#         with rasterio.open(pred_out, 'w', **profile) as dst:
#             dst.write(sgpt_uint8[:, :, 0], 1)
#             dst.write(sgpt_uint8[:, :, 1], 2)
#             dst.write(sgpt_uint8[:, :, 2], 3)

#         print(f"Saved: {chip_name}")

#     print(f"\nDone!")
#     print(f"GT GeoTIFFs:   {gt_output_dir}")
#     print(f"Pred GeoTIFFs: {pred_output_dir}")

# if __name__ == '__main__':
#     main()

# Prithvi
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import glob

COLORS = np.array([
    [34, 139, 34],    # 0: Natural Vegetation
    [0, 100, 0],      # 1: Forest
    [255, 215, 0],    # 2: Corn
    [139, 69, 19],    # 3: Soybeans
    [0, 191, 255],    # 4: Wetlands
    [128, 128, 128],  # 5: Developed/Barren
    [0, 0, 255],      # 6: Open Water
    [255, 165, 0],    # 7: Winter Wheat
    [147, 112, 219],  # 8: Alfalfa
    [210, 180, 140],  # 9: Fallow/Idle
    [255, 192, 203],  # 10: Cotton
    [165, 42, 42],    # 11: Sorghum
    [192, 192, 192]   # 12: Other
]) / 255.0

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other"
]

def class_to_rgb(array, colors):
    H, W = array.shape
    rgb = np.zeros((H, W, 3))
    for class_id in range(len(colors)):
        mask = array == class_id
        rgb[mask] = colors[class_id]
    return rgb

def main():
    base_dir      = '/bigdata/eldawylab/sdas050/MS_Research'
    gt_dir        = os.path.join(base_dir, 'data/multi_temporal_crop_segmentation/NWIA')
    prithvi_dir   = os.path.join(base_dir, 'predictions/prithvi_NWIA')

    gt_output_dir   = os.path.join(base_dir, 'visualizations_geotiff/prithvi_NWIA_GT')
    pred_output_dir = os.path.join(base_dir, 'visualizations_geotiff/prithvi_NWIA_pred')
    os.makedirs(gt_output_dir, exist_ok=True)
    os.makedirs(pred_output_dir, exist_ok=True)

    pred_files = sorted(glob.glob(os.path.join(prithvi_dir, 'chip_*.npy')))
    print(f"Found {len(pred_files)} Prithvi prediction files\n")

    for pred_path in pred_files:
        chip_name = os.path.basename(pred_path).replace('.npy', '')
        mask_path = os.path.join(gt_dir, f'{chip_name}.mask.tif')

        if not os.path.exists(mask_path):
            print(f"⚠️  Mask not found: {chip_name}")
            continue

        # Load GT (1-13, 0=NoData) → shift to 0-12
        with rasterio.open(mask_path) as src:
            gt_raw  = src.read(1)
            profile = src.profile.copy()

        gt = gt_raw.astype(np.int32) - 1   # 0-12, NoData → -1

        # Load Prithvi prediction (0-12)
        prithvi_pred = np.load(pred_path)

        # Convert to RGB
        gt_rgb   = class_to_rgb(np.clip(gt, 0, 12), COLORS)
        pred_rgb = class_to_rgb(prithvi_pred, COLORS)

        # NoData pixels → black
        nodata = gt < 0
        gt_rgb[nodata] = 0

        # Convert to uint8
        gt_uint8   = (gt_rgb   * 255).astype(np.uint8)
        pred_uint8 = (pred_rgb * 255).astype(np.uint8)

        # Update profile for RGB GeoTIFF
        profile.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'compress': 'lzw',
            'photometric': 'RGB',
            'nodata': None
        })

        # Save GT GeoTIFF
        gt_out = os.path.join(gt_output_dir, f'{chip_name}_GT.tif')
        with rasterio.open(gt_out, 'w', **profile) as dst:
            dst.write(gt_uint8[:, :, 0], 1)
            dst.write(gt_uint8[:, :, 1], 2)
            dst.write(gt_uint8[:, :, 2], 3)

        # Save Prediction GeoTIFF
        pred_out = os.path.join(pred_output_dir, f'{chip_name}_pred.tif')
        with rasterio.open(pred_out, 'w', **profile) as dst:
            dst.write(pred_uint8[:, :, 0], 1)
            dst.write(pred_uint8[:, :, 1], 2)
            dst.write(pred_uint8[:, :, 2], 3)

        print(f"Saved: {chip_name}")

    print(f"\nDone!")
    print(f"GT GeoTIFFs:   {gt_output_dir}")
    print(f"Pred GeoTIFFs: {pred_output_dir}")

if __name__ == '__main__':
    main()