import os
import numpy as np
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
])

def brightness_modulate(prediction, ground_truth, colors):
    """
    prediction:    (H, W) values 0-12
    ground_truth:  (H, W) values 1-13, 0=NoData
    """
    H, W = prediction.shape
    rgb_image = np.zeros((H, W, 3), dtype=np.uint8)

    # Shift GT from 1-13 to 0-12, NoData(0) becomes -1
    gt_shifted = ground_truth.astype(np.int32) - 1  # -1 = NoData, 0-12 = valid

    # Valid pixel mask (exclude NoData)
    valid = gt_shifted >= 0

    # Correct prediction mask (only on valid pixels)
    correct = (prediction == gt_shifted) & valid

    # Color each class based on GT
    for class_id in range(len(colors)):
        mask = gt_shifted == class_id
        rgb_image[mask] = colors[class_id]

    # Brightness: correct=full, incorrect=30%, nodata=black(0)
    brightness = np.where(correct, 1.0, np.where(valid, 0.3, 0.0))

    for i in range(3):
        rgb_image[:, :, i] = (rgb_image[:, :, i] * brightness).astype(np.uint8)

    acc = correct.sum() / valid.sum() * 100 if valid.sum() > 0 else 0.0

    return rgb_image, correct, valid, acc


def main():
    base_dir = '/bigdata/eldawylab/sdas050/MS_Research'
    gt_dir   = os.path.join(base_dir, 'SpectralGPT_chips_multitemporal/NWIA/ImageSets')
    pred_dir = os.path.join(base_dir, 'predictions/spectralgpt_NWIA')
    output_dir = os.path.join(base_dir, 'visualizations_geotiff/spectralgpt_NWIA_128x128')

    os.makedirs(output_dir, exist_ok=True)

    # NWIA chips: NWIA_chip_320_64_mask.tif
    mask_files = sorted(glob.glob(os.path.join(gt_dir, 'NWIA_chip_*_mask.tif')))

    print(f"SpectralGPT GeoTIFF Visualization (128×128 chips)")
    print(f"Found {len(mask_files)} chips\n")

    if len(mask_files) == 0:
        print("ERROR: No mask files found!")
        return

    results = []
    missing_pred = 0
    missing_chip = 0

    for mask_path in mask_files:
        # NWIA_chip_320_64_mask.tif -> NWIA_chip_320_64
        chip_name = os.path.basename(mask_path).replace('_mask.tif', '')
        pred_path = os.path.join(pred_dir, f'{chip_name}.npy')
        chip_path = os.path.join(gt_dir, f'{chip_name}.tif')   # no _merged suffix

        if not os.path.exists(pred_path):
            print(f"Prediction not found: {chip_name}")
            missing_pred += 1
            continue

        if not os.path.exists(chip_path):
            print(f"Image chip not found: {chip_name}")
            missing_chip += 1
            continue

        # Load GT mask (values 1-13, 0=NoData)
        with rasterio.open(mask_path) as src:
            gt = src.read(1)

        # Load prediction (values 0-12)
        pred = np.load(pred_path)

        # Create visualization
        rgb_vis, correct, valid, acc = brightness_modulate(pred, gt, COLORS)

        # Read spatial metadata from image chip
        with rasterio.open(chip_path) as src:
            profile = src.profile.copy()

        profile.update({
            'driver': 'GTiff',
            'dtype': 'uint8',
            'count': 3,
            'compress': 'lzw',
            'photometric': 'RGB',
            'nodata': None
        })

        # Save GeoTIFF
        output_path = os.path.join(output_dir, f'{chip_name}_visualization.tif')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(rgb_vis[:, :, 0], 1)
            dst.write(rgb_vis[:, :, 1], 2)
            dst.write(rgb_vis[:, :, 2], 3)
            dst.update_tags(
                description=f'SpectralGPT visualization - Accuracy: {acc:.2f}%',
                valid_pixels=str(int(valid.sum())),
                correct_pixels=str(int(correct.sum()))
            )

        results.append(acc)
        print(f"{chip_name}: {acc:.2f}% (valid px: {valid.sum()}) → saved")

    # Summary
    print(f"\n{'='*60}")
    if results:
        print(f"Processed:        {len(results)} chips")
        print(f"Missing preds:    {missing_pred}")
        print(f"Missing chips:    {missing_chip}")
        print(f"Average Accuracy: {np.mean(results):.2f}%")
        print(f"Min Accuracy:     {np.min(results):.2f}%")
        print(f"Max Accuracy:     {np.max(results):.2f}%")
        print(f"GeoTIFFs saved to: {output_dir}/")
    else:
        print("No chips processed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()