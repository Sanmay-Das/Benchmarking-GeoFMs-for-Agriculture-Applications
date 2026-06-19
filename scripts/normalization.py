import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


def compute_stats_fast(chips_dir):
    """
    Vectorized across ALL bands at once — no Python band loop.
    Uses Chan's parallel algorithm for constant memory.
    """
    chips_dir = Path(chips_dir)

    # Same glob + filter as original
    all_tif_files = sorted(chips_dir.glob("chip_*.tif"))
    chip_files    = [f for f in all_tif_files if '.mask' not in f.name]

    print(f"Found {len(chip_files)} image chips")
    if len(chip_files) == 0:
        print("No chips found!")
        return None, None

    with rasterio.open(chip_files[0]) as src:
        n_bands       = src.count
        height, width = src.height, src.width

    print(f"Chip specs: {n_bands} bands, {height}x{width} pixels")
    print(f"Processing {len(chip_files):,} chips...\n")

    # Running stats — all bands simultaneously
    count = np.zeros(n_bands, dtype=np.float64)
    mean  = np.zeros(n_bands, dtype=np.float64)
    M2    = np.zeros(n_bands, dtype=np.float64)

    for chip_path in tqdm(chip_files, desc="Computing stats"):
        with rasterio.open(chip_path) as src:
            if src.count != n_bands:
                continue
            chip = src.read().astype(np.float64)  # (n_bands, H, W)

        # VECTORIZED: valid mask across all bands at once
        # chip shape: (n_bands, H, W)
        valid_mask = (chip != -9999) & (chip != 0)  # (n_bands, H, W)

        for band_idx in range(n_bands):
            valid_pixels = chip[band_idx][valid_mask[band_idx]]
            if len(valid_pixels) == 0:
                continue

            n_b    = len(valid_pixels)
            mean_b = valid_pixels.mean()
            var_b  = valid_pixels.var()
            n_a    = count[band_idx]

            if n_a == 0:
                count[band_idx] = n_b
                mean[band_idx]  = mean_b
                M2[band_idx]    = var_b * n_b
            else:
                n_combined      = n_a + n_b
                delta           = mean_b - mean[band_idx]
                mean_combined   = (n_a * mean[band_idx] + n_b * mean_b) / n_combined
                M2_combined     = (M2[band_idx] +
                                   var_b * n_b +
                                   delta**2 * n_a * n_b / n_combined)
                count[band_idx] = n_combined
                mean[band_idx]  = mean_combined
                M2[band_idx]    = M2_combined

    std = np.sqrt(M2 / count)

    # Print results
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Chips processed: {len(chip_files):,}")
    print(f"Bands: {n_bands}\n")

    print("MEAN = np.array([")
    for i, m in enumerate(mean):
        print(f"    {m:.8f},  # Band {i}")
    print("])\n")

    print("STD = np.array([")
    for i, s in enumerate(std):
        print(f"    {s:.8f},  # Band {i}")
    print("])\n")

    if n_bands % 6 == 0:
        n_timesteps = n_bands // 6
        band_names  = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
        print("="*70)
        print(f"PER-TIMESTEP ({n_timesteps} timesteps x 6 bands):")
        print("="*70)
        for t in range(n_timesteps):
            print(f"\nTimestep {t}:")
            start_idx = t * 6
            for i in range(6):
                idx = start_idx + i
                print(f"  {band_names[i]}: mean={mean[idx]:>8.2f}, std={std[idx]:>8.2f}")

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    CHIPS_DIR = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/CentIA"

    print("="*70)
    print("FAST MEAN/STD COMPUTATION")
    print("="*70)
    print(f"Directory: {CHIPS_DIR}\n")

    mean, std = compute_stats_fast(CHIPS_DIR)

    if mean is not None:
        print("\nDone!")