import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

def compute_minmax_cd(chips_dir, suffix="_t1"):
    chips_dir = Path(chips_dir)
    chip_files = sorted(chips_dir.glob(f"images/*{suffix}.tif"))
    
    print(f"Found {len(chip_files)} {suffix} chips")
    if len(chip_files) == 0:
        return None, None

    with rasterio.open(chip_files[0]) as src:
        n_bands = src.count

    print(f"Bands: {n_bands}")

    global_min = np.full(n_bands, np.inf, dtype=np.float32)
    global_max = np.full(n_bands, -np.inf, dtype=np.float32)

    for chip_path in tqdm(chip_files, desc=f"Computing {suffix}"):
        with rasterio.open(chip_path) as src:
            chip = src.read().astype(np.float32)  # (6, H, W)

        valid_mask = (chip != -9999) & np.isfinite(chip)

        for b in range(n_bands):
            valid = chip[b][valid_mask[b]]
            if len(valid) > 0:
                global_min[b] = min(global_min[b], valid.min())
                global_max[b] = max(global_max[b], valid.max())

    band_names = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
    print(f"\n{suffix} stats:")
    for i in range(n_bands):
        print(f"  {band_names[i]}: min={global_min[i]:.2f}, max={global_max[i]:.2f}")

    return global_min.tolist(), global_max.tolist()


if __name__ == "__main__":
    # Run on CentIA ONLY (training location)
    CHIPS_DIR = "/bigdata/eldawylab/sdas050/MS_Research/change_detection_chips/spectralgpt/CentIA"

    t1_min, t1_max = compute_minmax_cd(CHIPS_DIR, suffix="_t1")
    t2_min, t2_max = compute_minmax_cd(CHIPS_DIR, suffix="_t2")