
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

def compute_minmax_for_spectralgpt(chips_dir):
    """
    Calculate GLOBAL MIN/MAX for SpectralGPT [0, 1] normalization
    
    According to SpectralGPT paper Section 2.8:
    "normalize the spectral images band by band, scaling their 
     values to a standardized range of 0 to 1"
    """
    chips_dir = Path(chips_dir)
    
    # Get image chips (exclude masks) - UPDATED NAMING CONVENTION
    all_tif_files = sorted(chips_dir.glob("NorthMN_chip_*.tif"))
    chip_files = [f for f in all_tif_files if 'mask' not in f.name.lower()]
    
    print(f"Found {len(chip_files)} image chips")
    
    if len(chip_files) == 0:
        print("No chips found!")
        return None, None
    
    # Auto-detect bands
    with rasterio.open(chip_files[0]) as src:
        n_bands = src.count
        height, width = src.height, src.width
    
    print(f"Chip specs: {n_bands} bands, {height}x{width} pixels")
    print(f"Processing {len(chip_files):,} chips...\n")
    
    # Initialize min/max trackers
    global_min = np.full(n_bands, np.inf, dtype=np.float32)
    global_max = np.full(n_bands, -np.inf, dtype=np.float32)
    
    # Process each chip
    for chip_path in tqdm(chip_files, desc="Computing min/max"):
        with rasterio.open(chip_path) as src:
            if src.count != n_bands:
                continue
            chip = src.read()  # (n_bands, H, W)
        
        # Update global min/max for each band
        for band_idx in range(n_bands):
            band_data = chip[band_idx]
            
            # Include zeros! Only exclude no-data (-9999)
            valid_mask = (band_data != -9999) & np.isfinite(band_data)
            valid_pixels = band_data[valid_mask]
            
            if len(valid_pixels) > 0:
                band_min = valid_pixels.min()
                band_max = valid_pixels.max()
                
                # Update global trackers
                if band_min < global_min[band_idx]:
                    global_min[band_idx] = band_min
                if band_max > global_max[band_idx]:
                    global_max[band_idx] = band_max
    
    # Calculate range
    global_range = global_max - global_min
    
    # Print results
    print("\n" + "="*70)
    print("SPECTRALGPT NORMALIZATION STATISTICS (MIN/MAX)")
    print("="*70)
    print(f"Chips processed: {len(chip_files):,}")
    print(f"Bands: {n_bands}\n")
    
    print("# FOR SPECTRALGPT: Use [0, 1] Min-Max Normalization\n")
    
    print("GLOBAL_MIN = np.array([")
    for i, val in enumerate(global_min):
        print(f"    {val:.2f},  # Band {i}")
    print("], dtype=np.float32)\n")
    
    print("GLOBAL_MAX = np.array([")
    for i, val in enumerate(global_max):
        print(f"    {val:.2f},  # Band {i}")
    print("], dtype=np.float32)\n")
    
    print("GLOBAL_RANGE = np.array([  # max - min")
    for i, val in enumerate(global_range):
        print(f"    {val:.2f},  # Band {i}")
    print("], dtype=np.float32)\n")
    
    # Per-timestep breakdown
    if n_bands == 18:
        band_names = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
        print("="*70)
        print("PER-TIMESTEP (3 timesteps x 6 bands):")
        print("="*70)
        for t in range(3):
            print(f"\nTimestep {t+1}:")
            start_idx = t * 6
            for i in range(6):
                idx = start_idx + i
                print(f"  {band_names[i]}: min={global_min[idx]:>8.2f}, "
                      f"max={global_max[idx]:>8.2f}, "
                      f"range={global_range[idx]:>8.2f}")
    
    # Show usage
    print("\n" + "="*70)
    print("USAGE IN SpectralGPTsegdataset.py:")
    print("="*70)
    print("""
class SegDataset(data.Dataset):
    def __init__(self, ...):
        # SpectralGPT [0, 1] normalization
        self.global_min = GLOBAL_MIN.reshape(18, 1, 1)
        self.global_max = GLOBAL_MAX.reshape(18, 1, 1)
        self.global_range = self.global_max - self.global_min
    
    def __getitem__(self, index):
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        
        # Normalize to [0, 1] (SpectralGPT method)
        img = (img - torch.from_numpy(self.global_min)) / torch.from_numpy(self.global_range)
        
        return img, target
""")
    
    return global_min.tolist(), global_max.tolist()


if __name__ == "__main__":
    CHIPS_DIR = f"{DATA_ROOT}/SpectralGPT_chips_multitemporal/NorthCentMN/ImageSets"
    
    print("="*70)
    print("SPECTRALGPT [0,1] NORMALIZATION")
    print("="*70)
    print(f"Directory: {CHIPS_DIR}\n")
    print("Method: Min-Max scaling to [0, 1] range")
    print("Formula: normalized = (x - min) / (max - min)\n")
    
    global_min, global_max = compute_minmax_for_spectralgpt(CHIPS_DIR)
    
    if global_min is not None:
        print("\nDone!")