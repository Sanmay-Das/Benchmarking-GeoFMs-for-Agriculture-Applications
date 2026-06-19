"""
Compute mean and std statistics for crop classification chips
"""
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def compute_statistics(csv_path, num_samples=None):
    """
    Compute per-band mean and std from crop classification chips
    
    Args:
        csv_path: Path to CSV file (train.csv)
        num_samples: Number of chips to sample (None = use all)
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Total chips in dataset: {len(df)}")
    
    # Sample chips OR use all
    if num_samples is None:
        print(f"Using ALL {len(df)} chips for statistics...")
        sampled_df = df
    else:
        print(f"Sampling {min(num_samples, len(df))} chips for statistics...")
        if len(df) > num_samples:
            sampled_df = df.sample(n=num_samples, random_state=42)
        else:
            sampled_df = df
    
    # Collect pixel values per band
    all_pixels = [[] for _ in range(10)]  # 10 bands
    
    print("\nLoading chips...")
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        data_root = Path("/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data")
        img_path = data_root / row["image_path"]
        
        try:
            with rasterio.open(img_path) as src:
                img = src.read()  # (10, 96, 96)
            
            # Skip if has NoData
            if np.any(img == -9999):
                continue
            
            # Collect valid pixels for each band
            for band_idx in range(10):
                band_data = img[band_idx].flatten()
                # Filter extreme outliers
                valid_pixels = band_data[(band_data > 0) & (band_data < 10000)]
                all_pixels[band_idx].extend(valid_pixels)
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    # Compute statistics
    print("\nComputing statistics...")
    means = []
    stds = []
    
    band_names = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']
    
    for band_idx in range(10):
        if len(all_pixels[band_idx]) == 0:
            print(f"Warning: No valid pixels for band {band_idx}")
            means.append(0.0)
            stds.append(1.0)
        else:
            band_pixels = np.array(all_pixels[band_idx])
            mean_val = float(np.mean(band_pixels))
            std_val = float(np.std(band_pixels))
            means.append(mean_val)
            stds.append(std_val)
    
    # Print results
    print("\n" + "="*70)
    print("COMPUTED STATISTICS FOR CROP CLASSIFICATION")
    print("="*70)
    
    print("\nPer-band statistics:")
    for i, (name, mean, std) in enumerate(zip(band_names, means, stds)):
        print(f"  Band {i:2d} ({name:4s}): mean={mean:10.2f}, std={std:8.2f}")
    
    print("\n" + "="*70)
    print("COPY THESE TO SentinelIndividualImageDataset IN datasets.py:")
    print("="*70)
    print(f"\nmean = {means}")
    print(f"std = {stds}")
    print("\n")
    
    return means, stds


if __name__ == "__main__":
    # Path to your training CSV
    csv_path = "/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/train.csv"
    
    print("Computing statistics from training data...")
    # Use ALL chips (no sampling)
    means, stds = compute_statistics(csv_path, num_samples=None)
    
    print("\nDone! Now:")
    print("1. Copy the mean and std values above")
    print("2. Replace the mean and std in SentinelIndividualImageDataset class")
    print("3. These will be used for your 10-band crop classification")