"""
Extract Labeled Sentinel-2 Patches Using Ground Truth Raster
================================================================

Your Setup:
-----------
1. Sentinel-2 6-band TIF files organized by date:
   CA_early_season_30m_2024/2024-03-11/*.tif
   
2. Ground truth TIF file with crop type labels:
   ground_truth_conus.tif (pixel values = crop class IDs)

Goal:
-----
Extract 224x224 patches where each patch has a known crop type label
from the ground truth raster, then organize for Scale-MAE KNN evaluation.

"""

import os
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds, reproject, Resampling
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from tqdm import tqdm


# ========================================
# STEP 1: Extract Co-Located Patches
# ========================================

def extract_labeled_patches(
    sentinel_tif_path,
    groundtruth_tif_path,
    output_dir,
    patch_size=224,
    stride=224,
    min_valid_pixels=0.9,
    min_label_consensus=0.7,
    exclude_classes=None
):
    """
    Extract Sentinel-2 patches with labels from ground truth raster
    
    Args:
        sentinel_tif_path: Path to Sentinel-2 6-band TIF
        groundtruth_tif_path: Path to ground truth crop type TIF
        output_dir: Where to save labeled patches
        patch_size: Patch size in pixels (224 for ViT)
        stride: Stride for sliding window
        min_valid_pixels: Min % of valid (non-zero) pixels in Sentinel patch
        min_label_consensus: Min % of dominant crop type in GT patch (0.7 = 70% same class)
        exclude_classes: List of class IDs to exclude (e.g., [0] for background/no-data)
    
    Returns:
        List of (patch_path, class_label) tuples
    """
    
    if exclude_classes is None:
        exclude_classes = [0]  # Typically 0 = background/no-data
    
    os.makedirs(output_dir, exist_ok=True)
    extracted_samples = []
    
    print(f"\nProcessing: {os.path.basename(sentinel_tif_path)}")
    
    # Open both files
    with rasterio.open(sentinel_tif_path) as s2_src, \
         rasterio.open(groundtruth_tif_path) as gt_src:
        
        # Check if they overlap
        s2_bounds = s2_src.bounds
        gt_bounds = gt_src.bounds
        
        print(f"Sentinel-2 bounds: {s2_bounds}")
        print(f"Ground truth bounds: {gt_bounds}")
        
        # Get Sentinel-2 data
        s2_img = s2_src.read()  # Shape: (6, H, W)
        height, width = s2_img.shape[1], s2_img.shape[2]
        
        print(f"Sentinel-2 shape: {s2_img.shape}")
        print(f"Ground truth shape: ({gt_src.count}, {gt_src.height}, {gt_src.width})")
        
        # Sliding window over Sentinel-2 image
        patch_count = 0
        class_distribution = defaultdict(int)
        
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                
                # Extract Sentinel-2 patch
                s2_patch = s2_img[:, i:i+patch_size, j:j+patch_size]
                
                # Check if Sentinel-2 patch has enough valid data
                valid_ratio = np.count_nonzero(s2_patch) / s2_patch.size
                if valid_ratio < min_valid_pixels:
                    continue
                
                # Get corresponding window in Sentinel-2 georeferenced space
                window = Window(j, i, patch_size, patch_size)
                s2_transform = s2_src.window_transform(window)
                s2_patch_bounds = rasterio.windows.bounds(window, s2_src.transform)
                
                # Read ground truth for this spatial extent
                try:
                    # Reproject ground truth bounds to match GT CRS if needed
                    if s2_src.crs != gt_src.crs:
                        gt_patch_bounds = transform_bounds(
                            s2_src.crs, 
                            gt_src.crs, 
                            *s2_patch_bounds
                        )
                    else:
                        gt_patch_bounds = s2_patch_bounds
                    
                    # Get window in ground truth image
                    gt_window = rasterio.windows.from_bounds(
                        *gt_patch_bounds, 
                        transform=gt_src.transform
                    )
                    
                    # Read ground truth patch
                    gt_patch = gt_src.read(1, window=gt_window)
                    
                    # Resize GT patch to match Sentinel-2 patch size if needed
                    if gt_patch.shape != (patch_size, patch_size):
                        gt_patch_resampled = np.zeros((patch_size, patch_size), dtype=gt_patch.dtype)
                        reproject(
                            source=gt_patch,
                            destination=gt_patch_resampled,
                            src_transform=gt_src.window_transform(gt_window),
                            src_crs=gt_src.crs,
                            dst_transform=s2_transform,
                            dst_crs=s2_src.crs,
                            resampling=Resampling.nearest
                        )
                        gt_patch = gt_patch_resampled
                    
                    # Determine dominant crop class in GT patch
                    gt_patch_flat = gt_patch.flatten()
                    
                    # Remove excluded classes
                    valid_labels = gt_patch_flat[~np.isin(gt_patch_flat, exclude_classes)]
                    
                    if len(valid_labels) == 0:
                        continue
                    
                    # Find most common class
                    class_counts = Counter(valid_labels)
                    dominant_class, dominant_count = class_counts.most_common(1)[0]
                    
                    # Check consensus (e.g., 70% of pixels must be same class)
                    consensus = dominant_count / len(valid_labels)
                    
                    if consensus < min_label_consensus:
                        continue  # Mixed classes, skip
                    
                    # Save patch with label
                    patch_filename = f"patch_{patch_count:06d}_class{dominant_class}.tif"
                    patch_path = os.path.join(output_dir, patch_filename)
                    
                    # Save Sentinel-2 patch
                    profile = s2_src.profile.copy()
                    profile.update({
                        'height': patch_size,
                        'width': patch_size,
                        'count': 6,
                        'transform': s2_transform
                    })
                    
                    with rasterio.open(patch_path, 'w', **profile) as dst:
                        dst.write(s2_patch)
                    
                    extracted_samples.append((patch_path, int(dominant_class)))
                    class_distribution[int(dominant_class)] += 1
                    patch_count += 1
                    
                except Exception as e:
                    # Skip patches that fail (e.g., outside GT bounds)
                    continue
        
        print(f"\nExtracted {patch_count} labeled patches")
        print(f"Class distribution: {dict(class_distribution)}")
        
    return extracted_samples


# ========================================
# STEP 2: Process All Sentinel-2 Files
# ========================================

def process_all_sentinel2_files(
    sentinel_base_dir,
    groundtruth_tif_path,
    output_base_dir,
    date_folders=None,
    **extraction_kwargs
):
    """
    Process all Sentinel-2 TIF files across multiple dates
    
    Args:
        sentinel_base_dir: Base directory (e.g., CA_early_season_30m_2024/)
        groundtruth_tif_path: Path to ground truth TIF
        output_base_dir: Base output directory
        date_folders: List of date folders to process (None = all)
        **extraction_kwargs: Additional args for extract_labeled_patches
    """
    
    all_samples = []
    
    # Find all date folders
    if date_folders is None:
        date_folders = sorted([
            d for d in os.listdir(sentinel_base_dir)
            if os.path.isdir(os.path.join(sentinel_base_dir, d))
        ])
    
    print(f"Found {len(date_folders)} date folders")
    
    for date_folder in tqdm(date_folders, desc="Processing dates"):
        date_path = os.path.join(sentinel_base_dir, date_folder)
        
        # Find TIF files in this date folder
        tif_files = [f for f in os.listdir(date_path) if f.endswith('.tif') and not f.endswith('.aux.xml')]
        
        for tif_file in tif_files:
            sentinel_tif_path = os.path.join(date_path, tif_file)
            output_dir = os.path.join(output_base_dir, "patches", date_folder)
            
            # Extract patches
            samples = extract_labeled_patches(
                sentinel_tif_path,
                groundtruth_tif_path,
                output_dir,
                **extraction_kwargs
            )
            
            all_samples.extend(samples)
    
    print(f"\n{'='*60}")
    print(f"TOTAL: Extracted {len(all_samples)} labeled patches")
    print(f"{'='*60}")
    
    return all_samples


# ========================================
# STEP 3: Organize into Train/Val Splits
# ========================================

def organize_train_val_splits(
    all_samples,
    output_dir,
    train_ratio=0.8,
    stratify=True
):
    """
    Organize extracted patches into train/val splits
    
    Args:
        all_samples: List of (patch_path, class_label) tuples
        output_dir: Output directory
        train_ratio: Fraction for training
        stratify: Maintain class balance in splits
    """
    from sklearn.model_selection import train_test_split
    import shutil
    
    # Separate paths and labels
    patch_paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]
    
    # Create stratified split
    if stratify and len(set(labels)) > 1:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            patch_paths, labels,
            train_size=train_ratio,
            stratify=labels,
            random_state=42
        )
    else:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            patch_paths, labels,
            train_size=train_ratio,
            random_state=42
        )
    
    print(f"\nTrain samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    
    # Create directory structure
    for split_name, paths, split_labels in [
        ('train', train_paths, train_labels),
        ('val', val_paths, val_labels)
    ]:
        for class_id in set(split_labels):
            class_dir = os.path.join(output_dir, split_name, f"class_{class_id}")
            os.makedirs(class_dir, exist_ok=True)
        
        # Copy files
        for patch_path, label in zip(paths, split_labels):
            dst_path = os.path.join(
                output_dir,
                split_name,
                f"class_{label}",
                os.path.basename(patch_path)
            )
            shutil.copy2(patch_path, dst_path)
    
    # Also create split text files (Scale-MAE format)
    create_split_files(output_dir, output_dir)
    
    print(f"\nOrganized into: {output_dir}")


def create_split_files(data_dir, output_dir):
    """Create train-sentinel2.txt and val-sentinel2.txt"""
    
    for split in ['train', 'val']:
        split_file = os.path.join(output_dir, f"{split}-sentinel2.txt")
        
        with open(split_file, 'w') as f:
            split_dir = os.path.join(data_dir, split)
            
            if not os.path.exists(split_dir):
                continue
            
            for class_folder in sorted(os.listdir(split_dir)):
                class_path = os.path.join(split_dir, class_folder)
                if not os.path.isdir(class_path):
                    continue
                
                # Extract class number
                class_id = int(class_folder.split('_')[1])
                
                for filename in sorted(os.listdir(class_path)):
                    if filename.endswith('.tif'):
                        rel_path = f"./{class_folder}/{filename}"
                        f.write(f"{rel_path} {class_id}\n")
        
        print(f"Created {split_file}")


# ========================================
# STEP 4: Verify Ground Truth Classes
# ========================================

def inspect_ground_truth(groundtruth_tif_path):
    """Inspect ground truth raster to see class distribution"""
    
    print(f"\n{'='*60}")
    print("GROUND TRUTH INSPECTION")
    print(f"{'='*60}")
    
    with rasterio.open(groundtruth_tif_path) as src:
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Shape: ({src.count}, {src.height}, {src.width})")
        print(f"Data type: {src.dtypes[0]}")
        
        # Sample the data
        gt_data = src.read(1)
        unique_classes = np.unique(gt_data)
        
        print(f"\nUnique classes found: {len(unique_classes)}")
        print(f"Class IDs: {unique_classes[:20]}...")  # Show first 20
        
        # Class distribution
        class_counts = Counter(gt_data.flatten())
        print(f"\nTop 10 classes by pixel count:")
        for class_id, count in class_counts.most_common(10):
            percentage = 100 * count / gt_data.size
            print(f"  Class {class_id}: {count:,} pixels ({percentage:.2f}%)")
    
    return unique_classes


# ========================================
# MAIN WORKFLOW
# ========================================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    
    SENTINEL_BASE_DIR = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/CA_early_season_30m_2024"
    GROUNDTRUTH_TIF = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/2023_30m_cdls.tif"  # Your ground truth file
    OUTPUT_DIR = "/bigdata/eldawylab/sdas050/MS_Research/scale-mae/mae/splits/sentinel2_labeled"
    
    # Extraction parameters
    PATCH_SIZE = 224
    STRIDE = 224  # No overlap; use 112 for 50% overlap
    MIN_VALID_PIXELS = 0.9  # 90% of patch must have valid data
    MIN_LABEL_CONSENSUS = 0.7  # 70% of patch must be same class
    EXCLUDE_CLASSES = [0]  # Exclude background/no-data class
    
    # Optional: Process only specific dates (None = all)
    DATE_FOLDERS = None  # ['2024-03-11', '2024-03-21']  # Or None for all
    
    
    # ===== STEP 1: Inspect Ground Truth =====
    # print("\nSTEP 1: Inspecting ground truth raster...")
    # inspect_ground_truth(GROUNDTRUTH_TIF)
    
    
    # ===== STEP 2: Extract Labeled Patches =====
    print(f"\nSTEP 2: Extracting labeled patches...")
    all_samples = process_all_sentinel2_files(
        sentinel_base_dir=SENTINEL_BASE_DIR,
        groundtruth_tif_path=GROUNDTRUTH_TIF,
        output_base_dir=OUTPUT_DIR,
        date_folders=DATE_FOLDERS,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_pixels=MIN_VALID_PIXELS,
        min_label_consensus=MIN_LABEL_CONSENSUS,
        exclude_classes=EXCLUDE_CLASSES
    )
    
    # Save sample list
    sample_list_path = os.path.join(OUTPUT_DIR, "all_samples.json")
    with open(sample_list_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"\nSaved sample list to: {sample_list_path}")
    
    
    # ===== STEP 3: Organize Train/Val Splits =====
    print(f"\nSTEP 3: Organizing into train/val splits...")
    organize_train_val_splits(
        all_samples=all_samples,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        stratify=True
    )
    
    
    # ===== DONE! =====
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nFinal structure:")
    print(f"{OUTPUT_DIR}/")
    print(f"├── train/")
    print(f"│   ├── class_0/")
    print(f"│   ├── class_1/")
    print(f"│   └── ...")
    print(f"├── val/")
    print(f"│   ├── class_0/")
    print(f"│   ├── class_1/")
    print(f"│   └── ...")
    print(f"├── train-sentinel2.txt")
    print(f"└── val-sentinel2.txt")
    print(f"\nNext step: Run Scale-MAE KNN evaluation!")