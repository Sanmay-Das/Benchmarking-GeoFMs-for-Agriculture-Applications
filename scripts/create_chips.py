import os
import numpy as np
import rasterio
from rasterio.windows import Window
import logging
from pathlib import Path

# --- Configuration ---
INPUT_STACKED_PATH = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/stacked_CA_test_2024.tif"
OUTPUT_CHIPS_DIR = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/inference_chips_CA_2024"
CHIP_SIZE = 224  # 224x224 pixels as required by Prithvi
OVERLAP = 0  # No overlap for inference (set to 112 for 50% overlap if needed)

# Create output directory
os.makedirs(OUTPUT_CHIPS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_chips_for_inference(input_path, output_dir, chip_size=224, overlap=0):
    """
    Create 224x224 chips from stacked multi-temporal image for Prithvi inference.
    
    Args:
        input_path: Path to stacked image (18 bands)
        output_dir: Directory to save chips
        chip_size: Size of chips (default 224 for Prithvi)
        overlap: Overlap between chips in pixels (default 0)
    
    Returns:
        List of created chip paths
    """
    
    logging.info("="*80)
    logging.info("CREATING CHIPS FOR PRITHVI INFERENCE")
    logging.info("="*80)
    
    chip_paths = []
    
    with rasterio.open(input_path) as src:
        # Verify input
        logging.info(f"\nInput file: {input_path}")
        logging.info(f"  Bands: {src.count}")
        logging.info(f"  Dimensions: {src.width}w x {src.height}h")
        logging.info(f"  Dtype: {src.dtypes[0]}")
        logging.info(f"  NoData: {src.nodata}")
        logging.info(f"  CRS: {src.crs}")
        
        if src.count != 18:
            raise ValueError(f"Expected 18 bands, got {src.count}")
        
        if src.dtypes[0] != 'int16':
            logging.warning(f"Expected int16 dtype, got {src.dtypes[0]}")
        
        # Calculate stride (distance between chip origins)
        stride = chip_size - overlap
        
        # Calculate number of chips in each dimension
        n_chips_x = (src.width - overlap) // stride
        n_chips_y = (src.height - overlap) // stride
        
        # Handle remainder
        if (src.width - overlap) % stride != 0:
            n_chips_x += 1
        if (src.height - overlap) % stride != 0:
            n_chips_y += 1
        
        total_chips = n_chips_x * n_chips_y
        
        logging.info(f"\nChip configuration:")
        logging.info(f"  Chip size: {chip_size}x{chip_size}")
        logging.info(f"  Overlap: {overlap} pixels")
        logging.info(f"  Stride: {stride} pixels")
        logging.info(f"  Chips (X): {n_chips_x}")
        logging.info(f"  Chips (Y): {n_chips_y}")
        logging.info(f"  Total chips: {total_chips}")
        
        # Create profile for chips
        chip_profile = src.profile.copy()
        chip_profile.update({
            'width': chip_size,
            'height': chip_size,
            'count': 18,
            'dtype': 'int16',
            'nodata': -9999
        })
        
        logging.info(f"\nCreating chips...")
        logging.info(f"Output directory: {output_dir}")
        
        chip_count = 0
        skipped_count = 0
        
        # Create chips
        for row_idx in range(n_chips_y):
            for col_idx in range(n_chips_x):
                # Calculate window position
                col_off = col_idx * stride
                row_off = row_idx * stride
                
                # Adjust window size if at edge
                actual_width = min(chip_size, src.width - col_off)
                actual_height = min(chip_size, src.height - row_off)
                
                # Skip if chip would be too small
                # For inference, we might want to pad instead of skip
                if actual_width < chip_size or actual_height < chip_size:
                    # Read what we can
                    window = Window(col_off, row_off, actual_width, actual_height)
                    chip_data = src.read(window=window)
                    
                    # Pad to full chip size
                    padded_chip = np.full((18, chip_size, chip_size), -9999, dtype=np.int16)
                    padded_chip[:, :actual_height, :actual_width] = chip_data
                    chip_data = padded_chip
                else:
                    # Read full chip
                    window = Window(col_off, row_off, chip_size, chip_size)
                    chip_data = src.read(window=window)
                
                # Check if chip has valid data
                valid_pixels = np.sum(chip_data != -9999)
                total_pixels = chip_data.size
                valid_percentage = (valid_pixels / total_pixels) * 100
                
                # For inference, we create ALL chips, even if mostly nodata
                # You can add a minimum threshold here if needed
                # if valid_percentage < 10.0:  # Skip if less than 10% valid
                #     skipped_count += 1
                #     continue
                
                # ✓ FIXED: Create chip filename with _merged.tif suffix
                chip_filename = f"chip_{row_idx:03d}_{col_idx:03d}_merged.tif"
                chip_path = os.path.join(output_dir, chip_filename)
                
                # Update transform for this chip
                chip_transform = rasterio.windows.transform(window, src.transform)
                chip_profile['transform'] = chip_transform
                
                # Write chip
                with rasterio.open(chip_path, 'w', **chip_profile) as dst:
                    dst.write(chip_data)
                    
                    # Copy band descriptions
                    for i in range(18):
                        if i < len(src.descriptions) and src.descriptions[i]:
                            dst.set_band_description(i + 1, src.descriptions[i])
                
                chip_paths.append(chip_path)
                chip_count += 1
                
                # Progress logging
                if chip_count % 100 == 0:
                    logging.info(f"  Created {chip_count}/{total_chips} chips...")
        
        logging.info(f"\n✓ Chip creation complete!")
        logging.info(f"  Total chips created: {chip_count}")
        logging.info(f"  Chips skipped: {skipped_count}")
        logging.info(f"  Output directory: {output_dir}")
    
    return chip_paths


def verify_chips(output_dir, num_samples=5):
    """Verify a sample of created chips."""
    
    logging.info("\n" + "="*80)
    logging.info("VERIFYING CHIPS")
    logging.info("="*80)
    
    chip_files = sorted(list(Path(output_dir).glob("chip_*_merged.tif")))
    
    if not chip_files:
        logging.error("No chips found!")
        return False
    
    logging.info(f"\nTotal chips found: {len(chip_files)}")
    
    # Sample chips to verify
    sample_size = min(num_samples, len(chip_files))
    sample_indices = np.linspace(0, len(chip_files)-1, sample_size, dtype=int)
    
    logging.info(f"\nVerifying {sample_size} sample chips:")
    
    all_valid = True
    
    for idx in sample_indices:
        chip_path = chip_files[idx]
        
        try:
            with rasterio.open(chip_path) as src:
                # Read first band to check
                band1 = src.read(1)
                valid_pixels = np.sum(band1 != -9999)
                total_pixels = band1.size
                valid_pct = (valid_pixels / total_pixels) * 100
                
                # Get value range
                valid_data = band1[band1 != -9999]
                if len(valid_data) > 0:
                    min_val = valid_data.min()
                    max_val = valid_data.max()
                    mean_val = valid_data.mean()
                else:
                    min_val = max_val = mean_val = -9999
                
                logging.info(f"\n  {chip_path.name}")
                logging.info(f"    Size: {src.width}x{src.height}")
                logging.info(f"    Bands: {src.count}")
                logging.info(f"    Dtype: {src.dtypes[0]}")
                logging.info(f"    NoData: {src.nodata}")
                logging.info(f"    Valid pixels: {valid_pct:.1f}%")
                logging.info(f"    Value range: [{min_val}, {max_val}]")
                logging.info(f"    Mean: {mean_val:.2f}")
                
                # Verify structure
                if src.count != 18:
                    logging.error(f"    ✗ Wrong band count: {src.count}")
                    all_valid = False
                if src.dtypes[0] != 'int16':
                    logging.error(f"    ✗ Wrong dtype: {src.dtypes[0]}")
                    all_valid = False
                if src.width != 224 or src.height != 224:
                    logging.error(f"    ✗ Wrong dimensions: {src.width}x{src.height}")
                    all_valid = False
                if src.nodata != -9999:
                    logging.error(f"    ✗ Wrong nodata: {src.nodata}")
                    all_valid = False
                
                if all([src.count == 18, src.dtypes[0] == 'int16', 
                       src.width == 224, src.height == 224, src.nodata == -9999]):
                    logging.info(f"    ✓ Valid chip")
                
        except Exception as e:
            logging.error(f"  ✗ Error reading {chip_path.name}: {e}")
            all_valid = False
    
    if all_valid:
        logging.info(f"\n✓ All sample chips are valid!")
    else:
        logging.warning(f"\n⚠ Some chips have issues")
    
    return all_valid


def create_chip_index(output_dir):
    """Create an index file listing all chips with metadata."""
    
    logging.info("\n" + "="*80)
    logging.info("CREATING CHIP INDEX")
    logging.info("="*80)
    
    chip_files = sorted(list(Path(output_dir).glob("chip_*_merged.tif")))
    
    index_path = os.path.join(output_dir, "chip_index.txt")
    
    with open(index_path, 'w') as f:
        f.write("# Chip Index for Prithvi Inference\n")
        f.write(f"# Total chips: {len(chip_files)}\n")
        f.write(f"# Chip size: 224x224\n")
        f.write(f"# Bands: 18 (6 bands × 3 seasons)\n")
        f.write(f"# Naming: chip_<row>_<col>_merged.tif\n")
        f.write(f"# Format: filename,row,col,valid_percentage,min_value,max_value\n")
        f.write("#\n")
        
        for chip_path in chip_files:
            try:
                with rasterio.open(chip_path) as src:
                    band1 = src.read(1)
                    valid_pixels = np.sum(band1 != -9999)
                    total_pixels = band1.size
                    valid_pct = (valid_pixels / total_pixels) * 100
                    
                    valid_data = band1[band1 != -9999]
                    if len(valid_data) > 0:
                        min_val = valid_data.min()
                        max_val = valid_data.max()
                    else:
                        min_val = max_val = -9999
                    
                    # Extract row and col from filename: chip_<row>_<col>_merged.tif
                    parts = chip_path.stem.replace('_merged', '').split('_')
                    row = int(parts[1])
                    col = int(parts[2])
                    
                    f.write(f"{chip_path.name},{row},{col},{valid_pct:.2f},{min_val},{max_val}\n")
                    
            except Exception as e:
                logging.error(f"Error processing {chip_path.name}: {e}")
    
    logging.info(f"✓ Index created: {index_path}")
    logging.info(f"  Total chips indexed: {len(chip_files)}")


if __name__ == "__main__":
    logging.info("\n" + "="*80)
    logging.info("PRITHVI INFERENCE CHIP CREATION")
    logging.info("="*80)
    
    # Check if input file exists
    if not os.path.exists(INPUT_STACKED_PATH):
        logging.error(f"Input file not found: {INPUT_STACKED_PATH}")
        exit(1)
    
    # Create chips
    try:
        chip_paths = create_chips_for_inference(
            INPUT_STACKED_PATH,
            OUTPUT_CHIPS_DIR,
            chip_size=CHIP_SIZE,
            overlap=OVERLAP
        )
        
        # Verify chips
        verify_chips(OUTPUT_CHIPS_DIR, num_samples=5)
        
        # Create index
        create_chip_index(OUTPUT_CHIPS_DIR)
        
        logging.info("\n" + "="*80)
        logging.info("✓✓✓ SUCCESS! ✓✓✓")
        logging.info("="*80)
        logging.info(f"\nChips ready for Prithvi inference!")
        logging.info(f"  Location: {OUTPUT_CHIPS_DIR}")
        logging.info(f"  Total chips: {len(chip_paths)}")
        logging.info(f"  Chip size: {CHIP_SIZE}x{CHIP_SIZE}")
        logging.info(f"  Bands per chip: 18")
        logging.info(f"  Format: Int16, NoData=-9999")
        logging.info(f"  Naming: chip_<row>_<col>_merged.tif")
        logging.info("\nNext step: Run Prithvi inference on these chips")
        logging.info("="*80)
        
    except Exception as e:
        logging.error(f"\n❌ Error creating chips: {e}")
        import traceback
        traceback.print_exc()
        exit(1)