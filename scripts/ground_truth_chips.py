import os
import rasterio
from rasterio.windows import Window
import math
import numpy as np
import logging


# --- Configuration ---
INPUT_GROUND_TRUTH_FILE = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/CDL_2024_california_clipped.tif"
OUTPUT_MASK_CHIPS_DIR = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/inference_chips_CA_2024"
CHIP_SIZE = 224
MASK_SUFFIX = ".mask.tif"
OVERLAP = 0


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_mask_chips_from_raster(input_raster_path, output_dir, chip_size=224,
                                  mask_suffix=".mask.tif", overlap=0):
    """
    Creates mask chips - NO SKIPPING to ensure perfect 1:1 pairing with image chips.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
       
        logging.info("=" * 80)
        logging.info("CREATING MASK CHIPS (NO SKIPPING - ALL CHIPS)")
        logging.info("=" * 80)
        logging.info(f"Input file: {input_raster_path}")
        logging.info(f"Output directory: {output_dir}")


        with rasterio.open(input_raster_path) as src:
            if src.count != 1:
                logging.error(f"Input must have 1 band, found {src.count}. Exiting.")
                return


            width = src.width
            height = src.height
           
            logging.info(f"\nInput properties:")
            logging.info(f"  Size: {width}w x {height}h pixels")
            logging.info(f"  CRS: {src.crs}")
            logging.info(f"  Bounds: {src.bounds}")


            profile = src.profile.copy()
            profile.update({
                'height': chip_size,
                'width': chip_size,
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
                'nodata': 255
            })
           
            # Calculate stride
            stride = chip_size - overlap
           
            # Calculate number of chips
            n_chips_x = math.ceil((width - overlap) / stride)
            n_chips_y = math.ceil((height - overlap) / stride)
           
            logging.info(f"\nChip configuration:")
            logging.info(f"  Chip size: {chip_size}x{chip_size}")
            logging.info(f"  Overlap: {overlap}")
            logging.info(f"  Grid: {n_chips_x} x {n_chips_y} = {n_chips_x * n_chips_y} total chips")
            logging.info(f"  NO SKIPPING - creating ALL chips")
            logging.info("")


            chip_count = 0
            edge_chips = 0


            for row in range(n_chips_y):
                for col in range(n_chips_x):
                    x_off = col * stride
                    y_off = row * stride
                   
                    # Check dimensions
                    current_chip_width = min(chip_size, width - x_off)
                    current_chip_height = min(chip_size, height - y_off)
                   
                    # For edge chips that are smaller, pad with nodata (255)
                    is_edge_chip = (current_chip_width < chip_size or current_chip_height < chip_size)
                   
                    if is_edge_chip:
                        edge_chips += 1
                        # Create padded chip with nodata
                        chip_data = np.full((chip_size, chip_size), 255, dtype='uint8')
                       
                        # Read available data
                        window = Window(x_off, y_off, current_chip_width, current_chip_height)
                        available_data = src.read(1, window=window)
                       
                        # Place in padded array
                        chip_data[:current_chip_height, :current_chip_width] = available_data
                       
                        # Use default transform
                        chip_transform = src.window_transform(Window(x_off, y_off, chip_size, chip_size))
                    else:
                        # Normal chip
                        window = Window(x_off, y_off, chip_size, chip_size)
                        chip_data = src.read(1, window=window)
                        chip_transform = src.window_transform(window)
                   
                    # Update profile
                    chip_profile = profile.copy()
                    chip_profile.update({
                        'transform': chip_transform,
                        'height': chip_size,
                        'width': chip_size
                    })
                   
                    # Generate filename
                    output_filename = f"chip_{row:03d}_{col:03d}{mask_suffix}"
                    output_path = os.path.join(output_dir, output_filename)
                   
                    # Write chip
                    with rasterio.open(output_path, 'w', **chip_profile) as dst:
                        dst.write(chip_data, 1)
                   
                    chip_count += 1
                   
                    if chip_count % 100 == 0:
                        logging.info(f"Created {chip_count} mask chips...")


            logging.info("")
            logging.info("=" * 80)
            logging.info("MASK CHIP CREATION COMPLETE")
            logging.info("=" * 80)
            logging.info(f"Total chips created: {chip_count}")
            logging.info(f"Edge chips (padded): {edge_chips}")
           
            # Check pairing
            logging.info("\nVerifying image-mask pairing...")
            image_chips = sorted([f for f in os.listdir(output_dir) if f.endswith('_merged.tif')])
            mask_chips = sorted([f for f in os.listdir(output_dir) if f.endswith('.mask.tif')])
           
            logging.info(f"  Image chips: {len(image_chips)}")
            logging.info(f"  Mask chips: {len(mask_chips)}")
           
            if len(image_chips) == len(mask_chips):
                logging.info(f"  ✓ Perfect 1:1 pairing!")
            else:
                logging.warning(f"  ✗ Mismatch: {len(image_chips)} images vs {len(mask_chips)} masks")
           
            # Verify first pair
            if image_chips and mask_chips:
                first_image = image_chips[0]
                expected_mask = first_image.replace('_merged.tif', '.mask.tif')
                if expected_mask == mask_chips[0]:
                    logging.info(f"  ✓ First pair verified: {first_image} <-> {mask_chips[0]}")
                else:
                    logging.warning(f"  ✗ First pair mismatch!")
           
            logging.info("=" * 80)
           
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not os.path.exists(INPUT_GROUND_TRUTH_FILE):
        logging.error(f"Input file not found: {INPUT_GROUND_TRUTH_FILE}")
    else:
        create_mask_chips_from_raster(
            INPUT_GROUND_TRUTH_FILE,
            OUTPUT_MASK_CHIPS_DIR,
            CHIP_SIZE,
            MASK_SUFFIX,
            OVERLAP
        )
