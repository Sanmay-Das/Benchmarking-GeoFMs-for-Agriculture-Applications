import os
import argparse
import logging
import rasterio
from rasterio.enums import Resampling
from glob import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def downsample_to_30m(src_path, dst_path, target_res=30.0, resampling_method=Resampling.average):
    """
    Downsamples a GeoTIFF from its current resolution to a target resolution.
    
    :param src_path: Path to the source 10m GeoTIFF.
    :param dst_path: Path to save the new 30m GeoTIFF.
    :param target_res: The target resolution (e.g., 30.0 for 30m).
    :param resampling_method: The rasterio resampling method to use.
    """
    try:
        with rasterio.open(src_path) as src:
            # Get the current resolution
            src_res = src.res[0]
            if not (src_res > 0):
                raise ValueError("Source resolution is not valid.")

            # Calculate the scaling factor
            scale_factor = src_res / target_res  # e.g., 10m / 30m = 0.333
            
            # Calculate new width and height
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            
            # Get the source metadata
            meta = src.meta.copy()

            # Calculate new transform
            new_transform = src.transform * src.transform.scale(
                (src.width / new_width),
                (src.height / new_height)
            )
            
            # Update metadata for the output file
            meta.update({
                "driver": "GTiff",
                "height": new_height,
                "width": new_width,
                "transform": new_transform,
                "compress": "LZW"
            })
            
            with rasterio.open(dst_path, "w", **meta) as dst:
                for i in range(1, src.count + 1):
                    # Read data and resample in one step
                    data = src.read(
                        i,
                        out_shape=(new_height, new_width),
                        resampling=resampling_method
                    )
                    
                    # Write the resampled data to the new band
                    dst.write(data, i)
                    
                    # --- THIS IS THE CORRECTED PART ---
                    # Copy band description (e.g., 'B02', 'B03', etc.)
                    # We read from the .descriptions tuple (which is 0-indexed)
                    if src.descriptions and len(src.descriptions) >= i:
                        band_desc = src.descriptions[i-1]
                        if band_desc:
                            dst.set_band_description(i, band_desc)
                    # --- END OF FIX ---

        logger.debug(f"Successfully downsampled: {dst_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {src_path}: {e}")
        # Clean up partially created file on error
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="Downsample 10m GeoTIFFs to 30m.")
    parser.add_argument("--input-dir", required=True, 
                        help="Root directory containing 10m TIF files (e.g., /path/to/sentinel_data_10m/)")
    parser.add_argument("--output-dir", required=True, 
                        help="Root directory to save 30m TIF files (e.g., /path/to/sentinel_data_30m/)")
    parser.add_argument("--resampling", default="average", 
                        choices=["average", "bilinear", "cubic", "nearest"], 
                        help="Resampling method to use (default: 'average')")
    
    args = parser.parse_args()

    # Get the chosen resampling method from rasterio.enums
    resampling_method = getattr(Resampling, args.resampling)
    
    logger.info(f"Starting downsampling from {args.input_dir} to {args.output_dir}")
    logger.info(f"Using resampling method: {args.resampling}")
    
    # Find all TIF files in the input directory and its subdirectories
    search_path = os.path.join(args.input_dir, "**", "*.tif")
    
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    # Use glob to find all TIF files recursively
    tif_files = list(glob(search_path, recursive=True))
    if not tif_files:
        logger.warning(f"No .tif files found in {args.input_dir}")
        return

    logger.info(f"Found {len(tif_files)} .tif files to process.")

    for src_path in tif_files:
        
        # Create the corresponding output path
        relative_path = os.path.relpath(src_path, args.input_dir)
        dst_path = os.path.join(args.output_dir, relative_path)
        
        # Get the directory part of the destination path
        dst_dir = os.path.dirname(dst_path)
        
        # Create the output subdirectory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        # Skip if file already exists
        if os.path.exists(dst_path):
            logger.debug(f"Skipping, already exists: {dst_path}")
            skipped_count += 1
            continue

        logger.info(f"Processing: {src_path} -> {dst_path}")
        if downsample_to_30m(src_path, dst_path, resampling_method=resampling_method):
            processed_count += 1
        else:
            failed_count += 1

    logger.info("--- Downsampling Complete ---")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Failed: {failed_count}")

if __name__ == "__main__":
    main()