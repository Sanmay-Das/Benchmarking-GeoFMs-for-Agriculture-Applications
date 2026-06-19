import os
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import geopandas as gpd
import logging
from pyproj import Transformer
from shapely.ops import transform as shapely_transform

# --- Configuration ---
# Input CDL raster (raw, full CONUS)
INPUT_CDL_FILE = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/2023_30m_cdls.tif"

# Your AOI GeoJSON
AOI_GEOJSON_PATH = "/bigdata/eldawylab/sdas050/MS_Research/scripts/agriculture_CA_UTM11N.geojson"

# Output paths
OUTPUT_CLIPPED_REMAPPED = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/CA_groundtruth_processed.tif"

# Target CRS (must match stacked image)
TARGET_CRS = "EPSG:5070"

# --- Class Remapping ---
# Map CDL values (255 classes) to your 13 model classes
# Based on your CLASSES from config:
# 0: Natural Vegetation
# 1: Forest
# 2: Corn
# 3: Soybeans
# 4: Wetlands
# 5: Developed/Barren
# 6: Open Water
# 7: Winter Wheat
# 8: Alfalfa
# 9: Fallow/Idle Cropland
# 10: Cotton
# 11: Sorghum
# 12: Other

CDL_TO_MODEL_MAPPING = {
    # Corn
    1: 2,    # Corn -> Corn (class 2)
    12: 2,   # Sweet Corn -> Corn
    13: 2,   # Pop or Orn Corn -> Corn
    
    # Cotton
    2: 10,   # Cotton -> Cotton (class 10)
    
    # Soybeans
    5: 3,    # Soybeans -> Soybeans (class 3)
    26: 3,   # Dbl Crop WinWht/Soybeans -> Soybeans
    
    # Sorghum
    4: 11,   # Sorghum -> Sorghum (class 11)
    
    # Winter Wheat
    24: 7,   # Winter Wheat -> Winter Wheat (class 7)
    
    # Alfalfa
    36: 8,   # Alfalfa -> Alfalfa (class 8)
    
    # Fallow/Idle Cropland
    61: 9,   # Fallow/Idle Cropland -> Fallow (class 9)
    
    # Forest (Deciduous, Evergreen, Mixed)
    141: 1,  # Deciduous Forest -> Forest (class 1)
    142: 1,  # Evergreen Forest -> Forest
    143: 1,  # Mixed Forest -> Forest
    
    # Wetlands
    87: 4,   # Wetlands -> Wetlands (class 4)
    190: 4,  # Woody Wetlands -> Wetlands
    195: 4,  # Herbaceous Wetlands -> Wetlands
    
    # Open Water
    83: 6,   # Water -> Open Water (class 6)
    111: 6,  # Open Water -> Open Water
    
    # Developed/Barren
    121: 5,  # Developed/Open Space -> Developed/Barren (class 5)
    122: 5,  # Developed/Low Intensity -> Developed/Barren
    123: 5,  # Developed/Med Intensity -> Developed/Barren
    124: 5,  # Developed/High Intensity -> Developed/Barren
    131: 5,  # Barren -> Developed/Barren
    
    # Natural Vegetation (Shrubland, Grassland)
    152: 0,  # Shrubland -> Natural Vegetation (class 0)
    176: 0,  # Grassland/Pasture -> Natural Vegetation
    
    # All other crops -> Other (class 12)
    # This will be handled by default mapping below
}

# Default mapping for all other CDL values -> "Other" (class 12)
# Any CDL crop not explicitly mapped above goes to "Other"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_full_mapping_array():
    """
    Create a lookup array for fast remapping.
    Array index = CDL value, Array value = Model class
    """
    # Initialize all to 12 (Other class)
    mapping = np.full(256, 12, dtype=np.uint8)
    
    # Apply explicit mappings
    for cdl_val, model_class in CDL_TO_MODEL_MAPPING.items():
        mapping[cdl_val] = model_class
    
    # Set nodata/background to 255 (ignore class)
    mapping[0] = 255  # Background/NoData
    
    return mapping

def process_ground_truth(input_cdl_path, aoi_geojson_path, output_path, target_crs):
    """
    Process CDL ground truth: clip to AOI, reproject to target CRS, and remap classes.
    """
    try:
        logging.info("=" * 80)
        logging.info("PROCESSING GROUND TRUTH DATA")
        logging.info("=" * 80)
        
        # Step 1: Load AOI
        logging.info("\nStep 1: Loading AOI...")
        aoi_gdf = gpd.read_file(aoi_geojson_path)
        if aoi_gdf.crs is None:
            raise ValueError("AOI GeoJSON missing CRS information")
        
        aoi_geometry = aoi_gdf.geometry.iloc[0]
        aoi_crs = aoi_gdf.crs
        
        logging.info(f"  AOI CRS: {aoi_crs}")
        logging.info(f"  Target CRS: {target_crs}")
        
        # Step 2: Reproject AOI to target CRS
        logging.info("\nStep 2: Reprojecting AOI to target CRS...")
        transformer = Transformer.from_crs(aoi_crs, target_crs, always_xy=True)
        aoi_geometry_target = shapely_transform(transformer.transform, aoi_geometry)
        
        minx, miny, maxx, maxy = aoi_geometry_target.bounds
        logging.info(f"  AOI bounds in {target_crs}:")
        logging.info(f"    Min X: {minx:.2f}, Max X: {maxx:.2f}")
        logging.info(f"    Min Y: {miny:.2f}, Max Y: {maxy:.2f}")
        
        # Step 3: Open and check input CDL
        logging.info("\nStep 3: Opening input CDL raster...")
        with rasterio.open(input_cdl_path) as src:
            logging.info(f"  Input file: {input_cdl_path}")
            logging.info(f"  Input CRS: {src.crs}")
            logging.info(f"  Input size: {src.width} x {src.height}")
            logging.info(f"  Input bounds: {src.bounds}")
            
            # Step 4: Calculate output transform and dimensions
            logging.info("\nStep 4: Calculating output dimensions...")
            
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, target_crs,
                src.width, src.height,
                *src.bounds
            )
            
            # Get pixel size
            pixel_size_x = dst_transform[0]
            pixel_size_y = -dst_transform[4]
            
            # Calculate clipped dimensions
            clip_width = int(np.ceil((maxx - minx) / pixel_size_x))
            clip_height = int(np.ceil((maxy - miny) / pixel_size_y))
            
            from rasterio.transform import from_bounds
            clip_transform = from_bounds(minx, miny, maxx, maxy, clip_width, clip_height)
            
            logging.info(f"  Output size (clipped): {clip_width} x {clip_height}")
            logging.info(f"  Pixel size: {pixel_size_x:.2f} x {pixel_size_y:.2f} meters")
            
            # Step 5: Reproject to target CRS and clip
            logging.info("\nStep 5: Reprojecting and clipping to AOI...")
            
            # Create temporary array for reprojected data
            temp_reprojected = np.zeros((clip_height, clip_width), dtype=np.uint8)
            
            reproject(
                source=src.read(1),
                destination=temp_reprojected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=clip_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,  # Use nearest for categorical data
                src_nodata=0,
                dst_nodata=0
            )
            
            logging.info(f"  Reprojection complete")
            logging.info(f"  Unique CDL values in clipped area: {len(np.unique(temp_reprojected))}")
            
            # Step 6: Remap classes
            logging.info("\nStep 6: Remapping CDL classes to model classes...")
            
            mapping_array = create_full_mapping_array()
            remapped = mapping_array[temp_reprojected]
            
            unique_classes = np.unique(remapped)
            unique_classes = unique_classes[unique_classes != 255]  # Exclude ignore class
            
            logging.info(f"  Model classes present: {sorted(unique_classes.tolist())}")
            
            # Count pixels per class
            class_names = [
                "Natural Vegetation",
                "Forest",
                "Corn",
                "Soybeans",
                "Wetlands",
                "Developed/Barren",
                "Open Water",
                "Winter Wheat",
                "Alfalfa",
                "Fallow/Idle Cropland",
                "Cotton",
                "Sorghum",
                "Other"
            ]
            
            logging.info("\n  Class distribution:")
            for class_id in sorted(unique_classes):
                count = np.sum(remapped == class_id)
                percentage = (count / remapped.size) * 100
                class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"
                logging.info(f"    Class {class_id} ({class_name}): {count:,} pixels ({percentage:.2f}%)")
            
            # Step 7: Save output
            logging.info("\nStep 7: Saving processed ground truth...")
            
            profile = src.profile.copy()
            profile.update({
                'crs': target_crs,
                'transform': clip_transform,
                'width': clip_width,
                'height': clip_height,
                'count': 1,
                'dtype': 'uint8',
                'compress': 'lzw',
                'nodata': 255
            })
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(remapped, 1)
            
            logging.info(f"  Saved to: {output_path}")
            logging.info(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # Step 8: Verify output
        logging.info("\nStep 8: Verifying output...")
        with rasterio.open(output_path) as dst:
            logging.info(f"  Output CRS: {dst.crs}")
            logging.info(f"  Output size: {dst.width} x {dst.height}")
            logging.info(f"  Output bounds: {dst.bounds}")
            logging.info(f"  Data type: {dst.dtypes[0]}")
            
            sample = dst.read(1, window=((0, 100), (0, 100)))
            logging.info(f"  Sample values: {np.unique(sample)[:10]}")
        
        logging.info("\n" + "=" * 80)
        logging.info("✓ GROUND TRUTH PROCESSING COMPLETE")
        logging.info("=" * 80)
        
        return True
        
    except Exception as e:
        logging.error(f"Error processing ground truth: {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting ground truth processing...")
    
    if not os.path.exists(INPUT_CDL_FILE):
        logging.error(f"Input CDL file not found: {INPUT_CDL_FILE}")
    elif not os.path.exists(AOI_GEOJSON_PATH):
        logging.error(f"AOI GeoJSON not found: {AOI_GEOJSON_PATH}")
    else:
        success = process_ground_truth(
            INPUT_CDL_FILE,
            AOI_GEOJSON_PATH,
            OUTPUT_CLIPPED_REMAPPED,
            TARGET_CRS
        )
        
        if success:
            logging.info("\n" + "=" * 80)
            logging.info("NEXT STEP: Create mask chips")
            logging.info("=" * 80)
            logging.info(f"Use this file as input: {OUTPUT_CLIPPED_REMAPPED}")