import os
import logging
from datetime import datetime, timedelta
from cdsetool.query import query_features
from cdsetool.download import download_feature
from cdsetool.credentials import Credentials
from cdsetool.monitor import StatusMonitor
from multiprocessing import Manager
import json
import zipfile
import rasterio
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import shape, box
from shapely.wkt import loads as parse_wkt
from queue import Queue
from threading import Thread, Lock
import shutil
import tempfile

# Set up logging
logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging based on log level."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_file = "sentinel2_downloader.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(numeric_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    formatter = logging.Formatter("%(asctime)s %(threadName)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info(f"Logging to file: {os.path.abspath(log_file)}")


def create_grid(geometry, cell_size=10.0):
    """Create a uniform grid of polygons over the bounding box."""
    bounds = geometry.bounds
    minx, miny, maxx, maxy = bounds
    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            grid_cell = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(grid_cell)
            y += cell_size
        x += cell_size
    sub_geometries = [geometry.intersection(cell) for cell in grid_cells if geometry.intersects(cell)]
    return sub_geometries


def split_date_range(start_date, end_date):
    """Split date range into daily ranges."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    ranges = []
    while start <= end:
        ranges.append(start.strftime("%Y-%m-%d"))
        start += timedelta(days=1)
    return ranges


def process_zip_to_ndvi(zip_path, output_dir, keep_raw=False):
    """
    Extract ZIP and create 6-band GeoTIFF at 10m resolution.
    Bands: B02, B03, B04, B8A, B11, B12 (all resampled to 10m)
    """
    tile_id = os.path.basename(zip_path).split(".")[0]
    output_file = os.path.join(output_dir, f"{tile_id}.tif")
    logger.debug(f"Processing {zip_path} into {output_file}")

    temp_extract_dir = tempfile.mkdtemp(dir=output_dir, prefix=f"extract_{tile_id}_")
    logger.debug(f"Extracting {tile_id} to temporary directory: {temp_extract_dir}")

    try:
        # Extract ZIP
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        # Locate .SAFE directory
        safe_dirs = [d for d in os.listdir(temp_extract_dir) if d.endswith(".SAFE")]
        logger.debug(f"Found {len(safe_dirs)} .SAFE directories: {safe_dirs}")

        safe_dir = None
        for d in safe_dirs:
            if tile_id in d or d.startswith(tile_id):
                safe_dir = os.path.join(temp_extract_dir, d)
                break
        if not safe_dir and safe_dirs:
            safe_dir = os.path.join(temp_extract_dir, safe_dirs[0])
        if not safe_dir:
            raise FileNotFoundError(f"Could not find .SAFE directory for {tile_id}")

        # Locate GRANULE
        granule_dir = next((os.path.join(safe_dir, "GRANULE", d) 
                           for d in os.listdir(os.path.join(safe_dir, "GRANULE"))), None)
        if not granule_dir:
            raise FileNotFoundError("Could not find GRANULE directory")

        # Locate band folders
        r10m_dir = os.path.join(granule_dir, "IMG_DATA", "R10m")
        r20m_dir = os.path.join(granule_dir, "IMG_DATA", "R20m")

        # Find band files
        band_paths = {}
        
        # 10m bands: B02, B03, B04
        if os.path.isdir(r10m_dir):
            for file in os.listdir(r10m_dir):
                if file.endswith("_B02_10m.jp2"):
                    band_paths['B02'] = os.path.join(r10m_dir, file)
                elif file.endswith("_B03_10m.jp2"):
                    band_paths['B03'] = os.path.join(r10m_dir, file)
                elif file.endswith("_B04_10m.jp2"):
                    band_paths['B04'] = os.path.join(r10m_dir, file)

        # 20m bands: B8A, B11, B12
        if os.path.isdir(r20m_dir):
            for file in os.listdir(r20m_dir):
                if file.endswith("_B8A_20m.jp2"):
                    band_paths['B8A'] = os.path.join(r20m_dir, file)
                elif file.endswith("_B11_20m.jp2"):
                    band_paths['B11'] = os.path.join(r20m_dir, file)
                elif file.endswith("_B12_20m.jp2"):
                    band_paths['B12'] = os.path.join(r20m_dir, file)

        # Verify all bands found
        required_bands = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
        missing = [b for b in required_bands if b not in band_paths]
        if missing:
            raise FileNotFoundError(f"Missing bands: {missing}")

        logger.debug(f"Found all 6 bands for {tile_id}")

        # Read 10m bands (reference resolution)
        with rasterio.open(band_paths['B02']) as src:
            meta = src.meta.copy()
            target_height = src.height
            target_width = src.width
            target_transform = src.transform

        # Update metadata for 6 bands
        meta.update({
            "driver": "GTiff",
            "dtype": "uint16",  # Sentinel-2 bands are 16-bit
            "count": 6,
            "compress": "LZW",
            "nodata": 0
        })

        # Read and resample all bands to 10m
        bands_data = []
        band_order = ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']
        
        for band_name in band_order:
            with rasterio.open(band_paths[band_name]) as src:
                if src.height == target_height and src.width == target_width:
                    # Already 10m resolution
                    data = src.read(1)
                    logger.debug(f"Read {band_name} at native 10m")
                else:
                    # Resample from 20m to 10m
                    data = src.read(
                        1,
                        out_shape=(target_height, target_width),
                        resampling=Resampling.bilinear
                    )
                    logger.debug(f"Resampled {band_name} from 20m to 10m")
                bands_data.append(data)

        # Write 6-band GeoTIFF
        with rasterio.open(output_file, "w", **meta) as dst:
            for i, data in enumerate(bands_data, 1):
                dst.write(data, i)
                dst.set_band_description(i, band_order[i-1])

        logger.info(f"Created 6-band GeoTIFF: {output_file}")

        # Handle keep_raw option
        if keep_raw:
            # Keep only the 6 band files
            for file in os.listdir(r10m_dir):
                file_path = os.path.join(r10m_dir, file)
                if not any(file.endswith(f"_{b}_10m.jp2") for b in ['B02', 'B03', 'B04']):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            for file in os.listdir(r20m_dir):
                file_path = os.path.join(r20m_dir, file)
                if not any(file.endswith(f"_{b}_20m.jp2") for b in ['B8A', 'B11', 'B12']):
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            # Remove R60m folder
            r60m_path = os.path.join(granule_dir, "IMG_DATA", "R60m")
            if os.path.exists(r60m_path):
                shutil.rmtree(r60m_path)

            # Remove unnecessary folders
            for folder in ["QI_DATA", "AUX_DATA"]:
                folder_path = os.path.join(granule_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)

            for folder in ["HTML", "rep_info", "DATASTRIP"]:
                folder_path = os.path.join(safe_dir, folder)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path)

            # Move cleaned SAFE to output
            safe_name = os.path.basename(safe_dir)
            final_safe_path = os.path.join(output_dir, safe_name)
            if os.path.exists(final_safe_path):
                shutil.rmtree(final_safe_path)
            shutil.move(safe_dir, final_safe_path)
            logger.debug(f"Moved SAFE directory to {final_safe_path}")

    finally:
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)

    return output_file


def download_and_process(feature, credentials, output_dir, keep_raw=False):
    """Downloads and processes a single feature to 6-band GeoTIFF."""
    tile_id = feature["properties"]["title"].removesuffix('.SAFE')
    tile_parts = tile_id.split('_')
    tile_code = tile_parts[5] if len(tile_parts) >= 6 else "UNKNOWN"

    logger.info(f"Processing feature: {tile_id} (Tile code: {tile_code})")

    try:
        date = feature["properties"]["startDate"][:10]
        date_dir = os.path.join(output_dir, date)
        os.makedirs(date_dir, exist_ok=True)

        output_tif = os.path.join(date_dir, f"{tile_id}.tif")
        zip_path = os.path.join(date_dir, f"{tile_id}.zip")

        # Skip if already exists
        if os.path.exists(output_tif):
            logger.debug(f"Skipping '{tile_id}' - TIF already exists")
            return "skip"

        if os.path.exists(zip_path):
            logger.debug(f"Skipping download for '{tile_id}' - ZIP exists")
        else:
            monitor = StatusMonitor()
            logger.info(f"Downloading '{tile_id}' (Tile: {tile_code})")
            downloaded_filename = download_feature(feature, date_dir, 
                                                  {"credentials": credentials, "monitor": monitor})
            zip_path = os.path.join(date_dir, downloaded_filename)

        logger.info(f"Processing ZIP to 6-band for '{tile_id}' (Tile: {tile_code})")
        output_file = process_zip_to_ndvi(zip_path, date_dir, keep_raw)

        # Cleanup ZIP
        os.remove(zip_path)

        logger.info(f"Successfully processed '{tile_id}' -> {output_file}")
        return "success"
    except Exception as e:
        logger.error(f"Error processing {tile_id}: {e}")
        return "error"


def download_sentinel2_data(date_from, date_to, roi, output_dir, keep_raw=False):
    """Download Sentinel2 data and create 6-band GeoTIFFs."""
    max_retries = 3
    manager = Manager()
    all_files = manager.dict()
    processed_files = manager.list()
    skipped_files = manager.list()
    failed_files = manager.list()
    work_queue = Queue(maxsize=100)
    completion_lock = Lock()

    def producer():
        logger.debug("Starting search process...")
        sub_geometries = create_grid(roi)
        date_ranges = split_date_range(date_from, date_to)

        for date in date_ranges:
            complete_file_path = os.path.join(output_dir, date, ".complete")
            if os.path.exists(complete_file_path):
                logger.debug(f"Skipping completed day: {date}")
                continue

            for sub_geometry in sub_geometries:
                search_terms = {
                    "startDate": date,
                    "completionDate": (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "processingLevel": "S2MSI2A",
                    "geometry": sub_geometry.wkt,
                    "cloudCover": "[0,10]",
                }

                features = list(query_features("Sentinel2", search_terms))
                logger.debug(f"Found {len(features)} on [{search_terms['startDate']}, {search_terms['completionDate']}]")
                
                if features:
                    with completion_lock:
                        if date not in all_files:
                            all_files[date] = manager.list()
                        for feature in features:
                            if feature not in all_files[date]:
                                all_files[date].append(feature)
                                work_queue.put((feature, max_retries))

            with completion_lock:
                for file in list(processed_files) + list(skipped_files):
                    file_date = file["properties"]["startDate"][:10]
                    if file_date in all_files and file in all_files[file_date]:
                        all_files[file_date].remove(file)
                        if not all_files[file_date]:
                            del all_files[file_date]
                            day_dir = os.path.join(output_dir, file_date)
                            os.makedirs(day_dir, exist_ok=True)
                            with open(os.path.join(day_dir, ".complete"), "w") as f:
                                f.write("")

        work_queue.put(None)

    def consumer():
        logger.debug("Starting downloader")
        credentials = Credentials(username="sanmay4119@gmail.com", password="Qscnkp&*()7890")
        
        while True:
            try:
                task = work_queue.get()
                if task is None:
                    logger.debug("Downloader is done")
                    work_queue.task_done()
                    work_queue.put(None)
                    break

                feature, retries = task

                while retries >= 0:
                    status = download_and_process(feature, credentials, output_dir, keep_raw)

                    if status == "success":
                        processed_files.append(feature)
                        break
                    elif status == "error" and retries > 0:
                        feature_tile_id = feature["properties"]["title"].removesuffix('.SAFE')
                        logger.warning(f"Retrying {feature_tile_id}, remaining: {retries}")
                        retries -= 1
                    elif status == "error" and retries == 0:
                        failed_files.append(feature)
                        break
                    elif status == "skip":
                        skipped_files.append(feature)
                        break
                    else:
                        logger.error(f"Unexpected status {status}")
                        break

                work_queue.task_done()
            except Exception as e:
                work_queue.task_done()
                logger.error(f"Error in consumer: {e}")
                continue

    producer_thread = Thread(target=producer, name="producer")
    producer_thread.start()
    
    consumers = []
    for i in range(4):
        consumer_thread = Thread(target=consumer, name=f"consumer #{i}")
        consumers.append(consumer_thread)
        consumer_thread.start()

    while producer_thread.is_alive() or any(t.is_alive() for t in consumers):
        logger.info("Checking thread statuses...")
        producer_thread.join(timeout=120)
        for consumer_thread in consumers:
            consumer_thread.join(timeout=120)

    return {
        "success": len(processed_files),
        "skipped": len(skipped_files),
        "failed": len(failed_files),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Sentinel2 6-band data.")
    parser.add_argument("--date-from", required=True, help="Start date (yyyy-mm-dd)")
    parser.add_argument("--date-to", required=True, help="End date (yyyy-mm-dd)")
    parser.add_argument("--roi", required=True, help="GeoJSON file or WKT")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--keep-raw", action="store_true", default=False, 
                       help="Keep raw band files")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="INFO", help="Logging level")

    args = parser.parse_args()
    setup_logging(args.log_level)

    roi = args.roi
    if os.path.exists(roi) and roi.lower().endswith(".geojson"):
        with open(roi, "r") as f:
            geojson = json.load(f)
            geometry = geojson["features"][0]["geometry"]
            roi = shape(geometry)
    else:
        roi = parse_wkt(roi)
    
    results = download_sentinel2_data(args.date_from, args.date_to, roi, 
                                     args.output, args.keep_raw)
    logger.info(f"Summary: {results['success']} processed, {results['skipped']} skipped, "
               f"{results['failed']} errors.")