import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

CDL_PATH   = "/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/SouthMN/SouthMN_cdl_epsg5070_10m.tif"
STACK_PATH = "/bigdata/eldawylab/sdas050/MS_Research/scripts/processed_stacks/SouthMN/SouthMN_multitemporal_stack.tif"
OUTPUT_DIR = "/bigdata/eldawylab/sdas050/MS_Research/predictions/gt_maps"


# ============================================================================
# CDL → 13-CLASS MAPPING
# ============================================================================

def create_cdl_mapping():
    mapping = {}
    # 0: Natural Vegetation
    mapping.update({code: 0 for code in [37, 60, 62, 65, 152, 176]})
    # 1: Forest
    mapping.update({code: 1 for code in [63, 141, 142, 143, 190]})
    # 2: Corn
    mapping[1] = 2
    # 3: Soybeans
    mapping[5] = 3
    # 4: Wetlands
    mapping.update({code: 4 for code in [87, 195]})
    # 5: Developed/Barren
    mapping.update({code: 5 for code in [82, 121, 122, 123, 124, 131]})
    # 6: Open Water
    mapping.update({code: 6 for code in [83, 111]})
    # 7: Winter Wheat
    mapping[24] = 7
    # 8: Alfalfa
    mapping[36] = 8
    # 9: Fallow/Idle Cropland
    mapping[61] = 9
    # 10: Cotton
    mapping[2] = 10
    # 11: Sorghum
    mapping[4] = 11
    # 12: Other (everything else)
    return mapping


# ============================================================================
# EXACT ESRI RGB COLORS FROM CDL EXCEL
# ============================================================================

CLASS_COLORS = {
    0:   (233, 255, 190),  # Natural Vegetation — CDL 176 Grassland/Pasture
    1:   (149, 206, 147),  # Forest             — CDL 141 Deciduous Forest
    2:   (255, 212, 0),    # Corn               — CDL 1
    3:   (38,  115, 0),    # Soybeans           — CDL 5
    4:   (128, 179, 179),  # Wetlands           — CDL 87
    5:   (156, 156, 156),  # Developed/Barren   — CDL 121
    6:   (77,  112, 163),  # Open Water         — CDL 111
    7:   (168, 112, 0),    # Winter Wheat       — CDL 24
    8:   (255, 168, 227),  # Alfalfa            — CDL 36
    9:   (191, 191, 122),  # Fallow/Idle        — CDL 61
    10:  (255, 38,  38),   # Cotton             — CDL 2
    11:  (255, 158, 15),   # Sorghum            — CDL 4
    12:  (204, 191, 163),  # Other              — CDL 131 Barren
    255: (255,   255,   255),    # NoData             — white
}

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans",
    "Wetlands", "Developed/Barren", "Open Water", "Winter Wheat",
    "Alfalfa", "Fallow/Idle", "Cotton", "Sorghum", "Other"
]


# ============================================================================
# COLORIZE
# ============================================================================

def colorize_map(class_array):
    h, w = class_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[class_array == class_id] = color
    return rgb


def save_legend(output_path):
    n = len(CLASS_NAMES)
    box_size = 30
    padding = 10
    text_width = 220
    width = box_size + padding * 2 + text_width
    height = n * (box_size + padding) + padding

    legend = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(legend)

    for i, name in enumerate(CLASS_NAMES):
        y = padding + i * (box_size + padding)
        color = CLASS_COLORS.get(i, (128, 128, 128))
        draw.rectangle([padding, y, padding + box_size, y + box_size],
                       fill=color, outline=(0, 0, 0))
        draw.text((padding + box_size + 8, y + 6), f"{i}: {name}", fill=(0, 0, 0))

    legend.save(output_path)
    print(f"  ✅ Legend saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def generate_gt_map(cdl_path, stack_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read CDL
    print(f"Reading CDL: {cdl_path}")
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1)
        profile = src.profile
        print(f"  Shape: {cdl.shape}, CRS: {src.crs}")

    # Remap CDL → 13 classes
    print("\nRemapping CDL → 13 classes...")
    mapping = create_cdl_mapping()
    lookup = np.full(512, 12, dtype=np.uint8)  # Default → Other
    for cdl_code, class_id in mapping.items():
        if cdl_code < 512:
            lookup[cdl_code] = class_id
    lookup[0] = 255  # Background → NoData

    mapped = lookup[np.clip(cdl, 0, 511).astype(np.uint16)]
    mapped[cdl == 0] = 255

    # Apply stack NoData mask so GT has same shape as prediction
    print("\nApplying stack NoData mask...")
    with rasterio.open(stack_path) as src:
        stack_band1 = src.read(1)
        nodata_mask = stack_band1 == -9999
    mapped[nodata_mask] = 255
    print(f"  Masked {nodata_mask.sum():,} NoData border pixels")

    print(f"  Unique mapped classes: {np.unique(mapped)}")

    # Class distribution
    print("\nClass distribution:")
    total_valid = np.sum(mapped != 255)
    for c in range(13):
        count = np.sum(mapped == c)
        if count > 0:
            print(f"  Class {c:2d} ({CLASS_NAMES[c]:25s}): {count:10,d} px ({100*count/total_valid:.2f}%)")

    # Save GT GeoTIFF (class 0-12, NoData=255)
    gt_tif_path = output_dir / "SouthMN_gt_13class.tif"
    out_profile = profile.copy()
    out_profile.update({'dtype': 'uint8', 'count': 1, 'nodata': 255})
    with rasterio.open(gt_tif_path, 'w', **out_profile) as dst:
        dst.write(mapped.astype(np.uint8), 1)
    print(f"\n  ✅ GT GeoTIFF saved: {gt_tif_path}")

    # Save colored PNG
    rgb = colorize_map(mapped)
    gt_png_path = output_dir / "SouthMN_gt_colored.png"
    Image.fromarray(rgb).save(gt_png_path)
    print(f"  ✅ GT colored PNG saved: {gt_png_path}")

    # Save legend
    save_legend(output_dir / "legend.png")

    return gt_tif_path, gt_png_path


if __name__ == "__main__":
    generate_gt_map(CDL_PATH, STACK_PATH, OUTPUT_DIR)