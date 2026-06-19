import rasterio
import numpy as np
from pathlib import Path
from collections import Counter

chip_dir = Path("/bigdata/eldawylab/sdas050/MS_Research/data/multi_temporal_crop_segmentation/EastIA")

label_files = list(chip_dir.glob("*.mask.tif"))
print(f"Total label chips: {len(label_files)}")

class_names = [
    'No Data', 'Natural Vegetation', 'Forest', 'Corn', 'Soybeans',
    'Wetlands', 'Developed/Barren', 'Open Water', 'Winter Wheat',
    'Alfalfa', 'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]

total_counts = Counter()

for f in label_files[:50]:  # Sample 50 chips
    with rasterio.open(f) as src:
        label = src.read(1)
    unique, counts = np.unique(label, return_counts=True)
    for u, c in zip(unique, counts):
        total_counts[int(u)] += int(c)

print("\nClass distribution in chips (raw label values):")
for class_id in sorted(total_counts.keys()):
    name = class_names[class_id] if class_id < len(class_names) else "UNKNOWN"
    print(f"  {class_id:3d} ({name:20s}): {total_counts[class_id]:10,} px")