
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

#!/usr/bin/env python3
"""
Get crop statistics using exact cdl_remap_rules.txt mapping
"""
import numpy as np
import rasterio
from pathlib import Path
from collections import Counter

def load_remap_rules(rules_file):
    """
    Load mapping from cdl_remap_rules.txt
    Format: CDL_VALUE = CLASS_ID
    """
    mapping = {}
    
    with open(rules_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '=' in line and not line.startswith('*'):
                parts = line.split('=')
                cdl_value = int(parts[0].strip())
                class_id = int(parts[1].strip())
                mapping[cdl_value] = class_id
    
    return mapping


def get_crop_statistics(cdl_file, mapping):
    """
    Calculate crop distribution using the mapping
    """
    crop_names = {
        1: "Natural Vegetation",
        2: "Forest",
        3: "Corn",
        4: "Soybeans",
        5: "Wetlands",
        6: "Developed/Barren",
        7: "Open Water",
        8: "Winter Wheat",
        9: "Alfalfa",
        10: "Fallow/Idle Cropland",
        11: "Cotton",
        12: "Sorghum",
        13: "Other"
    }
    
    print(f"\n{'='*80}")
    print(f"CROP STATISTICS: {Path(cdl_file).name}")
    print(f"{'='*80}")
    
    # Read CDL file
    with rasterio.open(cdl_file) as src:
        cdl_data = src.read(1)
        
        # Count CDL values
        cdl_counts = Counter(cdl_data.flatten())
        total_pixels = cdl_data.size
        
        # Map to classes 1-13
        class_counts = {i: 0 for i in range(1, 14)}
        
        for cdl_value, count in cdl_counts.items():
            if cdl_value == 0:  # Skip nodata
                continue
            
            if cdl_value in mapping:
                class_id = mapping[cdl_value]
                class_counts[class_id] += count
            else:
                # Anything not in mapping = class 13 (Other)
                class_counts[13] += count
        
        # Print results
        print(f"Total pixels: {total_pixels:,}\n")
        print(f"{'Class':<7} {'Crop Type':<27} {'Pixel Count':<15} {'Percentage'}")
        print("-"*80)
        
        for cls in range(1, 14):
            count = class_counts[cls]
            pct = 100 * count / total_pixels if total_pixels > 0 else 0
            crop_name = crop_names[cls]
            
            status = ""
            if count == 0:
                status = "[WARN] MISSING"
            elif pct < 0.5:
                status = "[WARN] RARE"
            elif pct > 20:
                status = "[OK] DOMINANT"
            
            print(f"{cls:<7} {crop_name:<27} {count:<15,} {pct:>6.2f}%  {status}")
        
        print("="*80)
        
        return class_counts, total_pixels


# Main execution
if __name__ == "__main__":
    # Load remap rules
    rules_file = f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/cdl_remap_rules.txt'
    mapping = load_remap_rules(rules_file)
    
    print(f"[OK] Loaded {len(mapping)} CDL mappings from cdl_remap_rules.txt")
    
    # Define regions
    regions = {
        'NorthCA': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/NorthCA/NorthCA_cdl_epsg5070_10m.tif',
        'CentCA': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/CentCA/CentCA_cdl_epsg5070_10m.tif',
        'SouthCA': f'{DATA_ROOT}/data/multi_temporal_crop_segmentation/SouthCA/SouthCA_cdl_epsg5070_10m.tif'
    }
    
    all_results = {}
    
    # Process each region
    for region_name, cdl_file in regions.items():
        if Path(cdl_file).exists():
            counts, total = get_crop_statistics(cdl_file, mapping)
            all_results[region_name] = (counts, total)
        else:
            print(f"[FAIL] File not found: {cdl_file}")
    
    # Comparison table
    if all_results:
        print("\n\n" + "="*120)
        print("CROP DISTRIBUTION COMPARISON - ALL REGIONS")
        print("="*120)
        print(f"{'Class':<7} {'Crop Type':<27} {'NorthCA %':<15} {'CentCA %':<15} {'SouthCA %':<15}")
        print("-"*120)
        
        crop_names = {
            1: "Natural Vegetation", 2: "Forest", 3: "Corn", 4: "Soybeans",
            5: "Wetlands", 6: "Developed/Barren", 7: "Open Water", 8: "Winter Wheat",
            9: "Alfalfa", 10: "Fallow/Idle", 11: "Cotton", 12: "Sorghum", 13: "Other"
        }
        
        for cls in range(1, 14):
            crop_name = crop_names[cls]
            
            percentages = []
            for region in ['NorthCA', 'CentCA', 'SouthCA']:
                if region in all_results:
                    counts, total = all_results[region]
                    pct = 100 * counts[cls] / total if total > 0 else 0
                    percentages.append(f"{pct:.2f}%")
                else:
                    percentages.append("N/A")
            
            print(f"{cls:<7} {crop_name:<27} {percentages[0]:>13}  {percentages[1]:>13}  {percentages[2]:>13}")
        
        print("="*120)