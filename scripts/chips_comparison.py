import os
import re
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

PRITHVI_CHIPS_DIR = r"C:\MS_Research\data\multi_temporal_crop_segmentation\SouthCA"
SPECTRALGPT_CHIPS_DIR = r"C:\MS_Research\SpectralGPT_chips_multitemporal\SouthCA\ImageSets"
PRITHVI_PREDS_DIR = r"C:\MS_Research\predictions\prithvi_southCA"
SPECTRALGPT_PREDS_DIR = r"C:\MS_Research\predictions\spectralgpt_southCA"
OUTPUT_DIR = r"C:\MS_Research\comparisons\southCA"

PRITHVI_CHIP_SIZE = 224
SPECTRALGPT_CHIP_SIZE = 128

# ============================================================================
# CLASS COLORS AND NAMES
# ============================================================================

CLASS_NAMES = [
    'Natural Vegetation', 'Forest', 'Corn', 'Soybeans', 'Wetlands',
    'Developed/Barren', 'Open Water', 'Winter Wheat', 'Alfalfa',
    'Fallow/Idle', 'Cotton', 'Sorghum', 'Other'
]

CLASS_COLORS = [
    [0, 128, 0],     # Natural Vegetation - green
    [0, 64, 0],      # Forest - dark green
    [255, 215, 0],   # Corn - yellow
    [0, 191, 255],   # Soybeans - deep sky blue
    [64, 224, 208],  # Wetlands - turquoise
    [128, 128, 128], # Developed/Barren - gray
    [0, 0, 255],     # Open Water - blue
    [210, 180, 140], # Winter Wheat - tan
    [144, 238, 144], # Alfalfa - light green
    [210, 105, 30],  # Fallow/Idle - chocolate
    [255, 255, 255], # Cotton - white
    [165, 42, 42],   # Sorghum - brown
    [200, 200, 200], # Other - light gray
]

def label_to_rgb(label_array):
    """Convert class label array to RGB image"""
    h, w = label_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        mask = label_array == class_id
        rgb[mask] = color
    return rgb

# ============================================================================
# PARSE ROW/COL FROM FILENAME
# ============================================================================

def parse_row_col(filename):
    """Extract row and col from chip filename like chip_224_1792"""
    stem = Path(filename).stem
    # Remove suffixes like _merged, _mask
    stem = stem.replace('_merged', '').replace('_mask', '')
    match = re.search(r'chip_(\d+)_(\d+)', stem)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

# ============================================================================
# FIND MATCHING SPECTRALGPT CHIPS FOR A PRITHVI CHIP
# ============================================================================

def find_matching_spectralgpt_chips(prithvi_row, prithvi_col, sgpt_chip_list):
    """
    Find SpectralGPT chips that fall completely inside a Prithvi chip region.
    
    Prithvi chip covers: rows [prithvi_row, prithvi_row+224]
                         cols [prithvi_col, prithvi_col+224]
    
    SpectralGPT chip at (r,c) covers: rows [r, r+128], cols [c, c+128]
    It fits inside if:
        r >= prithvi_row AND r+128 <= prithvi_row+224
        c >= prithvi_col AND c+128 <= prithvi_col+224
    """
    matches = []
    for chip_name in sgpt_chip_list:
        r, c = parse_row_col(chip_name)
        if r is None:
            continue
        # Check if SpectralGPT chip falls inside Prithvi chip
        if (r >= prithvi_row and r + SPECTRALGPT_CHIP_SIZE <= prithvi_row + PRITHVI_CHIP_SIZE and
            c >= prithvi_col and c + SPECTRALGPT_CHIP_SIZE <= prithvi_col + PRITHVI_CHIP_SIZE):
            matches.append((r, c, chip_name))
    
    return matches

# ============================================================================
# LOAD PREDICTION
# ============================================================================

def load_prediction(pred_path):
    """Load prediction from .npy file"""
    return np.load(pred_path)

def load_mask(mask_path):
    """Load ground truth mask from .tif file"""
    with rasterio.open(mask_path) as src:
        return src.read(1)

# ============================================================================
# CROP PRITHVI PREDICTION TO MATCH SPECTRALGPT CHIP REGION
# ============================================================================

def crop_prithvi_to_sgpt(prithvi_pred, prithvi_row, prithvi_col, sgpt_row, sgpt_col):
    """
    Crop the Prithvi 224x224 prediction to match the SpectralGPT 128x128 region.
    
    Local offset within Prithvi chip:
        row_offset = sgpt_row - prithvi_row
        col_offset = sgpt_col - prithvi_col
    """
    row_offset = sgpt_row - prithvi_row
    col_offset = sgpt_col - prithvi_col
    
    cropped = prithvi_pred[
        row_offset : row_offset + SPECTRALGPT_CHIP_SIZE,
        col_offset : col_offset + SPECTRALGPT_CHIP_SIZE
    ]
    return cropped

# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of Prithvi chips
    prithvi_masks = [f for f in os.listdir(PRITHVI_CHIPS_DIR) if f.endswith('.mask.tif')]
    prithvi_preds = [f for f in os.listdir(PRITHVI_PREDS_DIR) if f.endswith('.npy')]
    
    # Get list of SpectralGPT chips (image files, not masks)
    sgpt_chips = [f for f in os.listdir(SPECTRALGPT_CHIPS_DIR) 
                  if f.endswith('.tif') and '_mask' not in f]
    sgpt_preds = [f for f in os.listdir(SPECTRALGPT_PREDS_DIR) if f.endswith('.npy')]
    
    print(f"Prithvi chips: {len(prithvi_masks)}")
    print(f"SpectralGPT chips: {len(sgpt_chips)}")
    print(f"Prithvi predictions: {len(prithvi_preds)}")
    print(f"SpectralGPT predictions: {len(sgpt_preds)}")
    
    comparison_count = 0
    
    for pred_file in prithvi_preds:
        p_row, p_col = parse_row_col(pred_file)
        if p_row is None:
            continue
        
        print(f"\nPrithvi chip: chip_{p_row}_{p_col} (rows {p_row}-{p_row+224}, cols {p_col}-{p_col+224})")
        
        # Find matching SpectralGPT chips
        matches = find_matching_spectralgpt_chips(p_row, p_col, sgpt_preds)
        
        if not matches:
            print(f"  No matching SpectralGPT chips found")
            continue
        
        print(f"  Found {len(matches)} matching SpectralGPT chips")
        
        # Load Prithvi prediction
        prithvi_pred_path = os.path.join(PRITHVI_PREDS_DIR, pred_file)
        prithvi_pred = load_prediction(prithvi_pred_path)
        
        # Load Prithvi ground truth mask
        mask_file = f"chip_{p_row}_{p_col}.mask.tif"
        mask_path = os.path.join(PRITHVI_CHIPS_DIR, mask_file)
        if not os.path.exists(mask_path):
            print(f"  Ground truth mask not found: {mask_path}")
            continue
        gt_full = load_mask(mask_path)
        
        # For each matching SpectralGPT chip, create a comparison figure
        for s_row, s_col, sgpt_pred_file in matches:
            print(f"  Comparing with SpectralGPT chip_{s_row}_{s_col}")
            
            # Load SpectralGPT prediction
            sgpt_pred_path = os.path.join(SPECTRALGPT_PREDS_DIR, sgpt_pred_file)
            if not os.path.exists(sgpt_pred_path):
                print(f"    SpectralGPT prediction not found: {sgpt_pred_path}")
                continue
            sgpt_pred = load_prediction(sgpt_pred_path)
            
            # Crop Prithvi prediction and GT to match SpectralGPT region
            prithvi_cropped = crop_prithvi_to_sgpt(prithvi_pred, p_row, p_col, s_row, s_col)
            gt_cropped = crop_prithvi_to_sgpt(gt_full, p_row, p_col, s_row, s_col)
            
            # Convert to RGB
            gt_rgb = label_to_rgb(gt_cropped)
            prithvi_rgb = label_to_rgb(prithvi_cropped)
            sgpt_rgb = label_to_rgb(sgpt_pred)
            
            # Create comparison figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(gt_rgb)
            axes[0].set_title('Ground Truth (CDL)', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(prithvi_rgb)
            axes[1].set_title('Prithvi Prediction', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(sgpt_rgb)
            axes[2].set_title('SpectralGPT Prediction', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Add legend
            legend_patches = [
                mpatches.Patch(color=[c/255 for c in color], label=name)
                for name, color in zip(CLASS_NAMES, CLASS_COLORS)
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=7, 
                      fontsize=8, bbox_to_anchor=(0.5, -0.05))
            
            plt.suptitle(
                f'SouthCA - Region: rows {s_row}-{s_row+128}, cols {s_col}-{s_col+128}',
                fontsize=12
            )
            plt.tight_layout()
            
            # Save
            out_name = f"comparison_r{s_row}_c{s_col}.png"
            out_path = output_dir / out_name
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Saved: {out_path}")
            comparison_count += 1
    
    print(f"\nDone! Generated {comparison_count} comparison figures in {OUTPUT_DIR}")
    
    # Also print a summary of all available matches
    print("\n" + "="*60)
    print("SUMMARY: Available Prithvi-SpectralGPT chip matches")
    print("="*60)
    for pred_file in prithvi_preds:
        p_row, p_col = parse_row_col(pred_file)
        if p_row is None:
            continue
        matches = find_matching_spectralgpt_chips(p_row, p_col, sgpt_preds)
        print(f"Prithvi chip_{p_row}_{p_col} → {len(matches)} SpectralGPT matches")
        for s_row, s_col, name in matches:
            print(f"    SpectralGPT chip_{s_row}_{s_col}")


if __name__ == "__main__":
    run_comparison()