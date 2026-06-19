import os
from pathlib import Path
import random

def create_train_test_split(chips_dir, train_ratio=0.8):
 """
 Create train.txt and test.txt for SpectralGPT
 Exactly as paper expects
 """
 chips_dir = Path(chips_dir)
 
 # Find all image chips (exclude masks)
 image_files = sorted(chips_dir.glob('chip_*.tif'))
 image_files = [f for f in image_files if '_mask' not in f.name]
 
 print(f"Found {len(image_files)} chips")
 
 # Extract chip names (without extension)
 chip_names = [f.stem for f in image_files]
 
 # Shuffle
 random.seed(42)
 random.shuffle(chip_names)
 
 # Split 80/20
 n_train = int(len(chip_names) * train_ratio)
 train_chips = chip_names[:n_train]
 test_chips = chip_names[n_train:]
 
 print(f"Train: {len(train_chips)} chips ({len(train_chips)/len(chip_names)*100:.1f}%)")
 print(f"Test: {len(test_chips)} chips ({len(test_chips)/len(chip_names)*100:.1f}%)")
 
 # Create ImageSets directory
 imagesets_dir = chips_dir / 'ImageSets'
 imagesets_dir.mkdir(exist_ok=True)
 
 # Write train.txt
 train_txt = imagesets_dir / 'train.txt'
 with open(train_txt, 'w') as f:
 for name in train_chips:
 f.write(f"{name}\n")
 print(f" Saved: {train_txt}")
 
 # Write test.txt
 test_txt = imagesets_dir / 'test.txt'
 with open(test_txt, 'w') as f:
 for name in test_chips:
 f.write(f"{name}\n")
 print(f" Saved: {test_txt}")
 
 return train_chips, test_chips


if __name__ == "__main__":
 
 CHIPS_DIR = r"C:\MS_Research\SpectralGPT_chips\CentIA"
 
 print("="*70)
 print(" CREATING TRAIN/TEST SPLIT")
 print("="*70)
 
 create_train_test_split(CHIPS_DIR, train_ratio=0.8)
 
 print("\n" + "="*70)
 print(" DONE!")
 print("="*70)