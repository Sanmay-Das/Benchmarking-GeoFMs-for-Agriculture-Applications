import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import sys
from pathlib import Path
import rasterio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# DON'T load config yet - manually set the data root
data_root = r"C:\MS_Research\data\multi_temporal_crop_segmentation\training"

class SimpleGeoDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split='training'):
        self.data_root = Path(data_root)
        self.chip_dir = self.data_root / f"{split}_chips"
        self.image_files = sorted(list(self.chip_dir.glob("*_merged.tif")))
        print(f"Found {len(self.image_files)} {split} images")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = str(img_path).replace('_merged.tif', '.mask.tif')
        
        with rasterio.open(img_path) as src:
            img = src.read()  # (18, H, W)
            
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # (H, W)
            
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        
        # Reshape to (6, 3, H, W)
        img = img.reshape(6, 3, img.shape[1], img.shape[2])
        
        return img, mask

print("Testing dataset loading...")
train_dataset = SimpleGeoDataset(data_root, 'training')
val_dataset = SimpleGeoDataset(data_root, 'validation')

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

if len(train_dataset) > 0:
    img, mask = train_dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {torch.unique(mask)}")
    print("\nSuccess! Data loads correctly.")
else:
    print("ERROR: No training data found!")