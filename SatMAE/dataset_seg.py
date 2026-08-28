"""
SatMAE Crop Segmentation Dataset
Reads 96x96 chips (18-band Int16 GeoTIFF) + mask (Uint8 GeoTIFF)
Applies SatMAE normalization during __getitem__
Labels: 0=NoData (ignored in loss), 1-13=crop classes

Your directory structure:
    SatMAE_chips_multitemporal/
        CentIA/          <- train chips (actual .tif files)
        EastIA/          <- val chips   (actual .tif files)
        NWIA/            <- test chips  (actual .tif files)
        Iowa/
            train.txt    <- chip stems from CentIA
            val.txt      <- chip stems from EastIA
            test.txt     <- chip stems from NWIA

Pass data_path = SatMAE_chips_multitemporal/
The dataset scans ALL subdirectories to find each chip by name.
No copying needed.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from pathlib import Path


# ============================================================================
# SatMAE normalization stats -- Table 10 from SatMAE paper (Appendix A.2.2)
# 6 bands per timestep: B02, B03, B04, B8A, B11, B12
# Repeated 3x for 3 timesteps -> 18 values total
# ============================================================================

_MEAN_6 = np.array([
    1184.3824625,   # B02
    1120.77120066,  # B03
    1136.26026392,  # B04
    1972.62420416,  # B8A
    1732.16362238,  # B11
    1247.91870117,  # B12
], dtype=np.float32)

_STD_6 = np.array([
    650.2842772,    # B02
    712.12507725,   # B03
    965.23119807,   # B04
    1364.38688993,  # B8A
    1310.36996126,  # B11
    1087.6020813,   # B12
], dtype=np.float32)

# Repeat for 3 timesteps -> 18 values
SATMAE_MEAN = np.tile(_MEAN_6, 3).astype(np.float32)  # (18,)
SATMAE_STD  = np.tile(_STD_6,  3).astype(np.float32)  # (18,)

IGNORE_INDEX = 0


def build_chip_index(root_dir: Path) -> dict:
    """
    Recursively scan root_dir and all subdirectories.
    Returns dict: chip_stem -> full Path to image .tif

    e.g. {'chip_0_0': Path('.../CentIA/chip_0_0.tif'), ...}

    Mask files (_mask.tif) are excluded from index --
    mask path is derived from image path at load time.
    """
    index = {}
    duplicates = 0
    for dirpath, _, files in os.walk(str(root_dir), followlinks=True):
        for fname in files:
            if fname.startswith("chip_") and fname.endswith(".tif") and "_mask" not in fname:
                tif = Path(dirpath) / fname
                stem = tif.stem
                if stem not in index:
                    index[stem] = tif
                else:
                    duplicates += 1
    if duplicates:
        print(f"  WARNING: {duplicates} duplicate chip names found across subdirectories -- kept first occurrence of each.")
    return index


class SatMAESegDataset(Dataset):
    """
    Dataset for SatMAE multi-temporal crop segmentation.

    Chips can be in subdirectories -- dataset scans recursively
    so no need to copy/merge chips into one flat directory.

    Args:
        data_dir:   root dir to scan for chips (e.g. SatMAE_chips_multitemporal/)
        split_file: .txt file listing chip stems for this split
        augment:    random horizontal flip if True (train split only)
    """

    def __init__(self, data_dir: str, split_file: str, augment: bool = False):
        self.data_dir = Path(data_dir)
        self.augment  = augment

        # Read chip names from split file
        with open(split_file, 'r') as f:
            self.chip_names = [line.strip() for line in f if line.strip()]

        print(f"SatMAESegDataset loaded: {len(self.chip_names)} chips")
        print(f"  Split file:  {split_file}")
        print(f"  Scanning:    {self.data_dir}")

        # Build chip index by scanning all subdirectories
        self.chip_index = build_chip_index(self.data_dir)
        print(f"  Total chips found in index: {len(self.chip_index)}")

        # Verify all chips in split are found
        missing = [n for n in self.chip_names if n not in self.chip_index]
        if missing:
            print(f"  WARNING: {len(missing)} chips missing from index!")
            print(f"  First 5: {missing[:5]}")
        else:
            print(f"  All {len(self.chip_names)} chips found.")

    def __len__(self):
        return len(self.chip_names)

    def __getitem__(self, idx):
        chip_name = self.chip_names[idx]

        # Resolve paths from index
        if chip_name not in self.chip_index:
            # Return a zero sample for missing chips -- will be ignored by ignore_index=0
            image = np.zeros((18, 96, 96), dtype=np.float32)
            mask  = np.zeros((96, 96), dtype=np.int64)
            return torch.from_numpy(image), torch.from_numpy(mask)
        img_path  = self.chip_index[chip_name]
        mask_path = img_path.parent / f"{chip_name}_mask.tif"

        # Load image chip (18, 96, 96) -- raw Int16
        with rasterio.open(img_path) as src:
            image = src.read().astype(np.float32)

        # Load mask (96, 96) -- Uint8 labels 0-13
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)  # (96, 96)

        # Clamp to valid range -- any label >13 gets mapped to 0 (NoData/ignore)
        mask = np.where(mask > 13, 0, mask)
        mask = np.where(mask < 0, 0, mask)

        # Replace nodata (-9999) with 0 before normalizing
        image = np.where(image == -9999, 0.0, image)

        # SatMAE z-score normalization -- paper Table 10 stats
        image = (image - SATMAE_MEAN[:, None, None]) / (SATMAE_STD[:, None, None] + 1e-8)

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()  # (18, 96, 96)
        mask_tensor  = torch.from_numpy(mask).long()    # (96, 96)

        # Augmentation -- random horizontal flip for train split
        if self.augment and torch.rand(1).item() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[-1])
            mask_tensor  = torch.flip(mask_tensor,  dims=[-1])

        return image_tensor, mask_tensor

    @property
    def num_classes(self):
        return 13

    @property
    def in_c(self):
        return 18


def build_seg_dataset(is_train: bool, args) -> SatMAESegDataset:
    """
    Builds SatMAESegDataset from args.

    args.data_path  -- root dir to scan for chips
                      (e.g. SatMAE_chips_multitemporal/ -- parent of CentIA, EastIA etc.)
    args.train_path -- path to train.txt (e.g. Iowa/train.txt)
    args.test_path  -- path to val.txt   (e.g. Iowa/val.txt)
    """
    split_file = args.train_path if is_train else args.test_path
    dataset    = SatMAESegDataset(
        data_dir   = args.data_path,
        split_file = split_file,
        augment    = is_train,
    )
    return dataset