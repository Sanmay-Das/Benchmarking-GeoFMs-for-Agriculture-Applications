"""
dataset_cd_prithvi.py
---------------------
Change Detection Dataset for Prithvi benchmark.

Identical structure to dataset_cd.py (SpectralGPT) with one key difference:
    Normalization: z-score with global mean/std (Iowa stats)
    instead of per-image min-max.

This matches Prithvi's segmentation config (TorchNormalize with means/stds).

Input chips  : 6-band int16 GeoTIFFs, 224x224 (Prithvi chip size)
Mask         : 1-band uint8, values 0=unchanged, 1=changed, 255=nodata

Iowa stats computed from CentIA training chips (6 bands x 1 date):
    Bands: B02, B03, B04, B8A, B11, B12

Place this file at:
    downstream_tasks/ChangeDetection/dataset_cd_prithvi.py
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import rasterio
import pandas as pd


# ============================================================================
# Iowa normalization stats for Prithvi CD
# 6 bands: B02, B03, B04, B8A, B11, B12
# Computed from CentIA 2023 (T1) and 2024 (T2) single-date chips
# Using the same IA mean/std values from your segmentation config
# (first 6 bands of T1 timestep)
# ============================================================================

# T1 (2023) stats -- first 6 bands from your IA segmentation config
T1_MEANS = np.array([
    1861.19006065,  # B02
    2033.17032775,  # B03
    2273.37933660,  # B04
    3262.91588412,  # B8A
    4457.44718789,  # B11
    3994.99188433,  # B12
], dtype=np.float32)

T1_STDS = np.array([
    307.48006869,   # B02
    351.67526808,   # B03
    447.86017086,   # B04
    654.81295100,   # B8A
    697.75477528,   # B11
    788.08159599,   # B12
], dtype=np.float32)

# T2 (2024) stats -- second 6 bands from your IA segmentation config
T2_MEANS = np.array([
    1704.63697798,  # B02
    1961.37926168,  # B03
    2034.19159947,  # B04
    3929.94252524,  # B8A
    4352.22479367,  # B11
    3695.04113396,  # B12
], dtype=np.float32)

T2_STDS = np.array([
    341.76676414,   # B02
    357.44447699,   # B03
    513.43405597,   # B04
    842.65097823,   # B8A
    920.07998270,   # B11
    1054.49693239,  # B12
], dtype=np.float32)


# ============================================================================
# Helpers
# ============================================================================

def _load_tif(path: str) -> np.ndarray:
    """Read a GeoTIFF and return float32 numpy array (C, H, W)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def _normalize_zscore(img: np.ndarray,
                       means: np.ndarray,
                       stds: np.ndarray) -> np.ndarray:
    """
    Per-band z-score normalization.
    img   : (C, H, W) float32
    means : (C,) float32
    stds  : (C,) float32
    Returns (C, H, W) float32
    """
    means = means.reshape(-1, 1, 1)
    stds  = stds.reshape(-1, 1, 1)
    return (img - means) / (stds + 1e-8)


def _joint_augment(t1, t2, mask):
    """
    Apply identical random spatial augmentation to T1, T2, and mask.
    t1, t2 : (C, H, W) numpy float32
    mask   : (H, W)    numpy int64
    """
    # Random horizontal flip
    if random.random() > 0.5:
        t1   = t1[:, :, ::-1].copy()
        t2   = t2[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()

    # Random vertical flip
    if random.random() > 0.5:
        t1   = t1[:, ::-1, :].copy()
        t2   = t2[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()

    # Random 90 degree rotation
    k = random.randint(0, 3)
    if k > 0:
        t1   = np.rot90(t1,   k, axes=(1, 2)).copy()
        t2   = np.rot90(t2,   k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

    return t1, t2, mask


# ============================================================================
# Dataset
# ============================================================================

class CDDatasetPrithvi(data.Dataset):
    """
    Change Detection dataset for Prithvi.

    Args:
        csv_paths : list of CSV manifest paths (one per location).
                    Each CSV has columns:
                        location, row, col, t1, t2, mask, change_pct
        training  : if True, apply joint random augmentation.
    """

    def __init__(self, csv_paths: list, training: bool = False):
        super().__init__()

        self.training = training
        self.samples  = []

        for csv_path in csv_paths:
            assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append((
                    str(row['t1']),
                    str(row['t2']),
                    str(row['mask']),
                ))

        print(f"Loaded {len(self.samples):,} Prithvi CD chip triplets "
              f"({'train' if training else 'val/test'})")

        self.weights = self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        """
        Compute NLLLoss class weights from pixel counts.
        FP_MODIFIER=1 for balanced data (~60% changed pixels).
        OSCD used FP_MODIFIER=10 but that was for rare change (<5%).
        """
        FP_MODIFIER = 1
        n_pix    = 0
        true_pix = 0

        print("Computing class weights from masks...")
        for _, _, mask_path in self.samples:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.int32)
            valid    = (mask != 255)
            n_pix    += int(valid.sum())
            true_pix += int((mask == 1).sum())

        if n_pix == 0:
            return [1.0, 1.0]

        w_changed   = FP_MODIFIER * 2 * true_pix / n_pix
        w_unchanged = 2 * (n_pix - true_pix) / n_pix
        print(f"  changed pixels  : {true_pix:,} ({true_pix/n_pix*100:.1f}%)")
        print(f"  unchanged pixels: {n_pix-true_pix:,} "
              f"({(n_pix-true_pix)/n_pix*100:.1f}%)")
        print(f"  weights -> unchanged={w_unchanged:.4f}, "
              f"changed={w_changed:.4f}")
        return [w_unchanged, w_changed]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        t1_path, t2_path, mask_path = self.samples[idx]

        # load
        t1 = _load_tif(t1_path)    # (6, 224, 224) int16 -> float32
        t2 = _load_tif(t2_path)    # (6, 224, 224)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)   # (224, 224)

        # replace nodata with mean value before normalizing
        for b in range(t1.shape[0]):
            t1[b] = np.where(t1[b] == -9999, T1_MEANS[b], t1[b])
            t2[b] = np.where(t2[b] == -9999, T2_MEANS[b], t2[b])

        # z-score normalization with Iowa stats
        t1 = _normalize_zscore(t1, T1_MEANS, T1_STDS)   # (6, 224, 224)
        t2 = _normalize_zscore(t2, T2_MEANS, T2_STDS)   # (6, 224, 224)

        # joint augmentation (training only)
        if self.training:
            t1, t2, mask = _joint_augment(t1, t2, mask)

        # to tensors
        t1   = torch.from_numpy(t1.copy()).float()     # (6, 224, 224)
        t2   = torch.from_numpy(t2.copy()).float()     # (6, 224, 224)
        mask = torch.from_numpy(mask.copy()).long()    # (224, 224)

        return {'t1': t1, 't2': t2, 'mask': mask}

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        t1s   = torch.stack([b['t1']   for b in batch])   # (B, 6, 224, 224)
        t2s   = torch.stack([b['t2']   for b in batch])   # (B, 6, 224, 224)
        masks = torch.stack([b['mask'] for b in batch])   # (B, 224, 224)
        return {'t1': t1s, 't2': t2s, 'mask': masks}