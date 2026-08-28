"""
dataset_cd_satmae.py
--------------------
Change Detection Dataset for SatMAE benchmark.

Identical structure to dataset_cd_prithvi.py with SatMAE-specific
normalization: z-score with SatMAE paper stats (Table 10, Appendix A.2.2).

For CD we have 6 bands per date (B02, B03, B04, B8A, B11, B12)
so we use the 6-band stats directly (not repeated 3x as in segmentation).

Input chips  : 6-band int16 GeoTIFFs, 96x96 (SatMAE chip size)
Mask         : 1-band uint8, values 0=unchanged, 1=changed, 255=nodata

Place this file at:
    SatMAE/ChangeDetection/dataset_cd_satmae.py
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import rasterio
import pandas as pd


# ============================================================================
# SatMAE normalization stats -- Table 10 from SatMAE paper (Appendix A.2.2)
# 6 bands: B02, B03, B04, B8A, B11, B12
# Same stats for T1 and T2 (sensor-level stats, not year-specific)
# ============================================================================

SATMAE_MEAN = np.array([
    1184.3824625,    # B02
    1120.77120066,   # B03
    1136.26026392,   # B04
    1972.62420416,   # B8A
    1732.16362238,   # B11
    1247.91870117,   # B12
], dtype=np.float32)

SATMAE_STD = np.array([
    650.2842772,     # B02
    712.12507725,    # B03
    965.23119807,    # B04
    1364.38688993,   # B8A
    1310.36996126,   # B11
    1087.6020813,    # B12
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
    means : (C,)
    stds  : (C,)
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
    if random.random() > 0.5:
        t1   = t1[:, :, ::-1].copy()
        t2   = t2[:, :, ::-1].copy()
        mask = mask[:, ::-1].copy()

    if random.random() > 0.5:
        t1   = t1[:, ::-1, :].copy()
        t2   = t2[:, ::-1, :].copy()
        mask = mask[::-1, :].copy()

    k = random.randint(0, 3)
    if k > 0:
        t1   = np.rot90(t1,   k, axes=(1, 2)).copy()
        t2   = np.rot90(t2,   k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

    return t1, t2, mask


# ============================================================================
# Dataset
# ============================================================================

class CDDatasetSatMAE(data.Dataset):
    """
    Change Detection dataset for SatMAE.

    Args:
        csv_paths : list of CSV manifest paths (one per location).
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

        print(f"Loaded {len(self.samples):,} SatMAE CD chip triplets "
              f"({'train' if training else 'val/test'})")

        self.weights = self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        """NLLLoss class weights with FP_MODIFIER=10."""
        FP_MODIFIER = 1   # balanced data ~60% changed pixels (OSCD used 10 for <5%)
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
        t1 = _load_tif(t1_path)    # (6, 96, 96)
        t2 = _load_tif(t2_path)    # (6, 96, 96)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)   # (96, 96)

        # replace nodata with band mean before normalizing
        for b in range(t1.shape[0]):
            t1[b] = np.where(t1[b] == -9999, SATMAE_MEAN[b], t1[b])
            t2[b] = np.where(t2[b] == -9999, SATMAE_MEAN[b], t2[b])

        # z-score normalization with SatMAE paper stats
        t1 = _normalize_zscore(t1, SATMAE_MEAN, SATMAE_STD)
        t2 = _normalize_zscore(t2, SATMAE_MEAN, SATMAE_STD)

        # joint augmentation (training only)
        if self.training:
            t1, t2, mask = _joint_augment(t1, t2, mask)

        # to tensors
        t1   = torch.from_numpy(t1.copy()).float()
        t2   = torch.from_numpy(t2.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        return {'t1': t1, 't2': t2, 'mask': mask}

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        t1s   = torch.stack([b['t1']   for b in batch])
        t2s   = torch.stack([b['t2']   for b in batch])
        masks = torch.stack([b['mask'] for b in batch])
        return {'t1': t1s, 't2': t2s, 'mask': masks}