"""
dataset_cd.py
-------------
Change Detection Dataset for SpectralGPT / Prithvi / SatMAE benchmark.

Reads pre-chipped triplets (T1, T2, change mask) from a CSV manifest.

Normalization:
    Per-image min-max to [0, 1] per band -- exactly as SpectralGPT's
    OSCD dataset.py does for their change detection experiments.

Augmentation (training only):
    Random horizontal flip, vertical flip, and 90 degree rotation
    applied JOINTLY to T1, T2, and mask so spatial correspondence
    is preserved.

Mask convention (from chip generation):
    0   = unchanged
    1   = changed
    255 = nodata / ignore

Place this file at:
    downstream_tasks/ChangeDetection/dataset_cd.py
"""

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import rasterio
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tif(path: str) -> np.ndarray:
    """Read a GeoTIFF and return a float32 numpy array (C, H, W)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def _normalize_per_image(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-band min-max normalization to [0, 1].
    Matches SpectralGPT OSCD dataset.py normalization exactly.
    img: (C, H, W) float32
    Returns (C, H, W) float32 in [0, 1].
    """
    mn = img.min(axis=(1, 2), keepdims=True)   # (C, 1, 1)
    mx = img.max(axis=(1, 2), keepdims=True)   # (C, 1, 1)
    return (img - mn) / (mx - mn + eps)


def _joint_augment(t1, t2, mask):
    """
    Apply identical random spatial augmentation to T1, T2, and mask.
    t1, t2 : (C, H, W) numpy float32
    mask   : (H, W)    numpy int64
    Returns augmented t1, t2, mask with same shapes.
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

    # Random 90 degree rotation (0, 90, 180, 270)
    k = random.randint(0, 3)
    if k > 0:
        t1   = np.rot90(t1,   k, axes=(1, 2)).copy()
        t2   = np.rot90(t2,   k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

    return t1, t2, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CDDataset(data.Dataset):
    """
    Change Detection dataset.

    Args:
        csv_paths : list of CSV manifest paths (one per location).
                    Each CSV has columns:
                        location, row, col, t1, t2, mask, change_pct
        training  : if True, apply joint random augmentation.
    """

    def __init__(self, csv_paths: list, training: bool = False):
        super().__init__()

        self.training = training
        self.samples  = []   # list of (t1_path, t2_path, mask_path)

        for csv_path in csv_paths:
            assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.samples.append((
                    str(row['t1']),
                    str(row['t2']),
                    str(row['mask']),
                ))

        print(f"Loaded {len(self.samples):,} CD chip triplets "
              f"({'train' if training else 'val/test'})")

        # Compute class weights from all masks in this split
        self.weights = self._compute_weights()

    # ------------------------------------------------------------------
    def _compute_weights(self):
        """
        Compute NLLLoss class weights from pixel counts across all masks.
        weights[0] = weight for unchanged (class 0)
        weights[1] = weight for changed   (class 1)
        Pixels with value 255 are ignored.

        Mirrors SpectralGPT OSCD train.py:
            weights = [FP_MODIFIER * 2 * true_pix / n_pix,
                       2 * (n_pix - true_pix) / n_pix]
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
        t1 = _load_tif(t1_path)    # (6, H, W) int16 -> float32
        t2 = _load_tif(t2_path)    # (6, H, W)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)   # (H, W)

        # replace nodata (-9999) with 0 before normalizing
        t1 = np.where(t1 == -9999, 0.0, t1)
        t2 = np.where(t2 == -9999, 0.0, t2)

        # per-image min-max normalization (SpectralGPT OSCD style)
        t1 = _normalize_per_image(t1)   # (6, H, W) in [0, 1]
        t2 = _normalize_per_image(t2)   # (6, H, W) in [0, 1]

        # joint augmentation (training only)
        if self.training:
            t1, t2, mask = _joint_augment(t1, t2, mask)

        # to tensors
        t1   = torch.from_numpy(t1.copy()).float()     # (6, H, W)
        t2   = torch.from_numpy(t2.copy()).float()     # (6, H, W)
        mask = torch.from_numpy(mask.copy()).long()    # (H, W)

        # clamp after augmentation
        t1 = t1.clamp(0.0, 1.0)
        t2 = t2.clamp(0.0, 1.0)

        return {'t1': t1, 't2': t2, 'mask': mask}

    # ------------------------------------------------------------------
    @staticmethod
    def collate_fn(batch):
        t1s   = torch.stack([b['t1']   for b in batch])   # (B, 6, H, W)
        t2s   = torch.stack([b['t2']   for b in batch])   # (B, 6, H, W)
        masks = torch.stack([b['mask'] for b in batch])   # (B, H, W)
        return {'t1': t1s, 't2': t2s, 'mask': masks}