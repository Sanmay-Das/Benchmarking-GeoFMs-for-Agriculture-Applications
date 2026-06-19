"""
SatMAE + FCNHead  —  Sliding-window inference on NWIA multitemporal stack
Follows the same TerraTorch tiled-inference protocol as Prithvi / SpectralGPT.
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SATMAE_DIR = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE'
sys.path.insert(0, SATMAE_DIR)

import models_vit_group_channels
from models_satmae_fcn import SatMAEFCN

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT  = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE/output_seg_Iowa_fcn/checkpoint-best.pth'
STACK_PATH  = '/bigdata/eldawylab/sdas050/MS_Research/scripts/processed_stacks/NWIA/NWIA_multitemporal_stack.tif'
OUTPUT_DIR  = '/bigdata/eldawylab/sdas050/MS_Research/predictions/satmae_fcn_NWIA'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'NWIA_SatMAE_FCN_Prediction.tif')

CHIP_SIZE   = 96
STRIDE      = 48
DELTA       = 8
NUM_CLASSES = 14
BATCH_SIZE  = 32
NODATA_VAL  = -9999

_MEAN_6 = np.array([
    1184.3824625, 1120.77120066, 1136.26026392,
    1972.62420416, 1732.16362238, 1247.91870117,
], dtype=np.float32)
_STD_6 = np.array([
    650.2842772,  712.12507725,  965.23119807,
    1364.38688993, 1310.36996126, 1087.6020813,
], dtype=np.float32)

SATMAE_MEAN = np.tile(_MEAN_6, 3).reshape(18, 1, 1)
SATMAE_STD  = np.tile(_STD_6,  3).reshape(18, 1, 1)

GROUPED_BANDS = [[0,1,2,3,4,5], [6,7,8,9,10,11], [12,13,14,15,16,17]]
PATCH_SIZE    = 8
INPUT_SIZE    = 96
IN_CHANS      = 18
DROP_PATH     = 0.2


def cosine_blend_mask(chip_size, stride, delta):
    size     = chip_size - 2 * delta
    overlap  = chip_size - stride
    ramp_len = max(min(chip_size // 2, overlap) - delta, 0)
    x = torch.ones(size, dtype=torch.float32)
    if ramp_len > 0:
        ramp = torch.cos(math.pi * (torch.arange(ramp_len, dtype=torch.float32) + 1)
                         / (ramp_len + 1)) / 2 + 0.5
        x[:ramp_len] = ramp.flip(0)
        x[-ramp_len:] = ramp
    mask = x[:, None] * x[None, :]
    return mask + 1e-6


class SlidingWindowDataset(Dataset):
    def __init__(self, stack_path, chip_size, stride, delta):
        with rasterio.open(stack_path) as src:
            self.profile = src.profile.copy()
            self.orig_H  = src.height
            self.orig_W  = src.width
            data         = src.read().astype(np.float32)

        self.nodata_mask = np.any(data == NODATA_VAL, axis=0)
        data[data == NODATA_VAL] = 0.0

        self.pad  = chip_size // 2
        self.data = np.pad(data,
                           ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                           mode='reflect')

        self.chip_size = chip_size
        self.stride    = stride
        self.delta     = delta
        self.H         = self.data.shape[1]
        self.W         = self.data.shape[2]

        y_steps = list(range(0, self.H - chip_size + 1, stride))
        x_steps = list(range(0, self.W - chip_size + 1, stride))
        if not y_steps or y_steps[-1] + chip_size < self.H:
            y_steps.append(self.H - chip_size)
        if not x_steps or x_steps[-1] + chip_size < self.W:
            x_steps.append(self.W - chip_size)
        self.windows = [(y, x) for y in y_steps for x in x_steps]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y, x = self.windows[idx]
        chip = self.data[:, y:y+self.chip_size, x:x+self.chip_size].copy()
        chip = (chip - SATMAE_MEAN) / (SATMAE_STD + 1e-8)
        return torch.from_numpy(chip).float(), y, x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\nBuilding SatMAE + FCNHead model...")
    encoder = models_vit_group_channels.vit_large_patch16(
        patch_size    = PATCH_SIZE,
        img_size      = INPUT_SIZE,
        in_chans      = IN_CHANS,
        channel_groups= GROUPED_BANDS,
        num_classes   = 0,
        drop_path_rate= DROP_PATH,
        global_pool   = False,
    )
    model = SatMAEFCN(encoder=encoder, nb_classes=NUM_CLASSES)

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    print(f"  Loaded epoch: {ckpt.get('epoch', 'best')}")
    model = model.to(device)
    model.eval()

    dataset    = SlidingWindowDataset(STACK_PATH, CHIP_SIZE, STRIDE, DELTA)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    H, W       = dataset.orig_H, dataset.orig_W
    pad        = dataset.pad
    prob_accum = torch.zeros((NUM_CLASSES, H, W), dtype=torch.float32)
    count_map  = torch.zeros((1, H, W),           dtype=torch.float32)
    blend_mask = cosine_blend_mask(CHIP_SIZE, STRIDE, DELTA)
    inner_size = CHIP_SIZE - 2 * DELTA

    print(f"\nStack: {H}×{W} | Padded: {dataset.H}×{dataset.W} | Windows: {len(dataset)}")
    print(f"Chip: {CHIP_SIZE}×{CHIP_SIZE} | Stride: {STRIDE} | Delta: {DELTA}\n")

    start = time.time()
    with torch.no_grad():
        for chips, ys, xs in tqdm(dataloader, desc="SatMAE FCN Inference"):
            chips     = chips.to(device)
            batch_len = chips.size(0)

            logits = model(chips, is_train=False)        # (B, 14, 96, 96)
            probs  = F.softmax(logits, dim=1).cpu()

            for i in range(batch_len):
                y_pad, x_pad = ys[i].item(), xs[i].item()
                probs_inner  = probs[i][:, DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA]

                y_orig = y_pad - pad + DELTA
                x_orig = x_pad - pad + DELTA

                y0 = max(y_orig, 0);  y1 = min(y_orig + inner_size, H)
                x0 = max(x_orig, 0);  x1 = min(x_orig + inner_size, W)

                py0 = y0 - y_orig;  py1 = py0 + (y1 - y0)
                px0 = x0 - x_orig;  px1 = px0 + (x1 - x0)

                bm = blend_mask[py0:py1, px0:px1]
                prob_accum[:, y0:y1, x0:x1] += probs_inner[:, py0:py1, px0:px1] * bm
                count_map[:,  y0:y1, x0:x1] += bm

    print(f"\nInference done in {time.time()-start:.1f}s")

    count_map[count_map == 0] = 1
    prob_accum /= count_map
    pred = torch.argmax(prob_accum, dim=0).numpy().astype(np.uint8)
    pred[dataset.nodata_mask] = 255

    print(f"Predicted classes: {np.unique(pred)}")

    profile = dataset.profile
    profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw', 'nodata': 255})
    with rasterio.open(OUTPUT_FILE, 'w', **profile) as dst:
        dst.write(pred, 1)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Output size: {pred.shape}  —  matches original stack: {H}×{W}")


if __name__ == '__main__':
    main()
