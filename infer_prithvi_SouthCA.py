
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS, seg_checkpoint, stack_path

import os
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys

sys.path.insert(0, f'{MSR_ROOT}/prithvi_finetune')
from mmcv import Config
from mmseg.models import build_segmentor
from mmcv.runner import load_checkpoint

# ============================================================
# CONFIGURATION
# ============================================================
# CHECKPOINT   = f'{OUTPUT_ROOT}/experiments/prithvi_multi_temporal_crop_classification/best_mIoU_epoch_20.pth'
CHECKPOINT   = str(seg_checkpoint('prithvi', 'SouthCA', ''))
CONFIG_FILE  = f'{MSR_ROOT}/prithvi_finetune/configs/multi_temporal_crop_classification.py'
STACK_PATH   = str(stack_path('SouthCA'))
OUTPUT_DIR   = f'{PREDICTIONS}/prithvi_SouthCA_terratorch'
OUTPUT_FILE  = os.path.join(OUTPUT_DIR, 'SouthCA_Prithvi_Prediction_Stitched.tif')

CHIP_SIZE    = 224
STRIDE       = 112   # 50% overlap
DELTA        = 8     # TerraTorch: border pixels to discard from each chip
NUM_CLASSES  = 13
BATCH_SIZE   = 16
NODATA_VAL   = -9999

# CentIA z-score normalization stats (used during finetuning)
# MEANS = torch.tensor([
#     1861.15487260, 2033.31513843, 2273.56778324, 3264.31809357, 4457.67865344, 3993.98225713,
#     1704.26900815, 1961.18125080, 2033.85207114, 3931.60397257, 4351.87295767, 3694.09881752,
#     1518.01208966, 1767.57494316, 2015.06161967, 3425.27676236, 3618.00166154, 2742.72472052
# ], dtype=torch.float32).view(18, 1, 1)

# STDS = torch.tensor([
#     307.15351465, 351.41085957, 447.99935412, 655.15274943, 698.52735671, 788.87837039,
#     341.42365808, 357.10833130, 513.54092979, 843.17614847, 921.40904876, 1055.85466537,
#     367.93081322, 422.83202252, 602.21383506, 688.45668426, 779.09201742, 671.24211383
# ], dtype=torch.float32).view(18, 1, 1)

# CA stats
MEANS = torch.tensor([
    1399.18814785, 1624.09021280, 1530.61747529, 3433.65512704, 2625.27257993, 2036.76201361, # Frame 1
    1690.31674406, 1965.86899305, 2231.22908437, 3764.86120507, 3585.95243342, 2720.77963287, # Frame 2
    1547.22463694, 1832.03965290, 2079.88629347, 3402.57135275, 3377.24651308, 2538.78450119  # Frame 3
], dtype=torch.float32).view(18, 1, 1)

STDS = torch.tensor([
    281.63779618, 300.33935069, 375.98413426, 1116.25457000, 684.00131194, 559.83401327, # Frame 1
    382.69301356, 482.81761256, 787.29874717, 1098.04393375, 1231.90817451, 884.90944039, # Frame 2
    349.85626067, 448.42248082, 723.75199174, 1019.93200015, 1201.66504561, 846.55948618  # Frame 3
], dtype=torch.float32).view(18, 1, 1)


# ============================================================
# TERRATORCH: COSINE BLEND MASK
# ============================================================
def cosine_blend_mask(chip_size, stride, delta):
    """
    Cosine-tapered blend mask following TerraTorch tiled_inference protocol.
    Center pixels weighted higher, edge pixels weighted lower.
    Output size: (chip_size - 2*delta, chip_size - 2*delta)
    """
    size = chip_size - 2 * delta
    overlap = chip_size - stride
    ramp_len = min(chip_size // 2, overlap) - delta
    ramp_len = max(ramp_len, 0)

    x = torch.ones(size, dtype=torch.float32)
    if ramp_len > 0:
        ramp = torch.cos(math.pi * (torch.arange(ramp_len, dtype=torch.float32) + 1) / (ramp_len + 1)) / 2 + 0.5
        x[:ramp_len] = ramp.flip(0)   # left/top edge
        x[-ramp_len:] = ramp           # right/bottom edge

    mask = x[:, None] * x[None, :]    # (size, size) outer product
    mask = mask + 1e-6                 # ensure no zero weights
    return mask


# ============================================================
# DATASET  (with reflect padding)
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, stack_path, chip_size, stride, delta):
        with rasterio.open(stack_path) as src:
            self.profile  = src.profile.copy()
            self.orig_H   = src.height
            self.orig_W   = src.width
            data          = src.read().astype(np.float32)  # (18, H, W)

        # Pre-compute NoData mask on ORIGINAL stack (before padding)
        self.nodata_mask = np.any(data == NODATA_VAL, axis=0)  # (H, W)

        # Replace -9999 with 0 before normalization
        data[data == NODATA_VAL] = 0.0

        # TerraTorch: reflect padding at borders (pad = chip_size // 2)
        self.pad      = chip_size // 2
        self.data     = np.pad(data,
                               ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                               mode='reflect')  # (18, H+2*pad, W+2*pad)

        self.chip_size = chip_size
        self.stride    = stride
        self.delta     = delta
        self.H         = self.data.shape[1]
        self.W         = self.data.shape[2]

        # Sliding window steps over padded stack
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
        chip = torch.from_numpy(chip)  # (18, 224, 224)

        # Z-score normalize
        chip = (chip - MEANS) / STDS

        # Reshape to Prithvi format: (6, 3, 224, 224)
        chip = chip.view(3, 6, self.chip_size, self.chip_size)
        chip = chip.permute(1, 0, 2, 3)

        return chip, y, x


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading Prithvi model...")
    cfg   = Config.fromfile(CONFIG_FILE)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, CHECKPOINT, map_location='cpu')
    model = model.to(device)
    model.eval()
    print("Model loaded\n")

    # Dataset + dataloader
    dataset    = SlidingWindowDataset(STACK_PATH, CHIP_SIZE, STRIDE, DELTA)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Accumulation buffers over ORIGINAL (unpadded) size
    H, W       = dataset.orig_H, dataset.orig_W
    pad        = dataset.pad
    prob_accum = torch.zeros((NUM_CLASSES, H, W), dtype=torch.float32)
    count_map  = torch.zeros((1, H, W),           dtype=torch.float32)

    print(f"Original stack: {H}x{W} | Padded: {dataset.H}x{dataset.W} | Windows: {len(dataset)}\n")

    # Precompute cosine blend mask (on CPU, moved to GPU in loop)
    blend_mask = cosine_blend_mask(CHIP_SIZE, STRIDE, DELTA)  # (chip-2delta, chip-2delta)
    inner_size = CHIP_SIZE - 2 * DELTA                         # 208 for delta=8

    start = time.time()
    with torch.no_grad():
        for chips, ys, xs in tqdm(dataloader, desc="Prithvi Inference"):
            chips     = chips.to(device)
            batch_len = chips.size(0)

            img_metas = [
                dict(
                    img_shape=(CHIP_SIZE, CHIP_SIZE, 6),
                    ori_shape=(CHIP_SIZE, CHIP_SIZE, 6),
                    pad_shape=(CHIP_SIZE, CHIP_SIZE, 6),
                    scale_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    flip=False,
                    filename=None,
                    ori_filename=None
                )
                for _ in range(batch_len)
            ]

            logits = model.encode_decode(chips, img_metas)       # (B, 13, 224, 224)
            probs  = F.softmax(logits, dim=1).cpu()              # (B, 13, 224, 224)

            for i in range(batch_len):
                y_pad, x_pad = ys[i].item(), xs[i].item()

                # TerraTorch delta: crop border pixels from prediction
                probs_inner = probs[i][:, DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA]  # (13, 208, 208)

                # Map padded coords back to original image coords
                y_orig = y_pad - pad + DELTA
                x_orig = x_pad - pad + DELTA

                # Clamp to valid original image bounds
                y0 = max(y_orig, 0)
                x0 = max(x_orig, 0)
                y1 = min(y_orig + inner_size, H)
                x1 = min(x_orig + inner_size, W)

                # Corresponding slice in probs_inner
                py0 = y0 - y_orig
                px0 = x0 - x_orig
                py1 = py0 + (y1 - y0)
                px1 = px0 + (x0 - x_orig) + (x1 - x0)
                px1 = px0 + (x1 - x0)

                # Blend mask slice
                bm = blend_mask[py0:py1, px0:px1]  # (h, w)

                prob_accum[:, y0:y1, x0:x1] += probs_inner[:, py0:py1, px0:px1] * bm
                count_map[:,  y0:y1, x0:x1] += bm

    print(f"\nInference done in {time.time()-start:.1f}s")

    # Average + argmax
    count_map[count_map == 0] = 1
    prob_accum /= count_map
    pred = torch.argmax(prob_accum, dim=0).numpy().astype(np.uint8)

    # Post-inference: overwrite NoData border pixels with 255
    pred[dataset.nodata_mask] = 255

    print(f"Predicted classes: {np.unique(pred)}")

    # Save -- use original profile (unpadded)
    profile = dataset.profile
    profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw', 'nodata': 255})
    with rasterio.open(OUTPUT_FILE, 'w', **profile) as dst:
        dst.write(pred, 1)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Output size: {pred.shape} -- matches original stack: {H}x{W}")


if __name__ == '__main__':
    main()