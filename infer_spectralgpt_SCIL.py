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

sys.path.insert(0, '/bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich')
from src.models_vit_tensor_CD_2 import vit_base_patch8

# ============================================================
# CONFIGURATION
# ============================================================
CHECKPOINT  = '/bigdata/eldawylab/sdas050/MS_Research/IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich/multi_train/best_mIoU_WCECIL_model.pth'
STACK_PATH  = '/bigdata/eldawylab/sdas050/MS_Research/scripts/processed_stacks/SCIL/SCIL_multitemporal_stack.tif'
OUTPUT_DIR  = '/bigdata/eldawylab/sdas050/MS_Research/predictions/spectralgpt_SCIL_terratorch'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'SCIL_SpectralGPT_Prediction_Stitched.tif')

CHIP_SIZE   = 128
STRIDE      = 64     # 50% overlap
DELTA       = 8      # TerraTorch: border pixels to discard from each chip
NUM_CLASSES = 13
BATCH_SIZE  = 32
NODATA_VAL  = -9999

# IL Min-Max normalization stats (used during finetuning)
GLOBAL_MIN = np.array([
    675.00, 1012.00, 1052.00, 984.00, 1031.00, 1005.00,    # T1: B02, B03, B04, B8A, B11, B12
    872.00,  694.00,  785.00, 1047.00, 1116.00, 1099.00,   # T2: B02, B03, B04, B8A, B11, B12
    517.00,  787.00,  829.00, 1040.00, 1013.00,  996.00    # T3: B02, B03, B04, B8A, B11, B12
], dtype=np.float32).reshape(18, 1, 1)

GLOBAL_MAX = np.array([
    18688.00, 17872.00, 17231.00, 16633.00, 15868.00, 16061.00,  # T1
    12834.00, 13445.00, 14211.00, 10290.00, 13408.00, 14751.00,  # T2
    18131.00, 17536.00, 16976.00, 13643.00, 16100.00, 16052.00   # T3
], dtype=np.float32).reshape(18, 1, 1)

GLOBAL_RANGE = GLOBAL_MAX - GLOBAL_MIN


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
# DATASET (with reflect padding)
# ============================================================
class SlidingWindowDataset(Dataset):
    def __init__(self, stack_path, chip_size, stride, delta):
        with rasterio.open(stack_path) as src:
            self.profile = src.profile.copy()
            self.orig_H  = src.height
            self.orig_W  = src.width
            data         = src.read().astype(np.float32)  # (18, H, W)

        # Pre-compute NoData mask on ORIGINAL stack (before padding)
        self.nodata_mask = np.any(data == NODATA_VAL, axis=0)  # (H, W)

        # Replace -9999 with 0 before normalization
        data[data == NODATA_VAL] = 0.0

        # TerraTorch: reflect padding at borders (pad = chip_size // 2)
        self.pad  = chip_size // 2
        self.data = np.pad(data,
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

        # Min-Max normalize to [0, 1] (NoData already replaced with 0)
        chip = (chip - GLOBAL_MIN) / GLOBAL_RANGE
        chip = np.clip(chip, 0.0, 1.0)

        return torch.from_numpy(chip), y, x   # (18, 128, 128)


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading SpectralGPT model...")
    model = vit_base_patch8(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded | Epoch: {checkpoint['epoch']} | Best mIoU: {checkpoint.get('best_miou', 'N/A')}\n")

    # Dataset + dataloader
    dataset    = SlidingWindowDataset(STACK_PATH, CHIP_SIZE, STRIDE, DELTA)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Accumulation buffers over ORIGINAL (unpadded) size
    H, W       = dataset.orig_H, dataset.orig_W
    pad        = dataset.pad
    prob_accum = torch.zeros((NUM_CLASSES, H, W), dtype=torch.float32)
    count_map  = torch.zeros((1, H, W),           dtype=torch.float32)

    print(f"Original stack: {H}×{W} | Padded: {dataset.H}×{dataset.W} | Windows: {len(dataset)}\n")

    # Precompute cosine blend mask
    blend_mask = cosine_blend_mask(CHIP_SIZE, STRIDE, DELTA)  # (chip-2δ, chip-2δ)
    inner_size = CHIP_SIZE - 2 * DELTA                         # 112 for delta=8

    start = time.time()
    with torch.no_grad():
        for chips, ys, xs in tqdm(dataloader, desc="SpectralGPT Inference"):
            chips     = chips.to(device)
            batch_len = chips.size(0)

            output = model(chips)
            logits = output['out']                        # (B, 13, 128, 128)
            probs  = F.softmax(logits, dim=1).cpu()       # (B, 13, 128, 128)

            for i in range(batch_len):
                y_pad, x_pad = ys[i].item(), xs[i].item()

                # TerraTorch delta: crop border pixels from prediction
                probs_inner = probs[i][:, DELTA:CHIP_SIZE-DELTA, DELTA:CHIP_SIZE-DELTA]  # (13, 112, 112)

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

    # Save stitched GeoTIFF
    profile = dataset.profile
    profile.update({'count': 1, 'dtype': 'uint8', 'compress': 'lzw', 'nodata': 255})
    with rasterio.open(OUTPUT_FILE, 'w', **profile) as dst:
        dst.write(pred, 1)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Output size: {pred.shape} — matches original stack: {H}×{W}")


if __name__ == '__main__':
    main()