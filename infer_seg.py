"""
Semantic-segmentation inference for any backbone on any region.

Replaces the fourteen infer_<model>[_<head>]_<region>.py scripts. Within a
backbone the per-region copies were identical apart from names; across
backbones what genuinely differed was the normalization, the model
construction and the input mode. All of that now comes from
configs/registry.py.

    python infer_seg.py --model satmae --head fpn --region NWIA
    python infer_seg.py --model prithvi --region all

Two input modes, chosen per (model, region) by the registry:

    stack   slide a window over a stitched multitemporal raster, blending
            overlapping predictions with a cosine-tapered mask (the
            TerraTorch tiled_inference protocol)
    chips   run over pre-cut chips listed in a split file and stitch them

Writes <region>_<Model>[_<HEAD>]_Prediction.tif under
$MSR_OUTPUT_ROOT/predictions/.
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
from paths import (MSR_ROOT, DATA_ROOT, PREDICTIONS,  # noqa: E402
                   seg_checkpoint, stack_path)
import registry as R  # noqa: E402


# ---------------------------------------------------------------------------
# Normalization -- one scheme per backbone, matching how each was trained.
# ---------------------------------------------------------------------------

def make_normalizer(spec):
    """Return f(chip) -> normalized chip for an (18, h, w) float32 array.

    Epsilon placement follows each original script exactly: SatMAE divides by
    (std + 1e-8), Prithvi and SpectralGPT divide by the bare denominator.
    """
    scheme = spec["seg_norm"]

    if scheme == "zscore_tiled":
        # 6-band sensor stats tiled across the 3 dates of the stack.
        mean = np.tile(np.asarray(spec["mean6"], dtype=np.float32), 3).reshape(18, 1, 1)
        std = np.tile(np.asarray(spec["std6"], dtype=np.float32), 3).reshape(18, 1, 1)
        return lambda chip: (chip - mean) / (std + 1e-8)

    if scheme == "zscore18":
        mean = np.asarray(spec["mean18"], dtype=np.float32).reshape(18, 1, 1)
        std = np.asarray(spec["std18"], dtype=np.float32).reshape(18, 1, 1)
        return lambda chip: (chip - mean) / std

    if scheme == "global_minmax":
        lo = np.asarray(spec["gmin"], dtype=np.float32).reshape(18, 1, 1)
        rng = np.asarray(spec["gmax"], dtype=np.float32).reshape(18, 1, 1) - lo
        return lambda chip: np.clip((chip - lo) / rng, 0.0, 1.0)

    raise SystemExit("Unknown segmentation normalization: {}".format(scheme))


def reshape_for_model(tensor, spec, chip_size):
    """Apply a backbone's expected input layout.

    Prithvi consumes (bands, dates, H, W); the stack is stored as 18 channels
    ordered date-major, so it is viewed as (3, 6, H, W) and transposed. The
    other backbones take the 18 channels as-is.
    """
    if spec.get("layout") != "bands_dates":
        return tensor
    return tensor.view(3, 6, chip_size, chip_size).permute(1, 0, 2, 3)


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def cosine_blend_mask(chip_size, stride, delta):
    """Cosine-tapered blend weights, per the TerraTorch tiled_inference protocol.

    Centre pixels are weighted higher than edges so overlapping windows join
    without seams. Returns a (chip_size - 2*delta) square, strictly positive so
    no pixel can end up with zero total weight.
    """
    size = chip_size - 2 * delta
    overlap = chip_size - stride
    ramp_len = max(min(chip_size // 2, overlap) - delta, 0)

    x = torch.ones(size, dtype=torch.float32)
    if ramp_len > 0:
        ramp = torch.cos(
            math.pi * (torch.arange(ramp_len, dtype=torch.float32) + 1)
            / (ramp_len + 1)) / 2 + 0.5
        x[:ramp_len] = ramp.flip(0)
        x[-ramp_len:] = ramp
    return (x[:, None] * x[None, :]) + 1e-6


class SlidingWindowDataset(Dataset):
    """Windows over a stitched multitemporal raster, reflect-padded at borders."""

    def __init__(self, raster, chip_size, stride, delta, normalize, spec):
        with rasterio.open(raster) as src:
            data = src.read().astype(np.float32)
            self.profile = src.profile
            self.crs = src.crs
            self.transform = src.transform

        self.nodata_mask = (data[0] == -9999)
        data[data == -9999] = 0.0

        self.pad = chip_size // 2
        self.data = np.pad(
            data, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)), mode='reflect')
        self.chip_size = chip_size
        self.normalize = normalize
        self.spec = spec
        self.height, self.width = data.shape[1], data.shape[2]

        padded_h, padded_w = self.data.shape[1], self.data.shape[2]
        ys = list(range(0, padded_h - chip_size + 1, stride))
        xs = list(range(0, padded_w - chip_size + 1, stride))
        if ys and ys[-1] != padded_h - chip_size:
            ys.append(padded_h - chip_size)
        if xs and xs[-1] != padded_w - chip_size:
            xs.append(padded_w - chip_size)
        self.windows = [(y, x) for y in ys for x in xs]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        y, x = self.windows[idx]
        chip = self.data[:, y:y + self.chip_size, x:x + self.chip_size]
        chip = torch.from_numpy(self.normalize(chip)).float()
        return reshape_for_model(chip, self.spec, self.chip_size), y, x


def compute_iou(pred, gt, num_classes):
    """Per-class IoU in percent, skipping class 0 (NoData). NaN where absent."""
    ious = []
    for c in range(1, num_classes):
        mask = gt != 0
        pred_c = (pred == c) & mask
        gt_c = (gt == c) & mask
        inter = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        ious.append(float('nan') if union == 0 else 100.0 * inter / union)
    return ious


# ---------------------------------------------------------------------------
# Model construction -- three genuinely different paths.
# ---------------------------------------------------------------------------

def build_model(model_name, head, checkpoint, device):
    spec = R.SEG_MODELS[model_name]
    tree = MSR_ROOT / spec["tree"]

    if model_name == "satmae":
        sys.path.insert(0, str(MSR_ROOT / "SatMAE"))
        import models_vit_group_channels
        enc = spec["encoder"]
        encoder = models_vit_group_channels.vit_large_patch16(
            patch_size=enc["patch_size"], img_size=spec["chip"],
            in_chans=enc["in_chans"], channel_groups=enc["channel_groups"],
            num_classes=0, drop_path_rate=enc["drop_path"], global_pool=False)
        module_name, class_name = spec["head_classes"][head]
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        model = cls(encoder=encoder, nb_classes=spec["num_classes"])
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    elif model_name == "spectralgpt":
        sys.path.insert(0, str(tree))
        module_name, fn_name = spec["seg_builder"]
        module = __import__(module_name, fromlist=[fn_name])
        model = getattr(module, fn_name)(num_classes=spec["num_classes"])
        ckpt = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])

    elif model_name == "prithvi":
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmseg.models import build_segmentor
        cfg = Config.fromfile(str(MSR_ROOT / spec["config"]))
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, str(checkpoint), map_location='cpu')

    else:
        raise SystemExit("Unknown segmentation model: {}".format(model_name))

    return model.to(device).eval()


def run_stack(model_name, region, head, batch, out_dir, spec):
    chip, stride, delta = spec["chip"], spec["stride"], spec["delta"]
    num_classes = spec["num_classes"]
    suffix = "_{}".format(head) if head else ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    model = build_model(model_name, head, seg_checkpoint(model_name, region, head), device)
    normalize = make_normalizer(spec)

    raster = stack_path(region)
    dataset = SlidingWindowDataset(raster, chip, stride, delta, normalize, spec)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    print("Windows: {}  ({}x{} raster)".format(
        len(dataset), dataset.height, dataset.width))

    blend = cosine_blend_mask(chip, stride, delta).to(device)
    pad = dataset.pad
    H, W = dataset.height, dataset.width
    logits_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    label = "{}{} {}".format(spec["label"], suffix, region)
    with torch.no_grad():
        for chips, ys, xs in tqdm(loader, desc=label):
            out = model(chips.to(device))
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out[:, :, delta:chip - delta, delta:chip - delta]

            for i in range(out.shape[0]):
                y = int(ys[i]) - pad + delta
                x = int(xs[i]) - pad + delta
                h = w = chip - 2 * delta
                y0, x0 = max(y, 0), max(x, 0)
                y1, x1 = min(y + h, H), min(x + w, W)
                if y1 <= y0 or x1 <= x0:
                    continue
                sy, sx = y0 - y, x0 - x
                tile = out[i][:, sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
                mask = blend[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
                logits_sum[:, y0:y1, x0:x1] += tile * mask
                weight_sum[:, y0:y1, x0:x1] += mask

    weight_sum[weight_sum == 0] = 1
    pred = (logits_sum / weight_sum).argmax(0).cpu().numpy().astype(np.uint8)
    pred[dataset.nodata_mask] = 0

    profile = dict(dataset.profile)
    profile.update(driver='GTiff', dtype='uint8', count=1,
                   compress='lzw', nodata=0)
    name = "{}_{}{}_Prediction.tif".format(
        region, spec["label"], "_" + head.upper() if head else "")
    out_file = out_dir / name
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(pred, 1)
    print("Saved: {}".format(out_file))
    return out_file


def run_chips(model_name, region, head, batch, out_dir, spec):
    """Chip-based inference: predict over pre-cut chips listed in a split file.

    Used where a region was evaluated against its own test split rather than a
    stitched raster. Unlike the stack path this also stitches the ground truth
    and reports per-class IoU, because the chips carry their own masks.
    """
    chip, stride, delta = spec["chip"], spec["stride"], spec["delta"]
    num_classes = spec["num_classes"]
    suffix = "_{}".format(head) if head else ""

    run_name = R.region(region)["seg_split"]
    data_dir = DATA_ROOT / spec["chips_dir"].format(region=region, run=run_name)
    splits = DATA_ROOT / spec["splits"].format(region=region, run=run_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    with open(splits) as fh:
        names = [l.strip() for l in fh if l.strip()]
    print("Test chips: {}".format(len(names)))

    coords = [(int(n.split('_')[1]), int(n.split('_')[2])) for n in names]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_row, min_col = min(rows), min(cols)
    H = max(rows) - min_row + chip
    W = max(cols) - min_col + chip
    print("Canvas: {}x{}".format(H, W))

    with rasterio.open(os.path.join(data_dir, names[0] + '.tif')) as src:
        ref_profile = src.profile.copy()
        first_transform = src.transform
    first_row, first_col = coords[0]
    res = first_transform.a
    canvas_transform = rasterio.transform.from_origin(
        first_transform.c - (first_col - min_col) * res,
        first_transform.f + (first_row - min_row) * res,
        res, res)

    model = build_model(model_name, head,
                        seg_checkpoint(model_name, region, head), device)
    normalize = make_normalizer(spec)
    blend = cosine_blend_mask(chip, stride, delta).to(device)

    logits_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, H, W), dtype=torch.float32, device=device)
    gt_canvas = np.zeros((H, W), dtype=np.uint8)

    label = "{}{} {}".format(spec["label"], suffix, region)
    with torch.no_grad():
        for start in tqdm(range(0, len(names), batch), desc=label):
            block = names[start:start + batch]
            chips, metas = [], []
            for name in block:
                with rasterio.open(os.path.join(data_dir, name + '.tif')) as src:
                    arr = src.read().astype(np.float32)
                arr[arr == -9999] = 0.0
                chips.append(normalize(arr))
                r, c = coords[names.index(name)]
                metas.append((r - min_row, c - min_col))

                mask_file = os.path.join(data_dir, name + '_mask.tif')
                if os.path.exists(mask_file):
                    with rasterio.open(mask_file) as src:
                        gt_canvas[r - min_row:r - min_row + chip,
                                  c - min_col:c - min_col + chip] = src.read(1)

            batch_t = torch.from_numpy(np.stack(chips)).float()
            if spec.get('layout') == 'bands_dates':
                batch_t = batch_t.view(-1, 3, 6, chip, chip).permute(0, 2, 1, 3, 4)
            out = model(batch_t.to(device))
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out[:, :, delta:chip - delta, delta:chip - delta]

            for i, (y, x) in enumerate(metas):
                y0, x0 = y + delta, x + delta
                h = w = chip - 2 * delta
                logits_sum[:, y0:y0 + h, x0:x0 + w] += out[i] * blend
                weight_sum[:, y0:y0 + h, x0:x0 + w] += blend

    weight_sum[weight_sum == 0] = 1
    pred = (logits_sum / weight_sum).argmax(0).cpu().numpy().astype(np.uint8)

    profile = dict(ref_profile)
    profile.update(driver='GTiff', dtype='uint8', count=1, height=H, width=W,
                   transform=canvas_transform, compress='lzw', nodata=0)
    name = "{}_{}{}_Prediction.tif".format(
        region, spec["label"], "_" + head.upper() if head else "")
    out_file = out_dir / name
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(pred, 1)
    print("Saved: {}".format(out_file))

    ious = compute_iou(pred, gt_canvas, num_classes)
    valid = [v for v in ious if not math.isnan(v)]
    if valid:
        print("mIoU: {:.2f}%  over {} classes".format(
            sum(valid) / len(valid), len(valid)))
    return out_file


def input_mode(spec, region):
    """Which inference path this (model, region) uses."""
    mode = spec.get("input_mode", "stack")
    return mode.get(region, "stack") if isinstance(mode, dict) else mode


def run(model_name, region, head="", batch=None):
    spec = R.SEG_MODELS[model_name]

    if head and head not in spec["heads"]:
        raise SystemExit("{} has no head {!r}; choose from {}".format(
            model_name, head, spec["heads"]))
    if not head:
        head = spec["heads"][0]

    batch = batch or spec["batch"]
    suffix = "_{}".format(head) if head else ""
    out_dir = PREDICTIONS / "{}{}_{}".format(model_name, suffix, region)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = input_mode(spec, region)
    runner = run_chips if mode == "chips" else run_stack
    print("Input mode: {}".format(mode))
    return runner(model_name, region, head, batch, out_dir, spec)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(R.SEG_MODELS))
    ap.add_argument('--head', default='',
                    help="SatMAE decoder: fcn, fpn or psanet (ignored otherwise)")
    ap.add_argument('--region', required=True,
                    help="region name, or 'all' for every region with a checkpoint")
    ap.add_argument('--batch', type=int, default=None,
                    help='override the per-model batch size (memory only)')
    args = ap.parse_args()

    if args.region == 'all':
        regions = sorted({r for m, h, r in R.seg_pairs()
                          if m == args.model and (not args.head or h == args.head)})
    else:
        regions = [args.region]

    for region in regions:
        print("\n==== {} {} / {} ====".format(args.model, args.head or '', region))
        run(args.model, region, args.head, args.batch)


if __name__ == '__main__':
    main()
