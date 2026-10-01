"""
Semantic-segmentation inference for any backbone on any region.

Replaces the fourteen infer_<model>[_<head>]_<region>.py scripts. Within a
backbone the per-region copies were identical apart from names; across
backbones what genuinely differed was the normalization, the model
construction and the input mode. All of that now comes from
configs/registry.py.

    python infer_seg.py --model satmae --head fpn --region NWIA
    python infer_seg.py --model prithvi --region all

Two input modes; chips is what the published data supports:

    chips   run over pre-cut chips listed in a split file and stitch them
    stack   slide a window over a stitched multitemporal raster, if you have
            one (never published)

Both blend overlapping predictions with a cosine-tapered mask (the TerraTorch
tiled_inference protocol). Chip runs are then scored against the chip masks
with the method each model's column of Table 4 used -- see seg_metrics.py.

Writes <region>_<Model>[_<HEAD>]_Prediction.tif and a _metrics.json under
$GFM_OUTPUT_ROOT/predictions/.
"""

import os
import re
import sys
import json
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Trailing "_<row>_<col>" of a chip name, whatever prefix precedes it.
CHIP_COORDS_RE = re.compile(r'_(\d+)_(\d+)$')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
from runtime import resolve_device  # noqa: E402
import seg_metrics as SM  # noqa: E402
from paths import (GFM_ROOT, DATA_ROOT, PREDICTIONS,  # noqa: E402
                   seg_checkpoint, stack_path, seg_chips_dir,
                   seg_splits_file)
import registry as R  # noqa: E402


# ---------------------------------------------------------------------------
# Normalization -- one scheme per backbone, matching how each was trained.
# ---------------------------------------------------------------------------

def make_normalizer(spec, region):
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
        # Bounds are per region: each scene was scaled by its own observed
        # range, so using one region's numbers everywhere silently shifts the
        # inputs for the other three.
        bounds = spec["gminmax"].get(region)
        if bounds is None:
            raise SystemExit(
                "No min-max bounds for region {}; known regions are {}".format(
                    region, sorted(spec["gminmax"])))
        lo = np.asarray(bounds["gmin"], dtype=np.float32).reshape(18, 1, 1)
        rng = np.asarray(bounds["gmax"], dtype=np.float32).reshape(18, 1, 1) - lo
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


# ---------------------------------------------------------------------------
# Model construction -- three genuinely different paths.
# ---------------------------------------------------------------------------

def build_model(model_name, head, checkpoint, device):
    spec = R.SEG_MODELS[model_name]
    tree = GFM_ROOT / spec["tree"]

    if model_name == "satmae":
        sys.path.insert(0, str(GFM_ROOT / "SatMAE"))
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
        cfg = Config.fromfile(str(GFM_ROOT / spec["config"]))
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, str(checkpoint), map_location='cpu')

    else:
        raise SystemExit("Unknown segmentation model: {}".format(model_name))

    return model.to(device).eval()


def logits_from(out):
    """Pull the class logits out of whatever forward() returned.

    The three backbones do not agree: SpectralGPT's segmentation head returns
    a dict keyed "out" (the original scripts did output['out']), some heads
    return a tuple of scales with the finest first, and others return the
    tensor directly.
    """
    if isinstance(out, dict):
        for key in ("out", "logits", "pred"):
            if key in out:
                return out[key]
        raise SystemExit(
            "model returned a dict with no logits key; got {}".format(
                sorted(out)))
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def run_stack(model_name, region, head, batch, out_dir, spec, allow_cpu=False):
    chip, stride, delta = spec["chip"], spec["stride"], spec["delta"]
    num_classes = spec["num_classes"]
    suffix = "_{}".format(head) if head else ""

    device = resolve_device(allow_cpu)
    print("Device: {}".format(device))

    model = build_model(model_name, head, seg_checkpoint(model_name, region, head), device)
    normalize = make_normalizer(spec, region)

    raster = stack_path(region)
    dataset = SlidingWindowDataset(raster, chip, stride, delta, normalize, spec)
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)
    print("Windows: {}  ({}x{} raster)".format(
        len(dataset), dataset.height, dataset.width))

    blend = cosine_blend_mask(chip, stride, delta).to(device)
    pad = dataset.pad
    H, W = dataset.height, dataset.width
    prob_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    label = "{}{} {}".format(spec["label"], suffix, region)
    with torch.no_grad():
        for chips, ys, xs in tqdm(loader, desc=label):
            # Blend probabilities, not logits. The originals softmaxed before
            # accumulating, and averaging logits across overlapping windows is
            # not the same operation.
            out = F.softmax(logits_from(model(chips.to(device))), dim=1)
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
                prob_sum[:, y0:y1, x0:x1] += tile * mask
                weight_sum[:, y0:y1, x0:x1] += mask

    weight_sum[weight_sum == 0] = 1
    pred = (prob_sum / weight_sum).argmax(0).cpu().numpy().astype(np.uint8)
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


def run_chips(model_name, region, head, batch, out_dir, spec, allow_cpu=False):
    """Chip-based inference: predict over pre-cut chips listed in a split file.

    Used where a region was evaluated against its own test split rather than a
    stitched raster. Unlike the stack path this also stitches the ground truth
    and reports per-class IoU, because the chips carry their own masks.
    """
    chip, stride, delta = spec["chip"], spec["stride"], spec["delta"]
    num_classes = spec["num_classes"]
    suffix = "_{}".format(head) if head else ""

    data_dir = seg_chips_dir(model_name, region)
    splits = seg_splits_file(model_name, region)

    device = resolve_device(allow_cpu)
    print("Device: {}".format(device))

    # Chip names carry their row and col as the last two fields. The working
    # copies name them chip_<row>_<col>; the tarballs published on the Hub add
    # a region prefix, SouthCA_chip_<row>_<col>. Read the trailing pair rather
    # than fixed positions so both spellings work.
    #
    # Split files were generated by listing a directory, so some also contain
    # entries that are not chips at all -- SouthCA_test.txt starts with the
    # region's full CDL raster. Those are dropped here rather than left to fail
    # on an unhelpful int() conversion.
    with open(splits) as fh:
        entries = [l.strip() for l in fh if l.strip()]

    matched = [(n, CHIP_COORDS_RE.search(n)) for n in entries]
    skipped = [n for n, m in matched if m is None]
    names = [n for n, m in matched if m is not None]
    coords = [(int(m.group(1)), int(m.group(2))) for _, m in matched if m]

    if skipped:
        print("Ignoring {} non-chip entries in {} (e.g. {})".format(
            len(skipped), os.path.basename(str(splits)), skipped[0]))
    if not names:
        raise SystemExit(
            "No chip names in {} match <row>_<col>; cannot place chips on the "
            "canvas.".format(splits))
    print("Test chips: {}".format(len(names)))
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
    normalize = make_normalizer(spec, region)
    blend = cosine_blend_mask(chip, stride, delta).to(device)

    prob_sum = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    weight_sum = torch.zeros((1, H, W), dtype=torch.float32, device=device)

    label = "{}{} {}".format(spec["label"], suffix, region)
    with torch.no_grad():
        for start in tqdm(range(0, len(names), batch), desc=label):
            block = names[start:start + batch]
            chips, metas = [], []
            for offset, name in enumerate(block):
                with rasterio.open(os.path.join(data_dir, name + '.tif')) as src:
                    arr = src.read().astype(np.float32)
                arr[arr == -9999] = 0.0
                chips.append(normalize(arr))
                # Index directly: names.index() here was a linear scan per
                # chip, quadratic over a region with 33,000 of them.
                r, c = coords[start + offset]
                metas.append((r - min_row, c - min_col))


            batch_t = torch.from_numpy(np.stack(chips)).float()
            if spec.get('layout') == 'bands_dates':
                batch_t = batch_t.view(-1, 3, 6, chip, chip).permute(0, 2, 1, 3, 4)
            out = F.softmax(logits_from(model(batch_t.to(device))), dim=1)
            out = out[:, :, delta:chip - delta, delta:chip - delta]

            for i, (y, x) in enumerate(metas):
                y0, x0 = y + delta, x + delta
                h = w = chip - 2 * delta
                prob_sum[:, y0:y0 + h, x0:x0 + w] += out[i] * blend
                weight_sum[:, y0:y0 + h, x0:x0 + w] += blend

    # Pixels no window reached -- the outer delta-wide rim of the chip set --
    # have no prediction. Mark them NoData rather than letting argmax over
    # zeros call them class 0, which is Natural Veg, a real class.
    uncovered = (weight_sum[0] == 0).cpu().numpy()
    weight_sum[weight_sum == 0] = 1
    pred = (prob_sum / weight_sum).argmax(0).cpu().numpy().astype(np.uint8)
    pred[uncovered] = SM.NO_PREDICTION

    profile = dict(ref_profile)
    profile.update(driver='GTiff', dtype='uint8', count=1, height=H, width=W,
                   transform=canvas_transform, compress='lzw',
                   nodata=SM.NO_PREDICTION)
    name = "{}_{}{}_Prediction.tif".format(
        region, spec["label"], "_" + head.upper() if head else "")
    out_file = out_dir / name
    with rasterio.open(out_file, 'w', **profile) as dst:
        dst.write(pred, 1)
    print("Saved: {}".format(out_file))

    # Score with the method this model's column of Table 4 used.
    scorer = spec["scorer"]
    common = dict(pred=pred, names=names, coords=coords,
                  origin=(min_row, min_col), data_dir=data_dir, chip=chip,
                  mask_offset=R.per_region(spec["mask_offset"], region),
                  pred_offset=R.per_region(spec["pred_offset"], region))
    if scorer == "per_chip":
        scores = SM.score_per_chip(**common)
    elif scorer == "canvas":
        scores = SM.score_canvas(
            delta=delta, gt_inner=R.per_region(spec["gt_inner"], region),
            **common)
    else:
        raise SystemExit("Unknown segmentation scorer: {}".format(scorer))
    SM.print_report(scores, "{} {}".format(spec["label"], region))

    # Recorded as well as printed, so reproduce.sh can tabulate it. The change
    # detection path does the same.
    metrics = {
        "model": model_name, "region": region, "head": head,
        "state": R.state_of_region(region), "scorer": scorer,
        "prediction": str(out_file),
    }
    metrics.update(scores)
    metrics_path = out_dir / (out_file.stem + "_metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
    print("Saved metrics: {}".format(metrics_path))
    return out_file


def input_mode(spec, region, model_name=None, requested="auto"):
    """Which inference path this (model, region) uses.

    The published data is chips, and the paper's numbers are scored on chip
    ground truth, so chips is the path anyone who clones this will take.

    Some of the original runs predicted by sliding a window over a stitched
    multitemporal raster instead. Those rasters were never published. Where a
    working copy still has one, "auto" uses it so that copy reproduces its own
    output. Scoring is identical either way; on SpectralGPT California the two
    inputs score within 0.3 IoU of each other per class and identically on
    crops, the difference coming from chip edges, which the raster gives
    surrounding context. Pass --input to force one.
    """
    if requested in ("chips", "stack"):
        return requested

    if model_name is not None:
        if stack_path(region, must_exist=False).exists():
            return "stack"
        if seg_chips_dir(model_name, region, must_exist=False).is_dir():
            return "chips"

    declared = spec.get("input_mode", "chips")
    return declared.get(region, "chips") if isinstance(declared, dict) else declared


def run(model_name, region, head="", batch=None, requested="auto",
        allow_cpu=False):
    spec = R.SEG_MODELS[model_name]

    if head and head not in spec["heads"]:
        raise SystemExit("{} has no head {!r}; choose from {}".format(
            model_name, head, spec["heads"]))
    if not head:
        # The published head, not heads[0]: for SatMAE those differ -- fcn is
        # listed first but fpn is the checkpoint on the Hub, so defaulting to
        # the list order downloaded one file and then looked for another.
        head = R.seg_published_head(model_name) or spec["heads"][0]

    batch = batch or spec["batch"]
    suffix = "_{}".format(head) if head else ""
    out_dir = PREDICTIONS / "{}{}_{}".format(model_name, suffix, region)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = input_mode(spec, region, model_name, requested)
    print("Input mode: {}".format(mode))
    runner = run_chips if mode == "chips" else run_stack
    return runner(model_name, region, head, batch, out_dir, spec, allow_cpu)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(R.SEG_MODELS))
    ap.add_argument('--head', default='',
                    help="SatMAE decoder head; fpn is the only one published "
                         "(ignored for other backbones)")
    ap.add_argument('--region', required=True,
                    help="region name, or 'all' for every region with a checkpoint")
    ap.add_argument('--batch', type=int, default=None,
                    help='override the per-model batch size (memory only)')
    ap.add_argument('--input', default='auto', choices=['auto', 'chips', 'stack'],
                    dest='requested',
                    help="inference input: chips (the published data) or "
                         "stack, which needs your own processed_stacks/ "
                         "(default: auto)")
    ap.add_argument('--allow-cpu', action='store_true', dest='allow_cpu',
                    help='run on CPU without asking (for batch jobs)')
    args = ap.parse_args()

    if args.region == 'all':
        regions = sorted({r for m, h, r in R.seg_pairs()
                          if m == args.model and (not args.head or h == args.head)})
    else:
        regions = [args.region]

    for region in regions:
        print("\n==== {} {} / {} ====".format(args.model, args.head or '', region))
        run(args.model, region, args.head, args.batch, args.requested,
            args.allow_cpu)


if __name__ == '__main__':
    main()
