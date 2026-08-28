"""
Change-detection inference for any backbone on any region.

Replaces the twelve infer_cd_<model>_<region>.py scripts, which were copies of
this routine differing only in a checkpoint path, a chip-size constant, a
normalization scheme and the activation applied to the head. Those facts now
live in configs/registry.py.

    python infer_cd.py --model satmae --region SouthMN
    python infer_cd.py --model prithvi --region all

Writes, under $MSR_OUTPUT_ROOT/predictions/cd_<model>_<region>/:

    <region>_<Model>_CD_pred.tif   binary change map (0=unchanged, 1=changed,
                                   255=nodata)
    <region>_<Model>_CD_gt.tif     ground-truth change map, same encoding

Chips are read through paths.load_chips_csv(), so the manifests resolve
against MSR_DATA_ROOT wherever the data was unpacked.
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, PREDICTIONS, chips_csv, load_chips_csv, cd_checkpoint  # noqa: E402
import registry as R  # noqa: E402


# ---------------------------------------------------------------------------
# Normalization
#
# Each backbone was trained with its own scheme; these mirror the dataset
# classes used during training (dataset_cd_<model>.py).
# ---------------------------------------------------------------------------

def _zscore(img, mean, std):
    """Per-band (x - mean) / std, with nodata filled by the band mean first."""
    img = img.copy()
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    for b in range(img.shape[0]):
        img[b] = np.where(img[b] == -9999, mean[b], img[b])
    return (img - mean.reshape(-1, 1, 1)) / (std.reshape(-1, 1, 1) + 1e-8)


def _minmax(img):
    """Per-chip, per-band rescale to [0,1]. Nodata becomes 0."""
    img = img.copy()
    img[img == -9999] = 0.0
    for c in range(img.shape[0]):
        lo, hi = img[c].min(), img[c].max()
        img[c] = (img[c] - lo) / (hi - lo) if hi > lo else 0.0
    return np.clip(img, 0.0, 1.0)


def make_normalizers(spec):
    """Return (normalize_t1, normalize_t2) for a backbone."""
    scheme = spec["norm"]
    if scheme == "zscore_shared":
        fn = lambda img: _zscore(img, spec["mean"], spec["std"])
        return fn, fn
    if scheme == "zscore_paired":
        return (lambda img: _zscore(img, spec["t1_mean"], spec["t1_std"]),
                lambda img: _zscore(img, spec["t2_mean"], spec["t2_std"]))
    if scheme == "minmax":
        return _minmax, _minmax
    raise SystemExit("Unknown normalization scheme: {}".format(scheme))


def change_probability(output, head):
    """Probability of the 'changed' class from the model's raw output.

    A LogSoftmax head ("logprob") needs exp(); a raw-logit head needs softmax.
    Note the two are equivalent when applied to log-probabilities, which is why
    the original scripts disagreed harmlessly on this line.
    """
    if head == "logprob":
        return torch.exp(output)[0, 1]
    if head == "logits":
        return F.softmax(output, dim=1)[0, 1]
    raise SystemExit("Unknown head scheme: {}".format(head))


def build_model(model_name, checkpoint, device):
    """Import the vendored builder and load the fine-tuned weights."""
    spec = R.model(model_name)
    sys.path.insert(0, str(MSR_ROOT / spec["code_dir"]))
    module_name, fn_name = spec["builder"]
    module = __import__(module_name, fromlist=[fn_name])
    model = getattr(module, fn_name)(pretrain_path=None)

    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    print("Loaded checkpoint: epoch={}  best_F1={:.2f}%".format(
        ckpt['epoch'], ckpt['best_f1'] * 100))
    return model


def run(model_name, region, threshold=0.5):
    spec = R.model(model_name)
    R.region(region)
    chip = spec["chip"]

    csv_path = chips_csv(model_name, region)
    checkpoint = cd_checkpoint(model_name, region)
    out_dir = PREDICTIONS / "cd_{}_{}".format(model_name, region)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))

    model = build_model(model_name, checkpoint, device)
    norm_t1, norm_t2 = make_normalizers(spec)

    df = load_chips_csv(csv_path)
    print("{} chips: {}".format(region, len(df)))

    # Full-scene canvas spanning every chip's row/col offset.
    rows, cols = df['row'].values, df['col'].values
    min_row, min_col = int(rows.min()), int(cols.min())
    H = int(rows.max()) + chip - min_row
    W = int(cols.max()) + chip - min_col
    print("Canvas: {}x{}".format(H, W))

    with rasterio.open(df['t1'].iloc[0]) as src:
        crs = src.crs
    first = df.loc[df['row'] == min_row].iloc[0]
    with rasterio.open(first['t1']) as src:
        origin = src.transform
    res_x, res_y = origin.a, origin.e
    full_transform = rasterio.transform.from_bounds(
        origin.c, origin.f + res_y * H, origin.c + res_x * W, origin.f, W, H)

    prob_changed = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    gt_canvas = np.full((H, W), 255, dtype=np.uint8)

    label = "{} CD {}".format(spec["label"], region)
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
            r = int(row['row']) - min_row
            c = int(row['col']) - min_col

            with rasterio.open(row['t1']) as src:
                t1 = src.read().astype(np.float32)
            with rasterio.open(row['t2']) as src:
                t2 = src.read().astype(np.float32)

            t1_t = torch.from_numpy(norm_t1(t1)).unsqueeze(0).float().to(device)
            t2_t = torch.from_numpy(norm_t2(t2)).unsqueeze(0).float().to(device)
            probs = change_probability(model(t1_t, t2_t), spec["head"]).cpu().numpy()

            prob_changed[r:r + chip, c:c + chip] += probs
            count_map[r:r + chip, c:c + chip] += 1.0

            with rasterio.open(row['mask']) as src:
                gt_canvas[r:r + chip, c:c + chip] = src.read(1)

    count_map[count_map == 0] = 1
    prob_changed /= count_map
    pred = (prob_changed > threshold).astype(np.uint8)
    pred[gt_canvas == 255] = 255

    profile = {
        'driver': 'GTiff', 'dtype': 'uint8', 'count': 1,
        'height': H, 'width': W, 'crs': crs,
        'transform': full_transform, 'compress': 'lzw', 'nodata': 255,
    }

    stem = "{}_{}_CD".format(region, spec["label"])
    pred_path = out_dir / (stem + "_pred.tif")
    with rasterio.open(pred_path, 'w', **profile) as dst:
        dst.write(pred, 1)
    print("Saved prediction: {}".format(pred_path))

    gt_path = out_dir / (stem + "_gt.tif")
    with rasterio.open(gt_path, 'w', **profile) as dst:
        dst.write(gt_canvas, 1)
    print("Saved GT:         {}".format(gt_path))

    report(pred, gt_canvas, region)


def report(pred, gt_canvas, region):
    valid = gt_canvas != 255
    p, g = pred[valid], gt_canvas[valid]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    tn = int(((p == 0) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    oa = (tp + tn) / (tp + fp + tn + fn + 1e-8)
    print("\n{} Results:".format(region))
    print("  OA={:.2f}%  Precision={:.2f}%  Recall={:.2f}%  F1={:.2f}%".format(
        oa * 100, prec * 100, rec * 100, f1 * 100))
    print("  TP={}  FP={}  TN={}  FN={}".format(tp, fp, tn, fn))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, choices=sorted(R.MODELS))
    ap.add_argument('--region', required=True,
                    help="region name, or 'all' for every region")
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='probability above which a pixel is called changed')
    args = ap.parse_args()

    regions = sorted(R.REGIONS) if args.region == 'all' else [args.region]
    for region in regions:
        print("\n==== {} / {} ====".format(args.model, region))
        run(args.model, region, args.threshold)


if __name__ == '__main__':
    main()
