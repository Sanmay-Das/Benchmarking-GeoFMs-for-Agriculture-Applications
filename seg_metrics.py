"""
Segmentation scoring, matching the scripts that produced Table 4.

The paper's segmentation columns were not all scored the same way, so this
module keeps both methods and infer_seg.py picks one per model:

  per_chip   Prithvi, SpectralGPT -- scripts/infer_finder.py
             For each test chip, take its mask and the matching window of the
             stitched prediction, and add both to one confusion matrix. Chips
             overlap, so a pixel covered by several chips is counted once per
             chip.

  canvas     SatMAE -- evaluate_seg_fpn_corrected.py
             Stitch the chip masks into one ground-truth canvas first, then
             compare it with the prediction. Every pixel is counted once.

The two give slightly different numbers from the same prediction, which is why
each model is scored with the method its published column used.

Class indices are the 13 CDL classes, 0-12, in gt_extractor.py order. Masks and
predictions are shifted onto that index by an offset that depends on the model
and region (see mask_offset / pred_offset in configs/registry.py).
"""

import os

import numpy as np
import rasterio

CLASS_NAMES = [
    "Natural Veg", "Forest", "Corn", "Soybean", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other",
]
NUM_CLASSES = 13
CROP = [2, 3, 7, 8, 9, 10, 11]   # Corn, Soybean, Winter Wheat, Alfalfa, Fallow, Cotton, Sorghum
LAND = [0, 1, 4, 5, 6, 12]       # Natural Veg, Forest, Wetlands, Developed, Water, Other

NO_PREDICTION = 255


def find_mask(data_dir, name):
    """A chip's mask file. The Hub tarballs use <name>_mask.tif; Prithvi's
    working copies used <name>.mask.tif."""
    for suffix in ("_mask.tif", ".mask.tif"):
        path = os.path.join(data_dir, name + suffix)
        if os.path.exists(path):
            return path
    return None


def _to_class_index(arr, offset):
    """Shift onto 0-12; anything outside that range becomes -1 (excluded)."""
    idx = arr.astype(np.int64) - offset
    return np.where((arr == NO_PREDICTION) | (idx < 0) | (idx >= NUM_CLASSES), -1, idx)


def metrics_from_confusion(cm):
    """Per-class IoU, accuracy, precision, recall, F1, plus the summaries.

    Same arithmetic as compute_metrics() in scripts/infer_finder.py. Rows are
    ground truth, columns are prediction. Values are percentages; NaN where a
    class is absent from both.
    """
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp

    def ratio(num, den):
        return np.where(den > 0, 100.0 * num / np.maximum(den, 1), np.nan)

    iou = ratio(tp, tp + fp + fn)
    acc = ratio(tp, tp + fn)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), np.nan)

    total = cm.sum()
    return {
        "per_class_iou": iou.tolist(),
        "per_class_acc": acc.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "mIoU": float(np.nanmean(iou)),
        "mIoU_crops": float(np.nanmean(iou[CROP])),
        "mIoU_land": float(np.nanmean(iou[LAND])),
        "OA": float(100.0 * np.trace(cm) / total) if total else float("nan"),
        "classes_present": int(np.sum(~np.isnan(iou))),
    }


def score_per_chip(pred, names, coords, origin, data_dir, chip,
                   mask_offset, pred_offset):
    """scripts/infer_finder.py: confusion matrix summed chip by chip.

    pred is the stitched prediction; origin is the (row, col) of its top-left
    corner in chip coordinates, so a chip at (r, c) reads
    pred[r - origin_row : ... + chip, c - origin_col : ... + chip].
    Chips whose window runs off the raster are skipped, as there.
    """
    r0, c0 = origin
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    used = skipped = 0
    for name, (r, c) in zip(names, coords):
        path = find_mask(data_dir, name)
        if path is None:
            continue
        window = pred[r - r0:r - r0 + chip, c - c0:c - c0 + chip]
        if r - r0 < 0 or c - c0 < 0 or window.shape != (chip, chip):
            skipped += 1
            continue
        with rasterio.open(path) as src:
            g = _to_class_index(src.read(1), mask_offset)
        p = _to_class_index(window, pred_offset)
        valid = (g >= 0) & (p >= 0)
        cm += np.bincount(NUM_CLASSES * g[valid] + p[valid],
                          minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        used += 1
    out = metrics_from_confusion(cm)
    out.update(chips_scored=used, chips_skipped=skipped)
    return out


def score_canvas(pred, names, coords, origin, data_dir, chip, delta,
                 mask_offset, pred_offset, gt_inner):
    """evaluate_seg_fpn_corrected.py: stitch ground truth, then score once.

    gt_inner says whether each chip contributes only its centre
    [delta, chip - delta) -- the canvas regions -- or the whole chip, as for
    NWIA. Later chips overwrite earlier ones where they overlap, as there.
    """
    r0, c0 = origin
    H, W = pred.shape
    lo, hi = (delta, chip - delta) if gt_inner else (0, chip)
    gt = np.full((H, W), -1, dtype=np.int64)
    for name, (r, c) in zip(names, coords):
        path = find_mask(data_dir, name)
        if path is None:
            continue
        with rasterio.open(path) as src:
            m = _to_class_index(src.read(1)[lo:hi, lo:hi], mask_offset)
        y0, x0 = r - r0 + lo, c - c0 + lo
        y1, x1 = min(y0 + (hi - lo), H), min(x0 + (hi - lo), W)
        if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
            continue
        block = m[:y1 - y0, :x1 - x0]
        np.copyto(gt[y0:y1, x0:x1], block, where=block >= 0)

    p = _to_class_index(pred, pred_offset)
    valid = (gt >= 0) & (p >= 0)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (gt[valid], p[valid]), 1)
    out = metrics_from_confusion(cm)
    out.update(pixels_scored=int(valid.sum()))
    return out


def print_report(m, label):
    """Per-class table followed by the summaries Table 4 reports."""
    print("\n{:<18} {:>8} {:>8} {:>10} {:>8} {:>8}".format(
        label, "IoU", "Acc", "Precision", "Recall", "F1"))
    print("-" * 64)

    def fmt(v):
        return "     NaN" if v != v else "{:8.2f}".format(v)

    for k, cls in enumerate(CLASS_NAMES):
        print("{:<18} {} {} {:>10} {} {}".format(
            cls, fmt(m["per_class_iou"][k]), fmt(m["per_class_acc"][k]),
            fmt(m["per_class_precision"][k]).strip(),
            fmt(m["per_class_recall"][k]), fmt(m["per_class_f1"][k])))
    print("-" * 64)
    print("mIoU (all)   : {:6.2f}%".format(m["mIoU"]))
    print("mIoU (crops) : {:6.2f}%".format(m["mIoU_crops"]))
    print("mIoU (land)  : {:6.2f}%".format(m["mIoU_land"]))
    print("OA           : {:6.2f}%".format(m["OA"]))
