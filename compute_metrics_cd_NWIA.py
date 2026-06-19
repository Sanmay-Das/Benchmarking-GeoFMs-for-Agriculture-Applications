"""
compute_metrics_cd_NWIA.py
--------------------------
Computes full-scene change detection metrics for NWIA test set.

Metrics per model:
    OA         — Overall Accuracy
    Precision  — for changed class
    Recall     — for changed class
    F1         — for changed class
    IoU_change — IoU for changed class (TP / TP+FP+FN)
    IoU_nochange — IoU for unchanged class
    mIoU       — mean IoU across both classes
    Kappa      — Cohen's Kappa coefficient
"""

import numpy as np
import rasterio
import rasterio.windows

# ── paths ────────────────────────────────────────────────────────────────────
BASE = '/bigdata/eldawylab/sdas050/MS_Research'

MODELS = [
    ('SpectralGPT',
     f'{BASE}/predictions/cd_spectralgpt_NWIA/NWIA_SpectralGPT_CD_pred.tif',
     f'{BASE}/predictions/cd_spectralgpt_NWIA/NWIA_SpectralGPT_CD_gt.tif'),
    ('Prithvi',
     f'{BASE}/predictions/cd_prithvi_NWIA/NWIA_Prithvi_CD_pred.tif',
     f'{BASE}/predictions/cd_prithvi_NWIA/NWIA_Prithvi_CD_gt.tif'),
    ('SatMAE',
     f'{BASE}/predictions/cd_satmae_NWIA/NWIA_SatMAE_CD_pred.tif',
     f'{BASE}/predictions/cd_satmae_NWIA/NWIA_SatMAE_CD_gt.tif'),
]


STRIP_H = 256   # read this many rows at a time


def compute_metrics_strip(pred_path, gt_path):
    """Compute metrics strip by strip — never loads full arrays."""
    tp = fp = tn = fn = 0

    with rasterio.open(pred_path) as psrc, rasterio.open(gt_path) as gsrc:
        H = min(psrc.height, gsrc.height)
        W = min(psrc.width,  gsrc.width)
        for y0 in range(0, H, STRIP_H):
            y1  = min(y0 + STRIP_H, H)
            win = rasterio.windows.Window(0, y0, W, y1 - y0)
            p   = psrc.read(1, window=win).astype(np.int32)
            g   = gsrc.read(1, window=win).astype(np.int32)
            valid = g != 255
            p, g  = p[valid], g[valid]
            tp += int(((p==1)&(g==1)).sum())
            fp += int(((p==1)&(g==0)).sum())
            tn += int(((p==0)&(g==0)).sum())
            fn += int(((p==0)&(g==1)).sum())

    total = tp + fp + tn + fn

    oa        = (tp + tn) / (total + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    iou_change   = tp / (tp + fp + fn + 1e-8)
    iou_nochange = tn / (tn + fn + fp + 1e-8)
    miou         = (iou_change + iou_nochange) / 2

    # Cohen's Kappa
    pred_pos  = tp + fp
    pred_neg  = tn + fn
    actual_pos = tp + fn
    actual_neg = tn + fp
    expected  = (pred_pos * actual_pos + pred_neg * actual_neg) / (total ** 2 + 1e-8)
    kappa     = (oa - expected) / (1 - expected + 1e-8)

    return {
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'OA':        oa,
        'Precision': precision,
        'Recall':    recall,
        'F1':        f1,
        'IoU_change':   iou_change,
        'IoU_nochange': iou_nochange,
        'mIoU':      miou,
        'Kappa':     kappa,
    }


def main():
    print("\n" + "="*75)
    print("  Change Detection Metrics — NWIA Test Set")
    print("="*75)

    results = {}
    for model_name, pred_path, gt_path in MODELS:
        print(f"  Computing {model_name}...")
        m = compute_metrics_strip(pred_path, gt_path)
        results[model_name] = m

    # ── print table ──────────────────────────────────────────────────────────
    metrics = ['OA', 'Precision', 'Recall', 'F1',
               'IoU_change', 'IoU_nochange', 'mIoU', 'Kappa']

    col_w = 14
    header = f"  {'Metric':<16}" + "".join(f"{m:>{col_w}}" for m in results)
    print(header)
    print("  " + "-" * (16 + col_w * len(results)))

    for metric in metrics:
        row = f"  {metric:<16}"
        for model_name, m in results.items():
            val = m[metric]
            if metric == 'Kappa':
                row += f"{val:>{col_w}.4f}"
            else:
                row += f"{val*100:>{col_w-1}.2f}%"
        print(row)

    print("  " + "-" * (16 + col_w * len(results)))

    # ── confusion matrix counts ───────────────────────────────────────────────
    print(f"\n  {'':16}" + "".join(f"{'TP':>{col_w}}" for _ in results))
    for key in ['TP', 'FP', 'TN', 'FN']:
        row = f"  {key:<16}"
        for m in results.values():
            row += f"{m[key]:>{col_w},}"
        print(row)

    print("\n" + "="*75)


if __name__ == '__main__':
    main()
