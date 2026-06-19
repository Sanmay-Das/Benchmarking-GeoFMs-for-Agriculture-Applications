import os
import numpy as np
import rasterio
import glob
from tqdm import tqdm

CLASS_NAMES = [
    "Natural Vegetation", "Forest", "Corn", "Soybeans", "Wetlands",
    "Developed/Barren", "Open Water", "Winter Wheat", "Alfalfa",
    "Fallow/Idle", "Cotton", "Sorghum", "Other"
]
NUM_CLASSES = 13

def compute_confusion_matrix(pred, gt, num_classes):
    # Also exclude NoData predictions (255)
    mask = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    cm = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return cm

def compute_metrics(cm):
    iou_list      = []
    acc_list      = []
    precision_list = []
    recall_list   = []
    f1_list       = []

    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp  # GT is c, predicted something else
        fp = cm[:, c].sum() - tp  # Predicted c, GT is something else

        denom_iou  = tp + fp + fn
        denom_acc  = tp + fn
        denom_prec = tp + fp
        denom_rec  = tp + fn

        iou       = tp / denom_iou  if denom_iou  > 0 else float('nan')
        acc       = tp / denom_acc  if denom_acc  > 0 else float('nan')
        precision = tp / denom_prec if denom_prec > 0 else float('nan')
        recall    = tp / denom_rec  if denom_rec  > 0 else float('nan')

        if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float('nan')

        iou_list.append(iou)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    miou = np.nanmean(iou_list)
    oa   = cm.diagonal().sum() / cm.sum()
    return iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa


def evaluate_prithvi(stitched_pred_path, gt_chip_dir, chip_size=224):
    print("=" * 70)
    print("EVALUATING PRITHVI")
    print("=" * 70)

    with rasterio.open(stitched_pred_path) as src:
        pred_full = src.read(1).astype(np.int32)  # 0-12, 255=NoData
    print(f"Loaded prediction: {pred_full.shape}")

    mask_files = sorted(glob.glob(os.path.join(gt_chip_dir, 'chip_*.mask.tif')))
    print(f"Found {len(mask_files)} GT chip masks\n")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    skipped = 0

    for mask_path in tqdm(mask_files, desc="Evaluating"):
        fname = os.path.basename(mask_path)
        parts = fname.replace('.mask.tif', '').split('_')
        row, col = int(parts[1]), int(parts[2])

        with rasterio.open(mask_path) as src:
            gt_raw = src.read(1).astype(np.int32)
        gt = gt_raw - 1  # 1-13 → 0-12, NoData(0) → -1

        pred_chip = pred_full[row:row+chip_size, col:col+chip_size]

        if pred_chip.shape != (chip_size, chip_size):
            skipped += 1
            continue

        valid = gt >= 0
        cm += compute_confusion_matrix(pred_chip[valid], gt[valid], NUM_CLASSES)

    print(f"Skipped {skipped} chips (boundary)\n")
    iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa = compute_metrics(cm)
    print_results(iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa, "Prithvi")


def evaluate_spectralgpt(stitched_pred_path, gt_chip_dir, chip_size=128):
    print("=" * 70)
    print("EVALUATING SPECTRALGPT")
    print("=" * 70)

    with rasterio.open(stitched_pred_path) as src:
        pred_full = src.read(1).astype(np.int32)  # 0-12, 255=NoData
    print(f"Loaded prediction: {pred_full.shape}")

    mask_files = sorted(glob.glob(os.path.join(gt_chip_dir, 'SouthMN_chip_*_mask.tif')))
    print(f"Found {len(mask_files)} GT chip masks\n")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    skipped = 0

    for mask_path in tqdm(mask_files, desc="Evaluating"):
        fname = os.path.basename(mask_path)
        parts = fname.replace('_mask.tif', '').split('_')
        row, col = int(parts[2]), int(parts[3])

        with rasterio.open(mask_path) as src:
            gt_raw = src.read(1).astype(np.int32)
        gt = gt_raw - 1  # 1-13 → 0-12, NoData(0) → -1

        pred_chip = pred_full[row:row+chip_size, col:col+chip_size]

        if pred_chip.shape != (chip_size, chip_size):
            skipped += 1
            continue

        valid = gt >= 0
        cm += compute_confusion_matrix(pred_chip[valid], gt[valid], NUM_CLASSES)

    print(f"Skipped {skipped} chips (boundary)\n")
    iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa = compute_metrics(cm)
    print_results(iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa, "SpectralGPT")


def print_results(iou_list, acc_list, precision_list, recall_list, f1_list, miou, oa, model_name):
    print(f"\n{'='*80}")
    print(f"{model_name} RESULTS")
    print(f"{'='*80}")
    print(f"{'Class':<25} {'IoU':>8} {'Acc':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 72)
    for i, name in enumerate(CLASS_NAMES):
        def fmt(v): return f"{v*100:.2f}" if not np.isnan(v) else "   nan"
        print(f"{name:<25} {fmt(iou_list[i]):>8} {fmt(acc_list[i]):>8} {fmt(precision_list[i]):>10} {fmt(recall_list[i]):>8} {fmt(f1_list[i]):>8}")
    print("-" * 72)
    print(f"{'mIoU':<25} {miou*100:>8.2f}")
    print(f"{'OA':<25} {oa*100:>8.2f}")
    print("=" * 80)


if __name__ == '__main__':
    base_dir = '/bigdata/eldawylab/sdas050/MS_Research'

    # Prithvi
    evaluate_prithvi(
        stitched_pred_path=os.path.join(base_dir, 'predictions/prithvi_SouthMN_terratorch/SouthMN_Prithvi_Prediction_Stitched.tif'),
        gt_chip_dir=os.path.join(base_dir, 'data/multi_temporal_crop_segmentation/SouthMN'),
        chip_size=224
    )

    # SpectralGPT
    evaluate_spectralgpt(
        stitched_pred_path=os.path.join(base_dir, 'predictions/spectralgpt_SouthMN_terratorch/SouthMN_SpectralGPT_Prediction_Stitched.tif'),
        gt_chip_dir=os.path.join(base_dir, 'SpectralGPT_chips_multitemporal/SouthMN'),
        chip_size=128
    )