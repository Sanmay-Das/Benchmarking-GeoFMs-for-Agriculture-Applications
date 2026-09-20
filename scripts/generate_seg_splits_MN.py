"""
generate_seg_splits_MN.py
-------------------------
Generate train.txt / val.txt / test.txt for SatMAE MN segmentation.

    Train : NorthMN
    Val   : CentMN
    Test  : SouthMN

Chip stems are written one per line (no extension, no path),
matching the format of Iowa/train.txt used by dataset_seg.py.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'configs'))
from paths import GFM_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS


import os
from pathlib import Path

CHIPS_ROOT = Path(f"{DATA_ROOT}/SatMAE_chips_multitemporal")
OUT_DIR    = CHIPS_ROOT / "MN"
OUT_DIR.mkdir(exist_ok=True)

SPLITS = {
    "train.txt": CHIPS_ROOT / "NorthMN",
    "val.txt":   CHIPS_ROOT / "CentMN",
    "test.txt":  CHIPS_ROOT / "SouthMN",
}

for fname, chip_dir in SPLITS.items():
    stems = sorted(
        p.stem
        for p in chip_dir.glob("chip_*.tif")
        if "_mask" not in p.name
    )
    out_path = OUT_DIR / fname
    with open(out_path, "w") as f:
        f.write("\n".join(stems) + "\n")
    print(f"Wrote {len(stems):>6} chips -> {out_path}")

print("\nDone. Split files written to:", OUT_DIR)
