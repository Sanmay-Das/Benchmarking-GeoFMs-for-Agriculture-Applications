"""
Rewrite change-detection chip manifests to use portable, relative paths.

The manifests were generated with absolute paths baked into every row:

    /bigdata/.../change_detection_chips/satmae/NWIA/images/NWIA_00048_00384_t1.tif

which pins them to the machine that produced them. This rewrites the t1/t2/mask
columns to be relative to the change_detection_chips root:

    satmae/NWIA/images/NWIA_00048_00384_t1.tif

Scripts resolve them at load time via paths.load_chips_csv(), so the same CSV
works wherever the user unpacked the data.

Idempotent: rows that are already relative are left alone, so re-running is a
no-op. Use --check to report without writing, and --backup to keep a .bak copy.

    python scripts/make_manifests_portable.py --check
    python scripts/make_manifests_portable.py /path/to/change_detection_chips
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "configs"))
from paths import CD_CHIPS, CHIP_PATH_COLUMNS, _relative_chip_path  # noqa: E402

import pandas as pd  # noqa: E402


def convert(csv_path, check, backup):
    """Return (rows, changed_cells) for one manifest."""
    df = pd.read_csv(csv_path)
    present = [c for c in CHIP_PATH_COLUMNS if c in df.columns]
    if not present:
        return len(df), 0

    changed = 0
    for col in present:
        new = df[col].map(lambda v: str(_relative_chip_path(v)))
        changed += int((new != df[col]).sum())
        df[col] = new

    if changed and not check:
        if backup:
            shutil.copy2(csv_path, str(csv_path) + ".bak")
        df.to_csv(csv_path, index=False)

    return len(df), changed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=None,
                    help="change_detection_chips directory (default: CD_CHIPS)")
    ap.add_argument("--check", action="store_true",
                    help="report what would change; write nothing")
    ap.add_argument("--backup", action="store_true",
                    help="keep a .bak copy of every rewritten manifest")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve() if args.root else CD_CHIPS
    if not root.is_dir():
        raise SystemExit("Not a directory: {}".format(root))

    manifests = sorted(root.rglob("*_chips.csv"))
    if not manifests:
        raise SystemExit("No *_chips.csv found under {}".format(root))

    total_rows = total_changed = touched = 0
    for csv_path in manifests:
        rows, changed = convert(csv_path, args.check, args.backup)
        total_rows += rows
        total_changed += changed
        touched += bool(changed)
        status = "already relative" if not changed else "{} cells".format(changed)
        print("  {:<48} {:>7} rows  {}".format(
            str(csv_path.relative_to(root)), rows, status))

    verb = "would rewrite" if args.check else "rewrote"
    print("\n{} {}/{} manifests ({} rows, {} path cells) under {}".format(
        verb, touched, len(manifests), total_rows, total_changed, root))
    if args.check and total_changed:
        print("Re-run without --check to apply.")


if __name__ == "__main__":
    main()
