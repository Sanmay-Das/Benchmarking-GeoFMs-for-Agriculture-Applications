#!/usr/bin/env python3
"""
Download the pretrained backbones that training starts from.

    python scripts/fetch_backbones.py                    # all three
    python scripts/fetch_backbones.py --model prithvi
    python scripts/fetch_backbones.py --dry-run

Inference does not need these. A benchmark cell runs from the fine-tuned
checkpoint fetch.py installs, which already contains the backbone weights.
Training does need them: --pretrain-path defaults to $GFM_WEIGHTS/<file>, and
weights/ is gitignored, so a fresh clone has nothing there.

Each file is published under weights/<model>/ on the Hub and lands flat in
$GFM_WEIGHTS, which is where the train_cd_*.py defaults look.

These are the original authors' checkpoints, mirrored so that a clone resolves
one dataset rather than three. The upstream sources, should you prefer them:

    prithvi      huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
    satmae       github.com/sustainlab-group/SatMAE (fMoW-Sentinel, ViT-Large)
    spectralgpt  zenodo.org/records/8412455 (SpectralGPT+, BigEarthNet)

Already-downloaded files are left alone, so an interrupted run is resumed by
running it again.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import paths as P       # noqa: E402

REPO_ID = "sanmay4119/geofm-agriculture-benchmark"
REPO_TYPE = "dataset"

# model -> filename, as published and as the training defaults expect it
BACKBONES = {
    "prithvi": "Prithvi_EO_V1_100M.pt",
    "satmae": "pretrain-vit-large-e199.pth",
    "spectralgpt": "SpectralGPT+.pth",
}


def remote(model):
    return "weights/{}/{}".format(model, BACKBONES[model])


def dest(model):
    """Flat in GFM_WEIGHTS, matching train_cd_*.py --pretrain-path defaults."""
    return P.WEIGHTS / BACKBONES[model]


def hub():
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub is not installed.\n"
            "    pip install huggingface_hub\n"
            "It is listed in requirements/, so this usually means the "
            "virtual environment is not active.")
    return HfApi(), hf_hub_download


def human(n):
    return "{:.2f} GB".format(n / 1e9) if n else "?"


def main():
    p = argparse.ArgumentParser(
        description="Download pretrained backbones from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Each file is published")[0].strip())
    p.add_argument("--model", default="all",
                   help="{}, or all (default: all)".format(
                       ", ".join(sorted(BACKBONES))))
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be downloaded, with sizes, and stop")
    args = p.parse_args()

    if args.model == "all":
        models = sorted(BACKBONES)
    elif args.model in BACKBONES:
        models = [args.model]
    else:
        raise SystemExit("Unknown model {!r}. Choose from: {}".format(
            args.model, ", ".join(sorted(BACKBONES))))

    api, download = hub()
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    sizes = {s.rfilename: (s.size or 0) for s in info.siblings}

    todo = []
    for model in models:
        target = dest(model)
        if target.exists():
            print("[have] {:<12} {}".format(model, target))
            continue
        if remote(model) not in sizes:
            raise SystemExit(
                "{} is not published at {}. The repository layout may have "
                "changed; see the upstream sources in this script's "
                "docstring.".format(remote(model), REPO_ID))
        todo.append(model)

    if not todo:
        print("\nNothing to download.")
        return

    total = sum(sizes[remote(m)] for m in todo)
    print("\nTo download ({} files, {}):".format(len(todo), human(total)))
    for model in todo:
        print("    {:>8}  {}".format(human(sizes[remote(model)]), remote(model)))

    if args.dry_run:
        return

    P.WEIGHTS.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(todo, 1):
        print("\n[{}/{}] {}".format(i, len(todo), remote(model)))
        cached = download(repo_id=REPO_ID, repo_type=REPO_TYPE,
                          filename=remote(model))
        # Move rather than copy: the Hub cache is a second full copy of a
        # multi-gigabyte file, and nothing else reads it.
        os.replace(cached, dest(model))
        print("  installed {}".format(dest(model)))

    print("\nDone. Training reads these from {}.".format(P.WEIGHTS))


if __name__ == "__main__":
    main()
