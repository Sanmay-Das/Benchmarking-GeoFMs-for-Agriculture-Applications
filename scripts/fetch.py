#!/usr/bin/env python3
"""
Download exactly the data one benchmark cell needs, and nothing else.

The published dataset is 482 GB; a single (task, model, state) cell is a few
gigabytes. Downloading the repository wholesale is not a reasonable thing to
ask of anyone, so this resolves a cell to its individual files.

    python scripts/fetch.py --task cd --model prithvi --state iowa
    python scripts/fetch.py --task cd --state iowa --dry-run
    python scripts/fetch.py --task cd                    # every model, every state

Each cell needs two things:

  chips       change_detection/cd_<model>_<region>.tar on the Hub, unpacked to
              $MSR_DATA_ROOT/change_detection_chips/<model>/<region>/
  checkpoint  weights/change_detection/<model>_cd_<state>.pth, installed as
              $MSR_WEIGHTS/cd/cd_train_<model>[_<run>]/best_F1_model.pth,
              which is where paths.cd_checkpoint() looks.

Manifests are not published -- they contain absolute paths and are cheap to
rebuild -- so this regenerates them from the chips after unpacking.

Everything is idempotent: files already in place are left alone, so an
interrupted run is resumed by running it again.
"""

import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import paths as P       # noqa: E402
import registry as R    # noqa: E402

REPO_ID = "sanmay4119/geofm-agriculture-benchmark"
REPO_TYPE = "dataset"
CD_CKPT_NAME = "best_F1_model.pth"


# --- what a cell is made of -------------------------------------------------

def cd_chip_file(model, region):
    return "change_detection/cd_{}_{}.tar".format(model, region)


def cd_weight_file(model, state):
    return "weights/change_detection/{}_cd_{}.pth".format(model, state)


def cd_chip_dest(model, region):
    return P.CD_CHIPS / model / region


def cd_weight_dest(model, state):
    """Where cd_checkpoint() will look for this checkpoint."""
    region = R.test_region(state)
    return P.CD_WEIGHTS / R.checkpoint_dirname(model, region) / CD_CKPT_NAME


def cd_plan(model, state, splits):
    """Every (remote, destination) pair this cell needs.

    splits: which of the state's regions to fetch. Inference needs only
    "test"; training needs all three.
    """
    regions = R.cd_regions(state)
    items = []
    for role in splits:
        region = regions[role]
        items.append({
            "kind": "chips", "role": role, "region": region,
            "remote": cd_chip_file(model, region),
            "dest": cd_chip_dest(model, region),
            "model": model,
        })
    items.append({
        "kind": "weights", "role": "checkpoint", "region": R.test_region(state),
        "remote": cd_weight_file(model, state),
        "dest": cd_weight_dest(model, state),
        "model": model,
    })
    return items


# --- the Hub ----------------------------------------------------------------

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


def remote_index(api):
    """Filename -> size for everything published, in one request."""
    info = api.repo_info(REPO_ID, repo_type=REPO_TYPE, files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings}


# --- doing the work ---------------------------------------------------------

def unpack(tar_path, parent, expect):
    """Unpack a region tar, verifying it contains the directory it claims.

    The archives wrap a single top-level region directory, so extracting into
    the model directory produces <model>/<region>/. A tar that disagrees would
    scatter files into the wrong place, so check before extracting.
    """
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tops = {name.split("/")[0] for name in tf.getnames()}
        if tops != {expect}:
            raise SystemExit(
                "{} contains top-level {}, expected only '{}'. Refusing to "
                "unpack into {}.".format(tar_path.name, sorted(tops), expect, parent))
        tf.extractall(parent)


def install_weight(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(str(src), str(dest))


MANIFEST_DEPS = ("pandas", "rasterio")


def missing_manifest_deps():
    """Which manifest-building imports are unavailable in this interpreter.

    Checked before downloading rather than after: discovering a missing
    dependency at the end of an 80 GB transfer helps nobody.
    """
    import importlib.util
    return [m for m in MANIFEST_DEPS if importlib.util.find_spec(m) is None]


def stale_manifests(models, states, splits):
    """Models whose chips are unpacked but whose manifest is absent.

    Chips on disk with no manifest is the state left behind when manifest
    generation fails -- a missing dependency, an interrupted run. Without this
    check a re-run reports "nothing to download" and exits, leaving the cell
    permanently unusable, because the chips themselves are all present.
    """
    stale = set()
    for model in models:
        for st in states:
            regions = R.cd_regions(st)
            for role in splits:
                region = regions[role]
                if (cd_chip_dest(model, region).is_dir()
                        and not P.chips_csv(model, region).exists()):
                    stale.add(model)
    return stale


def regenerate_manifests(models):
    """Rebuild chip manifests from the chips just unpacked.

    A failure here does not lose the download -- the chips are on disk and
    this can be re-run -- so it is reported rather than raised.
    """
    script = ROOT / "scripts" / "generate_cd_csv.py"
    ok = True
    for model in sorted(models):
        base = P.CD_CHIPS / model
        if not base.is_dir():
            continue
        print("\n== manifests for {} ==".format(model))
        result = subprocess.run([sys.executable, str(script), "--base",
                                 str(base), "--region", "all"])
        if result.returncode != 0:
            ok = False
            print("  manifest generation failed for {}".format(model))
    return ok


def human(n):
    return "{:.2f} GB".format(n / 1e9) if n else "?"


def main():
    p = argparse.ArgumentParser(
        description="Download one benchmark cell from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Each cell needs")[0].strip())
    p.add_argument("--task", default="cd", choices=["cd"],
                   help="only change detection is published in full (default: cd)")
    p.add_argument("--model", default="all",
                   help="satmae, spectralgpt, prithvi, or all (default: all)")
    p.add_argument("--state", default="all",
                   help="{}, or all (default: all)".format(", ".join(R.states())))
    p.add_argument("--splits", default="test", choices=["test", "all"],
                   help="test region only (inference) or all three (training)")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be downloaded, with sizes, and stop")
    p.add_argument("--partial", action="store_true",
                   help="download a cell's available half even though the "
                        "cell cannot run (default: skip it)")
    p.add_argument("--keep-tars", action="store_true",
                   help="keep archives after unpacking (default: delete)")
    args = p.parse_args()

    models = sorted(R.MODELS) if args.model == "all" else [args.model]
    for m in models:
        R.model(m)
    states = R.states() if args.state == "all" else [args.state]
    for s in states:
        R.state(s)
    splits = ["train", "val", "test"] if args.splits == "all" else ["test"]

    lacking = missing_manifest_deps()
    if lacking:
        print("Warning: {} not importable, so chip manifests cannot be built.\n"
              "         The download will still run; finish with\n"
              "             pip install -r requirements/{}.txt\n"
              "             ./make_manifests.sh {}\n"
              .format(", ".join(lacking), models[0],
                      "" if args.model == "all" else args.model))

    api, download = hub()
    print("Reading the file index from {} ...".format(REPO_ID))
    index = remote_index(api)

    # Resolve cell by cell. A cell is all-or-nothing: chips without their
    # checkpoint, or a checkpoint without its chips, cannot produce a number,
    # so downloading either half wastes tens of gigabytes. An incomplete cell
    # is reported and skipped unless --partial says otherwise.
    wanted, present, missing, skipped = [], [], [], []
    for model in models:
        for st in states:
            items = cd_plan(model, st, splits)
            for item in items:
                item["state"] = st
                item["size"] = index.get(item["remote"], 0)
            absent = [i for i in items
                      if not i["dest"].exists() and i["remote"] not in index]
            if absent and not args.partial:
                missing.extend(absent)
                skipped.extend(i for i in items
                               if i not in absent and not i["dest"].exists())
                present.extend(i for i in items if i["dest"].exists())
                continue
            for item in items:
                if item["dest"].exists():
                    present.append(item)
                elif item["remote"] not in index:
                    missing.append(item)
                else:
                    wanted.append(item)

    if present:
        print("\nAlready present ({}):".format(len(present)))
        for i in present:
            print("  {:<14} {:<16} {}".format(i["model"], i["state"], i["dest"].name
                                              if i["kind"] == "weights" else i["region"]))

    if missing:
        print("\nNot published ({}):".format(len(missing)))
        for i in missing:
            print("  {:<14} {:<16} {:<12} {}".format(
                i["model"], i["state"], i["kind"], i["remote"]))
        print("  These cells cannot be reproduced from the Hub as it stands.")

    if skipped:
        total = sum(i["size"] for i in skipped)
        print("\nSkipped ({} files, {}) -- the other half of an incomplete "
              "cell:".format(len(skipped), human(total)))
        for i in skipped:
            print("  {:>9}  {}".format(human(i["size"]), i["remote"]))
        print("  Pass --partial to download these anyway.")

    total = sum(i["size"] for i in wanted)
    stale = stale_manifests(models, states, splits)
    if not wanted:
        if stale:
            print("\nNothing to download, but manifests are missing.")
            if not regenerate_manifests(stale):
                print("\nManifest generation failed; see above.")
                return 1
            print("\nManifests rebuilt. Data root: {}".format(P.DATA_ROOT))
            return 0
        print("\nNothing to download; everything is already in place.")
        return 1 if missing and not present else 0

    print("\nTo download ({} files, {}):".format(len(wanted), human(total)))
    for i in wanted:
        print("  {:>9}  {}".format(human(i["size"]), i["remote"]))

    if args.dry_run:
        print("\nDry run; nothing was downloaded.")
        return 0

    cache = P.DATA_ROOT / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)
    touched = set()

    for n, i in enumerate(wanted, 1):
        print("\n[{}/{}] {}  ({})".format(n, len(wanted), i["remote"], human(i["size"])))
        local = Path(download(REPO_ID, i["remote"], repo_type=REPO_TYPE,
                              local_dir=str(cache)))
        if i["kind"] == "chips":
            unpack(local, i["dest"].parent, i["region"])
            touched.add(i["model"])
            if not args.keep_tars:
                local.unlink()
        else:
            install_weight(local, i["dest"])
        print("  -> {}".format(i["dest"]))

    manifests_ok = regenerate_manifests(touched | stale_manifests(models, states, splits))

    # The per-file download cache holds only lock and metadata files once the
    # archives are unpacked, but leaving it behind is untidy.
    import shutil
    shutil.rmtree(cache, ignore_errors=True)

    print("\nDone. Data root: {}".format(P.DATA_ROOT))
    print("Weights:         {}".format(P.WEIGHTS))
    if not manifests_ok:
        print("\nChips are in place but manifests were not built. Install the "
              "requirements and run:\n    ./make_manifests.sh")
        return 1
    if missing:
        print("\n{} file(s) were not published; those cells remain "
              "unreproducible.".format(len(missing)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
