"""
Central path resolution for the GeoFM agriculture benchmark.

All scripts import their paths from here instead of hardcoding them, so the
code runs unchanged on any machine.

Three roots, each overridable by an environment variable:

    MSR_ROOT         Repository location (where this code lives).
                     Default: auto-detected from this file's location.

    MSR_DATA_ROOT    Datasets downloaded from Hugging Face.
                     Default: $MSR_ROOT/data

    MSR_OUTPUT_ROOT  Predictions, logs, and checkpoints written by these
                     scripts. Default: $MSR_ROOT/outputs

Typical use after downloading data from Hugging Face:

    export MSR_DATA_ROOT=/scratch/me/geofm-data
    python infer_cd.py --model spectralgpt --region SouthMN

Nothing is created until a script actually writes; importing this module has
no side effects beyond resolving paths.
"""

import os
from pathlib import Path, PurePosixPath


def _env_path(var, default):
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


# Repository root: the parent of the directory holding this file.
MSR_ROOT = _env_path("MSR_ROOT", Path(__file__).resolve().parent.parent)

# Where the user downloaded the datasets.
DATA_ROOT = _env_path("MSR_DATA_ROOT", MSR_ROOT / "data")

# Where this code writes its results.
OUTPUT_ROOT = _env_path("MSR_OUTPUT_ROOT", MSR_ROOT / "outputs")

# Dataset subdirectories, matching the layout published on Hugging Face.
CD_CHIPS = DATA_ROOT / "change_detection_chips"
SEG_CHIPS = DATA_ROOT / "segmentation_chips"

# Pretrained backbone checkpoints (downloaded separately; see README).
WEIGHTS = _env_path("MSR_WEIGHTS", MSR_ROOT / "weights")

# Vendored upstream model code, needed on sys.path by the inference scripts.
SATMAE_DIR = MSR_ROOT / "SatMAE"
SPECTRALGPT_DIR = MSR_ROOT / "IEEE_TPAMI_SpectralGPT"
PRITHVI_DIR = MSR_ROOT / "prithvi_finetune"

# Outputs.
PREDICTIONS = OUTPUT_ROOT / "predictions"
LOGS = OUTPUT_ROOT / "logs"
CHECKPOINTS = OUTPUT_ROOT / "checkpoints"


def require(path, what):
    """Fail early with an actionable message instead of a deep stack trace."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            "Missing {}: {}\n"
            "Set MSR_DATA_ROOT to the directory holding the data you\n"
            "downloaded from Hugging Face, or see README.md for the layout."
            .format(what, p)
        )
    return p


# ---------------------------------------------------------------------------
# Chip manifests
#
# The CSVs shipped with the dataset store chip paths RELATIVE to CD_CHIPS, e.g.
#
#     satmae/NWIA/images/NWIA_00048_00384_t1.tif
#
# so the same file works on any machine. Absolute paths are resolved at load
# time against wherever the user actually put the data. Always read a manifest
# through load_chips_csv() rather than pd.read_csv(), or MSR_DATA_ROOT is
# silently bypassed.
# ---------------------------------------------------------------------------

# Columns holding chip file paths in a change-detection manifest.
CHIP_PATH_COLUMNS = ("t1", "t2", "mask")


def chips_csv(model, region):
    """Manifest for one backbone/region pair, e.g. chips_csv('satmae', 'NWIA')."""
    return CD_CHIPS / model / "{}_chips.csv".format(region)


def load_chips_csv(path, root=None):
    """Read a chip manifest, resolving relative chip paths to absolute ones.

    Drop-in replacement for pd.read_csv() at the inference/visualization call
    sites: the returned frame has the same columns, with t1/t2/mask made
    absolute against CD_CHIPS (or `root`).

    Legacy manifests that still hold absolute paths are re-rooted onto the
    current CD_CHIPS, so a CSV generated on another machine keeps working.
    """
    import pandas as pd

    root = Path(root) if root is not None else CD_CHIPS
    require(path, "chip manifest")
    df = pd.read_csv(path)

    for col in CHIP_PATH_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda v: str(root / _relative_chip_path(v)))

    return df


def _relative_chip_path(value):
    """Strip any absolute prefix from a manifest path, leaving <model>/<region>/...

    Relative values pass through untouched. Absolute values are cut at the last
    'change_detection_chips' segment, which is what makes manifests written on
    another machine portable.
    """
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if "change_detection_chips" in parts:
        cut = len(parts) - 1 - parts[::-1].index("change_detection_chips")
        return PurePosixPath(*parts[cut + 1:])
    return PurePosixPath(*[p for p in parts if p != "/"])


# ---------------------------------------------------------------------------
# Fine-tuned checkpoints
#
# Published layout (what a user gets after downloading the weights):
#
#     $MSR_WEIGHTS/cd/cd_train_satmae_MN/best_F1_model.pth
#
# Historically these were written next to the training code, inside the repo
# itself (SatMAE/ChangeDetection/cd_train_satmae_MN/...). Those files are
# gitignored, so a fresh clone never has them. We look in MSR_WEIGHTS first
# and fall back to the in-tree location, which keeps an existing working copy
# running while making a clean checkout work for everyone else.
# ---------------------------------------------------------------------------

CD_WEIGHTS = WEIGHTS / "cd"


def cd_checkpoint(model, region, filename="best_F1_model.pth", must_exist=True):
    """Path to the change-detection checkpoint for one (model, region) pair.

    Searches MSR_WEIGHTS first, then the legacy in-repo training directory.
    Raises with download instructions when neither exists.
    """
    import registry

    dirname = registry.checkpoint_dirname(model, region)
    candidates = [
        CD_WEIGHTS / dirname / filename,
        MSR_ROOT / registry.model(model)["ckpt_dir"] / dirname / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if not must_exist:
        return candidates[0]

    raise SystemExit(
        "Missing change-detection checkpoint for {} / {}.\n"
        "Looked in:\n  {}\n"
        "Download the fine-tuned weights and place them under\n"
        "  {}\n"
        "or set MSR_WEIGHTS to the directory that holds them. See README.md."
        .format(model, region,
                "\n  ".join(str(c) for c in candidates),
                CD_WEIGHTS / dirname)
    )


def save_chips_csv(df, out_csv):
    """Write a chip manifest with portable, relative chip paths.

    Counterpart to load_chips_csv(). The generators build their rows from
    absolute glob results; relativizing on write keeps the published CSVs
    machine-independent, so regenerating a manifest cannot reintroduce paths
    that only resolve on the machine that produced it.
    """
    df = df.copy()
    for col in CHIP_PATH_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: str(_relative_chip_path(v)))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# Segmentation: checkpoints and multitemporal stacks
# ---------------------------------------------------------------------------

SEG_WEIGHTS = WEIGHTS / "seg"

# Stitched multitemporal rasters the segmentation scripts run over. These were
# read from scripts/processed_stacks/, i.e. input data stored inside the code
# tree, which meant they could never travel with a clone. They belong with the
# rest of the data, under MSR_DATA_ROOT.
STACKS = DATA_ROOT / "processed_stacks"


def seg_checkpoint(model, region, head="", must_exist=True):
    """Path to the segmentation checkpoint for one (model, head, region).

    Searches MSR_WEIGHTS/seg first, then the run directory inside the model's
    own tree where training originally wrote it.
    """
    import registry

    key = (model, head, region)
    if key not in registry.SEG_CHECKPOINTS:
        raise SystemExit(
            "No segmentation checkpoint recorded for model={} head={!r} region={}.\n"
            "Known combinations:\n  {}".format(
                model, head, region,
                "\n  ".join("{} {!r} {}".format(*k) for k in registry.seg_pairs()))
        )

    relative = registry.SEG_CHECKPOINTS[key]
    tree = registry.SEG_MODELS[model]["tree"]
    candidates = [
        SEG_WEIGHTS / model / relative,
        MSR_ROOT / tree / relative,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    if not must_exist:
        return candidates[0]

    raise SystemExit(
        "Missing segmentation checkpoint for {} {} {}.\n"
        "Looked in:\n  {}\n"
        "Download the fine-tuned weights and place them under\n"
        "  {}\n"
        "or set MSR_WEIGHTS to the directory that holds them. See README.md."
        .format(model, head or "(single head)", region,
                "\n  ".join(str(c) for c in candidates),
                (SEG_WEIGHTS / model / relative).parent)
    )


def stack_path(region, must_exist=True):
    """Stitched multitemporal raster for one region."""
    p = STACKS / region / "{}_multitemporal_stack.tif".format(region)
    if must_exist:
        require(p, "multitemporal stack for {}".format(region))
    return p
