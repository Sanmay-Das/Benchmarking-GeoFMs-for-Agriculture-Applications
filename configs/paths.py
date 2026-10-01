"""
Central path resolution for the GeoFM agriculture benchmark.

All scripts import their paths from here instead of hardcoding them, so the
code runs unchanged on any machine.

Three roots, each overridable by an environment variable:

    GFM_ROOT         Repository location (where this code lives).
                     Default: auto-detected from this file's location.

    GFM_DATA_ROOT    Datasets downloaded from Hugging Face.
                     Default: $GFM_ROOT/data

    GFM_OUTPUT_ROOT  Predictions, logs, and checkpoints written by these
                     scripts. Default: $GFM_ROOT/outputs

Typical use after downloading data from Hugging Face:

    export GFM_DATA_ROOT=/scratch/me/geofm-data
    python infer_cd.py --model spectralgpt --region SouthMN

Nothing is created until a script actually writes; importing this module has
no side effects beyond resolving paths.
"""

import os
from pathlib import Path, PurePosixPath


def _env(var):
    """Read a GFM_* variable, accepting the older MSR_* spelling.

    These were named after the directory this work was developed in
    (MS_Research), which meant nothing to anyone else. GFM_* replaces them;
    the old names still work so existing shells and job scripts do not break.
    """
    value = os.environ.get(var)
    if value:
        return value
    if var.startswith("GFM_"):
        return os.environ.get("MSR_" + var[4:])
    return None


def _env_path(var, default):
    value = _env(var)
    return Path(value).expanduser().resolve() if value else default


# Repository root: the parent of the directory holding this file.
GFM_ROOT = _env_path("GFM_ROOT", Path(__file__).resolve().parent.parent)

# Where the user downloaded the datasets.
DATA_ROOT = _env_path("GFM_DATA_ROOT", GFM_ROOT / "data")

# Where this code writes its results.
OUTPUT_ROOT = _env_path("GFM_OUTPUT_ROOT", GFM_ROOT / "outputs")

# Dataset subdirectories, matching the layout published on Hugging Face.
CD_CHIPS = DATA_ROOT / "change_detection_chips"
SEG_CHIPS = DATA_ROOT / "segmentation_chips"

# Pretrained backbone checkpoints (downloaded separately; see README).
WEIGHTS = _env_path("GFM_WEIGHTS", GFM_ROOT / "weights")

# Vendored upstream model code, needed on sys.path by the inference scripts.
SATMAE_DIR = GFM_ROOT / "SatMAE"
SPECTRALGPT_DIR = GFM_ROOT / "IEEE_TPAMI_SpectralGPT"
PRITHVI_DIR = GFM_ROOT / "prithvi_finetune"

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
            "Set GFM_DATA_ROOT to the directory holding the data you\n"
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
# through load_chips_csv() rather than pd.read_csv(), or GFM_DATA_ROOT is
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
#     $GFM_WEIGHTS/cd/cd_train_satmae_MN/best_F1_model.pth
#
# Historically these were written next to the training code, inside the repo
# itself (SatMAE/ChangeDetection/cd_train_satmae_MN/...). Those files are
# gitignored, so a fresh clone never has them. We look in GFM_WEIGHTS first
# and fall back to the in-tree location, which keeps an existing working copy
# running while making a clean checkout work for everyone else.
# ---------------------------------------------------------------------------

CD_WEIGHTS = WEIGHTS / "cd"


def cd_checkpoint(model, region, filename="best_F1_model.pth", must_exist=True):
    """Path to the change-detection checkpoint for one (model, region) pair.

    Searches GFM_WEIGHTS first, then the legacy in-repo training directory.
    Raises with download instructions when neither exists.
    """
    import registry

    dirname = registry.checkpoint_dirname(model, region)
    candidates = [
        CD_WEIGHTS / dirname / filename,
        GFM_ROOT / registry.model(model)["ckpt_dir"] / dirname / filename,
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
        "or set GFM_WEIGHTS to the directory that holds them. See README.md."
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
# rest of the data, under GFM_DATA_ROOT.
STACKS = DATA_ROOT / "processed_stacks"


def seg_checkpoint(model, region, head="", must_exist=True):
    """Path to the segmentation checkpoint for one (model, head, region).

    Searches GFM_WEIGHTS/seg first, then the run directory inside the model's
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
        GFM_ROOT / tree / relative,
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
        "or set GFM_WEIGHTS to the directory that holds them. See README.md."
        .format(model, head or "(single head)", region,
                "\n  ".join(str(c) for c in candidates),
                (SEG_WEIGHTS / model / relative).parent)
    )


def seg_chips_dir(model, region, must_exist=True):
    """Directory of segmentation chips for one (model, region).

    Prefers the published layout, segmentation_chips/<model>/<region>/, and
    falls back to the ad hoc directories the original runs used so an existing
    working copy keeps running.
    """
    import registry

    candidates = [SEG_CHIPS / model / region]
    spec = registry.SEG_MODELS.get(model, {})
    legacy = spec.get("chips_dir")
    if legacy:
        run = registry.region(region)["seg_split"]
        candidates.append(DATA_ROOT / legacy.format(region=region, run=run))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if not must_exist:
        return candidates[0]
    raise SystemExit(
        "No segmentation chips for {} / {}.\n"
        "Looked in:\n  {}\n"
        "Download them with\n"
        "  python scripts/fetch.py --task seg --model {} --state {}"
        .format(model, region, "\n  ".join(str(c) for c in candidates),
                model, registry.state_of_region(region)))


def seg_splits_file(model, region, must_exist=True):
    """File listing the test chip basenames for one (model, region).

    The test sets the paper was scored on ship with the repository under
    splits/seg/<model>/, and are used first. Where none ships, fall back to
    the list fetch.py writes by listing the unpacked chips -- which can hold
    chips the paper did not score, so it is only a fallback. The legacy
    per-run test.txt is accepted last.
    """
    import registry

    candidates = [GFM_ROOT / "splits" / "seg" / model / "{}_test.txt".format(region),
                  SEG_CHIPS / model / "{}_test.txt".format(region)]
    spec = registry.SEG_MODELS.get(model, {})
    legacy = spec.get("splits")
    if legacy:
        run = registry.region(region)["seg_split"]
        candidates.append(DATA_ROOT / legacy.format(region=region, run=run))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not must_exist:
        return candidates[1]          # where fetch.py will write it
    raise SystemExit(
        "No test split for {} / {}.\n"
        "Looked in:\n  {}\n"
        "fetch.py writes this when it unpacks the chips."
        .format(model, region, "\n  ".join(str(c) for c in candidates)))


def stack_path(region, must_exist=True):
    """Stitched multitemporal raster for one region."""
    p = STACKS / region / "{}_multitemporal_stack.tif".format(region)
    if must_exist:
        require(p, "multitemporal stack for {}".format(region))
    return p


# ---------------------------------------------------------------------------
# Segmentation chip names
# ---------------------------------------------------------------------------
# One chip is stored as an image plus a mask, and the models spell the pair
# differently:
#
#   SpectralGPT  SouthCA_chip_1024_1280.tif   SouthCA_chip_1024_1280_mask.tif
#   SatMAE       chip_1440_2352.tif           chip_1440_2352_mask.tif
#   Prithvi      chip_10080_10080_merged.tif  chip_10080_10080.mask.tif
#
# Everything downstream works on the chip name -- the stem with the image or
# mask suffix removed -- and finds the files from it.

import re as _re

_CHIP_SUFFIXES = (".mask", "_mask", "_merged")
CHIP_NAME_RE = _re.compile(r"_(\d+)_(\d+)$")


def chip_name(stem):
    """The chip a file stem belongs to, or None if it is not a chip.

    Strips the image/mask suffixes above, then requires a trailing
    _<row>_<col>; that excludes region rasters such as
    SouthCA_cdl_epsg5070_10m and files like chip_stats.
    """
    for suffix in _CHIP_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    return stem if CHIP_NAME_RE.search(stem) else None


def chip_image(data_dir, name):
    """Path of a chip's image, whichever spelling the model uses."""
    import os
    for suffix in (".tif", "_merged.tif"):
        path = os.path.join(str(data_dir), name + suffix)
        if os.path.exists(path):
            return path
    raise SystemExit("No image for chip {} in {}".format(name, data_dir))
