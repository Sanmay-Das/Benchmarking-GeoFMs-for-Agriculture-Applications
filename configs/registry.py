"""
What varies between backbones and regions, in one place.

The inference/visualization scripts were originally one file per
(model, region) pair -- 12 change-detection scripts that differed only in a
handful of strings and three lines of tensor handling. Everything that
actually differs now lives here, so a reviewer can check the experimental
setup by reading one table instead of diffing twelve files.

MODELS  -- per-backbone facts: chip size, where the vendored code lives,
           which builder to call, how inputs are normalized, and how the
           head's output is turned into a change probability.

REGIONS -- per-region facts, chiefly which training run supplies the
           checkpoint used to score that region. This mapping used to be
           implicit in a hardcoded path (SouthMN scored with the MN run);
           stating it explicitly makes it reviewable.

Normalization schemes:
    "zscore_shared"  per-band (x - mean) / std, one stat set for T1 and T2.
                     Nodata (-9999) filled with the band mean first.
    "zscore_paired"  as above but with separate T1 and T2 stat sets.
    "minmax"         per-chip, per-band rescale to [0,1]. Nodata -> 0.

Head schemes:
    "logprob"        model emits log-probabilities; probability = exp(x).
    "logits"         model emits raw logits; probability = softmax(x).
"""

MODELS = {
    "satmae": {
        "label": "SatMAE",
        "chip": 96,
        "code_dir": "SatMAE/ChangeDetection",
        "builder": ("src.model_cd_satmae", "build_satmae_cd"),
        "ckpt_dir": "SatMAE/ChangeDetection",
        "norm": "zscore_shared",
        "head": "logprob",
        # SatMAE paper Table 10; sensor-level, not year-specific.
        # Bands B02, B03, B04, B08, B11, B12.
        "mean": [1184.3824625, 1120.77120066, 1136.26026392,
                 1972.62420416, 1732.16362238, 1247.91870117],
        "std":  [650.2842772, 712.12507725, 965.23119807,
                 1364.38688993, 1310.36996126, 1087.6020813],
    },
    "spectralgpt": {
        "label": "SpectralGPT",
        "chip": 128,
        "code_dir": "IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection",
        "builder": ("src.model_cd_spectralgpt", "build_spectralgpt_cd"),
        "ckpt_dir": "IEEE_TPAMI_SpectralGPT/downstream_tasks/ChangeDetection",
        "norm": "minmax",
        "head": "logits",
    },
    "prithvi": {
        "label": "Prithvi",
        "chip": 224,
        "code_dir": "prithvi_finetune/ChangeDetection",
        "builder": ("src.model_cd_prithvi", "build_prithvi_cd"),
        "ckpt_dir": "prithvi_finetune/ChangeDetection",
        "norm": "zscore_paired",
        "head": "logprob",
        # Iowa z-score, matching training via dataset_cd_prithvi.py.
        "t1_mean": [1861.19006065, 2033.17032775, 2273.3793366,
                    3262.91588412, 4457.44718789, 3994.99188433],
        "t1_std":  [307.48006869, 351.67526808, 447.86017086,
                    654.812951, 697.75477528, 788.08159599],
        "t2_mean": [1704.63697798, 1961.37926168, 2034.19159947,
                    3929.94252524, 4352.22479367, 3695.04113396],
        "t2_std":  [341.76676414, 357.44447699, 513.43405597,
                    842.65097823, 920.0799827, 1054.49693239],
    },
}

# region -> the training run whose checkpoint scores it.
# "" means the un-suffixed checkpoint directory (the original Iowa run).
# "train_run" names the change-detection checkpoint directory ("" is the
# original un-suffixed Iowa run). "seg_split" names the directory holding that
# region's segmentation train/val/test lists, which uses the state's own name
# rather than the CD run key.
REGIONS = {
    "NWIA":    {"label": "Northwest Iowa",     "train_run": "",   "seg_split": "Iowa"},
    "SouthMN": {"label": "Southern Minnesota", "train_run": "MN", "seg_split": "MN"},
    "EastNC":  {"label": "Eastern N. Carolina", "train_run": "NC", "seg_split": "NC"},
    "SouthCA": {"label": "Southern California", "train_run": "CA", "seg_split": "CA"},
}


def model(name):
    if name not in MODELS:
        raise SystemExit("Unknown model '{}'. Choose from: {}".format(
            name, ", ".join(sorted(MODELS))))
    return MODELS[name]


def region(name):
    if name not in REGIONS:
        raise SystemExit("Unknown region '{}'. Choose from: {}".format(
            name, ", ".join(sorted(REGIONS))))
    return REGIONS[name]


def checkpoint_dirname(model_name, region_name):
    """Directory holding the fine-tuned checkpoint for this pair.

    e.g. ('satmae', 'SouthMN') -> 'cd_train_satmae_MN'
         ('satmae', 'NWIA')    -> 'cd_train_satmae'
    """
    run = region(region_name)["train_run"]
    return "cd_train_{}{}".format(model_name, "_" + run if run else "")


def pairs():
    """Every (model, region) combination, for batch drivers and tests."""
    return [(m, r) for m in sorted(MODELS) for r in sorted(REGIONS)]


# ---------------------------------------------------------------------------
# Semantic segmentation
#
# The segmentation runs were trained ad hoc over a long period and their
# checkpoints were never given a consistent naming scheme: SatMAE writes
# output_seg_<run>[_<head>]/checkpoint-best.pth, Prithvi writes
# experiments/.../<run>/best_mIoU_epoch_<N>.pth with the epoch number baked in,
# and SpectralGPT writes multi_train/best_mIoU_<run>_model.pth. There is no
# rule that derives one from the other, so the mapping is recorded explicitly
# rather than reconstructed.
#
# Keys are (model, head, region); head is "" for models with a single head.
# Values are paths relative to the model's own tree, which is what
# paths.seg_checkpoint() falls back to when MSR_WEIGHTS has no copy.
# ---------------------------------------------------------------------------

SEG_MODELS = {
    "satmae": {
        "label": "SatMAE",
        "chip": 96,
        "stride": 48,
        # FPN only. SatMAE segmentation was also trained with an FCN head and
        # with PSANet, and those runs still exist locally
        # (output_seg_*_fcn, output_seg_* without a suffix), but FPN is the
        # head published on Hugging Face and the one this benchmark reports.
        # Offering the others meant --head fcn resolved to a checkpoint nobody
        # can download.
        "heads": ["fpn"],
        "tree": "SatMAE",
        "num_classes": 14,
        "delta": 8,
        "batch": 32,
        # NWIA was evaluated by sliding a window over a stitched raster;
        # SouthMN was evaluated against its own pre-cut test chips.
        "input_mode": {"NWIA": "stack", "SouthMN": "chips"},
        "chips_dir": "SatMAE_chips_{run}/{region}",
        "splits": "SatMAE_chips_multitemporal/{run}/test.txt",
        # Seg uses a 3-date stack: the 6-band stats tiled to 18 channels.
        "seg_norm": "zscore_tiled",
        "mean6": [1184.3824625, 1120.77120066, 1136.26026392,
                  1972.62420416, 1732.16362238, 1247.91870117],
        "std6":  [650.2842772, 712.12507725, 965.23119807,
                  1364.38688993, 1310.36996126, 1087.6020813],
        # Encoder geometry shared by all three heads.
        "encoder": {"patch_size": 8, "in_chans": 18, "drop_path": 0.2,
                    "channel_groups": [[0,1,2,3,4,5],[6,7,8,9,10,11],[12,13,14,15,16,17]]},
        "head_classes": {"fpn": ("models_satmae_fpn", "SatMAEFPN")},
    },
    "spectralgpt": {
        "label": "SpectralGPT",
        "chip": 128,
        "stride": 64,
        "heads": [""],
        "tree": "IEEE_TPAMI_SpectralGPT/downstream_tasks/SegMunich",
        "num_classes": 13,
        "delta": 8,
        "batch": 32,
        "input_mode": "stack",
        "seg_norm": "global_minmax",
        "gmin": [641.0, 1018.0, 1013.0, 921.0, 1030.0, 1022.0, 764.0, 918.0, 675.0,
                 713.0, 992.0, 991.0, 687.0, 1000.0, 977.0, 881.0, 1053.0, 1030.0],
        "gmax": [19467.0, 18496.0, 17621.0, 13263.0, 15384.0, 16124.0, 18711.0,
                 18035.0, 17259.0, 10610.0, 12480.0, 14638.0, 18656.0, 17712.0,
                 17120.0, 15848.0, 16108.0, 16053.0],
        "seg_builder": ("src.models_vit_tensor_CD_2", "vit_base_patch8"),
    },
    "prithvi": {
        "label": "Prithvi",
        "chip": 224,
        "stride": 112,
        "heads": [""],
        "tree": "experiments/prithvi_multi_temporal_crop_classification",
        "num_classes": 13,
        "delta": 8,
        "batch": 16,
        "input_mode": "stack",
        "seg_norm": "zscore18",
        "mean18": [1683.21816572, 1795.1286564, 2027.99224864, 2614.43523828,
                   4018.17499175, 3795.35160064, 1332.62496374, 1593.52693158,
                   1510.5718514, 4645.33493779, 2915.19120058, 2091.37382972,
                   1444.89219603, 1713.57935808, 1806.45808454, 3524.61079859,
                   3447.32103589, 2790.36075006],
        "std18":  [201.19653431, 253.75733532, 338.02958852, 492.03218955,
                   576.15203045, 595.7844964, 194.01547233, 226.79497442,
                   411.89867229, 1132.56897351, 477.13239916, 482.08074425,
                   203.44619838, 224.95563218, 366.10911266, 984.88041998,
                   610.66313299, 721.43276144],
        "config": "prithvi_finetune/configs/multi_temporal_crop_classification.py",
        # Prithvi consumes (bands, dates, H, W), not 18 flat channels.
        "layout": "bands_dates",
    },
}

SEG_CHECKPOINTS = {
    ("satmae", "fpn",    "NWIA"):    "output_seg_Iowa_fpn/checkpoint-best.pth",
    ("satmae", "fpn",    "SouthMN"): "output_seg_MN_fpn/checkpoint-best.pth",

    # FPN was also trained for North Carolina and California. These were
    # omitted here while only two regions had been scored; the runs exist
    # (output_seg_NC_fpn, output_seg_CA_fpn) and FPN is the head published on
    # Hugging Face, so all four states are recorded for it.
    ("satmae", "fpn",    "EastNC"):  "output_seg_NC_fpn/checkpoint-best.pth",
    ("satmae", "fpn",    "SouthCA"): "output_seg_CA_fpn/checkpoint-best.pth",

    ("spectralgpt", "", "NWIA"):    "multi_train/best_mIoU_CentEastIA_model.pth",
    ("spectralgpt", "", "SouthMN"): "multi_train/best_mIoU_NorthCentMN_model.pth",
    ("spectralgpt", "", "EastNC"):  "multi_train/best_mIoU_NEECNC_model.pth",
    ("spectralgpt", "", "SouthCA"): "multi_train/best_mIoU_NorthCentCA_model.pth",

    # Prithvi's SouthCA run wrote to the experiment root rather than a
    # per-region subdirectory, and each region stopped at a different epoch.
    ("prithvi", "", "NWIA"):    "IA/best_mIoU_epoch_60.pth",
    ("prithvi", "", "SouthMN"): "MN/best_mIoU_epoch_60.pth",
    ("prithvi", "", "EastNC"):  "IL/best_mIoU_epoch_30.pth",
    ("prithvi", "", "SouthCA"): "best_mIoU_epoch_15.pth",
}


def seg_pairs():
    """Every (model, head, region) combination that has a checkpoint."""
    return sorted(SEG_CHECKPOINTS)


# ---------------------------------------------------------------------------
# Change-detection training splits
#
# Each run trains on one sub-region, validates on a second and tests on a
# third, all within the same state. These were previously hardcoded three
# times per state -- once in train_cd_satmae_<S>.py, once in
# train_cd_spectralgpt_<S>.py and once in train_cd_prithvi_<S>.py -- which is
# why the per-state training scripts existed at all.
#
# Keyed by split name; the test region is the one the paper reports.
# ---------------------------------------------------------------------------

CD_SPLITS = {
    "IA": {"train": "CentIA",  "val": "EastIA", "test": "NWIA",
           "label": "Iowa"},
    "CA": {"train": "NorthCA", "val": "CentCA", "test": "SouthCA",
           "label": "California"},
    "MN": {"train": "NorthMN", "val": "CentMN", "test": "SouthMN",
           "label": "Minnesota"},
    "NC": {"train": "NENC",    "val": "ECNC",   "test": "EastNC",
           "label": "North Carolina"},
}


def split(name):
    if name not in CD_SPLITS:
        raise SystemExit("Unknown split '{}'. Choose from: {}".format(
            name, ", ".join(sorted(CD_SPLITS))))
    return CD_SPLITS[name]


# ---------------------------------------------------------------------------
# States
#
# The user-facing unit. A state is what the published checkpoints are named
# after (satmae_cd_iowa.pth), and it determines everything else about a run:
# which sub-regions are train/val/test, and which checkpoint to score with.
#
# Regions are an implementation detail -- nobody outside this project knows
# that NWIA is the Iowa test region or that NENC and ECNC are its North
# Carolina training partners. Scripts take --state and derive the rest.
#
# Two keys because a state's change-detection split key is not its
# segmentation split directory: Iowa is "IA" for CD and "Iowa" for seg. That
# distinction used to live implicitly in REGIONS' train_run/seg_split pair.
# ---------------------------------------------------------------------------

STATES = {
    "iowa":           {"cd": "IA", "seg": "Iowa", "label": "Iowa"},
    "minnesota":      {"cd": "MN", "seg": "MN",   "label": "Minnesota"},
    "north_carolina": {"cd": "NC", "seg": "NC",   "label": "North Carolina"},
    "california":     {"cd": "CA", "seg": "CA",   "label": "California"},
}


def state(name):
    if name not in STATES:
        raise SystemExit("Unknown state '{}'. Choose from: {}".format(
            name, ", ".join(sorted(STATES))))
    return STATES[name]


def cd_regions(state_name):
    """The train/val/test regions for a state's change-detection split."""
    return split(state(state_name)["cd"])


def test_region(state_name):
    """The single region a state's reported number is computed on."""
    return cd_regions(state_name)["test"]


def state_of_region(region_name):
    """Inverse of test_region: which state a region belongs to.

    Accepts any of a split's three regions, not just the test one, so
    training paths can resolve a state from whichever region they hold.
    """
    for name, meta in STATES.items():
        regions = split(meta["cd"])
        if region_name in (regions["train"], regions["val"], regions["test"]):
            return name
    raise SystemExit("Region '{}' belongs to no known state.".format(region_name))


def states():
    """Every state name, for batch drivers and tests."""
    return sorted(STATES)


# ---------------------------------------------------------------------------
# Segmentation on Hugging Face
#
# The published names and the names training wrote do not correspond. Training
# produced output_seg_<run>[_<head>]/checkpoint-best.pth for SatMAE,
# multi_train/best_mIoU_<run>_model.pth for SpectralGPT and
# <run>/best_mIoU_epoch_<N>.pth for Prithvi; the Hub carries one flat file per
# model and state, <model>_seg_<state>.pth. fetch.py installs a downloaded file
# under the name seg_checkpoint() expects, so this records which head each
# published file actually is.
#
# Only one head per model was uploaded. For SatMAE that is FPN, identified by
# size: the published files are 2,628,248,073 bytes, matching output_seg_*_fpn
# exactly, against 2.54 GB for psanet and 2.47 GB for fcn. So fcn and psanet
# are reproducible only by someone who trains them.
# ---------------------------------------------------------------------------

SEG_PUBLISHED_HEAD = {"satmae": "fpn", "spectralgpt": "", "prithvi": ""}


def seg_published_head(model_name):
    """The head whose checkpoint is published for this backbone."""
    if model_name not in SEG_PUBLISHED_HEAD:
        raise SystemExit("Unknown segmentation model '{}'. Choose from: {}".format(
            model_name, ", ".join(sorted(SEG_PUBLISHED_HEAD))))
    return SEG_PUBLISHED_HEAD[model_name]


def seg_test_region(state_name):
    """The region a state's segmentation number is computed on.

    The same region as change detection; segmentation reaches it through a
    different split directory, which is what STATES' "seg" key records.
    """
    return test_region(state_name)


def seg_checkpoint_relative(model_name, state_name, head=None):
    """Where seg_checkpoint() expects this state's checkpoint to live.

    Returns the path relative to MSR_WEIGHTS/seg/<model>/, so a flat file
    downloaded from the Hub can be installed where the code will find it.
    """
    if head is None:
        head = seg_published_head(model_name)
    region = seg_test_region(state_name)
    key = (model_name, head, region)
    if key not in SEG_CHECKPOINTS:
        raise SystemExit(
            "No segmentation checkpoint recorded for model={} head={!r} "
            "state={} (region {}).".format(model_name, head, state_name, region))
    return SEG_CHECKPOINTS[key]


def seg_states(model_name):
    """States this backbone has a recorded segmentation checkpoint for."""
    head = seg_published_head(model_name)
    out = []
    for name in states():
        try:
            seg_checkpoint_relative(model_name, name, head)
        except SystemExit:
            continue
        out.append(name)
    return out
