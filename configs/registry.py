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
# paths.seg_checkpoint() falls back to when GFM_WEIGHTS has no copy.
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
        # Scored as in evaluate_seg_fpn_corrected.py: ground truth stitched
        # onto one canvas, each pixel counted once. Each state's model learned
        # its own region's mask convention, so the same offset applies to
        # masks and predictions: Iowa masks are 1-13 (0 = NoData), the other
        # three are already 0-12. Iowa contributes whole chips to the ground
        # truth; the others only their centre, as that script does.
        # infer_satmae_fpn_*.py fed chips to the model as read; only the
        # Prithvi and SpectralGPT scripts replaced -9999 NoData with 0 first.
        "zero_nodata": False,
        "scorer": "canvas",
        "mask_offset": {"NWIA": 1, "default": 0},
        "pred_offset": {"NWIA": 1, "default": 0},
        "gt_inner": {"NWIA": False, "default": True},
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
        # Scored as in scripts/infer_finder.py: one confusion matrix summed
        # chip by chip, so overlapping chips count a pixel more than once.
        # Masks are 1-13 (0 = NoData); predictions are already 0-12.
        "scorer": "per_chip",
        "mask_offset": 1,
        "pred_offset": 0,
        "delta": 8,
        "batch": 32,
        "input_mode": "stack",
        "seg_norm": "global_minmax",
        # Min-max bounds are per region, not global: each scene was scaled by
        # its own observed range. Values copied from the infer_spectralgpt_*.py
        # scripts that produced the published numbers.
        "gminmax": {
            "NWIA": {
                "gmin": [296.0, 1090.0, 798.0, 929.0, 1079.0, 1079.0, 525.0,
                          1076.0, 892.0, 1009.0, 1072.0, 1028.0, 289.0, 748.0,
                          844.0, 978.0, 1031.0, 1022.0],
                "gmax": [12539.0, 12638.0, 14299.0, 10808.0, 15296.0,
                          16058.0, 17267.0, 16242.0, 16593.0, 11714.0,
                          12628.0, 14097.0, 18592.0, 17680.0, 17056.0,
                          16474.0, 16108.0, 16053.0],
            },
            "SouthMN": {
                "gmin": [641.0, 1018.0, 1013.0, 921.0, 1030.0, 1022.0, 764.0,
                          918.0, 675.0, 713.0, 992.0, 991.0, 687.0, 1000.0,
                          977.0, 881.0, 1053.0, 1030.0],
                "gmax": [19467.0, 18496.0, 17621.0, 13263.0, 15384.0,
                          16124.0, 18711.0, 18035.0, 17259.0, 10610.0,
                          12480.0, 14638.0, 18656.0, 17712.0, 17120.0,
                          15848.0, 16108.0, 16053.0],
            },
            "EastNC": {
                "gmin": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "gmax": [15325.0, 15699.0, 16115.0, 13628.0, 14242.0,
                          15093.0, 17100.0, 16488.0, 16918.0, 11342.0,
                          14423.0, 16374.0, 15930.0, 15386.0, 15342.0,
                          11933.0, 14433.0, 16067.0],
            },
            "SouthCA": {
                "gmin": [347.0, 958.0, 783.0, 896.0, 997.0, 994.0, 811.0,
                          934.0, 936.0, 506.0, 999.0, 991.0, 392.0, 854.0,
                          916.0, 934.0, 997.0, 993.0],
                "gmax": [18912.0, 17984.0, 16932.0, 15344.0, 15434.0,
                          16073.0, 18877.0, 17452.0, 17009.0, 15409.0,
                          15174.0, 16092.0, 19216.0, 17262.0, 17256.0,
                          16531.0, 16179.0, 16106.0],
            },
        },
        "seg_builder": ("src.models_vit_tensor_CD_2", "vit_base_patch8"),
    },
    "prithvi": {
        "label": "Prithvi",
        "chip": 224,
        "stride": 112,
        "heads": [""],
        "tree": "experiments/prithvi_multi_temporal_crop_classification",
        "num_classes": 13,
        # Scored as in scripts/infer_finder.py: one confusion matrix summed
        # chip by chip, so overlapping chips count a pixel more than once.
        # Masks are 1-13 (0 = NoData); predictions are already 0-12.
        "scorer": "per_chip",
        "mask_offset": 1,
        "pred_offset": 0,
        "delta": 8,
        "batch": 16,
        "input_mode": "stack",
        "seg_norm": "zscore18",
        # z-score statistics are per region: each scene was normalized by
        # its own band means and standard deviations. Values copied from the
        # infer_prithvi_*.py scripts that produced the published numbers.
        "zscore18": {
            "NWIA": {
                "mean": [1861.1548726, 2033.31513843, 2273.56778324,
                          3264.31809357, 4457.67865344, 3993.98225713,
                          1704.26900815, 1961.1812508, 2033.85207114,
                          3931.60397257, 4351.87295767, 3694.09881752,
                          1518.01208966, 1767.57494316, 2015.06161967,
                          3425.27676236, 3618.00166154, 2742.72472052],
                "std":  [307.15351465, 351.41085957, 447.99935412,
                          655.15274943, 698.52735671, 788.87837039,
                          341.42365808, 357.1083313, 513.54092979,
                          843.17614847, 921.40904876, 1055.85466537,
                          367.93081322, 422.83202252, 602.21383506,
                          688.45668426, 779.09201742, 671.24211383],
            },
            "SouthMN": {
                "mean": [1683.21816572, 1795.1286564, 2027.99224864,
                          2614.43523828, 4018.17499175, 3795.35160064,
                          1332.62496374, 1593.52693158, 1510.5718514,
                          4645.33493779, 2915.19120058, 2091.37382972,
                          1444.89219603, 1713.57935808, 1806.45808454,
                          3524.61079859, 3447.32103589, 2790.36075006],
                "std":  [201.19653431, 253.75733532, 338.02958852,
                          492.03218955, 576.15203045, 595.7844964,
                          194.01547233, 226.79497442, 411.89867229,
                          1132.56897351, 477.13239916, 482.08074425,
                          203.44619838, 224.95563218, 366.10911266,
                          984.88041998, 610.66313299, 721.43276144],
            },
            "EastNC": {
                "mean": [1615.22474433, 1919.88049677, 1907.57302125,
                          4391.6287779, 3639.24284624, 2772.8521438,
                          1402.42608964, 1646.00419386, 1490.9730054,
                          4668.59382492, 3072.37737136, 2090.55866365,
                          1377.28325829, 1549.71437673, 1419.84248805,
                          4144.4942076, 2819.86398661, 1957.26925153],
                "std":  [445.83556981, 508.25045142, 765.69735543,
                          659.43243087, 1163.39733358, 1223.01516809,
                          276.18115909, 339.45512187, 419.55339594,
                          1044.42051202, 757.69613288, 650.64448343,
                          243.51106275, 291.01566979, 364.4813079,
                          920.12017449, 660.86197671, 555.53329805],
            },
            "SouthCA": {
                "mean": [1399.18814785, 1624.0902128, 1530.61747529,
                          3433.65512704, 2625.27257993, 2036.76201361,
                          1690.31674406, 1965.86899305, 2231.22908437,
                          3764.86120507, 3585.95243342, 2720.77963287,
                          1547.22463694, 1832.0396529, 2079.88629347,
                          3402.57135275, 3377.24651308, 2538.78450119],
                "std":  [281.63779618, 300.33935069, 375.98413426,
                          1116.25457, 684.00131194, 559.83401327,
                          382.69301356, 482.81761256, 787.29874717,
                          1098.04393375, 1231.90817451, 884.90944039,
                          349.85626067, 448.42248082, 723.75199174,
                          1019.93200015, 1201.66504561, 846.55948618],
            },
        },
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

    Returns the path relative to GFM_WEIGHTS/seg/<model>/, so a flat file
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


def per_region(value, region):
    """A registry value that may vary by region: {region: v, "default": v}."""
    if isinstance(value, dict):
        return value.get(region, value.get("default"))
    return value
