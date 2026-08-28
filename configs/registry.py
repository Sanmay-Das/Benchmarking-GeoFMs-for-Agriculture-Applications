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
REGIONS = {
    "NWIA":    {"label": "Northwest Iowa",    "train_run": ""},
    "SouthMN": {"label": "Southern Minnesota", "train_run": "MN"},
    "EastNC":  {"label": "Eastern N. Carolina", "train_run": "NC"},
    "SouthCA": {"label": "Southern California", "train_run": "CA"},
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
