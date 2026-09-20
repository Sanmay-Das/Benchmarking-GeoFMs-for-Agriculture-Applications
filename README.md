# Benchmarking Geospatial Foundation Models for Agriculture Applications

Change detection and semantic segmentation with three geospatial foundation
model backbones - **SatMAE**, **SpectralGPT** and **Prithvi** - evaluated across
four US agricultural regions.

## Quick start

```bash
git clone https://github.com/Sanmay-Das/Benchmarking-GeoFMs-for-Agriculture-Applications
cd Benchmarking-GeoFMs-for-Agriculture-Applications

./setup.sh                                  # build the Python environments
./reproduce.sh --task cd --state iowa       # download data, score, print results
```

Data and checkpoints are downloaded automatically
from [Hugging Face](https://huggingface.co/datasets/sanmay4119/geofm-agriculture-benchmark)
on first use; no paths or environment variables need to be set.

Check what a run will download before committing to it:

```bash
./reproduce.sh --task cd --dry-run
```

## The benchmark grid

A **cell** is one `(task, model, state)` triple, and one row of results:

- **tasks** — `cd` (change detection), `seg` (semantic segmentation)
- **models** — `satmae`, `spectralgpt`, `prithvi`
- **states** — `iowa`, `minnesota`, `north_carolina`, `california`

That is 2 x 3 x 4 = 24 cells. Every command takes the same three flags:

```bash
./reproduce.sh --task cd --model satmae --state minnesota
```

Omit `--model` or `--state` to run every one of them.

State is the unit everything is organised by. Each state has three sub-regions —
train, validation and test — and the reported number is computed on the test
region. You never name regions directly; `--state iowa` resolves to them.

| state | train | val | test |
|---|---|---|---|
| iowa | CentIA | EastIA | NWIA |
| minnesota | NorthMN | CentMN | SouthMN |
| north_carolina | NENC | ECNC | EastNC |
| california | NorthCA | CentCA | SouthCA |

## What is reproducible today

Six of the twelve change-detection cells have both chips and a checkpoint
published:

| model | iowa | minnesota | north_carolina | california |
|---|---|---|---|---|
| **satmae** | — | ✅ | ✅ | ✅ |
| **prithvi** | — | — | ✅ | ✅ |
| **spectralgpt** | — | — | — | ✅ |

The gaps are missing published files, not missing code. `fetch.py` names exactly
what is absent for any cell you ask for, and skips downloading the usable half
of a cell that cannot run.

### Segmentation

```bash
./reproduce.sh --task seg --model satmae --state minnesota
```

Seven of the twelve segmentation cells are reproducible:

| model | iowa | minnesota | north_carolina | california |
|---|---|---|---|---|
| **satmae** | — | ✅ | ✅ | no checkpoint |
| **prithvi** | — | ✅ | ✅ | no chips |
| **spectralgpt** | — | ✅ | ✅ | ✅ |

Iowa has no segmentation chips published for any backbone.

Segmentation is scored from the published test chips.

SatMAE segmentation uses the **FPN** head. It was also trained with an FCN
head and with PSANet, but FPN is the head published on Hugging Face and the one
this benchmark reports, so it is the only one wired up here.

The split files listing those chips are not published. They are derivable from
the chips, so `fetch.py` writes them after unpacking.

## Commands

```bash
./setup.sh [model ...]                            # environments; all three by default
./reproduce.sh --task {cd,seg} [options]          # fetch + score + table
python scripts/fetch.py --task {cd,seg} [options] # download only
./benchmark-gfm infer --task {cd,seg} --model M --state S
./benchmark-gfm train --task {cd,seg} --model M --state S
./benchmark-gfm eval --task seg --region R        # segmentation metrics
./benchmark-gfm manifests [model]                 # rebuild chip manifests
```

`reproduce.sh` options: `--model`, `--state`, `--threshold`, `--dry-run`,
`--skip-fetch`.

### Training

Training runs as a SLURM job:

```bash
./benchmark-gfm --sbatch train --task cd --model satmae --state minnesota
```

Training needs all three regions of a state:

```bash
python scripts/fetch.py --task cd --model satmae --state minnesota --splits all
```

## Layout

```
setup.sh               build the per-model environments
reproduce.sh           fetch + score + results table

configs/paths.py       every path, resolved from three roots
configs/registry.py    what varies per model, state and region

infer_cd.py            change-detection inference, all models and states
infer_seg.py           segmentation inference
evaluate_seg.py        segmentation metrics
visualize_cd.py        change-detection maps

benchmark-gfm          infer, train, viz, eval, manifests
                       add --sbatch to submit to SLURM

scripts/fetch.py       download one cell from Hugging Face
scripts/               region-specific helpers and analysis one-offs
tests/                 registry and numerical-equivalence guards

SatMAE/                vendored upstream backbone code
IEEE_TPAMI_SpectralGPT/
prithvi_finetune/
```

Top-level scripts are the ones meant to be run. Anything region-specific or
single-purpose lives in `scripts/`.

The three backbone directories are vendored upstream code, imported by the
inference scripts but not modified beyond path handling.

## Configuration

Defaults are self-contained; override only if you want data elsewhere.

| variable | default | holds |
|---|---|---|
| `MSR_DATA_ROOT` | `./data` | downloaded chips |
| `MSR_WEIGHTS` | `./weights` | checkpoints |
| `MSR_OUTPUT_ROOT` | `./outputs` | predictions, metrics, logs |
| `MSR_VENVS` | `./venvs` | per-model environments |

Nothing is ever written inside the source tree.

## Data

Published at
[`sanmay4119/geofm-agriculture-benchmark`](https://huggingface.co/datasets/sanmay4119/geofm-agriculture-benchmark)
— 482 GB in total, so do not clone it. One cell is 3–30 GB and `fetch.py`
downloads only what a cell needs.

Chip manifests are **not** published: they contain absolute paths, and are
rebuilt from the chips after unpacking. `./benchmark-gfm manifests` does this if you
ever need to redo it.

## Environments

One per model, because the pins conflict:

| model | torch |
|---|---|
| satmae | 1.11.0+cu113 |
| prithvi | 1.11.0+cu113 |
| spectralgpt | 2.5.1+cu121 |

`setup.sh` builds them under `venvs/`; `reproduce.sh` picks the right one per
cell. Do not attempt to merge them into one environment.

## Tests

```bash
for t in tests/test_*.py; do python "$t"; done
```

These check that the registry stays complete and that the collapsed inference
scripts remain numerically identical to the per-region scripts they replaced —
the equivalence tests compare against frozen fixtures, since the originals were
deleted.

## Citation

If you use this benchmark, please cite the accompanying paper.
