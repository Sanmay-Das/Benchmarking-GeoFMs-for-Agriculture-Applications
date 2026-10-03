# Benchmarking Geospatial Foundation Models for Agriculture Applications

Change detection and semantic segmentation with three geospatial foundation
model backbones - **SatMAE**, **SpectralGPT** and **Prithvi** - evaluated across
four US agricultural regions.

## Requirements

- An NVIDIA GPU. Check with `nvidia-smi`: if it prints your GPU, you're ready.
- Python 3.9.

## Setup

```bash
git clone https://github.com/Sanmay-Das/Benchmarking-GeoFMs-for-Agriculture-Applications
cd Benchmarking-GeoFMs-for-Agriculture-Applications
./setup.sh
```

If `python3` on your machine is not 3.9:

```bash
GFM_PYTHON=/path/to/python3.9 ./setup.sh
```

## Run

```bash
./reproduce.sh --task seg --model satmae --state north_carolina
```

This downloads the data and weights, runs the model, and prints the results.

- `--task`: `seg` (segmentation) or `cd` (change detection)
- `--model`: `satmae`, `spectralgpt`, `prithvi`
- `--state`: `iowa`, `minnesota`, `north_carolina`, `california`

## Running on an HPC cluster

On high-performance computing (HPC) clusters the login node usually has no
GPU. Submit the same command as a job with `sbatch`:

```bash
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=6:00:00 \
  --wrap "./reproduce.sh --task seg --model satmae --state north_carolina"
```

Run it from inside the repository folder. Change `--partition=gpu` to your
cluster's GPU partition name. The results appear in `slurm-<jobid>.out`.

## The benchmark grid

A **cell** is one `(task, model, state)` triple, and one row of results:

- **tasks** - `cd` (change detection), `seg` (semantic segmentation)
- **models** - `satmae`, `spectralgpt`, `prithvi`
- **states** - `iowa`, `minnesota`, `north_carolina`, `california`

That is 2 x 3 x 4 = 24 cells. Every command takes the same three flags:

```bash
./reproduce.sh --task cd --model satmae --state minnesota
```

Omit `--model` or `--state` to run every one of them.

State is the unit everything is organised by. Each state has three sub-regions -
train, validation and test - and the reported number is computed on the test
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
| **satmae** | - | yes | yes | yes |
| **prithvi** | - | - | yes | yes |
| **spectralgpt** | - | - | - | yes |

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
| **satmae** | - | yes | yes | no checkpoint |
| **prithvi** | - | yes | yes | no chips |
| **spectralgpt** | - | yes | yes | yes |

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
./benchmark-gfm manifests [model]                 # rebuild chip manifests
```

`reproduce.sh` options: `--model`, `--state`, `--threshold`, `--dry-run`,
`--skip-fetch`.

### Where it runs

Inference and training want a GPU. Nothing here requires a cluster -- SLURM is
one of three ways to get to one.

**On a machine with a GPU**, run the command directly:

```bash
./benchmark-gfm infer --task seg --model prithvi --state minnesota
```

**On a cluster, interactively** -- useful while trying things out, because you
see the output as it happens:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=2:00:00 --pty bash
./benchmark-gfm infer --task seg --model prithvi --state minnesota
```

**On a cluster, as a batch job**, add `--sbatch` and the command is submitted
with the resources that command needs:

```bash
./benchmark-gfm --sbatch infer --task seg --model prithvi --state minnesota
```

With no GPU, the inference commands say so and ask before continuing, since
CPU runs are slow. Pass `--allow-cpu` to skip the question in a batch job.

### Training

```bash
./benchmark-gfm --sbatch train --task cd --model satmae --state minnesota
```

Training needs all three regions of a state:

```bash
python scripts/fetch.py --task cd --model satmae --state minnesota --splits all
```

## Configuration

Defaults are self-contained; override only if you want data elsewhere.

| variable | default | holds |
|---|---|---|
| `GFM_DATA_ROOT` | `./data` | downloaded chips |
| `GFM_WEIGHTS` | `./weights` | checkpoints |
| `GFM_OUTPUT_ROOT` | `./outputs` | predictions, metrics, logs |
| `GFM_VENVS` | `./venvs` | per-model environments |

Nothing is ever written inside the source tree.

These were once named `MSR_*`, after the directory this was developed in. Those
names still work.

## Data

Published at
[`sanmay4119/geofm-agriculture-benchmark`](https://huggingface.co/datasets/sanmay4119/geofm-agriculture-benchmark)
- 482 GB in total, so do not clone it. One cell is 3-30 GB and `fetch.py`
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
scripts remain numerically identical to the per-region scripts they replaced -
the equivalence tests compare against frozen fixtures, since the originals were
deleted.

## Troubleshooting

### Python version

Python 3.9 is a hard requirement. The dependency versions were resolved against
3.9.18, and two of them have no wheels for newer interpreters: `torch
1.11.0+cu113` and `mmcv-full`. On Python 3.10 or later, pip falls back to
building them from source and the install fails. `setup.sh` checks the version
and stops if it is wrong.

If your default `python3` is something else, install a 3.9 interpreter and point
`GFM_PYTHON` at it:

```bash
pyenv install 3.9.18
GFM_PYTHON=~/.pyenv/versions/3.9.18/bin/python ./setup.sh

# or, with conda
conda create -n gfm39 python=3.9 -y
GFM_PYTHON=$(conda run -n gfm39 which python) ./setup.sh
```

If an earlier attempt already created environments with the wrong interpreter,
delete them first -- they cannot be repaired in place:

```bash
rm -rf venvs
```

### Data download

Data and checkpoints are downloaded automatically
from [Hugging Face](https://huggingface.co/datasets/sanmay4119/geofm-agriculture-benchmark)
on first use; no paths or environment variables need to be set.

Check what a run will download before committing to it:

```bash
./reproduce.sh --task cd --dry-run
```

## Citation

If you use this benchmark, please cite the accompanying paper.
