# Benchmarking Geospatial Foundation Models for Agriculture Applications

Change detection and semantic segmentation with three geospatial foundation
model backbones - **SatMAE**, **SpectralGPT** and **Prithvi** - evaluated across
four US agricultural regions.

## Requirements

- Linux with an NVIDIA GPU. Check with `nvidia-smi`: if it prints your GPU, you're ready.
- Python 3.9 (recommended) or 3.10. If you don't have either, the setup below
  installs 3.9 for you. See [Python version](#python-version) for why other
  versions don't work.

## Setup

```bash
git clone https://github.com/Sanmay-Das/Benchmarking-GeoFMs-for-Agriculture-Applications
cd Benchmarking-GeoFMs-for-Agriculture-Applications
curl -LsSf https://astral.sh/uv/install.sh | sh      # install uv, a small tool that downloads Python
~/.local/bin/uv python install 3.9                   # download Python 3.9 (no admin rights needed)
GFM_PYTHON=$(~/.local/bin/uv python find 3.9) ./setup.sh   # build one environment per model
```

If `python3` on your machine is already 3.9 or 3.10, the last line can be
just `./setup.sh`.

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

Use Python 3.9 or 3.10. Both install exactly the same package versions, and
`setup.sh` stops with a message on any other version. 3.9 is recommended
because the published results were produced with it (3.9.18).

| Python | SatMAE | SpectralGPT | Prithvi |
|---|---|---|---|
| 3.8 or older | no | no | no |
| **3.9** | yes | yes | yes |
| **3.10** | yes | yes | yes |
| 3.11 | no | yes | no |
| 3.12 or newer | no | no | no |

Why other versions don't work:

- **3.8 or older:** `numpy 1.26.4`, `scipy 1.13.1`, `rasterio 1.4.3` and
  `torch 2.5.1` were only released for Python 3.9 and newer.
- **3.11:** SatMAE and Prithvi need `torch 1.11.0+cu113` and `mmcv-full`
  (1.7.1 and 1.6.2), which were only released up to Python 3.10. pip finds no
  `torch 1.11.0+cu113` for 3.11 at all, and SatMAE's older pins (`pandas
  1.3.4`, `matplotlib 3.5.1`, `Pillow 9.0.1`) fail to compile from source.
- **3.12 or newer:** in addition, SpectralGPT pins `numpy 1.23.5`, which was
  only released up to Python 3.11.

These versions can't simply be upgraded: the pins reproduce the exact
environments the published results came from, and newer releases of these
packages would change the environment.

If your default `python3` is something else, get a 3.9 interpreter with uv
and point `GFM_PYTHON` at it (use `3.10` in both commands for 3.10):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.9
GFM_PYTHON=$(~/.local/bin/uv python find 3.9) ./setup.sh
```

Any complete Python 3.9 or 3.10 works, including one from conda. Avoid a Python
compiled from source (for example with pyenv): a source build made without
system libraries such as libffi is missing modules that torch needs, and
`setup.sh` will refuse it. uv downloads a prebuilt Python, so it does not have
this problem.

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
