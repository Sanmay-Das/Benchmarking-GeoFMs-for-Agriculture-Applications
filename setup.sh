#!/bin/bash
#
# Build the Python environments this benchmark needs.
#
#     ./setup.sh              # all three models
#     ./setup.sh prithvi      # just one
#
# Requires Python 3.9. The pins were resolved against 3.9.18 and several of
# them -- torch 1.11.0+cu113 and mmcv-full in particular -- have no wheels for
# newer interpreters, so they would be built from source and fail. If your
# default python3 is not 3.9, point GFM_PYTHON at one:
#
#     GFM_PYTHON=/usr/bin/python3.9 ./setup.sh
#
# One environment per model, because the pins genuinely conflict: SpectralGPT
# needs torch 2.5.1+cu121 while SatMAE and Prithvi need 1.11.0+cu113. A single
# shared environment cannot satisfy both, so do not try to merge them.
#
# Environments are created under ./venvs/<model>/ and are what benchmark-gfm and
# reproduce.sh look for. Set GFM_VENVS to put them elsewhere (a scratch
# filesystem, say -- they total several GB).

set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
VENVS="${GFM_VENVS:-$ROOT/venvs}"
PYTHON="${GFM_PYTHON:-python3}"

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
    MODELS=(satmae spectralgpt prithvi)
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "No such interpreter: $PYTHON" >&2
    echo "Set GFM_PYTHON to a Python 3.9 interpreter." >&2
    exit 1
fi

version=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$version" != "3.9" ]; then
    cat >&2 <<MSG
Python 3.9 is required; '$PYTHON' is $version.

The pinned versions have no wheels for $version, so pip would try to build
them from source and fail (mmcv-full needs pkg_resources, which modern
setuptools no longer ships; torch 1.11.0+cu113 has no $version wheels at all).

Install a 3.9 interpreter and point GFM_PYTHON at it, for example:

    pyenv install 3.9.18
    GFM_PYTHON=~/.pyenv/versions/3.9.18/bin/python ./setup.sh

    # or, with conda
    conda create -n gfm39 python=3.9 -y
    GFM_PYTHON=\$(conda run -n gfm39 which python) ./setup.sh
MSG
    exit 1
fi

mkdir -p "$VENVS"

for model in "${MODELS[@]}"; do
    req="$ROOT/requirements/$model.txt"
    if [ ! -f "$req" ]; then
        echo "No requirements file for '$model' ($req)" >&2
        exit 1
    fi
    env="$VENVS/$model"

    echo
    echo "=============== $model ==============="

    # An environment left over from a different interpreter cannot be repaired
    # by installing into it, so refuse rather than fail obscurely later.
    if [ -x "$env/bin/python" ]; then
        have=$("$env/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo unknown)
        if [ "$have" != "3.9" ]; then
            echo "Existing environment uses Python $have, not 3.9: $env" >&2
            echo "Remove it and re-run:  rm -rf '$env'" >&2
            exit 1
        fi
        echo "Environment already exists: $env"
    else
        echo "Creating $env ..."
        "$PYTHON" -m venv "$env"
    fi

    "$env/bin/pip" install --quiet --upgrade pip

    # mmcv-full is served from the OpenMMLab index, not PyPI. Installing it
    # with plain pip triggers a source build that fails. Install everything
    # else first, then hand mmcv-full to mim.
    mmcv=$(grep -i '^mmcv-full==' "$req" || true)
    filtered=$(mktemp)
    grep -iv '^mmcv-full==' "$req" > "$filtered"

    # --no-deps is deliberate. These pins came from pip freeze on environments
    # grown incrementally over three years, so their declared metadata is
    # mutually unsatisfiable even though the installed set works: kornia and
    # classy-vision declare a newer torch than 1.11.0+cu113, and opencv-python
    # 4.12 declares a newer numpy than matplotlib 3.5.1 and mmsegmentation
    # 0.30.0 allow. A freeze is already a complete transitive closure, so there
    # is nothing to resolve; installing it verbatim reproduces the environment
    # the published results were produced in. Letting pip resolve instead would
    # silently build a different one.
    "$env/bin/pip" install --quiet --no-deps -r "$filtered" \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu121

    if [ -n "$mmcv" ]; then
        echo "Installing $mmcv via mim ..."
        "$env/bin/pip" install --quiet -U openmim
        "$env/bin/mim" install "$mmcv"

        # mim resolves dependencies normally and upgrades numpy to 2.x, which
        # torch 1.11 cannot use -- it is compiled against the numpy 1.x ABI, so
        # every array operation then fails with "Numpy is not available". Put
        # the pinned versions back. Restore from the requirements file rather
        # than a literal, because the environments need different numpy
        # versions (satmae 1.26.4, prithvi 1.23.5).
        "$env/bin/pip" install --quiet --no-deps -r "$filtered" \
            --extra-index-url https://download.pytorch.org/whl/cu113 \
            --extra-index-url https://download.pytorch.org/whl/cu121
    fi

    rm -f "$filtered"

    # Verify rather than report. Printing versions is not enough: a numpy the
    # torch build cannot use imports fine and only fails on first array use, so
    # exercise the bridge here and fail the install rather than hand back an
    # environment that looks correct and cannot run anything.
    if ! "$env/bin/python" - <<'PY'
import sys
import numpy as np
import rasterio
import torch

try:
    torch.from_numpy(np.zeros((2, 2), dtype="float32"))
except Exception as exc:
    print("  numpy {} is unusable from torch {}: {}".format(
        np.__version__, torch.__version__, exc))
    print("  The environment is broken; do not use it.")
    sys.exit(1)

print("  torch    {}  (cuda available: {})".format(
    torch.__version__, torch.cuda.is_available()))
print("  rasterio {}".format(rasterio.__version__))
print("  numpy    {}".format(np.__version__))
PY
    then
        echo "Verification failed for '$model'." >&2
        exit 1
    fi
done

echo
echo "Environments are under $VENVS"
echo "Next: ./reproduce.sh --task cd --state iowa"
