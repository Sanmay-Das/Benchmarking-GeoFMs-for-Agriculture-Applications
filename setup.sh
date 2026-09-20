#!/bin/bash
#
# Build the Python environments this benchmark needs.
#
#     ./setup.sh              # all three models
#     ./setup.sh prithvi      # just one
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

version=$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$version" != "3.9" ]; then
    echo "Warning: the pins were resolved against Python 3.9; this is $version."
    echo "         Set GFM_PYTHON to a 3.9 interpreter if installation fails."
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
    if [ -x "$env/bin/python" ]; then
        echo "Environment already exists: $env"
    else
        echo "Creating $env ..."
        "$PYTHON" -m venv "$env"
    fi

    # The CUDA-suffixed torch wheels are not on PyPI proper.
    "$env/bin/pip" install --quiet --upgrade pip
    "$env/bin/pip" install --quiet -r "$req" \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu121

    "$env/bin/python" - <<'PY'
import torch, rasterio, numpy  # noqa: F401
print("  torch    {}  (cuda available: {})".format(
    torch.__version__, torch.cuda.is_available()))
print("  rasterio {}".format(rasterio.__version__))
print("  numpy    {}".format(numpy.__version__))
PY
done

echo
echo "Environments are under $VENVS"
echo "Next: ./reproduce.sh --task cd --state iowa"
