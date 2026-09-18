# Shared shell environment for the job scripts.
#
# The CUDA module load, the virtual-environment activation and the legacy
# environment names used to be copy-pasted into eight scripts, so changing the
# CUDA version meant editing eight files and missing one meant a silent
# inconsistency. They live here instead.
#
# Source configs/paths.sh before this file.

# Load the CUDA module when running under an environment module system.
# Absent on a laptop, present on the cluster; both are fine.
msr_load_cuda() {
    if command -v module >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
        module purge
        module load cuda/12.1
    fi
}

# Environments predating setup.sh, kept so an existing working copy still runs.
msr_legacy_env() {
    case "$1" in
        satmae)      echo "satmae_env" ;;
        spectralgpt) echo "spectralgptenv" ;;
        prithvi)     echo "prithvifmenv" ;;
        *)           echo "" ;;
    esac
}

# Echo the interpreter for a model, in preference order:
#   venvs/<model>   built by setup.sh
#   $MSR_VENV       an environment the user names explicitly
#   legacy in-tree  satmae_env, spectralgptenv, prithvifmenv
#   python3         whatever is on PATH
msr_python() {
    local model="${1:-}" venvs legacy
    venvs="${MSR_VENVS:-$MSR_ROOT/venvs}"

    if [ -n "$model" ] && [ -x "$venvs/$model/bin/python" ]; then
        echo "$venvs/$model/bin/python"; return
    fi
    if [ -n "${MSR_VENV:-}" ] && [ -x "$MSR_VENV/bin/python" ]; then
        echo "$MSR_VENV/bin/python"; return
    fi
    legacy=$(msr_legacy_env "$model")
    if [ -n "$legacy" ] && [ -x "$MSR_ROOT/$legacy/bin/python" ]; then
        echo "$MSR_ROOT/$legacy/bin/python"; return
    fi
    command -v python3
}

# torch.distributed wants a port that does not collide between concurrent jobs.
msr_master_port() {
    echo $((29500 + ${SLURM_JOB_ID:-0} % 1000))
}

msr_banner() {
    echo "==== $* ===="
    echo "Job:  ${SLURM_JOB_ID:-local}"
    echo "Node: $(hostname)"
    echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo n/a)"
    echo ""
}
