#!/bin/bash
#
# Reproduce the benchmark: download what is needed, score it, print the table.
#
#     ./reproduce.sh --task cd                      # every model, every state
#     ./reproduce.sh --task cd --state iowa         # one state, all models
#     ./reproduce.sh --task cd --model prithvi --state california
#     ./reproduce.sh --task cd --dry-run            # show the plan and the size
#
# A cell is one (task, model, state) triple and one row of the results table.
# Each cell is fetched, scored, and recorded independently, so a failure in one
# does not lose the others -- the table at the end reports what succeeded.
#
# Requires ./setup.sh to have run. Data goes to $GFM_DATA_ROOT (default
# ./data), results to $GFM_OUTPUT_ROOT (default ./outputs); neither needs
# setting for a default install.

set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/configs/paths.sh"
# shellcheck disable=SC1091
source "$ROOT/configs/env.sh"

TASK=cd
MODEL=all
STATE=all
DRY_RUN=0
SKIP_FETCH=0
THRESHOLD=""

while [ $# -gt 0 ]; do
    case "$1" in
        --task)       TASK="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --state)      STATE="$2"; shift 2 ;;
        --threshold)  THRESHOLD="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --skip-fetch) SKIP_FETCH=1; shift ;;
        -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$TASK" in
    cd) ;;
    seg) ;;
    *) echo "unknown task: $TASK (expected cd or seg)" >&2; exit 2 ;;
esac

# Interpreter selection lives in configs/env.sh, shared with benchmark-gfm.
pick_python() { msr_python "$1"; }

# Expand "all" using the registry rather than a second copy of the lists.
list_of() {
    python3 - "$1" <<'PY'
import sys
sys.path.insert(0, "configs")
import registry as R
print(" ".join(sorted(R.MODELS) if sys.argv[1] == "models" else R.states()))
PY
}

cd "$ROOT"
[ "$MODEL" = all ] && MODELS=($(list_of models)) || MODELS=("$MODEL")
[ "$STATE" = all ] && STATES=($(list_of states)) || STATES=("$STATE")

if [ "$DRY_RUN" = 1 ]; then
    for m in "${MODELS[@]}"; do
        "$(pick_python "$m")" scripts/fetch.py --task "$TASK" --model "$m" \
            --state "$STATE" --dry-run
    done
    exit 0
fi

ok=(); failed=(); skipped=()

for model in "${MODELS[@]}"; do
    PY="$(pick_python "$model")"
    for state in "${STATES[@]}"; do
        cell="$model/$state"
        echo
        echo "################ $cell ################"

        if [ "$SKIP_FETCH" = 0 ]; then
            if ! "$PY" scripts/fetch.py --task "$TASK" --model "$model" --state "$state"; then
                echo "-- $cell: data unavailable or incomplete; skipping"
                skipped+=("$cell")
                continue
            fi
        fi

        args=(infer --task "$TASK" --model "$model" --state "$state")
        [ -n "$THRESHOLD" ] && args+=(--threshold "$THRESHOLD")

        if ./benchmark-gfm "${args[@]}"; then
            ok+=("$cell")
        else
            echo "-- $cell: inference failed"
            failed+=("$cell")
        fi
    done
done

echo
echo "############### results ###############"
GFM_TASK="$TASK" GFM_MODELS="${MODELS[*]}" GFM_STATES="${STATES[*]}" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "configs")
import paths as P

task = os.environ.get("GFM_TASK", "cd")
# Only the cells this run asked for. Results from earlier runs stay on disk
# under predictions/ but are not listed unless their model and state were
# requested again.
models = set(os.environ.get("GFM_MODELS", "").split())
states = set(os.environ.get("GFM_STATES", "").split())

rows = []
if P.PREDICTIONS.is_dir():
    for f in sorted(P.PREDICTIONS.rglob("*_metrics.json")):
        try:
            r = json.load(open(f))
        except (ValueError, OSError):
            continue
        # A change-detection record reports F1, a segmentation record mIoU.
        kind = "cd" if "F1" in r else "seg"
        if kind == task and r.get("model") in models and r.get("state") in states:
            rows.append(r)

if not rows:
    print("No {} metrics for the requested cells under {}".format(
        task, P.PREDICTIONS))
    raise SystemExit

def key(r):
    return (r.get("model", ""), r.get("state", ""), r.get("head", ""))

if task == "cd":
    hdr = "{:<13} {:<16} {:<9} {:>7} {:>7} {:>7} {:>7}"
    print(hdr.format("model", "state", "region", "OA%", "Prec%", "Rec%", "F1%"))
    print("-" * 72)
    for r in sorted(rows, key=key):
        print("{:<13} {:<16} {:<9} {:>7.2f} {:>7.2f} {:>7.2f} {:>7.2f}".format(
            r.get("model", "?"), r.get("state", "?"), r.get("region", "?"),
            r.get("OA", float("nan")), r.get("precision", float("nan")),
            r.get("recall", float("nan")), r.get("F1", float("nan"))))
else:
    # Same summaries as Table 4: mIoU over all available classes and over
    # the crop classes alone.
    hdr = "{:<13} {:<16} {:<9} {:<7} {:>10} {:>12}"
    print(hdr.format("model", "state", "region", "head", "mIoU(all)", "mIoU(crops)"))
    print("-" * 72)
    for r in sorted(rows, key=key):
        print("{:<13} {:<16} {:<9} {:<7} {:>10.2f} {:>12.2f}".format(
            r.get("model", "?"), r.get("state", "?"), r.get("region", "?"),
            r.get("head", "") or "-", r.get("mIoU", float("nan")),
            r.get("mIoU_crops", float("nan"))))

print("\nPredictions and metrics: {}".format(P.PREDICTIONS))
PY

echo
echo "cells scored:  ${#ok[@]}   ${ok[*]:-}"
[ ${#skipped[@]} -gt 0 ] && echo "cells skipped: ${#skipped[@]}   ${skipped[*]}"
[ ${#failed[@]} -gt 0 ] && echo "cells failed:  ${#failed[@]}   ${failed[*]}"
[ ${#failed[@]} -gt 0 ] && exit 1
exit 0
