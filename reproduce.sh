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
# Requires ./setup.sh to have run. Data goes to $MSR_DATA_ROOT (default
# ./data), results to $MSR_OUTPUT_ROOT (default ./outputs); neither needs
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
    seg)
        echo "Note: most segmentation cells infer over a stitched raster" >&2
        echo "      (processed_stacks/), which is not published. Only SatMAE" >&2
        echo "      minnesota runs from chips. Cells without their input are" >&2
        echo "      reported and skipped." >&2
        echo "" >&2 ;;
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
MSR_TASK="$TASK" python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "configs")
import paths as P

task = os.environ.get("MSR_TASK", "cd")

rows = []
if P.PREDICTIONS.is_dir():
    for f in sorted(P.PREDICTIONS.rglob("*_metrics.json")):
        try:
            r = json.load(open(f))
        except (ValueError, OSError):
            continue
        # A change-detection record reports F1, a segmentation record mIoU.
        kind = "cd" if "F1" in r else "seg"
        if kind == task:
            rows.append(r)

if not rows:
    print("No {} metrics found under {}".format(task, P.PREDICTIONS))
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
    hdr = "{:<13} {:<16} {:<9} {:<7} {:>8} {:>8}"
    print(hdr.format("model", "state", "region", "head", "mIoU%", "classes"))
    print("-" * 68)
    for r in sorted(rows, key=key):
        print("{:<13} {:<16} {:<9} {:<7} {:>8.2f} {:>8}".format(
            r.get("model", "?"), r.get("state", "?"), r.get("region", "?"),
            r.get("head", "") or "-", r.get("mIoU", float("nan")),
            r.get("classes_present", "?")))

print("\nPredictions and metrics: {}".format(P.PREDICTIONS))
PY

echo
echo "cells scored:  ${#ok[@]}   ${ok[*]:-}"
[ ${#skipped[@]} -gt 0 ] && echo "cells skipped: ${#skipped[@]}   ${skipped[*]}"
[ ${#failed[@]} -gt 0 ] && echo "cells failed:  ${#failed[@]}   ${failed[*]}"
[ ${#failed[@]} -gt 0 ] && exit 1
exit 0
