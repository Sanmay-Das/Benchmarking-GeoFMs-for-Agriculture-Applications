"""
Guard: every segmentation inference script must resolve its checkpoint and
input through the registry, not a hardcoded path.

Mirrors tests/test_registry_matches_legacy.py for the segmentation family.
Checks that each infer_<model>[_<head>]_<region>.py calls
seg_checkpoint(model, region, head) with the arguments its filename implies,
and that no in-tree checkpoint or processed_stacks path survives.

    python tests/test_seg_registry.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402

HEADS = {"fcn", "fpn", "psanet"}

# Patterns that indicate a path the refactor was supposed to remove. Commented
# lines are ignored -- a few scripts keep an alternative checkpoint in a
# comment as a record of which run was chosen.
STALE = [
    (r"output_seg_\w+/checkpoint-best\.pth", "in-tree SatMAE checkpoint"),
    (r"multi_train/best_mIoU_\w+\.pth", "in-tree SpectralGPT checkpoint"),
    (r"processed_stacks", "stack path inside the code tree"),
]


def script_name(model, head, region):
    return "infer_{}{}_{}.py".format(model, "_" + head if head else "", region)


def check(model, head, region):
    path = ROOT / script_name(model, head, region)
    if not path.exists():
        return ["{}: file absent".format(path.name)]

    src = path.read_text()
    problems = []

    call = re.search(
        r"seg_checkpoint\(\s*'(\w+)'\s*,\s*'(\w+)'\s*,\s*'(\w*)'\s*\)", src)
    if not call:
        problems.append("CHECKPOINT does not use seg_checkpoint(...)")
    else:
        got = (call.group(1), call.group(3), call.group(2))
        if got != (model, head, region):
            problems.append("seg_checkpoint{} != {}".format(got, (model, head, region)))

    live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    for pattern, label in STALE:
        if re.search(pattern, live):
            problems.append("still references {}".format(label))

    return ["{}: {}".format(path.name, p) for p in problems]


def main():
    failures = []
    for model, head, region in R.seg_pairs():
        found = check(model, head, region)
        print("  {:<9} {}".format("OK" if not found else "MISMATCH",
                                  script_name(model, head, region)))
        failures.extend(found)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nAll {} segmentation combinations match the registry.".format(
        len(R.seg_pairs())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
