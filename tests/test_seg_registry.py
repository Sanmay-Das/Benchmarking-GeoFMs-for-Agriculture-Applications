"""
Guard: the segmentation registry stays complete and resolvable.

Replaces the version that pinned the registry to the fourteen per-region
scripts. Those are gone -- infer_seg.py is the only implementation -- so what
needs guarding is that every backbone is fully described and every
(model, head, region) combination still resolves to a checkpoint path.

    python tests/test_seg_registry.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402
import paths as P  # noqa: E402

REQUIRED = ("label", "chip", "stride", "delta", "batch", "heads", "tree",
            "num_classes", "seg_norm")
NORM_FIELDS = {
    "zscore_tiled": ("mean6", "std6"),
    "zscore18": ("mean18", "std18"),
    "global_minmax": ("gmin", "gmax"),
}
EXPECTED_LEN = {"mean6": 6, "std6": 6, "mean18": 18, "std18": 18,
                "gmin": 18, "gmax": 18}


def main():
    problems = []

    for name in sorted(R.SEG_MODELS):
        spec = R.SEG_MODELS[name]
        for field in REQUIRED:
            if field not in spec:
                problems.append("{}: missing '{}'".format(name, field))

        scheme = spec.get("seg_norm")
        if scheme not in NORM_FIELDS:
            problems.append("{}: unknown seg_norm {!r}".format(name, scheme))
        else:
            for field in NORM_FIELDS[scheme]:
                stats = spec.get(field)
                if not stats:
                    problems.append("{}: {} requires '{}'".format(name, scheme, field))
                elif len(stats) != EXPECTED_LEN[field]:
                    problems.append("{}: '{}' has {} values, expected {}".format(
                        name, field, len(stats), EXPECTED_LEN[field]))

        if spec.get("stride", 0) * 2 != spec.get("chip", 0):
            problems.append("{}: stride should be half the chip size".format(name))

        print("  OK        {:<12} chip={} heads={} classes={} norm={}".format(
            name, spec["chip"], spec["heads"], spec["num_classes"],
            spec["seg_norm"]))

    for model, head, region in R.seg_pairs():
        try:
            P.seg_checkpoint(model, region, head, must_exist=False)
        except SystemExit as exc:
            problems.append("{} {} {}: {}".format(model, head, region, exc))
    print("  OK        {} (model, head, region) combinations resolve".format(
        len(R.seg_pairs())))

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nSegmentation registry is complete and consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
