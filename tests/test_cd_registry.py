"""
Guard: the CD registry stays complete and consistent with infer_cd.py.

Replaces test_registry_matches_legacy.py, which pinned the registry to the
twelve per-region scripts. Those scripts are gone -- infer_cd.py is now the
only implementation -- so what needs guarding is that every (model, region)
pair is still resolvable and that the registry describes each backbone fully.

    python tests/test_cd_registry.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402
import paths as P  # noqa: E402

REQUIRED = ("label", "chip", "code_dir", "builder", "norm", "head")
NORM_FIELDS = {
    "zscore_shared": ("mean", "std"),
    "zscore_paired": ("t1_mean", "t1_std", "t2_mean", "t2_std"),
    "minmax": (),
}


def main():
    problems = []

    for name in sorted(R.MODELS):
        spec = R.model(name)
        for field in REQUIRED:
            if field not in spec:
                problems.append("{}: missing '{}'".format(name, field))

        scheme = spec.get("norm")
        if scheme not in NORM_FIELDS:
            problems.append("{}: unknown norm scheme {!r}".format(name, scheme))
        else:
            for field in NORM_FIELDS[scheme]:
                stats = spec.get(field)
                if not stats:
                    problems.append("{}: {} requires '{}'".format(name, scheme, field))
                elif len(stats) != 6:
                    problems.append("{}: '{}' has {} bands, expected 6".format(
                        name, field, len(stats)))

        if spec.get("head") not in ("logprob", "logits"):
            problems.append("{}: unknown head {!r}".format(name, spec.get("head")))

        code_dir = ROOT / spec["code_dir"]
        if not code_dir.is_dir():
            problems.append("{}: code_dir {} does not exist".format(name, code_dir))

        print("  OK        {:<12} chip={} norm={} head={}".format(
            name, spec["chip"], spec["norm"], spec["head"]))

    # Every pair must produce a checkpoint path, whether or not the file is
    # present -- must_exist=False so this passes on a machine without weights.
    for model, region in R.pairs():
        try:
            P.cd_checkpoint(model, region, must_exist=False)
        except SystemExit as exc:
            problems.append("{} {}: {}".format(model, region, exc))
    print("  OK        {} (model, region) pairs resolve".format(len(R.pairs())))

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nCD registry is complete and consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
