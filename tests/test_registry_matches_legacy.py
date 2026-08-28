"""
Guard: the registry must reproduce the constants of the original per-region
scripts exactly.

The 12 infer_cd_<model>_<region>.py files are being collapsed into a single
parameterized driver. This test pins the registry to what those files actually
did, so the refactor cannot silently change an experiment. Delete it once the
legacy scripts are gone -- until then it is the proof the collapse is faithful.

    python tests/test_registry_matches_legacy.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402


def check(model_name, region_name):
    path = ROOT / "infer_cd_{}_{}.py".format(model_name, region_name)
    if not path.exists():
        return ["{}: file absent".format(path.name)]

    src = path.read_text()
    spec = R.model(model_name)
    problems = []

    def expect(label, actual, wanted):
        if actual != wanted:
            problems.append("{}: {} != {}".format(label, actual, wanted))

    chip = int(re.search(r"CHIP_SIZE\s*=\s*(\d+)", src).group(1))
    expect("chip", chip, spec["chip"])

    # Checkpoints resolve through paths.cd_checkpoint(model, region) rather
    # than a hardcoded path, so assert the call carries the right pair.
    call = re.search(r"CHECKPOINT\s*=\s*str\(cd_checkpoint\(\s*'(\w+)'\s*,\s*'(\w+)'\s*\)\)", src)
    if not call:
        problems.append("CHECKPOINT does not use cd_checkpoint(model, region)")
    else:
        expect("ckpt_model", call.group(1), model_name)
        expect("ckpt_region", call.group(2), region_name)
    if re.search(r"best_F1_model\.pth'", src):
        problems.append("hardcoded checkpoint path still present")

    csv = re.search(r"change_detection_chips/([a-z]+)/(\w+)_chips\.csv", src)
    expect("csv_model", csv.group(1), model_name)
    expect("csv_region", csv.group(2), region_name)

    builder_fn = spec["builder"][1]
    if builder_fn not in src:
        problems.append("builder {} not called".format(builder_fn))
    if spec["code_dir"] not in src:
        problems.append("code_dir {} not on sys.path".format(spec["code_dir"]))

    # exp(log_probs) and softmax(log_probs) are mathematically identical for a
    # LogSoftmax head, so both spellings are accepted for a "logprob" model.
    uses_exp = "torch.exp(" in src
    uses_softmax = "F.softmax(" in src
    if spec["head"] == "logprob" and not (uses_exp or uses_softmax):
        problems.append("no probability conversion found")
    if spec["head"] == "logits" and not uses_softmax:
        problems.append("expected softmax for logits head")

    return ["{}: {}".format(path.name, p) for p in problems]


def main():
    failures = []
    for model_name, region_name in R.pairs():
        found = check(model_name, region_name)
        status = "OK" if not found else "MISMATCH"
        print("  {:<9} infer_cd_{}_{}.py".format(status, model_name, region_name))
        failures.extend(found)

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nAll {} (model, region) pairs match the registry.".format(len(R.pairs())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
