"""
Guard: infer_seg.py must reproduce the legacy per-region scripts exactly.

infer_seg.py replaces fourteen infer_<model>[_<head>]_<region>.py files. What
can be checked without a GPU -- normalization and the input layout each
backbone expects -- is compared here against the formulas transcribed from the
originals' __getitem__.

The epsilon placement differs per backbone and is deliberate: SatMAE divides
by (std + 1e-8), Prithvi and SpectralGPT divide by the bare denominator. An
earlier version of this refactor added 1e-8 uniformly; the fixtures below
exist so that cannot happen again unnoticed.

    python tests/test_infer_seg_equivalence.py
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402


def _stub_stack():
    for name in ["torch", "torch.nn", "torch.nn.functional", "rasterio",
                 "tqdm", "torch.utils", "torch.utils.data"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["torch"].nn = sys.modules["torch.nn"]
    sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]
    sys.modules["tqdm"].tqdm = lambda x, **kw: x
    sys.modules["torch.utils.data"].DataLoader = object
    sys.modules["torch.utils.data"].Dataset = object
    sys.modules["rasterio"].transform = types.ModuleType("rasterio.transform")


def load_unified():
    _stub_stack()
    spec = importlib.util.spec_from_file_location("infer_seg", ROOT / "infer_seg.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    unified = load_unified()
    chip = np.load(FIXTURE_DIR / "seg_norm_input.npy")
    golden = json.loads((FIXTURE_DIR / "seg_norm_expected.json").read_text())

    problems = []
    for model in sorted(golden):
        got = unified.make_normalizer(R.SEG_MODELS[model])(chip)
        ref = golden[model]
        delta = max(abs(ref["sum"] - float(got.sum())),
                    max(abs(a - b) for a, b in
                        zip(ref["first16"], got.ravel()[:16].tolist())))
        if delta < 1e-3:
            print("  OK        {:<12} delta = {:.3e}".format(model, delta))
        else:
            problems.append("{}: normalization differs by {:.3e}".format(model, delta))

    # Prithvi is the only backbone needing a layout change; guard both branches.
    if R.SEG_MODELS["prithvi"].get("layout") != "bands_dates":
        problems.append("prithvi lost its bands_dates layout")
    for model in ("satmae", "spectralgpt"):
        if R.SEG_MODELS[model].get("layout"):
            problems.append("{} should not declare a layout".format(model))
    print("  OK        input layouts (prithvi reshapes, others do not)")

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\ninfer_seg.py matches the legacy scripts on every checkable path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
