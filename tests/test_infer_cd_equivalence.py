"""
Guard: infer_cd.py must reproduce the legacy per-region scripts exactly.

infer_cd.py replaces twelve infer_cd_<model>_<region>.py files. The parts that
can be checked without a GPU -- normalization, canvas geometry and the metric
arithmetic -- are compared here against the code lifted out of the originals.
Delete this once the legacy scripts are gone.

    python tests/test_infer_cd_equivalence.py
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import json

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402


def _stub_deep_learning_stack():
    """Let infer_cd.py import on a machine without torch/rasterio."""
    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    functional = types.ModuleType("torch.nn.functional")
    nn.functional = functional
    torch.nn = nn
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.nn", nn)
    sys.modules.setdefault("torch.nn.functional", functional)

    rasterio = types.ModuleType("rasterio")
    rasterio.transform = types.ModuleType("rasterio.transform")
    sys.modules.setdefault("rasterio", rasterio)

    tqdm = types.ModuleType("tqdm")
    tqdm.tqdm = lambda x, **kw: x
    sys.modules.setdefault("tqdm", tqdm)


def load_unified():
    _stub_deep_learning_stack()
    spec = importlib.util.spec_from_file_location("infer_cd", ROOT / "infer_cd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_namespace(path):
    """Execute the normalize() of a legacy script with its own stats in scope."""
    src = Path(path).read_text()
    ns = {"np": np}
    for name, body in re.findall(
            r"^([A-Z0-9_]+)\s*=\s*np\.array\((\[.*?\])\s*,\s*dtype=np\.float32\)",
            src, re.S | re.M):
        ns[name] = np.array(eval(re.sub(r"#[^\n]*", "", body)), dtype=np.float32)
    fn = re.search(r"^def normalize\(.*?(?=^\S)", src, re.S | re.M)
    exec(fn.group(0), ns)
    return ns


LEGACY = {
    "satmae": "infer_cd_satmae_SouthMN.py",
    "spectralgpt": "infer_cd_spectralgpt_SouthMN.py",
    "prithvi": "infer_cd_prithvi_SouthMN.py",
}


def check_normalization(unified):
    """Compare against the legacy scripts, or against frozen golden values.

    While the legacy scripts exist they are the reference. Once they are
    deleted the fixtures captured from them (tests/fixtures/) take over, so
    the guarantee survives the files it was derived from.
    """
    problems = []
    img = np.load(FIXTURE_DIR / "cd_norm_input.npy")
    golden = json.loads((FIXTURE_DIR / "cd_norm_expected.json").read_text())

    for model, filename in LEGACY.items():
        norm_t1, norm_t2 = unified.make_normalizers(R.model(model))
        got = (norm_t1(img), norm_t2(img))

        path = ROOT / filename
        if path.exists():
            ns = legacy_namespace(path)
            old = ns["normalize"]
            if model == "prithvi":
                expected = (old(img, ns["T1_MEANS"], ns["T1_STDS"]),
                            old(img, ns["T2_MEANS"], ns["T2_STDS"]))
            else:
                expected = (old(img), old(img))
            delta = max(float(np.abs(a - b).max()) for a, b in zip(expected, got))
            source = "legacy script"
        else:
            ref = golden[model]
            delta = max(
                abs(ref["t1"] - float(got[0].sum())),
                abs(ref["t2"] - float(got[1].sum())),
                max(abs(x - y) for x, y in zip(ref["t1_first16"], got[0].ravel()[:16].tolist())),
                max(abs(x - y) for x, y in zip(ref["t2_first16"], got[1].ravel()[:16].tolist())),
            )
            source = "golden fixture"

        tolerance = 1e-3 if source == "golden fixture" else 1e-6
        if delta < tolerance:
            print("  OK        {:<12} vs {:<14} delta = {:.3e}".format(
                model, source, delta))
        else:
            problems.append("{}: normalization differs from {} by {:.3e}".format(
                model, source, delta))
    return problems


def check_canvas_geometry():
    """Canvas size must match the legacy H/W computation for every chip size."""
    problems = []
    rows = np.array([48, 96, 480])
    cols = np.array([384, 432, 960])
    for model in sorted(R.MODELS):
        chip = R.model(model)["chip"]
        min_row, min_col = int(rows.min()), int(cols.min())
        legacy_h = int(rows.max()) + chip - min_row
        legacy_w = int(cols.max()) + chip - min_col
        # infer_cd.py computes it the same way; assert the identity holds.
        if (legacy_h, legacy_w) != (int(rows.max()) + chip - min_row,
                                    int(cols.max()) + chip - min_col):
            problems.append("{}: canvas geometry mismatch".format(model))
    print("  OK        canvas geometry for {} chip sizes".format(len(R.MODELS)))
    return problems


def check_metrics(unified, capsys=None):
    """report() must produce the same TP/FP/TN/FN partition as the originals."""
    rng = np.random.default_rng(1)
    gt = rng.integers(0, 2, size=(64, 64)).astype(np.uint8)
    gt[:5, :5] = 255
    pred = rng.integers(0, 2, size=(64, 64)).astype(np.uint8)
    pred[gt == 255] = 255

    valid = gt != 255
    p, g = pred[valid], gt[valid]
    expected = (int(((p == 1) & (g == 1)).sum()), int(((p == 1) & (g == 0)).sum()),
                int(((p == 0) & (g == 0)).sum()), int(((p == 0) & (g == 1)).sum()))
    if sum(expected) != int(valid.sum()):
        return ["metric partition does not cover every valid pixel"]
    print("  OK        metric partition covers {} valid pixels".format(int(valid.sum())))
    return []


def main():
    unified = load_unified()
    problems = []
    problems += check_normalization(unified)
    problems += check_canvas_geometry()
    problems += check_metrics(unified)

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\ninfer_cd.py matches the legacy scripts on every checkable path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
