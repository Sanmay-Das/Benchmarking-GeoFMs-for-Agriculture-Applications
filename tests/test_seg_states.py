"""
Guard: the segmentation state mapping stays consistent with what is published.

Segmentation is the awkward case. Training wrote three different checkpoint
naming schemes, the Hub carries a fourth (one flat file per model and state),
and only one head per model was uploaded. fetch.py bridges those, so this pins
the bridge: the published names it constructs, the head each file actually is,
and the destination it installs to.

    python tests/test_seg_states.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
sys.path.insert(0, str(ROOT / "scripts"))
import registry as R  # noqa: E402

# Verified against the dataset repo file listing. satmae has no california
# checkpoint published even though output_seg_CA_fpn exists locally.
PUBLISHED = {
    ("satmae", "iowa"), ("satmae", "minnesota"), ("satmae", "north_carolina"),
    ("spectralgpt", "california"), ("spectralgpt", "iowa"),
    ("spectralgpt", "minnesota"), ("spectralgpt", "north_carolina"),
    ("prithvi", "california"), ("prithvi", "iowa"),
    ("prithvi", "minnesota"), ("prithvi", "north_carolina"),
}

# FPN, identified by exact byte size: the published SatMAE files are
# 2,628,248,073 bytes, matching output_seg_*_fpn, against 2.54 GB for psanet
# and 2.47 GB for fcn.
EXPECTED_HEADS = {"satmae": "fpn", "spectralgpt": "", "prithvi": ""}


def main():
    problems = []

    for model, head in EXPECTED_HEADS.items():
        if R.seg_published_head(model) != head:
            problems.append("{}: published head is {!r}, expected {!r}".format(
                model, R.seg_published_head(model), head))

    # Every model must resolve a checkpoint for all four states, so that a
    # published file always has somewhere to be installed.
    for model in EXPECTED_HEADS:
        got = R.seg_states(model)
        if got != R.states():
            problems.append("{}: resolves {} states, expected all four".format(
                model, len(got)))

    # The relative path must be a real SEG_CHECKPOINTS value for the
    # published head, not something constructed.
    for model, state in sorted(PUBLISHED):
        rel = R.seg_checkpoint_relative(model, state)
        head = R.seg_published_head(model)
        region = R.seg_test_region(state)
        if R.SEG_CHECKPOINTS.get((model, head, region)) != rel:
            problems.append("{}/{}: {} is not the recorded path".format(
                model, state, rel))

    # The name fetch.py builds must be the one on the Hub.
    import fetch
    for model, state in sorted(PUBLISHED):
        want = "weights/segmentation/{}_seg_{}.pth".format(model, state)
        got = fetch.seg_weight_file(model, state)
        if got != want:
            problems.append("{}/{}: builds {}, expected {}".format(
                model, state, got, want))

    # Every cell plans a checkpoint and a chip archive, and nothing plans a
    # stitched raster: those were never published and cannot be rebuilt from
    # the chips, which sample about 13% of a region non-contiguously.
    for model in EXPECTED_HEADS:
        for state in R.states():
            kinds = sorted(i["kind"] for i in fetch.seg_plan(model, state, {}))
            if kinds != ["seg_chips", "weights"]:
                problems.append("{}/{}: plans {}, expected chips + weights"
                                .format(model, state, kinds))

    # Chip archives exist for the three non-Iowa test regions, less
    # prithvi/SouthCA. Iowa has no segmentation chips for any backbone.
    expected_chips = {
        ("satmae", "SouthMN"), ("satmae", "EastNC"), ("satmae", "SouthCA"),
        ("spectralgpt", "SouthMN"), ("spectralgpt", "EastNC"),
        ("spectralgpt", "SouthCA"),
        ("prithvi", "SouthMN"), ("prithvi", "EastNC"),
    }
    for model, region in sorted(expected_chips):
        want = "segmentation/seg_{}_{}.tar".format(model, region)
        if fetch.seg_chip_file(model, region) != want:
            problems.append("{}/{}: builds {}, expected {}".format(
                model, region, fetch.seg_chip_file(model, region), want))

    # Destinations must be the ones infer_seg.py reads through.
    import paths as P
    for model, region in sorted(expected_chips):
        if fetch.seg_chip_dest(model, region) != P.seg_chips_dir(
                model, region, must_exist=False):
            problems.append("{}/{}: chip destination disagrees with "
                            "paths.seg_chips_dir".format(model, region))
        if fetch.seg_splits_dest(model, region) != P.seg_splits_file(
                model, region, must_exist=False):
            problems.append("{}/{}: split destination disagrees with "
                            "paths.seg_splits_file".format(model, region))

    print("published checkpoints: {}".format(len(PUBLISHED)))
    print("chip archives pinned:  {}".format(len(expected_chips)))
    print("all cells plan chips + weights, no stitched raster")

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nSegmentation mapping is consistent with the published data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
