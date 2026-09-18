"""
Guard: STATES is the user-facing axis, so it must stay consistent with both
the CD splits it wraps and the checkpoint names published on Hugging Face.

The naming coupling is the fragile part. fetch.py builds a download URL as
    weights/change_detection/<model>_cd_<state>.pth
so a state key renamed here silently 404s for every user. This pins the four
keys to the tokens actually present in the dataset repo.

    python tests/test_states.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "configs"))
import registry as R  # noqa: E402

# Tokens as they appear in the published filenames, verified against
# HfApi().list_repo_files("sanmay4119/geofm-agriculture-benchmark").
HF_STATE_TOKENS = {"iowa", "minnesota", "north_carolina", "california"}

EXPECTED_TEST_REGIONS = {
    "iowa": "NWIA",
    "minnesota": "SouthMN",
    "north_carolina": "EastNC",
    "california": "SouthCA",
}


def main():
    problems = []

    if set(R.STATES) != HF_STATE_TOKENS:
        problems.append("state keys {} != published tokens {}".format(
            sorted(R.STATES), sorted(HF_STATE_TOKENS)))

    for name in R.states():
        meta = R.state(name)
        for key in ("cd", "seg", "label"):
            if key not in meta:
                problems.append("{}: missing '{}'".format(name, key))

        # The cd key must name a real split.
        if meta.get("cd") not in R.CD_SPLITS:
            problems.append("{}: cd key '{}' is not a CD_SPLITS entry".format(
                name, meta.get("cd")))
            continue

        # The label must round-trip to the state key, since that derivation
        # is what makes the HF token predictable rather than hardcoded.
        derived = R.CD_SPLITS[meta["cd"]]["label"].lower().replace(" ", "_")
        if derived != name:
            problems.append("{}: split label derives '{}', not the key".format(
                name, derived))

        got = R.test_region(name)
        want = EXPECTED_TEST_REGIONS[name]
        if got != want:
            problems.append("{}: test region {} != {}".format(name, got, want))

        # Every region of the split must invert back to this state.
        regions = R.cd_regions(name)
        for role in ("train", "val", "test"):
            back = R.state_of_region(regions[role])
            if back != name:
                problems.append("{}: {} ({}) inverts to {}".format(
                    name, regions[role], role, back))

    # Each of the twelve regions belongs to exactly one state.
    seen = {}
    for name in R.states():
        regions = R.cd_regions(name)
        for role in ("train", "val", "test"):
            r = regions[role]
            if r in seen:
                problems.append("{} claimed by both {} and {}".format(
                    r, seen[r], name))
            seen[r] = name
    if len(seen) != 12:
        problems.append("expected 12 distinct regions, found {}".format(len(seen)))

    # The four test regions must match REGIONS, which infer_cd.py uses.
    if set(EXPECTED_TEST_REGIONS.values()) != set(R.REGIONS):
        problems.append("test regions {} != REGIONS {}".format(
            sorted(set(EXPECTED_TEST_REGIONS.values())), sorted(R.REGIONS)))

    for bad in ("Iowa", "IA", "", "texas"):
        try:
            R.state(bad)
        except SystemExit:
            pass
        else:
            problems.append("state('{}') should have been rejected".format(bad))

    print("states:  {}".format(", ".join(R.states())))
    print("regions: {} mapped, each to one state".format(len(seen)))

    if problems:
        print("\nFAILURES:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nSTATES is consistent with CD_SPLITS, REGIONS and the HF names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
