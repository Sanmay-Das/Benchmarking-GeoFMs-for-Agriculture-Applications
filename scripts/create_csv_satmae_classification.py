from pathlib import Path
import csv
from collections import Counter

# =========================
# CONFIG
# =========================
DATA_ROOT = Path("/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data").resolve()  # Make absolute
CHIPS_DIR = DATA_ROOT / "chips"

TRAIN_LOCATIONS = ["CentIA"]
TEST_LOCATIONS  = ["EastIA"]

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV  = DATA_ROOT / "test.csv"

# CDL crop codes you kept
CDL_MIN = 1
CDL_MAX = 61


# =========================
# HELPERS
# =========================
def collect_samples(locations):
    samples = []
    class_counter = Counter()

    for loc in locations:
        loc_dir = CHIPS_DIR / loc
        if not loc_dir.exists():
            raise FileNotFoundError(f"Missing directory: {loc_dir}")

        for tif_path in sorted(loc_dir.glob("chip_*.tif")):
            label_path = tif_path.with_suffix(".txt")
            if not label_path.exists():
                continue

            label_raw = int(label_path.read_text().strip())

            # Safety check
            if not (CDL_MIN <= label_raw <= CDL_MAX):
                # skip unexpected labels (or raise error if you prefer)
                continue

            # convert 1..61 -> 0..60
            label = label_raw - 1

            # Use absolute path instead of relative
            abs_path = tif_path.resolve()
            samples.append((str(abs_path), label))
            class_counter[label] += 1

    return samples, class_counter


def write_csv(csv_path, samples):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "label"])
        writer.writerows(samples)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("=" * 70)
    print("CREATING SatMAE CSV FILES")
    print("=" * 70)

    # -------- TRAIN --------
    train_samples, train_dist = collect_samples(TRAIN_LOCATIONS)
    write_csv(TRAIN_CSV, train_samples)

    print(f"\nTRAIN CSV: {TRAIN_CSV}")
    print(f"  Samples: {len(train_samples)}")
    print(f"  Classes: {len(train_dist)}")
    print(f"  Sample path: {train_samples[0][0]}")  # Show first path
    print(f"  Distribution:")
    for k, v in train_dist.most_common():
        print(f"    Class {k}: {v}")

    # -------- TEST --------
    test_samples, test_dist = collect_samples(TEST_LOCATIONS)
    write_csv(TEST_CSV, test_samples)

    print(f"\nTEST CSV: {TEST_CSV}")
    print(f"  Samples: {len(test_samples)}")
    print(f"  Classes: {len(test_dist)}")
    print(f"  Sample path: {test_samples[0][0]}")  # Show first path
    print(f"  Distribution:")
    for k, v in test_dist.most_common():
        print(f"    Class {k}: {v}")

    print("\nCSV creation complete")