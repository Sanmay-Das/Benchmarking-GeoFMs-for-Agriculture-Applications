
import os as _os, sys as _sys
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isfile(
        _os.path.join(_d, 'configs', 'paths.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _os.path.join(_d, 'configs'))
from paths import CD_CHIPS, MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS  # noqa: E402

import pandas as pd

OLD_ROOT = "C:/MS_Research/scripts/change_detection_chips"
NEW_ROOT = f"{CD_CHIPS}"

MODELS    = ["spectralgpt", "prithvi", "satmae"]
LOCATIONS = ["CentIA", "EastIA", "NWIA"]

for model in MODELS:
    for loc in LOCATIONS:
        csv_path = f"{NEW_ROOT}/{model}/{loc}_chips.csv"
        try:
            df = pd.read_csv(csv_path)
            for col in ['t1', 't2', 'mask']:
                df[col] = df[col].str.replace(OLD_ROOT, NEW_ROOT, regex=False)
            df.to_csv(csv_path, index=False)
            print(f"Fixed: {csv_path}")
        except FileNotFoundError:
            print(f"Skipped (not found): {csv_path}")