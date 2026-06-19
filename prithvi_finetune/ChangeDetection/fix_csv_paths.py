import pandas as pd

OLD_ROOT = "C:/MS_Research/scripts/change_detection_chips"
NEW_ROOT = "/bigdata/eldawylab/sdas050/MS_Research/change_detection_chips"

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