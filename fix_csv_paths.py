
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'configs'))
from paths import MSR_ROOT, DATA_ROOT, OUTPUT_ROOT, WEIGHTS, PREDICTIONS

import pandas as pd

# Read CSV
df = pd.read_csv(f'{MSR_ROOT}/SatMAE/data/EuroSAT/train.csv')

# Add absolute path prefix
base_path = f'{MSR_ROOT}/SatMAE/data/EuroSAT/'
df['Filename'] = base_path + df['Filename']

# Save
df.to_csv(f'{MSR_ROOT}/SatMAE/data/EuroSAT/train_absolute.csv', index=False)

# Do the same for test.csv
df_test = pd.read_csv(f'{MSR_ROOT}/SatMAE/data/EuroSAT/test.csv')
df_test['Filename'] = base_path + df_test['Filename']
df_test.to_csv(f'{MSR_ROOT}/SatMAE/data/EuroSAT/test_absolute.csv', index=False)