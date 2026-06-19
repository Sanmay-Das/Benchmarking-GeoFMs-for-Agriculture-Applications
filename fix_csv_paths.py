import pandas as pd

# Read CSV
df = pd.read_csv('/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/EuroSAT/train.csv')

# Add absolute path prefix
base_path = '/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/EuroSAT/'
df['Filename'] = base_path + df['Filename']

# Save
df.to_csv('/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/EuroSAT/train_absolute.csv', index=False)

# Do the same for test.csv
df_test = pd.read_csv('/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/EuroSAT/test.csv')
df_test['Filename'] = base_path + df_test['Filename']
df_test.to_csv('/bigdata/eldawylab/sdas050/MS_Research/SatMAE/data/EuroSAT/test_absolute.csv', index=False)