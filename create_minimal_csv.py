import pandas as pd
import os

def create_minimal_csv(input_csv, output_csv):
    """Create CSV with only image_path and category (no timestamp)"""
    
    # Read original CSV
    df = pd.read_csv(input_csv)
    
    # Drop index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    print(f"\nProcessing: {input_csv}")
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Create new DataFrame with only required columns
    new_df = pd.DataFrame({
        'image_path': 'MS_Research/data/EuroSAT' + df['Filename'].astype(str),
        'category': df['ClassName']
    })
    
    # Save
    new_df.to_csv(output_csv, index=False)
    
    print(f"✅ Created: {output_csv}")
    print(f"New columns: {new_df.columns.tolist()}")
    print(f"New shape: {new_df.shape}")
    print(f"\nFirst 3 rows:")
    print(new_df.head(3))
    print(f"\nClass distribution:")
    print(new_df['category'].value_counts().sort_index())
    
    # Verify first image exists
    first_path = new_df.iloc[0]['image_path']
    exists = os.path.exists(first_path)
    print(f"\nFirst image path: {first_path}")
    print(f"Image exists: {'✅ YES' if exists else '❌ NO'}")
    
    return new_df

# Process all CSV files
print("="*60)
print("Creating CSV files WITHOUT timestamp")
print("="*60)

train_df = create_minimal_csv(
    '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/train.csv',
    '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/train_minimal.csv'
)

test_df = create_minimal_csv(
    '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/test.csv',
    '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/test_minimal.csv'
)

if os.path.exists('/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/validation.csv'):
    val_df = create_minimal_csv(
        '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/validation.csv',
        '/bigdata/eldawylab/sdas050/MS_Research/data/EuroSAT/validation_minimal.csv'
    )

print("\n" + "="*60)
print("✅ All minimal CSV files created successfully!")
print("="*60)
print("\nNow try running the training command and see if it crashes!")