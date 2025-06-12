import os, sys, glob
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

md_root = 'data/raw/metadata'
rgb1_path = os.path.join(md_root, 'final_sample_lists2.csv')
rgb2_path = os.path.join(md_root, 'full_sample_lists2_topview.csv')

check = [md_root, rgb1_path, rgb2_path]
for path in check:
    assert os.path.exists(path), f"Path {path} does not exist."
    
# Load RGB1
rgb1_df = pd.read_csv(rgb1_path)
rgb1_df = rgb1_df[[
    'species', 'experiment', 'Round Order', 'Plant Info', 'File Path', 
    'Analyse Date', 'PC1_bin', 'PC2_bin', 'PC3_bin']]

# Load RGB2
rgb2_df = pd.read_csv(rgb2_path)
rgb2_df = rgb2_df[rgb2_df['File Type'] == 'Image']
rgb2_df = rgb2_df[['species', 'experiment', 'Round Order', 'Plant Info', 'File Path', 'Analyse Date']]

# Merge RGB1 and RGB2
rgb2_df = pd.merge(
    rgb1_df.drop(columns=['File Path', 'Analyse Date']),
    rgb2_df,
    on=['species', 'experiment', 'Round Order', 'Plant Info'], 
    how='inner')

# Add modality info
rgb1_df['Modality'] = 'RGB1'
rgb2_df['Modality'] = 'RGB2'

# Concat RGB1 and RGB2
df = pd.concat([rgb1_df, rgb2_df], ignore_index=True)

# Update file paths
df['File Path'] = (
    df['File Path']
    .str.replace('a1\\', '')
    .str.replace('\\', '/')
    .apply(lambda x: '_'.join(x.split('/')))
)

# Convert bins to int
for col in ['PC1_bin', 'PC2_bin', 'PC3_bin']:
    df[col] = df[col].astype(int)

# Select and rename columns
df = df.rename(columns={
    'species': 'Species',
    'Modality': 'Modality',
    'Plant Info': 'Plant ID',
    'Analyse Date': 'Date',
    'PC1_bin': 'PC1',
    'PC2_bin': 'PC2',
    'PC3_bin': 'PC3',
    'File Path': 'File Name'
})
df = df[['Species', 'Modality', 'Plant ID', 'Date', 'PC1', 'PC2', 'PC3', 'File Name']]

# Sort and reset index
df = (
    df
    .sort_values(by=['Species', 'PC1', 'PC2', 'PC3', 'Plant ID', 'Modality'])
    .reset_index(drop=True)
)

# Check if our image files exist in the metadata
temp_dir = 'data/temp/'
assert os.path.exists(temp_dir), f"Temporary directory {temp_dir} does not exist."

expected_files = set(df['File Name'].unique())
actual_files = set(os.path.basename(f) for f in glob.glob(os.path.join(temp_dir, '*.png')))
missing = expected_files - actual_files
for name in sorted(missing):
    print(f"File {name} does not exist in {temp_dir}")

# Save to csv | Consider moving to metadata folder.
df.to_csv('data/splits/metadata.csv', index=False)
