import os, sys, glob
import pandas as pd
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
from multiprocessing import Pool
from modules import process_image

md_path = 'data/splits/metadata.csv'
assert os.path.exists(md_path), f"Metadata path {md_path} does not exist."
metadata = pd.read_csv(md_path)

raw_base = 'data/temp/'
imgs_base = 'data/images/'
assert os.path.exists(raw_base) and os.path.exists(imgs_base), f"Images path {raw_base} or {imgs_base} does not exist."

# Subdirectories for storing images
categories = ['rgb1-poplar', 'rgb1-switchgrass','rgb2-poplar', 'rgb2-switchgrass']
categories = sorted(categories)
for category in categories:
    if not os.path.exists(os.path.join(imgs_base, category)):
        os.makedirs(os.path.join(imgs_base, category))
        
# Get all images
raw_images = glob.glob(raw_base + '/*.png')
assert len(raw_images) == len(metadata), f"Number of images ({len(raw_images)}) does not match number of metadata entries ({len(metadata)})."
    
# Build modality_species_map
map = metadata.groupby('Species')['Modality'].unique().reset_index()
map['Species'] = map['Species'].str.lower()
map['Modality'] = map['Modality'].apply(lambda x: [s.lower() for s in x])
map = map.explode('Modality').reset_index(drop=True)

image_args = []
for path in raw_images:
    file = os.path.basename(path)
    if file in metadata['File Name'].values:
        
        row = metadata[metadata['File Name'] == file].iloc[0]
        species = row['Species'].lower()
        modality = row['Modality'].lower()
        
        match = map[
            (map['Species'] == species) &
            (map['Modality'].apply(lambda x: modality in x))
        ]
        
        if not match.empty:
            category = f"{match['Modality'].values[0]}-{match['Species'].values[0]}"
            image_args.append((path, os.path.join(imgs_base, category)))

assert len(image_args) == len(raw_images), f"Number of image arguments ({len(image_args)}) does not match number of raw images ({len(raw_images)})."

start = time.time()
with Pool(32) as p:
    list(p.starmap(process_image, image_args))
end = time.time()
print(f"Processed {len(image_args)} images in {end - start:.2f} seconds.")
