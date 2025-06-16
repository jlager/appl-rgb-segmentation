import os, sys, glob
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
from multiprocessing import Pool
from modules import (
    process_mask
)

md_path = 'data/splits/metadata.csv'
assert os.path.exists(md_path), f"Metadata path {md_path} does not exist."
metadata = pd.read_csv(md_path)

# Base Paths
raw_base = 'data/raw/'
img_base = 'data/images/'
mask_base = 'data/masks/'

base_paths = [raw_base, img_base, mask_base]
for path in base_paths:
    assert os.path.exists(path), f"Base directory {path} does not exist."
    
# Subdirectories for storing masks
categories = ['rgb1-poplar', 'rgb1-switchgrass','rgb2-poplar', 'rgb2-switchgrass']
categories = sorted(categories)
for category in categories:
    if not os.path.exists(os.path.join(mask_base, category)):
        os.makedirs(os.path.join(mask_base, category))

# Get raw masks
cvat_raw = glob.glob(raw_base + 'cvat_annotations/**/SegmentationClass/*.png')
manual_raw = glob.glob(raw_base + 'manual_annotations/**/SegmentationClass/*.png')
raw_masks = cvat_raw + manual_raw
assert len(raw_masks) == len(metadata), f"Missing masks. Expected {len(metadata)}, found {len(raw_masks)}."

# (Filename, folder) tuple for each category
print(f"Processing {len(raw_masks)} raw masks...")
with Pool(32) as p:
    
    mask_args = [(
        mask,
        os.path.join(mask_base, os.path.basename(os.path.dirname(os.path.dirname(mask)))))
        for mask in tqdm(raw_masks, desc="Preparing mask arguments")]

    list(p.starmap(process_mask, mask_args))