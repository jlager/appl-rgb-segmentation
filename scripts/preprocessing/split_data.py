import os, sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm
from scipy import ndimage
from multiprocessing import Pool
from torch.utils.data import DataLoader
from modules import (
    data_split,
    verify_disjoint,
    load_dataset,
    dilate_masks,
    Augmentations,
    TileDataset_DilationSampling as TDS
)

# === Data Splitting ===

# Load metadata
md_path = 'data/splits/metadata.csv'
assert os.path.exists(md_path), f"Metadata file not found at {md_path}"
metadata = pd.read_csv(md_path)
metadata['Group'] = metadata['Species'].astype(str) + '_' + metadata['Plant ID'].astype(str)

# Set split options
n_splits = 1000
n_samples = 250
group = 'Group'
random_state = 42

# Splitting into val/test with GroupShuffleSplit
train, test = data_split(
    metadata,
    n_splits=n_splits, 
    n_samples=n_samples, 
    group=group, 
    random_state=random_state
)

train, val = data_split(
    train, 
    n_splits=n_splits, 
    n_samples=n_samples, 
    group=group, 
    random_state=random_state
)

# Verify disjoint on the 'Group' column
verify_disjoint(train, test, 'Group', verbose=True)
verify_disjoint(train, val, 'Group', verbose=True)
verify_disjoint(val, test, 'Group', verbose=True)


# === Data Loading ===

# Options
data = 'data/'
images_base = os.path.join(data, 'images/')
masks_base = os.path.join(data, 'masks/')
image_ext = '.png'
assert os.path.exists(images_base), f"Images path {images_base} does not exist."
assert os.path.exists(masks_base), f"Masks path {masks_base} does not exist."

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Load names for each split
names_train = train['File Name'].tolist() 
names_val = val['File Name'].tolist()
names_test = test['File Name'].tolist()

# Get datasets
# train_images, train_masks = load_dataset((images_base, masks_base), names_train, "Train")
val_images, val_masks = load_dataset((images_base, masks_base), names_val, "Validation")
# test_images, test_masks = load_dataset((images_base, masks_base), names_test, "Test")

# Dilate the validation masks
print("Dilating validation masks...")
with Pool(32) as p:
        dmasks_val = list(tqdm(p.imap(dilate_masks, val_masks), total=len(val_masks)))
print(f"Dilated {len(dmasks_val)} validation masks.")

# Dataloaders
val_dataset = TDS(
    images=val_images,
    masks=val_masks,
    dilated_masks=dmasks_val,
    tile_size=512,
    transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
    tiles_per_image=100,
    mix_ratio=0.5,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)

    