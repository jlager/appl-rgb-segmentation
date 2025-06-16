"""
Functions used for preprocessing APPL RGB images and corresponding masks.
"""

import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from skimage import measure
from PIL import ImageDraw
from typing import List, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit
from multiprocessing import Pool
from scipy import ndimage
from tqdm import tqdm




# === Image & Mask Processing ===
def process_image(filename:str, dest_folder:str):
    """
    Categorizes an APPL RGB image by 'Modality'-'Species' and saves it to the specified destination folder.

    Args:
        filename (str): The path to the mask image file.
        processed_dir_root (str): The directory root for where the processed mask will be saved.
    """
    
    img = Image.open(filename)
    img.save(f'{dest_folder}/{os.path.basename(filename)}')


def process_mask(filename:str, dest_folder:str):
    """
    Processes a raw CVAT mask image by thresholding and saving the binary mask. Binary mask is categorized by 'Modality'-'Species' within processed_dir_root

    Args:
        filename (str): The path to the mask image file.
        processed_dir_root (str): The directory root for where the processed mask will be saved.
    """
 
    # # don't reprocess old files
    # if os.path.exists(f'{save_path}{os.path.basename(filename)}'):
    #     return
    
    # 1. get modality-species from filename
    # 2. load mask as an RGB image
    # 3. threshold the image, get list of pixels w/ values > 128     
    # 4. convert boolean array to binary (0 or 1) >> gives binary mask
    # 5. save the processed mask
    
    img = Image.open(filename).convert('RGB')
    img = np.array(img)[:, :, 0] > 128
    img = Image.fromarray(img.astype(np.uint8) * 255)
    img.save(f'{dest_folder}/{os.path.basename(filename)}')
    


def overlay(image_dir_root:str, mask_dir_root:str):
    """
    Overlays a binary mask on its original image.
    """
    
    # Some checks on arguments
    assert os.path.exists(image_dir_root), f"Image directory {image_dir_root} does not exist."
    assert os.path.exists(mask_dir_root), f"Mask directory {mask_dir_root} does not exist."
    
    categories = glob.glob(mask_dir_root + '*/')
    assert len(categories) > 0, f"No categories found in {mask_dir_root}. Check the directory structure."
    print(f"Found {len(categories)} categories in {mask_dir_root}")
    for category in categories:
        print(category)
    
    # Get all mask paths
    masks = [x for x in glob.glob(mask_dir_root + '**/*.png')]
    
    # Load random image and mask
    index = np.random.randint(len(masks))
    category = f"{os.path.basename(os.path.dirname(masks[index]))}/"
    name = os.path.basename(masks[index])
    image = Image.open(f'{image_dir_root}{category}{name}')
    mask = Image.open(f'{mask_dir_root}{category}{name}')
    
    # Get contours from mask
    contours = measure.find_contours(np.array(mask), level=0.5)
    
    # Draw contour on image (reference below)
    for contour in contours:
        contour = np.flip(contour, axis=1)
        draw = ImageDraw.Draw(image)
        draw.line([tuple(c) for c in contour.tolist()], fill='red', width=3)
        
    return image

# === Splitting & Data Loading ===


# sklearn gss to do our splits
def data_split(metadata: pd.DataFrame, n_splits, n_samples, group, random_state):
    """
    Split metadata into train and target sets using GroupShuffleSplit.
    """
    
    # options
    count = 0
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=n_samples/len(metadata), random_state=random_state)

    # storage for indices
    target_idx = []
    
    # Sample indices until we find a split with exactly n_samples in the target set
    for train_idx, target_idx in gss.split(metadata, groups=metadata[group]):

        if len(metadata.iloc[target_idx])==n_samples:
            break
        count += 1
    
    # If we reach the maximum number of splits without finding a valid target set, raise an error
    if len(metadata.iloc[target_idx]) != n_samples:
        raise ValueError(f"Could not find a split with exactly {n_samples} samples within {n_splits} iterations.")
    
    # Return the train and target sets
    target = metadata.iloc[target_idx].reset_index(drop=True)
    train = metadata.iloc[train_idx].reset_index(drop=True)
    
    return train, target

def verify_disjoint(df1, df2, column, verbose=False):
    """
    Verify that two DataFrames have disjoint sets of specified column values.
    """
    
    similar = []

    column1 = set(df1[column])
    column2 = set(df2[column])

    if column1.isdisjoint(column2):
        return True
    else:
        similar = column1.intersection(column2)

    if verbose:
        print(f"Overlapping groups: {len(similar)}")
        for group in similar:
            print(f"- {group}")
            
    assert len(similar) == 0, f"DataFrames are not disjoint on column '{column}'. See above for overlapping groups."
            
def _load_image_names(
    image_dir: str,
    mask_dir: str,
    names: List[str],
    ) -> Tuple[List[str], List[str]]:

    mask_paths = [glob.glob(mask_dir + f'**/{name}')[0] for name in names]
    image_paths = [glob.glob(image_dir + f'**/{name}')[0] for name in names]
    
    return image_paths, mask_paths

def _load_image_and_mask(paths):
    img_path, mask_path = paths
    image = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    return image, mask

def load_dataset(base_paths, names, dataset_name, pool_size=32):
    images_base, masks_base = base_paths
    image_paths, mask_paths = _load_image_names(images_base, masks_base, names)
    print(f"\nFound {len(image_paths)} images for {dataset_name}.")
    print(f"Loading {dataset_name} images into memory...")
    with Pool(pool_size) as p:
        images, masks = zip(*tqdm(p.imap(_load_image_and_mask, zip(image_paths, mask_paths)), total=len(image_paths)))
    print(f"{dataset_name} images loaded.")
    return images, masks

def dilate_masks(mask):
    mask = np.array(mask)
    mask = ndimage.binary_dilation(mask, iterations=512//2, structure=np.ones((3,3)))
    return mask

# === Utils ===
def execute_file(file: str):
    """
    Execute a Python file and return its output.
    
    Args:
    file (str): Path to the Python file to execute.
    """

    try:  
        with open(file) as f:
            exec_globals = {'__file__': file}
            exec(f.read(), exec_globals)
    except Exception as e:
        print(f"Error executing file {file}: {e}")
        raise
    
    
    

