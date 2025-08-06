"""
Functions used for preprocessing APPL RGB images and corresponding masks.
"""

import os
import glob
import numpy as np
import pandas as pd

from PIL import Image
from typing import List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from multiprocessing import Pool
from scipy import ndimage
from tqdm import tqdm


# === Image & Mask Processing ===
def image_to_memmap(image_path: str, save_path: str, dtype: str) -> None:
    """
    Convert an RGB image to a numpy memmap file.
    """
    # book keeping
    name = os.path.basename(image_path)
    sub_dir = os.path.dirname(image_path).split(os.sep)[-1]
    save_dir = os.path.join(save_path, sub_dir)
    save_name = os.path.join(save_dir, name.replace(image_path.split('.')[-1], 'memmap'))

    # skip if already exists
    if os.path.exists(save_name):
        return

    # load image
    image = np.array(Image.open(image_path).convert('RGB'))
            
    # create save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # save image as memmap
    memmap = np.memmap(save_name, dtype=dtype, mode='w+', shape=image.shape)
    memmap[:] = image[:]
    del memmap
    
def mask_to_memmap(mask_path: str, save_path: str, dtype: str) -> None:
    """
    Convert a mask image to a numpy memmap file.
    """
    # book keeping
    name = os.path.basename(mask_path)
    sub_dir = os.path.dirname(mask_path).split(os.sep)[-2]
    save_dir = os.path.join(save_path, sub_dir)
    save_name = os.path.join(save_dir, name.replace(mask_path.split('.')[-1], 'memmap'))

    # skip if already exists
    if os.path.exists(save_name):
        return

    # load mask
    mask = np.array(Image.open(mask_path).convert('RGB'))
    mask = mask[:, :, 0] > 128 # convert red channel to boolean mask

    # create save directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # save image as memmap
    memmap = np.memmap(save_name, dtype=dtype, mode='w+', shape=mask.shape)
    memmap[:] = mask[:]
    del memmap
    
# === Splitting & Data Loading ===

def split(df: pd.DataFrame, max_reshuffle_iters: int=1000, samples_per_split: int=250, random_state: int=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and target sets.
    """
    # build train/test splitter
    splitter = GroupShuffleSplit(
        n_splits=max_reshuffle_iters,
        test_size=samples_per_split / len(df),
        random_state=random_state
    )

    # reshuffle splits until we find samples_per_split in the test set
    for train_idx, target_idx in splitter.split(df, groups=df['Group']):
        if len(df.iloc[target_idx]) == samples_per_split:
            break

    # raise an error if the test set does not have samples_per_split samples
    if len(df.iloc[target_idx]) != samples_per_split:
        raise ValueError('Failed to find exact split')

    # split the metadata into train and test sets
    train_df = df.iloc[train_idx].reset_index(drop=True)
    target_df = df.iloc[target_idx].reset_index(drop=True)

    return train_df, target_df    

def load_memmap_paths(
    base_paths: Tuple[str, str],
    names: List[str],
    ) -> Tuple[List[str], List[str]]:
    
    # Unpack base paths for images and masks. Check for existence
    image_dir, mask_dir = base_paths
    assert os.path.exists(image_dir), f"Image directory not found: {image_dir}"
    assert os.path.exists(mask_dir), f"Mask directory not found: {mask_dir}"
    
    # Replace file extensions and find paths
    names = [name.replace('.png', '.memmap') for name in names]
    image_paths = [glob.glob(image_dir + f'**/*/{name}', recursive=True)[0] for name in names]
    mask_paths = [glob.glob(mask_dir + f'**/*/{name}', recursive=True)[0] for name in names]
    return image_paths, mask_paths

def load_memmap(path):
    """
    Load a memory-mapped file from the given path.
    Highly dependent on path structure.
    Args:
        path (str): The path to the memory-mapped file.
    """
    
    # Set h, w based on the path
    if '/rgb1' in os.path.dirname(path):
        h, w = 6556, 4104
    if '/rgb2' in os.path.dirname(path):
        h, w = 3006, 4104
        
    # set dtype and build shape based on the path.
    if '/images' in os.path.dirname(path):
        dtype = 'uint8'
        shape = (h, w, 3)
    if '/masks' in os.path.dirname(path):
        dtype = 'bool'
        shape = (h, w)

    np_arr = np.memmap(
        filename=path,
        dtype=np.dtype(dtype),
        mode='r+',
        shape=shape
    )
    return np_arr

def _dilate_masks(mask, tile_size):
    mask = np.array(mask)
    mask = ndimage.binary_dilation(mask, iterations=tile_size//2, structure=np.ones((3,3)))
    return mask

def mp_dilate_masks(masks, tile_size, pool_size=32):
    print("Dilating masks...")
    with Pool(pool_size) as p:
            dmasks = list(tqdm(p.starmap(_dilate_masks, [(mask, tile_size) for mask in masks]), total=len(masks)))
    print(f"Dilated {len(dmasks)} masks.")
    return dmasks

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
    
    
    

