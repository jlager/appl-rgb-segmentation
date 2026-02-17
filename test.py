import os
os.environ["OMP_NUM_THREADS"] = "1"                # stop using all CPU threads
os.environ["MKL_NUM_THREADS"] = "1"                # stop using all CPU threads
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
cv2.setNumThreads(1)                               # stop using all CPU threads

import torch
torch.set_num_threads(1)                           # stop using all CPU threads
torch.set_num_interop_threads(1)                   # stop using all CPU threads

import random
import numpy as np
import pandas as pd
import argparse
import collections
from tqdm import tqdm
from typing import List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2

import models
import config

random.seed(config.SEED)
np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.SEED)

# =============================================================================
# Data
# =============================================================================

def get_paths(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """ Extracts image and mask paths from the DataFrame. """
    modalities, species = df['Modality'], df['Species']
    names = df['File Name'].str.replace(config.IMAGE_EXT, config.DATA_EXT)
    image_paths, mask_paths, roi_paths = [], [], []
    for m, s, n in zip(modalities, species, names):
        subdir = '-'.join([m, s])
        image_paths.append(os.path.join(config.IMAGE_DIR, subdir, n))
        mask_paths.append(os.path.join(config.MASK_DIR, subdir, n))
        roi_n = n.replace(config.DATA_EXT, config.ROI_EXT)
        roi_paths.append(os.path.join(config.ROI_DIR, subdir, roi_n))
    return image_paths, mask_paths, roi_paths

def load_images_and_masks(
    image_paths: List[str], 
    mask_paths: List[str],
) -> Tuple[List[np.memmap], List[np.memmap]]:
    """Loads images and masks from memmap files."""
    images, masks = [], []
    for i, m in zip(image_paths, mask_paths):
        h = config.RGB1_HEIGHT if 'rgb1' in i.lower() else config.RGB2_HEIGHT
        w = config.RGB1_WIDTH if 'rgb1' in i.lower() else config.RGB2_WIDTH
        c = config.RGB1_CHANNELS if 'rgb1' in i.lower() else config.RGB2_CHANNELS
        images.append(np.memmap(i, dtype=config.IMAGE_DTYPE, mode='r', shape=(h, w, c)))
        masks.append(np.memmap(m, dtype=config.MASK_DTYPE, mode='r', shape=(h, w)))
    return images, masks

# =============================================================================
# Segmentation
# =============================================================================

def is_topview(width: int, height: int) -> bool:
    """Determine if image is topview (wider) or sideview (taller)"""
    return width > height

def get_tile_boundary(
    center_row: int, 
    center_col: int, 
    tile_size: int,
) -> Tuple[int, int, int, int]:
    """Get the boundary coordinates for a tile given its center"""
    left = center_col - tile_size // 2
    top = center_row - tile_size // 2
    right = center_col + tile_size // 2
    bottom = center_row + tile_size // 2
    return (left, top, right, bottom)

# inference transform
transform = A.Compose([
    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ToTensorV2(),
])

def segment(
    image: np.ndarray,
    model: torch.nn.Module, 
    tile_size: int,
    step: int, 
    batch_size: int = 64,
    threshold: float = 0.5,
    n_classes: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    
    # initialize
    H, W, _ = image.shape
    C = n_classes
    device = next(model.parameters()).device
    model.eval()
    
    # initialize book keeping
    prob_accum = torch.zeros((C, H, W), dtype=torch.float32, device=device)
    count_accum = torch.zeros((H, W), dtype=torch.float32, device=device)
    visited = set()
    
    # use a queue for flood-fill like processing
    queue = collections.deque()
    
    # initialize with a grid of non-overlapping tiles across the entire image
    for row in range(tile_size//2, H, tile_size):
        for col in range(tile_size//2, W, tile_size):
            queue.append((row, col))
    
    while queue:
        
        # process tiles in batches for efficiency
        tiles = []
        boundaries = []
        coords = []
        
        # collect up to batch_size tiles to process
        while queue and len(tiles) < batch_size:
            row, col = queue.popleft()
            
            # skip if already visited
            coord_key = (row, col)
            if coord_key in visited:
                continue
                
            # mark as visited
            visited.add(coord_key)
            
            # get tile boundary
            left, top, right, bottom = get_tile_boundary(row, col, tile_size)
            
            # skip if out of bounds
            if left < 0 or top < 0 or right > W or bottom > H:
                continue

            # extract tile
            tile = image[top:bottom, left:right, :]
            augmented = transform(image=tile)
            tiles.append(augmented['image'].unsqueeze(0).to(device))
            boundaries.append((left, top, right, bottom))
            coords.append((row, col))
        
        # if no tiles to process, we're done
        if not tiles:
            break

        # process batch
        with torch.no_grad():
            batch = torch.cat(tiles, dim=0)
            probs = torch.softmax(model(batch), dim=1)
        
        # add neighbors of tiles with segmented content to queue
        has_new_tiles = False
        for i, ((row, col), (left, top, right, bottom)) in enumerate(zip(coords, boundaries)):
            tile_probs = probs[i]
            
            # save tile probabilities
            prob_accum[:, top:bottom, left:right] += tile_probs
            count_accum[top:bottom, left:right] += 1
            
            # check if this tile detects segmentation
            if C == 1:
                has_content = (tile_probs[0] > threshold).any().item()
            else:
                has_content = (torch.argmax(tile_probs, dim=0) > 0).any().item()
            
            # if so, add 4-connected neighbors
            if has_content:
                has_new_tiles = True
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr * step, col + dc * step
                    if (nr, nc) not in visited:
                        queue.append((nr, nc))
        
        # if no tiles had content, and queue is empty, break early
        if not has_new_tiles and not queue:
            break
    
    # build final mask
    with torch.no_grad():
        mask = count_accum > 0
        final_probs = torch.zeros_like(prob_accum)
        final_probs[:, mask] = prob_accum[:, mask] / count_accum[mask]
        final_probs = final_probs.cpu()
        
        # threshold probabilities to get final mask [H, W]
        if C == 1:
            final_mask = (final_probs[0] > threshold).astype(np.uint8)
        else:
            final_mask = torch.argmax(final_probs, dim=0).numpy().astype(np.uint8)

    return final_mask, final_probs.numpy()

def compute_image_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds: [H, W]; targets: [H, W]
    preds = preds.astype(float)
    targets = targets.astype(float)
    intersection = (preds * targets).sum(axis=(0, 1))
    union = preds.sum(axis=(0, 1)) + targets.sum(axis=(0, 1))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

# =============================================================================
# Main
# =============================================================================

def parse_args():
    # example: python test.py --backbone "vit_base_patch8_224" --tile_size 448 --device 0
    parser = argparse.ArgumentParser(description='Test segmentation model')
    parser.add_argument('--backbone', type=str, default='vit_base_patch8_224', 
                       choices=['vit_small_patch8_224', 'vit_small_patch16_224',
                                'vit_base_patch8_224', 'vit_base_patch16_224',
                                'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='Model backbone')
    parser.add_argument('--tile_size', type=int, default=448, 
                        choices=[224, 448],
                        help='Tile size for training')
    parser.add_argument('--device', type=int, default=0, 
                        choices=list(range(8)),
                        help='CUDA device index (0, 1, ..., 7)')
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # set global variables based on args
    tile_size = args.tile_size
    backbone = args.backbone
    model_type = ['vit', 'unet']['resnet' in backbone] 
    device = torch.device(f'cuda:{args.device}')

    # get image and mask paths for each split
    train_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'test.csv'))
    train_image_paths, train_mask_paths, _ = get_paths(train_df)
    val_image_paths, val_mask_paths, _ = get_paths(val_df)
    test_image_paths, test_mask_paths, _ = get_paths(test_df)

    # load image/mask memmaps
    print("Loading images and masks...")
    train_images, train_masks = load_images_and_masks(train_image_paths, train_mask_paths)
    val_images, val_masks = load_images_and_masks(val_image_paths, val_mask_paths)
    test_images, test_masks = load_images_and_masks(test_image_paths, test_mask_paths)

    # print dataset sizes
    print(f"Train dataset size: {len(train_images)} images")
    print(f"Val dataset size: {len(val_images)} images")
    print(f"Test dataset size: {len(test_images)} images")

    # instantiate model
    model = models.build_model(
        model_type=model_type,
        backbone=backbone,
        tile_size=tile_size,
        device=device,
    )
    
    # load checkpoint
    checkpoint_path = config.CHECKPOINT_PATH.format(backbone, tile_size)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = torch.compile(model)

    # add columns to metrics file
    train_df['Dice'] = np.nan
    val_df['Dice'] = np.nan
    test_df['Dice'] = np.nan
    columns = list(train_df.columns)

    # create metrics directory if it doesn't exist
    os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)

    # evaluate on train, val, test sets
    train_split = ('train', train_images, train_masks, train_df)
    val_split = ('val', val_images, val_masks, val_df)
    test_split = ('test', test_images, test_masks, test_df)
    for split, images, masks, df in [train_split, val_split, test_split]:

        # initialize
        print(f"Evaluating on {split} set...")
        dices = []
        metrics_path = config.METRICS_PATH.format(backbone, tile_size, split)
        with open(metrics_path, 'w') as f:
            f.write(','.join(columns) + '\n')

        # evaluate each image
        for i in tqdm(range(len(images))):
            image = np.array(images[i])
            mask = np.array(masks[i])
            preds, _ = segment(
                image, model, 
                tile_size=tile_size, step=tile_size//2, 
                batch_size=config.EVAL_BATCH_SIZE, 
                threshold=0.5, n_classes=2)
            dice = compute_image_dice(preds, mask)
            dices.append(dice)
            df.at[i, 'Dice'] = dice
            with open(metrics_path, 'a') as f:
                f.write(','.join([str(df.iloc[i][c]) for c in columns]) + '\n')

        mean_dice = np.mean(dices)
        print(f"Mean Dice on {split} set: {mean_dice:.4f}")

if __name__ == '__main__':  
    main()
