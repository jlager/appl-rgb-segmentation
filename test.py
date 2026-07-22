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
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from typing import List, Tuple
from contextlib import nullcontext
import torch.nn.functional as F

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

def _normalize_image(
    image: np.ndarray,
    mean: List[float] = config.IMAGENET_MEAN,
    std: List[float] = config.IMAGENET_STD,
) -> np.ndarray:
    im = image.astype(np.float32, copy=False) / 255.0
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)
    return (im - mean_arr) / std_arr

def _generate_tile_coords(
    height: int,
    width: int,
    window_size: int,
) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    stride = window_size // 2

    for r in range(0, height - window_size + 1, stride):
        for c in range(0, width - window_size + 1, stride):
            coords.append((r, c))

    for c in range(0, width - window_size + 1, stride):
        coords.append((height - window_size, c))
    for r in range(0, height - window_size + 1, stride):
        coords.append((r, width - window_size))
    coords.append((height - window_size, width - window_size))

    seen = set()
    uniq_coords = []
    for rc in coords:
        if rc not in seen:
            seen.add(rc)
            uniq_coords.append(rc)
    return uniq_coords

def _generate_classical_tile_coords(
    height: int,
    width: int,
    window_size: int,
) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for r in range(0, height, window_size):
        for c in range(0, width, window_size):
            coords.append((r, c))
    return coords

def _generate_hann_weights(window_size: int) -> np.ndarray:
    w1d = torch.hann_window(window_size, periodic=False, dtype=torch.float32)
    weight2d = torch.outer(w1d, w1d).clamp_min(1e-6)
    return weight2d.cpu().numpy()

def segment(
    image: np.ndarray,
    model: torch.nn.Module,
    tile_size: int,
    batch_size: int = 64,
    n_classes: int = 2,
    inference_mode: str = 'hann',
) -> Tuple[np.ndarray, np.ndarray]:
    
    H, W, _ = image.shape
    C = n_classes
    normalized = _normalize_image(image, config.IMAGENET_MEAN, config.IMAGENET_STD)

    if inference_mode == 'hann':
        ws = min(tile_size, H, W)
        tile_image = normalized
        coords = _generate_tile_coords(H, W, ws)
        weight2d_np = _generate_hann_weights(ws)
    elif inference_mode == 'classical':
        ws = tile_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        tile_image = np.pad(
            normalized,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant')
        coords = _generate_classical_tile_coords(tile_image.shape[0], tile_image.shape[1], ws)
        weight2d_np = np.ones((ws, ws), dtype=np.float32)
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode}")

    tile_H, tile_W, _ = tile_image.shape
    prob_acc = np.zeros((C, tile_H, tile_W), dtype=np.float32)
    weight_acc = np.zeros((tile_H, tile_W), dtype=np.float32)

    device = next(model.parameters()).device
    model.eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    coord_idx = 0
    tiles_total = len(coords)
    with torch.inference_mode():
        while coord_idx < tiles_total:
            batch_coords = coords[coord_idx : coord_idx + batch_size]
            b = len(batch_coords)

            batch_np = np.empty((b, 3, ws, ws), dtype=np.float32)
            for i, (r, c) in enumerate(batch_coords):
                tile = tile_image[r : r + ws, c : c + ws, :]
                batch_np[i] = tile.transpose(2, 0, 1)

            batch = torch.from_numpy(batch_np)
            if device.type == "cuda":
                batch = batch.pin_memory().to(device, non_blocking=True)
            else:
                batch = batch.to(device)

            amp_ctx = (
                torch.amp.autocast('cuda', dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_ctx:
                logits = model(batch)
            probs = F.softmax(logits, dim=1).float().cpu().numpy()

            for i, (r, c) in enumerate(batch_coords):
                prob_acc[:, r : r + ws, c : c + ws] += probs[i] * weight2d_np[None, :, :]
                weight_acc[r : r + ws, c : c + ws] += weight2d_np

            coord_idx += b

    weight_acc = np.clip(weight_acc, 1e-6, None)
    final_probs = prob_acc / weight_acc[None, :, :]
    final_probs = final_probs[:, :H, :W]
    final_mask = final_probs.argmax(axis=0).astype(np.uint8)
    return final_mask, final_probs

def compute_image_dice(preds: np.ndarray, targets: np.ndarray) -> float:
    # preds: [H, W]; targets: [H, W]
    preds = preds.astype(float)
    targets = targets.astype(float)
    intersection = (preds * targets).sum(axis=(0, 1))
    union = preds.sum(axis=(0, 1)) + targets.sum(axis=(0, 1))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

def get_metrics_path(
    run_name: str,
    tile_size: int,
    split: str,
    inference_mode: str,
) -> str:
    metrics_path = config.METRICS_PATH.format(run_name, tile_size, split)
    if inference_mode == 'hann':
        return metrics_path
    root, ext = os.path.splitext(metrics_path)
    return f"{root}_inference-{inference_mode}{ext}"

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
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction,
                        default=True, help='Use pretrained backbone weights')
    parser.add_argument('--tile_size', type=int, default=448, 
                        choices=[224, 448],
                        help='Tile size for training')
    parser.add_argument('--device', type=int, default=0, 
                        choices=list(range(8)),
                        help='CUDA device index (0, 1, ..., 7)')
    parser.add_argument('--inference_mode', type=str, default='hann',
                        choices=['hann', 'classical'],
                        help='Tile inference mode')
    return parser.parse_args()
    
def main():
    args = parse_args()
    
    # set global variables based on args
    tile_size = args.tile_size
    backbone = args.backbone
    run_name = f"{backbone}_{'pretrained' if args.pretrained else 'scratch'}"
    inference_mode = args.inference_mode
    model_type = ['vit', 'unet']['resnet' in backbone] 
    device = torch.device(f'cuda:{args.device}')

    # get image and mask paths for each split
    train_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'test.csv'))
    gen_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'generalization.csv'))
    train_image_paths, train_mask_paths, _ = get_paths(train_df)
    val_image_paths, val_mask_paths, _ = get_paths(val_df)
    test_image_paths, test_mask_paths, _ = get_paths(test_df)
    gen_image_paths, gen_mask_paths, _ = get_paths(gen_df)

    # load image/mask memmaps
    print("Loading images and masks...")
    train_images, train_masks = load_images_and_masks(train_image_paths, train_mask_paths)
    val_images, val_masks = load_images_and_masks(val_image_paths, val_mask_paths)
    test_images, test_masks = load_images_and_masks(test_image_paths, test_mask_paths)
    gen_images, gen_masks = load_images_and_masks(gen_image_paths, gen_mask_paths)

    # print dataset sizes
    print(f"Train dataset size: {len(train_images)} images")
    print(f"Val dataset size: {len(val_images)} images")
    print(f"Test dataset size: {len(test_images)} images")
    print(f"Generalization dataset size: {len(gen_images)} images")

    # instantiate model
    model = models.build_model(
        model_type=model_type,
        backbone=backbone,
        tile_size=tile_size,
        device=device,
        pretrained=args.pretrained,
    )
    
    # load checkpoint
    checkpoint_path = config.CHECKPOINT_PATH.format(run_name, tile_size)
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = torch.compile(model)

    # add columns to metrics file
    train_df['Dice'] = np.nan
    val_df['Dice'] = np.nan
    test_df['Dice'] = np.nan
    gen_df['Dice'] = np.nan
    train_df['Time (s)'] = np.nan
    val_df['Time (s)'] = np.nan
    test_df['Time (s)'] = np.nan
    gen_df['Time (s)'] = np.nan
    tvt_columns = list(train_df.columns)
    gen_columns = list(gen_df.columns)

    # create metrics directory if it doesn't exist
    os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)

    # evaluate on train, val, test sets
    train_split = ('train', train_images, train_masks, train_df)
    val_split = ('val', val_images, val_masks, val_df)
    test_split = ('test', test_images, test_masks, test_df)
    gen_split = ('generalization', gen_images, gen_masks, gen_df)
    for split, images, masks, df in [train_split, val_split, test_split, gen_split]:


        # initialize
        print(f"Evaluating on {split} set...")
        dices = []
        metrics_path = get_metrics_path(run_name, tile_size, split, inference_mode)
        columns = gen_columns if split == 'generalization' else tvt_columns
        with open(metrics_path, 'w') as f:
            f.write(','.join(columns) + '\n')

        # evaluate each image
        for i in tqdm(range(len(images))):
            image = np.array(images[i])
            mask = np.array(masks[i])

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            preds, _ = segment(
                image, model, 
                tile_size=tile_size,
                batch_size=config.EVAL_BATCH_SIZE, 
                n_classes=2,
                inference_mode=inference_mode)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_time = time.perf_counter() - start_time

            dice = compute_image_dice(preds, mask)
            dices.append(dice)
            df.at[i, 'Dice'] = dice
            df.at[i, 'Time (s)'] = inference_time
            with open(metrics_path, 'a') as f:
                f.write(','.join([str(df.iloc[i][c]) for c in columns]) + '\n')

        mean_dice = np.mean(dices)
        print(f"Mean Dice on {split} set: {mean_dice:.4f}")

if __name__ == '__main__':  
    main()
