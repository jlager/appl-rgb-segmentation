import os
os.environ["OMP_NUM_THREADS"] = "1"                # stop using all CPU threads
os.environ["MKL_NUM_THREADS"] = "1"                # stop using all CPU threads
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import cv2
cv2.setNumThreads(1)                               # stop using all CPU threads

import torch
torch.set_num_threads(1)                           # stop using all CPU threads
torch.set_num_interop_threads(1)                   # stop using all CPU threads

import math
import random
import numpy as np
import pandas as pd
import argparse
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

def load_roi(roi_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads tile center coordinates (ROIs) from .npz file."""
    roi = np.load(roi_path)
    foreground_coords = roi['f_idx']
    background_coords = roi['b_idx']
    return foreground_coords, background_coords

def build_roi(height, width, tile_size) -> np.ndarray:
    """Builds a regular grid tile centers (ROIs) for validation/testing."""
    row_start = (tile_size // 2) + (height % tile_size) // 2 # center the grid
    col_start = (tile_size // 2) + (width % tile_size) // 2 # center the grid
    rows = list(range(row_start, height - tile_size//2 + 1, tile_size))
    cols = list(range(col_start, width - tile_size//2 + 1, tile_size))
    coords = np.array([(r, c) for r in rows for c in cols])
    coords = np.ravel_multi_index((coords[:, 0], coords[:, 1]), (height, width))
    return coords, np.array([], dtype=np.int64)

# train augmentation
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(scale=(0.8, 1.2), shear=(-10, 10), border_mode=1, 
             mask_interpolation=cv2.INTER_NEAREST, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ToTensorV2(),
])

# val / test augmentation
valid_transform = A.Compose([
    A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ToTensorV2(),
])

class TileGenerator(torch.utils.data.Dataset):

    def __init__(
        self, 
        images: List[np.memmap],
        masks: List[np.ndarray],
        rois: List[Tuple[np.ndarray, np.ndarray]],
        tile_size: int,
        tiles_per_image: int = 10,
        deterministic: bool = False,
    ) -> None:
        """
        Args:
            images: list of images as memmaps.
            masks: list of masks as boolean numpy arrays.
            rois: list of ROIs as tuples of (foreground_coords, background_coords).
            transform: image augmentation.
            tiles_per_image: # tiles per image to virtually expand the dataset.
            tile_size: size of the tiles to extract.
            deterministic: if False, randomly sample tiles.
        """
        self.images = images
        self.masks = masks
        self.rois = rois
        self.tiles_per_image = tiles_per_image
        self.tile_size = tile_size
        self.deterministic = deterministic

        if self.deterministic:
            self.transform = valid_transform
            self.roi_cumsum = np.cumsum([len(f) + len(b) for f, b in self.rois])
        else:
            self.transform = train_transform
            self.roi_cumsum = np.cumsum([self.tiles_per_image] * len(self.images))
        self.roi_cumsum = np.insert(self.roi_cumsum, 0, 0)

    def __len__(self) -> int:
        """Returns the total number of tiles in the dataset."""
        if self.deterministic:
            return sum(len(f) + len(b) for f, b in self.rois)
        else:
            return self.tiles_per_image * len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a single tile and its corresponding mask as tensors."""
        # unpack values
        img_idx = np.searchsorted(self.roi_cumsum, idx, side='right') - 1
        image = self.images[img_idx]
        mask = self.masks[img_idx]
        foreground_idx, background_idx = self.rois[img_idx]
        height, width, channels = image.shape

        # choose fg or bg (note, validation has no distinction, "bg" is empty)
        coin_flip = np.random.rand() > 0.5
        if foreground_idx.size == 0:
            coin_flip = False
        if background_idx.size == 0 or self.deterministic:
            coin_flip = True
        coords = foreground_idx if coin_flip else background_idx

        # sample coordinate
        if self.deterministic:
            coord_idx = idx - self.roi_cumsum[img_idx]
        else:
            coord_idx = np.random.randint(len(coords))
        row, col = np.unravel_index(coords[coord_idx], [height, width])

        # compute crop coordinates
        left = col.item() - self.tile_size // 2
        upper = row.item() - self.tile_size // 2
        right = left + self.tile_size
        lower = upper + self.tile_size

        # crop and apply transformation
        tile_image = np.array(image[upper:lower, left:right, :], dtype=np.uint8)
        tile_mask  = 255 * np.array(mask[upper:lower, left:right], dtype=np.uint8)
        augmented = self.transform(image=tile_image, mask=tile_mask)
        image_tensor = augmented['image']
        mask_tensor = (augmented['mask'] > 0).long() # [H, W] for binary segmentation

        return image_tensor, mask_tensor

# =============================================================================
# Metrics
# =============================================================================

class CELoss(torch.nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits: [B, C, H, W]; targets: [B, H, W]
        return self.ce(logits, targets)

class DiceLoss(torch.nn.Module):
    
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, H, W]; targets: [B, H, W]
        num_classes = logits.shape[1]
        probas = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        intersection = torch.sum(probas * targets_one_hot, dim=(2, 3))
        union = torch.sum(probas + targets_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(torch.nn.Module):
    
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.ce = CELoss()
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss = self.weight_ce * loss_ce + self.weight_dice * loss_dice
        return loss_ce, loss_dice, loss

def compute_batch_dice(logits, targets):
    # logits: [B, C, H, W]; targets: [B, H, W]
    preds = logits.argmax(dim=1).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(1,2))
    union = preds.sum(dim=(1,2)) + targets.sum(dim=(1,2))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

def compute_image_dice(logits, targets):
    # logits: [C, H, W]; targets: [H, W]
    preds = logits.argmax(dim=0).float()
    targets = targets.float()
    intersection = (preds * targets).sum(dim=(0,1))
    union = preds.sum(dim=(0,1)) + targets.sum(dim=(0,1))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return dice.mean().item()

# =============================================================================
# Training
# =============================================================================

# cosine lr decay scheduler with warmup -- Thanks Andrej Karpathy
def get_lr(epoch, max_lr, min_lr):
    
    # linear warmup
    if epoch < config.LR_WARMUP_EPOCHS:
        return max_lr * epoch / config.LR_WARMUP_EPOCHS

    # min learning rate
    if epoch > config.LR_DECAY_EPOCHS + config.LR_WARMUP_EPOCHS:
        return min_lr
    
    # intermediate cosine decay
    decay_ratio = (epoch - config.LR_WARMUP_EPOCHS) / (config.LR_DECAY_EPOCHS)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coeff*(max_lr - min_lr)

# =============================================================================
# Main
# =============================================================================

def parse_args():
    # example: python train.py --backbone "vit_base_patch8_224" --tile_size 448 --device 0
    parser = argparse.ArgumentParser(description='Train segmentation model')
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

    # create output directories
    log_path = config.LOG_PATH.format(backbone, tile_size)
    checkpoint_path = config.CHECKPOINT_PATH.format(backbone, tile_size)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # get image and mask paths for each split
    train_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(config.SPLIT_DIR, 'val.csv'))
    train_image_paths, train_mask_paths, train_roi_paths = get_paths(train_df)
    val_image_paths, val_mask_paths, _ = get_paths(val_df)

    # load image/mask memmaps
    print("Loading images and masks...")
    train_images, train_masks = load_images_and_masks(train_image_paths, train_mask_paths)
    val_images, val_masks = load_images_and_masks(val_image_paths, val_mask_paths)

    # load or build ROIs
    print("Loading regions of interest...")
    train_rois = [load_roi(path.format(tile_size)) for path in train_roi_paths]
    val_rois = [build_roi(image.shape[0], image.shape[1], tile_size) for image in val_images]

    # create training dataset
    print("Creating datasets...")
    train_dataset = TileGenerator(
        images=train_images,
        masks=train_masks,
        rois=train_rois,
        tile_size=tile_size,
        tiles_per_image=config.TRAIN_TILES_PER_IMAGE,
        deterministic=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.MICRO_BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.N_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=1)

    # print dataset sizes
    print(f"Train dataset size: {len(train_dataset)} tiles")
    print(f"Val dataset size: {len(val_images)} images")

    # instantiate model
    model = models.build_model(
        model_type=model_type,
        backbone=backbone,
        tile_size=tile_size,
        device=device,
    )
    if model_type == 'vit':
        encoder_params = model.encoder.parameters()
        decoder_params = model.decoder.parameters()
        param_groups = [
            {'params': encoder_params, 'lr': config.MAX_LR * config.BACKBONE_LR_FACTOR},
            {'params': decoder_params, 'lr': config.MAX_LR}
        ]
    if model_type == 'unet':
        params = model.model.parameters()
        param_groups = [{'params': params, 'lr': config.MAX_LR}]
    
    # instantiate optimizer
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
    scaler = torch.amp.GradScaler()

    # start a new log file
    columns = [
        'epoch', 'encoder_lr', 'decoder_lr', 
        'train_ce_loss', 'train_dice_loss', 'train_loss', 'train_dice',
        'val_ce_loss', 'val_dice_loss', 'val_loss', 'val_dice']
    with open(log_path, 'w') as log_file:
        log_file.write(','.join(columns) + '\n')

    # initialize best validation dice
    best_val_dice = 0.0

    # training loop
    for epoch in range(1, config.EPOCHS + 1):

        #
        # train
        #
        
        model.train()
        optimizer.zero_grad()
        train_ce_losses, train_dice_losses, train_losses, train_dices = [], [], [], []
        batch_ce_loss, batch_dice_loss, batch_loss, batch_dices = 0.0, 0.0, 0.0, []

        # update learning rate
        if model_type == 'vit':
            encoder_lr = get_lr(epoch, config.MAX_LR * config.BACKBONE_LR_FACTOR, config.MIN_LR)
            decoder_lr = get_lr(epoch, config.MAX_LR, config.MIN_LR)
            for pg, lr in zip(optimizer.param_groups, [encoder_lr, decoder_lr]):
                pg['lr'] = lr
        else:  # unet
            lr = get_lr(epoch, config.MAX_LR, config.MIN_LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        
        # loop over micro-batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EPOCHS} [Train]")
        for i, (images_batch, masks_batch) in enumerate(pbar):
            images_batch = images_batch.to(device)
            masks_batch  = masks_batch.to(device)

            # forward pass
            with torch.amp.autocast('cuda'):
                logits = model(images_batch)
                ce_loss, dice_loss, loss = criterion(logits, masks_batch)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS # scale loss
            
            # backward pass
            scaler.scale(loss).backward()
            
            # accumulate metrics
            dice = compute_batch_dice(logits, masks_batch)
            batch_dices.append(dice)
            batch_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS  # scale back
            batch_ce_loss += ce_loss.item()
            batch_dice_loss += dice_loss.item()
            
            # optimizer step
            step_iter = (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0
            last_iter = (i + 1) == len(train_loader)
            if step_iter or last_iter:

                # step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # log metrics
                train_losses.append(batch_loss)
                train_ce_losses.append(batch_ce_loss)
                train_dice_losses.append(batch_dice_loss)
                avg_dice = np.mean(batch_dices)
                train_dices.append(avg_dice)
                
                # reset accumulators
                batch_loss = 0.0
                batch_ce_loss = 0.0
                batch_dice_loss = 0.0
                batch_dices = []
                
                pbar.set_postfix(loss=f"{train_losses[-1]:.4e}", dice=f"{avg_dice:.4f}")

        avg_train_loss = np.mean(train_losses)
        avg_train_ce_loss = np.mean(train_ce_losses)
        avg_train_dice_loss = np.mean(train_dice_losses)
        avg_train_dice = np.mean(train_dices)

        #
        # validation
        #

        model.eval()
        val_losses, val_ce_losses, val_dice_losses, val_dices = [], [], [], []

        # loop over validation images
        pbar_val = tqdm(range(len(val_images)), desc=f"Epoch {epoch}/{config.EPOCHS} [Val]")
        with torch.no_grad():
            for i in pbar_val:

                # make validation dataset for the i-th image
                val_dataset = TileGenerator(
                    images=val_images[i:i+1],
                    masks=val_masks[i:i+1],
                    rois=val_rois[i:i+1],
                    tile_size=tile_size,
                    deterministic=True)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, 
                    batch_size=config.MICRO_BATCH_SIZE, 
                    shuffle=False)

                # loop over micro-batches
                mask_batches, logits_batches = [], []
                for images_batch, masks_batch in val_loader:

                    images_batch = images_batch.to(device)
                    masks_batch  = masks_batch.to(device)

                    with torch.amp.autocast('cuda'):
                        logits = model(images_batch)
                        ce_loss, dice_loss, loss = criterion(logits, masks_batch)
                    
                    val_losses.append(loss.item())
                    val_ce_losses.append(ce_loss.item())
                    val_dice_losses.append(dice_loss.item())
                    
                    mask_batches.append(masks_batch)
                    logits_batches.append(logits)
                
                # build pseudo-images for validation dice (class imbalance fix)
                masks = torch.cat(mask_batches, dim=0)
                logits = torch.cat(logits_batches, dim=0)
                B, C, H, W = logits.shape
                masks = masks.contiguous().view(B*H, W) # only works for binary segmentation
                logits = logits.permute(1, 0, 2, 3).contiguous().view(C, B*H, W)

                # compute image-level dice
                val_dice = compute_image_dice(logits, masks)
                val_dices.append(val_dice)

                pbar_val.set_postfix(loss=f"{loss.item():.4e}", dice=f"{val_dices[-1]:.4f}")

        avg_val_loss = np.mean(val_losses)
        avg_val_ce_loss = np.mean(val_ce_losses)
        avg_val_dice_loss = np.mean(val_dice_losses)
        avg_val_dice = np.mean(val_dices)
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]

        # log metrics
        with open(log_path, 'a') as log_file:
            line = '{},{},{},{},{},{},{},{},{},{},{}'.format(
                epoch,
                current_lrs[0],
                current_lrs[1] if len(current_lrs) > 1 else np.nan,
                avg_train_ce_loss,
                avg_train_dice_loss,
                avg_train_loss,
                avg_train_dice,
                avg_val_ce_loss,
                avg_val_dice_loss,
                avg_val_loss,
                avg_val_dice,
            )
            log_file.write(line + '\n')

        print(f"Epoch {epoch} -- Train Loss: {avg_train_loss:.4e}, Train Dice: {avg_train_dice:.4f} | "
              f"Val Loss: {avg_val_loss:.4e}, Val Dice: {avg_val_dice:.4f}")

        # save the best model checkpoint
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice,
            }, checkpoint_path)
            print(f"*** New best model saved with Val Dice: {best_val_dice:.4f} ***")

if __name__ == '__main__':  
    main()
