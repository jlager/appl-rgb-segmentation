import numpy as np
import random
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


# Augmentations
class Augmentations:

    def get_train_transform(IMAGENET_MEAN, IMAGENET_STD):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5, border_mode=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def get_valid_transform(IMAGENET_MEAN, IMAGENET_STD):
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

class TileDataset_DilationSampling(Dataset):

    def __init__(self, images, masks, dilated_masks, tile_size, transform, 
                 tiles_per_image=100, mix_ratio=0.7):
        
        self.images = images
        self.masks = masks
        self.dilated_masks = dilated_masks
        self.tile_size = tile_size
        self.transform = transform
        self.tiles_per_image = tiles_per_image
        self.bound = tile_size // 2
        self.mix_ratio = mix_ratio

    def __len__(self):
        return self.tiles_per_image * len(self.images)

    def __getitem__(self, idx):
        # Cycle through images evenly
        img_idx = idx % len(self.images)
        image = self.images[img_idx]
        mask = self.masks[img_idx]
        dilated_mask = self.dilated_masks[img_idx]

        if random.random() < self.mix_ratio:
            # Foreground sampling
            row, col = self._get_fg_px(dilated_mask, self.tile_size)
        else:
            # Background sampling
            row, col = self._get_bg_px(dilated_mask, self.tile_size)
        
        # Compute crop coordinates. If the tile would crop beyond the
        left = col - self.bound
        upper = row - self.bound
        right = left + self.tile_size
        lower = upper + self.tile_size

        # Crop and transform
        tile_image = np.asarray(image.crop((left, upper, right, lower)), copy=True)
        tile_mask  = np.asarray(mask.crop((left, upper, right, lower)), copy=True)

        # Augment tile and normalize to binary
        augmented = self.transform(image=tile_image, mask=tile_mask)
        image_tensor = augmented['image']
        mask_tensor  = (augmented['mask'] > 0).long() # [0 - 255] -> [0 - 1]

        # Change back to only return tensors after testing
        return image_tensor, mask_tensor
    
    def _get_fg_px(self, mask: np.ndarray, tile_size: int):
        
        # check for empty mask. if so, sample a bg_px instead
        if np.count_nonzero(mask[tile_size//2:-tile_size//2, tile_size//2:-tile_size//2]) == 0:
            return self._get_bg_px(mask, tile_size)
            

        foreground = np.flatnonzero(mask[tile_size//2:-tile_size//2, tile_size//2:-tile_size//2])
        sampled_flat = np.random.choice(foreground, size=1)
        f_coords = np.unravel_index(sampled_flat, [mask.shape[0] - tile_size, mask.shape[1] - tile_size])
        f_coords = np.stack(f_coords, axis=1) + tile_size//2
        return f_coords[0][0], f_coords[0][1]
    
    def _get_bg_px(self, mask: np.ndarray, tile_size: int):

        background = np.flatnonzero(~mask[tile_size//2:-tile_size//2, tile_size//2:-tile_size//2])
        sampled_flat = np.random.choice(background, size=1)
        b_coords = np.unravel_index(sampled_flat, [mask.shape[0] - tile_size, mask.shape[1] - tile_size])
        b_coords = np.stack(b_coords, axis=1) + tile_size//2
        return b_coords[0][0], b_coords[0][1]


class BuildDataloader:
    
    @staticmethod
    def build_dataloader(images, masks, dilated_masks, batch_size, tile_size, tiles_per_image, transform, mix_ratio, **kwargs):
        
        dataset = TileDataset_DilationSampling(
            images=images,
            masks=masks,
            dilated_masks=dilated_masks,
            tile_size=tile_size,
            transform=transform,
            tiles_per_image=tiles_per_image,
            mix_ratio=mix_ratio,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            **kwargs,
        )

        return dataset, dataloader
