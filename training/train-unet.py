import os, sys
import pandas as pd
import torch
import random
import segmentation_models_pytorch as smp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from multiprocessing import Pool
from torch.optim import AdamW
from modules import (
    data_split,
    verify_disjoint,
    load_memmap_paths,
    mp_dilate_masks,
    Augmentations,
    BuildDataloader,
    CombinedLoss,
)
from training import Trainer
                            
# ===============
# === OPTIONS ===
# ===============

PROJECT_ = '/mnt/DGX01/Personal/milliganj/codebase/projects/appl-rgb-segmentation'
METADATA_ = os.path.join(os.getcwd(), 'data', 'splits', '')
DATA_ = os.path.join(os.getcwd(), 'data')
IMAGES_ = os.path.join(DATA_, 'images')
MASKS_ = os.path.join(DATA_, 'masks')
IMAGE_EXT = 'png'

TILE_SIZE = 224
TRAIN_TILES_PER_IMAGE = 10
VAL_TILES_PER_IMAGE = 100
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BATCH_SIZE = 112
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE

EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 32

MODEL = 'resnet34'
WEIGHTS = None
NUM_CLASSES = 2
AUTOCAST_DTYPE = torch.float16

DEVICE = torch.device("cuda:0")
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    
# ======================    
# === DATA SPLITTING ===
# ======================    

# Load metadata
md_path = os.path.join(METADATA_, 'metadata.csv')
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

# Load names for each split
names_train = train['File Name'].tolist() 
names_val = val['File Name'].tolist()
names_test = test['File Name'].tolist()

# Get memory map paths for images and masks
train_images, train_masks = load_memmap_paths((IMAGES_, MASKS_), names_train)
print(f"Found {len(train_images)} training images and {len(train_masks)} training masks.")

val_images, val_masks = load_memmap_paths((IMAGES_, MASKS_), names_val)
print(f"Found {len(val_images)} validation images and {len(val_masks)} validation masks.\n")

# test_images, test_masks = load_dataset((images_base, masks_base), names_test, "Test")
# dmasks_test = mp_dilate_masks(test_masks, TILE_SIZE, pool_size=32)

# Build datasets and dataloaders
train_dataset, train_dataloader = BuildDataloader.build_dataloader(
    images=train_images,
    masks=train_masks,
    batch_size=BATCH_SIZE,
    tile_size=TILE_SIZE, 
    transform=Augmentations.get_train_transform(IMAGENET_MEAN, IMAGENET_STD),
    tiles_per_image=TRAIN_TILES_PER_IMAGE,
    mix_ratio=0.5,

    shuffle=True,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)

val_dataset, val_dataloader = BuildDataloader.build_dataloader(
    images=val_images,
    masks=val_masks,
    batch_size=BATCH_SIZE,
    tile_size=TILE_SIZE,
    transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
    tiles_per_image=VAL_TILES_PER_IMAGE,
    mix_ratio=0.5,
    
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)

# test_dataset, test_dataloader = BuildDataloader.build_dataloader(
#     images=test_images,
#     masks=test_masks,
#     dilated_masks=dmasks_test,
#     batch_size=BATCH_SIZE,
#     tile_size=TILE_SIZE,
#     transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
#     tiles_per_image=TEST_TILES_PER_IMAGE,
#     mix_ratio=0.25,
    
#     num_workers=32,
#     pin_memory=True,
#     persistent_workers=True,
# )


# === Model Training ===    

# Initialize the model
model = smp.Unet(
    encoder_name="resnet34",       
    encoder_weights=None,    
    in_channels=3,                   
    classes=2,                       
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
scaler = torch.amp.GradScaler('cuda')

trainer = Trainer(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader,
    optimizer=optimizer, 
    criterion=criterion,
    scaler=scaler,
    device=DEVICE,
    autocast_dtype=AUTOCAST_DTYPE,
    patience=None,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)

trainer.fit(
    max_epochs=EPOCHS,
    verbose=False
)