# %%
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
    Augmentations,
    BuildDataloader,
    CombinedLoss,
)
from training import Trainer

from models import (
    ViTSegmentation
)

# %%
# ===============
# === OPTIONS ===
# ===============

PROJECT_ = '/mnt/DGX01/Personal/milliganj/codebase/projects/appl-rgb-segmentation'
METADATA_ = os.path.join(os.getcwd(), 'data', 'splits', '')
DATA_ = os.path.join(os.getcwd(), 'data')
IMAGES_ = os.path.join(DATA_, 'images')
MASKS_ = os.path.join(DATA_, 'masks')

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

VIT_MODEL = 'vit_small_patch8_224'
VIT_PRETRAINED = True
NUM_CLASSES = 2
BACKBONE_LR_FACTOR = 0.1

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


# %%
# instantiate ViT model with pre-trained weights
model = ViTSegmentation(
    vit_model=VIT_MODEL, 
    img_size=TILE_SIZE,
    pretrained=VIT_PRETRAINED,
    num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# configure parameter groups with different learning rates
encoder_params = model.encoder.parameters()
decoder_params = model.decoder.parameters()
param_groups = [
    {'params': encoder_params, 'lr': LR * BACKBONE_LR_FACTOR},
    {'params': decoder_params, 'lr': LR}
]

# instantiate optimizer
optimizer = AdamW(param_groups, weight_decay=WEIGHT_DECAY)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, 
#     T_max=EPOCHS, 
#     eta_min=LR / 10)
criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5)
scaler = torch.amp.GradScaler()

trainer = Trainer(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader,
    optimizer=optimizer, 
    criterion=criterion,
    scaler=scaler,
    device=DEVICE,
    autocast_dtype='float16',  # Using bfloat16 for ViT
    patience=None,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
)

trainer.fit(
    max_epochs=EPOCHS,
    verbose=True
)

# %%



