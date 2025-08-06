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
SPLITS_ = os.path.join(os.getcwd(), 'data', 'metadata')
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
MICRO_BATCH_SIZE = 7
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

# Load metadata splits
md_train =  os.path.join(SPLITS_, 'train.csv')
md_val =    os.path.join(SPLITS_, 'val.csv')
md_test =   os.path.join(SPLITS_, 'test.csv')
assert os.path.exists(md_train), f"Metadata file not found at {md_train}"
assert os.path.exists(md_val), f"Metadata file not found at {md_val}"
assert os.path.exists(md_test), f"Metadata file not found at {md_test}"

# Dataframes for each split
md_train = pd.read_csv(md_train)
md_val = pd.read_csv(md_val)
md_test = pd.read_csv(md_test)

# Load names for each split
names_train = md_train['File Name'].tolist() 
names_val = md_val['File Name'].tolist()
names_test = md_test['File Name'].tolist()

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



