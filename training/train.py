import os, sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from multiprocessing import Pool
from trainer import SegmentationModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from modules import (
    data_split,
    verify_disjoint,
    load_dataset,
    mp_dilate_masks,
    Augmentations,
    BuildDataloader
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

BATCH_SIZE = 512
ACCUMULATION_STEPS = 32
TILE_SIZES = {
    "rn_224": 224,
    "rn_448": 448,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TILES_PER_IMAGE = 10
OTHER_TILES_PER_IMAGE = 100

# Load names for each split
names_train = train['File Name'].tolist() 
names_val = val['File Name'].tolist()
names_test = test['File Name'].tolist()

# Get datasets and dataloaders
train_images, train_masks = load_dataset((images_base, masks_base), names_train[:50], "Train")
dmasks_train = mp_dilate_masks(train_masks, TILE_SIZES["rn_224"], pool_size=32)

val_images, val_masks = load_dataset((images_base, masks_base), names_val[:25], "Validation")
dmasks_val = mp_dilate_masks(val_masks, TILE_SIZES["rn_224"], pool_size=32)

test_images, test_masks = load_dataset((images_base, masks_base), names_test[:25], "Test")
dmasks_test = mp_dilate_masks(test_masks, TILE_SIZES["rn_224"], pool_size=32)

train_dataset, train_dataloader = BuildDataloader.build_dataloader(
    images=train_images,
    masks=train_masks,
    dilated_masks=dmasks_train,
    batch_size=BATCH_SIZE,
    tile_size=TILE_SIZES["rn_224"], 
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
    dilated_masks=dmasks_val,
    batch_size=BATCH_SIZE,
    tile_size=TILE_SIZES["rn_224"],
    transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
    tiles_per_image=OTHER_TILES_PER_IMAGE,
    mix_ratio=0.25,
    
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)

test_dataset, test_dataloader = BuildDataloader.build_dataloader(
    images=val_images,
    masks=val_masks,
    dilated_masks=dmasks_val,
    batch_size=BATCH_SIZE,
    tile_size=TILE_SIZES["rn_224"],
    transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
    tiles_per_image=OTHER_TILES_PER_IMAGE,
    mix_ratio=0.25,
    
    num_workers=32,
    pin_memory=True,
    persistent_workers=True,
)


# === Model Training (w/ Pytorch Lightning) ===

# Seeding and paths
seed_everything(42, workers=True, verbose=True)
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

# Create log directory. Lightning will create tensorboard logs and save model checkpoints.
log_dir = os.path.join(root, 'logs', timestamp)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
# Create logger
logger = TensorBoardLogger(save_dir=log_dir, name='lightning_logs')

# Initialize the model
model = SegmentationModel(
    arch="Unet",
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    out_classes=1,
    max_epochs=100
)

# Set up the trainer
trainer = pl.Trainer(
    max_epochs=100, 
    log_every_n_steps=1,
    precision="bf16-mixed",
    accumulate_grad_batches=ACCUMULATION_STEPS,
    logger=logger,
    callbacks=[
        EarlyStopping(
            monitor="valid_dataset_f1",
            min_delta=0.005, 
            patience=50, 
            mode="max", 
            verbose=True,
            )
        ],
)

# Begin training
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)