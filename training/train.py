import os, sys
import pandas as pd
import torch
import trainer
import segmentation_models_pytorch as smp

from multiprocessing import Pool
from torch.optim import AdamW
from modules import (
    data_split,
    verify_disjoint,
    load_dataset,
    mp_dilate_masks,
    Augmentations,
    BuildDataloader,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
print("test")
print(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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

BATCH_SIZE = 256
ACCUMULATION_STEPS = 32
TILE_SIZES = {
    "rn_224": 224,
    "rn_448": 448,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_TILES_PER_IMAGE = 10
OTHER_TILES_PER_IMAGE = 100

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Load names for each split
names_train = train['File Name'].tolist() 
names_val = val['File Name'].tolist()
names_test = test['File Name'].tolist()

# Get datasets and dataloaders
image_count = 200
if image_count is not None:
    train_images, train_masks = load_dataset((images_base, masks_base), names_train[:image_count], "Train")
else:
    train_images, train_masks = load_dataset((images_base, masks_base), names_train, "Train")
dmasks_train = mp_dilate_masks(train_masks, TILE_SIZES["rn_224"], pool_size=32)

val_images, val_masks = load_dataset((images_base, masks_base), names_val[:(image_count*.08)], "Validation")
dmasks_val = mp_dilate_masks(val_masks, TILE_SIZES["rn_224"], pool_size=32)

# test_images, test_masks = load_dataset((images_base, masks_base), names_test, "Test")
# dmasks_test = mp_dilate_masks(test_masks, TILE_SIZES["rn_224"], pool_size=32)

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
#     tile_size=TILE_SIZES["rn_224"],
#     transform=Augmentations.get_valid_transform(IMAGENET_MEAN, IMAGENET_STD),
#     tiles_per_image=OTHER_TILES_PER_IMAGE,
#     mix_ratio=0.25,
    
#     num_workers=32,
#     pin_memory=True,
#     persistent_workers=True,
# )


# === Model Training ===    
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize the model
model = smp.Unet(
    encoder_name="resnet34",       
    encoder_weights=None,    
    in_channels=3,                   
    classes=2,                       
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler('cuda')

trainer = trainer.Trainer(
    model=model, 
    train_loader=train_dataloader, 
    val_loader=val_dataloader,
    optimizer=optimizer, 
    criterion=criterion,
    scaler=scaler,
    device=DEVICE
)


if __name__ == '__main__':
    trainer.fit(
        max_epochs=1,
    )