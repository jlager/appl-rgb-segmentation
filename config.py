import os

# data
IMAGE_DIR = os.path.join('data', 'images')
MASK_DIR = os.path.join('data', 'masks')
ROI_DIR = os.path.join('data', 'rois_{}')
SPLIT_DIR = os.path.join('data', 'metadata')
DATA_EXT = '.memmap'
IMAGE_EXT = '.png'
MASK_EXT = '.png'
ROI_EXT = '.npz'

# image
RGB1_HEIGHT = 6556
RGB1_WIDTH = 4104
RGB1_CHANNELS = 3
RGB2_HEIGHT = 3006
RGB2_WIDTH = 4104
RGB2_CHANNELS = 3
IMAGE_DTYPE = 'uint8'
MASK_DTYPE = 'bool'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# training
EPOCHS = 100
TRAIN_TILES_PER_IMAGE = 10
LR_WARMUP_EPOCHS = 10
LR_DECAY_EPOCHS = 50
BATCH_SIZE = 126 # must be divisible by MICRO_BATCH_SIZE
MICRO_BATCH_SIZE = 14
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
MAX_LR = 1e-4
MIN_LR = 1e-6
BACKBONE_LR_FACTOR = 0.1
WEIGHT_DECAY = 1e-4
N_WORKERS = 8

# checkpointing
CHECKPOINT_PATH = os.path.join(
    'outputs', 'checkpoints', 'checkpoint_model-{}_tile-{}.pt')
LOG_PATH = os.path.join(
    'outputs', 'logs', 'log_model-{}_tile-{}.csv')

# random seeds
SEED = 42