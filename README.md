# APPL RGB Segmentation

Code for training and evaluating RGB plant segmentation models on APPL imagery.

The pipeline trains binary plant/background models from large RGB images, evaluates
whole-image Dice, and compares U-Nets with ViT segmentation models.

## Setup

```bash
uv sync
```

This project uses Python 3.12+ and CUDA GPUs. Training and evaluation select a
GPU with `--device`.

## Layout

```text
config.py          shared paths and constants
models.py          segmentation models
train.py           training script
test.py            evaluation script
preprocessing/     raw data conversion, ROI generation, split creation
evaluation/        plot and table generation from logs/metrics
outputs/           logs and metrics
```

## Data

Expected raw inputs:

```text
data/raw/images/<modality-species>/*.png
data/raw/masks/**/SegmentationClass/*.png
data/raw/metadata/metadata.csv
```

Processed data is stored as memmaps:

```text
data/images/<modality-species>/*.memmap
data/masks/<modality-species>/*.memmap
data/rois_224/<modality-species>/*.npz
data/rois_448/<modality-species>/*.npz
data/metadata/{train,val,test,generalization}.csv
```

## Preprocess

Run from the repository root.

```bash
uv run python preprocessing/01_convert_raw_images_to_memmaps.py
uv run python preprocessing/03_convert_raw_masks_to_memmaps.py
uv run python preprocessing/07_train_val_test_split.py
uv run python preprocessing/08_generalization_split.py
```

Generate ROIs for each tile size used in training. Set `TILE_SIZE` in
`preprocessing/05_convert_masks_to_rois.py` to `224` or `448`, then run:

```bash
uv run python preprocessing/05_convert_masks_to_rois.py
```

Optional visual checks:

```bash
uv run python preprocessing/02_visualize_memmap_images.py
uv run python preprocessing/04_visualize_memmap_masks.py
uv run python preprocessing/06_visualize_rois.py
```

## Train

```bash
uv run python train.py --backbone vit_base_patch8_224 --tile_size 448 --device 0
```

Pretrained backbone weights are enabled by default. Use `--no-pretrained` to
train from scratch.

Supported backbones:

```text
resnet34, resnet50, resnet101, resnet152
vit_small_patch8_224, vit_small_patch16_224
vit_base_patch8_224, vit_base_patch16_224
```

Supported tile sizes: `224`, `448`.

Training writes:

```text
outputs/checkpoints/checkpoint_model-<backbone>_<pretrained|scratch>_tile-<tile>.pt
outputs/logs/log_model-<backbone>_<pretrained|scratch>_tile-<tile>.csv
```

## Evaluate

```bash
uv run python test.py --backbone vit_base_patch8_224 --tile_size 448 --device 0 --inference_mode hann
```

Use the same `--pretrained` or `--no-pretrained` setting used for training.

Inference modes:

```text
hann        overlapping tiles with Hann-window blending
classical   non-overlapping padded tiles
```

Evaluation writes one metrics CSV per split:

```text
outputs/metrics/metrics_model-<backbone>_<pretrained|scratch>_tile-<tile>_split-<split>.csv
outputs/metrics/metrics_model-<backbone>_<pretrained|scratch>_tile-<tile>_split-<split>_inference-classical.csv
```

It also writes predicted PNG masks under:

```text
outputs/masks/model-<backbone>_<pretrained|scratch>_tile-<tile>_inference-<mode>/<split>/<modality-species>/*.png
```

Evaluation reuses saved prediction masks when present, and runs inference only
for missing masks. To force mask regeneration:

```bash
uv run python test.py --backbone vit_base_patch8_224 --tile_size 448 --device 0 --inference_mode hann --overwrite_masks
```

Splits are `train`, `val`, `test`, and `generalization`.

## Summaries

Run scripts in `evaluation/` from that directory. They read `../outputs/logs`,
`../outputs/metrics`, and `../data/metadata`, then show plots or print LaTeX
tables.
