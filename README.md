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
outputs/metrics/metrics_model-<backbone>_<pretrained|scratch>_tile-<tile>_split-<split>_inference-hann.csv
outputs/metrics/metrics_model-<backbone>_<pretrained|scratch>_tile-<tile>_split-<split>_inference-classical.csv
```

It also writes predicted PNG masks under:

```text
outputs/masks/model-<backbone>_<pretrained|scratch>_tile-<tile>_inference-<mode>/<split>/<modality-species>/*.png
```

To evaluate the fixed-threshold baseline:

```bash
uv run python test.py --backbone threshold --threshold 0.2
```

This baseline uses normalized `4G - 3B - R`, requires no trained checkpoint,
and always uses `tile-448` and `inference-none` in output names. It omits the
`pretrained`/`scratch` suffix:

```text
outputs/metrics/metrics_model-threshold_tile-448_split-<split>_inference-none.csv
outputs/masks/model-threshold_tile-448_inference-none/<split>/<modality-species>/*.png
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

`1_segmentation_accuracy.py` prints a LaTeX summary for the threshold baseline
and scratch/pretrained U-Net and ViT categories. It selects each category's highest
mean validation Dice configuration across backbones and tile sizes, then reports
that same configuration's validation, test, and generalization Dice, using Hann
inference for all trained models.

`2_testing_accuracy.py` reports the same validation-selected
configurations on the test set, overall and by species/modality.
`3_generalization_accuracy.py` reports them on the generalization
set, overall and by species. Both use the summary table's categories and style.

### Supplemental tables

The scripts print the final supplemental LaTeX, including captions and page layout:
S1 has side-by-side U-Net/ViT panels on one landscape page; S2 and S6 use one
landscape page each with shared model-column headings and rotated row labels;
S3 uses three continued landscape pages; S4 and S5 each use one portrait page.
Table text is `\small`, with alternating gray rows. S1 retains validation-selection
bolding, and S2/S3 retain validation-selection daggers without winner bolding.
The generated fragments can be pasted directly into the active table section of
`supplementary.tex`; no manual panel rearrangement or caption rewriting is needed.
Keep the historical commented tables when replacing active blocks.

| Script | Table |
| --- | --- |
| `S1_segmentation_accuracy.py` | S1: all 32 trained configurations, Hann validation/test/generalization Dice |
| `S2_testing_accuracy_by_species_modality.py` | S2: all configurations, test species/modality scores |
| `S3_generalization_accuracy_by_species_modality.py` | S3: all configurations, generalization species/modality scores |
| `S4_effect_of_pretraining.py` | S4: matched pretrained-minus-scratch Dice on test and generalization |
| `S5_effect_of_overlapping_inference.py` | S5: matched Hann-minus-classical Dice on both evaluation splits |
| `S6_inference_time.py` | S6: actual generalization timing runs, by initialization and inference method |

Run from the repository root, for example:

```bash
mkdir -p outputs/tables
.venv/bin/python evaluation/S1_segmentation_accuracy.py --results-csv outputs/tables/supplement_results.csv > outputs/tables/S1.tex
.venv/bin/python evaluation/S2_testing_accuracy_by_species_modality.py > outputs/tables/S2.tex
.venv/bin/python evaluation/S3_generalization_accuracy_by_species_modality.py > outputs/tables/S3.tex
.venv/bin/python evaluation/S4_effect_of_pretraining.py > outputs/tables/S4.tex
.venv/bin/python evaluation/S5_effect_of_overlapping_inference.py > outputs/tables/S5.tex
.venv/bin/python evaluation/S6_inference_time.py > outputs/tables/S6.tex
```

The CSV option exports overall and species/modality statistics for every trained
configuration and the threshold baseline on train, validation, test, and
generalization splits, including exhaustive classical results. It records image
counts, source files, units, population SD (`ddof=0`), and validation-selection
flags. Export paths are relative to the working directory; input paths are
relative to the repository. The original image-level CSVs remain the source data.

All images are weighted equally. Input identifiers must be unique and match the
split metadata exactly. S4/S5 align by species, modality, and filename before
computing differences; reported SDs describe paired image-level differences in
percentage points. S1 bolds the four Hann validation-selected rows; S2–S3 mark
those configurations with a dagger. These marks do not identify test/generalization
winners. Initializations are
adjacent within each configuration. S2 repeats overall scores across modality
panels; S3 repeats them across species panels and reports only observed modalities.
Arabidopsis and pennycress have only RGB2. S6 reports each measured run separately,
including both initializations, and groups rhizotron views with their RGB1/RGB2
modality. Source filenames are recorded in the machine-readable CSV export.

The LaTeX uses `xcolor` with the `table` option, `graphicx`, `geometry`,
`pdflscape`, and `caption` (or `subcaption`) for continued tables. These are
already loaded by the supplement. Landscape fragments save and restore the
current portrait margins. Before S1, the manuscript sets supplemental numbering:

```latex
\clearpage
\setcounter{table}{0}
\renewcommand{\thetable}{S\arabic{table}}
```

The manuscript still supplies the document preamble, figures and title page.
The evaluation scripts do not edit it automatically. Legacy underscore-prefixed
scripts are kept locally in the Git-ignored `scratch/` directory.
