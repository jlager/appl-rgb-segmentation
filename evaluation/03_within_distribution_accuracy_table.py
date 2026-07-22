import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
import config

# options
models = [
    ('ResNet-34', '--', 'resnet34'),
    ('ResNet-50', '--', 'resnet50'),
    ('ResNet-101', '--', 'resnet101'),
    ('ResNet-152', '--', 'resnet152'),
    ('ViT-Small', '16', 'vit_small_patch16_224'),
    ('ViT-Small', '8', 'vit_small_patch8_224'),
    ('ViT-Base', '16', 'vit_base_patch16_224'),
    ('ViT-Base', '8', 'vit_base_patch8_224'),
]
tile_sizes = [224, 448]
groups = [
    ('Overall', None, None),
    ('Poplar RGB1', 'Poplar', 'RGB1'),
    ('Poplar RGB2', 'Poplar', 'RGB2'),
    ('Switchgrass RGB1', 'Switchgrass', 'RGB1'),
    ('Switchgrass RGB2', 'Switchgrass', 'RGB2'),
]

def compute_stats(df, species, modality):
    if species is None:
        dice = df['Dice']
    else:
        dice = df[(df['Species'] == species) & (df['Modality'] == modality)]['Dice']
    mean = 100 * dice.mean()
    std = 100 * dice.std(ddof=0)
    return mean, std

def format_stats(mean, std, is_best):
    stats = f'{mean:.1f} \\pm {std:.1f}'
    if is_best:
        return f'$\\mathbf{{{stats}}}$'
    return f'${stats}$'

def get_run_name(model_name):
    suffix = 'pretrained' if model_name.startswith('vit') else 'scratch'
    return f'{model_name}_{suffix}'

# compute accuracy metrics
rows = []
for backbone, patch_size, model_name in models:
    for tile_size in tile_sizes:
        run_name = get_run_name(model_name)
        metrics_file = os.path.join('..', config.METRICS_PATH.format(run_name, tile_size, 'test'))
        df = pd.read_csv(metrics_file)
        stats = [compute_stats(df, species, modality) for _, species, modality in groups]
        rows.append((backbone, patch_size, tile_size, stats))

# bold all models tied for the best displayed mean
rounded_means = np.array([[round(mean, 1) for mean, _ in stats] for _, _, _, stats in rows])
best_means = rounded_means.max(axis=0)

# build LaTeX table
lines = [
    r'\begin{table}[ht]',
    r'    \centering',
    r'    \caption{\textbf{Testing accuracy.} Image-level Dice scores (mean $\pm$ standard deviation, percentage points) are reported for each U-Net and Vision Transformer (ViT) configuration evaluated on the poplar and switchgrass test split. Results are shown overall and separately for each species and imaging modality (RGB1: side-view; RGB2: top-view). Higher Dice values indicate greater overlap between predicted and manually annotated segmentation masks. Best-performing models for each column are shown in bold.\vspace{1em}}',
    r'    \rowcolors{2}{gray!25}{white}',
    r'    \renewcommand{\arraystretch}{1.5}',
    r'    \begin{tabular}{lccccccc}',
    r'        Backbone & Patch & Tile & Overall & \shortstack{Poplar\\RGB1} & \shortstack{Poplar\\RGB2} & \shortstack{Switchgrass\\RGB1} & \shortstack{Switchgrass\\RGB2} \\',
    r'        \hline',
]
for backbone, patch_size, tile_size, stats in rows:
    values = []
    for i, (mean, std) in enumerate(stats):
        is_best = np.isclose(round(mean, 1), best_means[i])
        values.append(format_stats(mean, std, is_best))
    row = ' & '.join([backbone, patch_size, str(tile_size)] + values)
    lines.append(f'        {row} \\\\')
lines += [
    r'    \end{tabular}',
    r'    \label{tab:id_quantitative_comparison}',
    r'\end{table}',
]

print('\n'.join(lines))
