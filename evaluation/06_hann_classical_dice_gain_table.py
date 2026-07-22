import os
import sys
import numpy as np
import pandas as pd

sys.path.append('../')
import config

# options
models = [
    ('ResNet-34', '-', 'resnet34'),
    ('ResNet-50', '-', 'resnet50'),
    ('ResNet-101', '-', 'resnet101'),
    ('ResNet-152', '-', 'resnet152'),
    ('ViT-Small', '16', 'vit_small_patch16_224'),
    ('ViT-Small', '8', 'vit_small_patch8_224'),
    ('ViT-Base', '16', 'vit_base_patch16_224'),
    ('ViT-Base', '8', 'vit_base_patch8_224'),
]
tile_sizes = [224, 448]
groups = [
    ('Overall', None),
    ('RGB1', ['rgb1', 'rgb1-rhizo']),
    ('RGB2', ['rgb2', 'rgb2-rhizo']),
]
key_columns = ['Species', 'Modality', 'File Name']

def get_metrics_path(run_name, tile_size, inference_mode):
    metrics_file = os.path.join(
        '..', config.METRICS_PATH.format(run_name, tile_size, 'generalization'))
    if inference_mode == 'hann':
        return metrics_file
    root, ext = os.path.splitext(metrics_file)
    return f'{root}_inference-{inference_mode}{ext}'

def load_metrics(run_name, tile_size, inference_mode):
    metrics_file = get_metrics_path(run_name, tile_size, inference_mode)
    if not os.path.isfile(metrics_file):
        raise FileNotFoundError(metrics_file)

    df = pd.read_csv(metrics_file)
    required_columns = key_columns + ['Dice']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing columns in {metrics_file}: {missing_columns}')
    if df['Dice'].isna().any():
        raise ValueError(f'Missing Dice values in {metrics_file}')
    if df.duplicated(key_columns).any():
        raise ValueError(f'Duplicate metric rows in {metrics_file}')

    df = df[required_columns].copy()
    df['Species'] = df['Species'].str.lower()
    df['Modality'] = df['Modality'].str.lower()
    return df

def load_gain_metrics(run_name, tile_size):
    hann_df = load_metrics(run_name, tile_size, 'hann')
    classical_df = load_metrics(run_name, tile_size, 'classical')
    merged = hann_df.merge(
        classical_df,
        on=key_columns,
        how='outer',
        suffixes=('_hann', '_classical'),
        indicator=True,
    )

    n_missing_classical = (merged['_merge'] == 'left_only').sum()
    n_missing_hann = (merged['_merge'] == 'right_only').sum()
    if n_missing_classical or n_missing_hann:
        context = f'{run_name}, tile {tile_size}'
        raise ValueError(
            f'Mismatched Hann/classical rows for {context}: '
            f'{n_missing_classical} missing classical rows, '
            f'{n_missing_hann} missing Hann rows')

    merged['Dice Gain'] = 100 * (merged['Dice_hann'] - merged['Dice_classical'])
    return merged

def compute_stats(df, group_label, modalities, context):
    if modalities is None:
        gains = df['Dice Gain']
    else:
        gains = df[df['Modality'].isin(modalities)]['Dice Gain']

    if gains.empty:
        raise ValueError(
            f'No Dice gain rows found for {group_label} in {context}; '
            f'expected modalities: {modalities}')

    mean = gains.mean()
    std = gains.std(ddof=0)
    return mean, std

def format_stats(mean, std, is_best):
    stats = f'{mean:+.1f} \\pm {std:.1f}'
    if is_best:
        return f'$\\mathbf{{{stats}}}$'
    return f'${stats}$'

def get_run_name(model_name):
    suffix = 'pretrained' if model_name.startswith('vit') else 'scratch'
    return f'{model_name}_{suffix}'

# compute Dice gain metrics
rows = []
for backbone, patch_size, model_name in models:
    for tile_size in tile_sizes:
        run_name = get_run_name(model_name)
        df = load_gain_metrics(run_name, tile_size)
        stats = []
        for group_label, modalities in groups:
            context = f'{model_name}, tile {tile_size}'
            stats.append(compute_stats(df, group_label, modalities, context))
        rows.append((backbone, patch_size, tile_size, stats))

# bold all models tied for the best displayed mean gain
rounded_means = np.array([[round(mean, 1) for mean, _ in stats] for _, _, _, stats in rows])
best_means = rounded_means.max(axis=0)

# build LaTeX table
lines = [
    r'\begin{table}[ht]',
    r'    \centering',
    r'    \caption{\textbf{Dice gain from Hann inference.} Image-level Dice-score gains (Hann minus classical inference, mean $\pm$ standard deviation, percentage points) are reported for each U-Net and Vision Transformer (ViT) configuration evaluated on the generalization split. RGB1 includes RGB1 and RGB1-rhizo images; RGB2 includes RGB2 and RGB2-rhizo images. Higher values indicate larger gains from Hann-windowed inference. Best gains for each column are shown in bold.}',
    r'    \rowcolors{2}{gray!25}{white}',
    r'    \renewcommand{\arraystretch}{1.5}',
    r'    \begin{tabular}{lccccc}',
    r'        Backbone & Patch & Tile & Overall & RGB1 & RGB2 \\',
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
    r'    \label{tab:hann_classical_dice_gain}',
    r'\end{table}',
]

print('\n'.join(lines))
