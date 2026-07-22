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
inference_modes = [
    ('Hann', 'hann'),
    ('Classical', 'classical'),
]
groups = [
    ('Overall', None),
    ('RGB1', ['rgb1', 'rgb1-rhizo']),
    ('RGB2', ['rgb2', 'rgb2-rhizo']),
]

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
    if 'Time (s)' not in df.columns:
        raise ValueError(f'Missing Time (s) column in {metrics_file}')
    if 'Modality' not in df.columns:
        raise ValueError(f'Missing Modality column in {metrics_file}')
    if df['Time (s)'].isna().any():
        raise ValueError(f'Missing timing values in {metrics_file}')

    df = df.copy()
    df['Modality'] = df['Modality'].str.lower()
    return df

def compute_stats(df, group_label, modalities, context):
    if modalities is None:
        times = df['Time (s)']
    else:
        times = df[df['Modality'].isin(modalities)]['Time (s)']

    if times.empty:
        raise ValueError(
            f'No timing rows found for {group_label} in {context}; '
            f'expected modalities: {modalities}')

    mean = times.mean()
    std = times.std(ddof=0)
    return mean, std

def format_stats(mean, std, is_best):
    stats = f'{mean:.2f} \\pm {std:.2f}'
    if is_best:
        return f'$\\mathbf{{{stats}}}$'
    return f'${stats}$'

def get_run_name(model_name):
    suffix = 'pretrained' if model_name.startswith('vit') else 'scratch'
    return f'{model_name}_{suffix}'

# compute timing metrics
rows = []
for backbone, patch_size, model_name in models:
    for tile_size in tile_sizes:
        run_name = get_run_name(model_name)
        mode_dfs = {
            mode_name: load_metrics(run_name, tile_size, mode_name)
            for _, mode_name in inference_modes
        }
        stats = []
        for group_label, modalities in groups:
            for _, mode_name in inference_modes:
                context = f'{model_name}, tile {tile_size}, {mode_name}'
                stats.append(compute_stats(
                    mode_dfs[mode_name], group_label, modalities, context))
        rows.append((backbone, patch_size, tile_size, stats))

# bold all models tied for the fastest displayed mean
rounded_means = np.array([[round(mean, 2) for mean, _ in stats] for _, _, _, stats in rows])
best_means = rounded_means.min(axis=0)

# build LaTeX table
lines = [
    r'\begin{table*}[t]',
    r'    \centering',
    r'    \caption{\textbf{Inference time by modality.} Per-image inference time (mean $\pm$ standard deviation, seconds) is reported for each U-Net and Vision Transformer (ViT) configuration evaluated on the generalization split. RGB1 includes RGB1 and RGB1-rhizo images; RGB2 includes RGB2 and RGB2-rhizo images. Lower values indicate faster inference. Fastest models for each column are shown in bold.}',
    r'    \rowcolors{3}{gray!25}{white}',
    r'    \renewcommand{\arraystretch}{1.35}',
    r'    \begin{tabular}{lcccccccc}',
    r'        Backbone & Patch & Tile & \multicolumn{2}{c}{Overall} & \multicolumn{2}{c}{RGB1} & \multicolumn{2}{c}{RGB2} \\',
    r'        \cline{4-5}\cline{6-7}\cline{8-9}',
    r'        & & & Hann & Classical & Hann & Classical & Hann & Classical \\',
    r'        \hline',
]
for backbone, patch_size, tile_size, stats in rows:
    values = []
    for i, (mean, std) in enumerate(stats):
        is_best = np.isclose(round(mean, 2), best_means[i])
        values.append(format_stats(mean, std, is_best))
    row = ' & '.join([backbone, patch_size, str(tile_size)] + values)
    lines.append(f'        {row} \\\\')
lines += [
    r'    \end{tabular}',
    r'    \label{tab:inference_time_by_modality}',
    r'\end{table*}',
]

print('\n'.join(lines))
