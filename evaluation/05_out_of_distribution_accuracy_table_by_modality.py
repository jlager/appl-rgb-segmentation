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

species_order = ['sorghum', 'soybean', 'eucalyptus', 'arabidopsis', 'pennycress']
modality_order = ['rgb1', 'rgb1-rhizo', 'rgb2', 'rgb2-rhizo']
species_labels = {
    'sorghum': 'Sorghum',
    'soybean': 'Soybean',
    'eucalyptus': 'Eucalyptus',
    'arabidopsis': 'Arabidopsis',
    'pennycress': 'Pennycress',
}

def sort_key(row):
    species_rank = (
        species_order.index(row['Species'])
        if row['Species'] in species_order
        else len(species_order)
    )
    modality_rank = (
        modality_order.index(row['Modality'])
        if row['Modality'] in modality_order
        else len(modality_order)
    )
    return species_rank, modality_rank

def load_groups():
    metadata_file = os.path.join('..', config.SPLIT_DIR, 'generalization.csv')
    df = pd.read_csv(metadata_file)
    pairs = df[['Species', 'Modality']].drop_duplicates().to_dict('records')
    pairs = sorted(pairs, key=sort_key)
    return [(row['Species'], row['Modality']) for row in pairs]

def compute_stats(df, species, modality):
    dice = df[(df['Species'] == species) & (df['Modality'] == modality)]['Dice']
    if dice.empty:
        raise ValueError(f'No metrics found for {species} / {modality}')
    mean = 100 * dice.mean()
    std = 100 * dice.std()
    return mean, std

def compute_overall_stats(df):
    dice = df['Dice']
    mean = 100 * dice.mean()
    std = 100 * dice.std()
    return mean, std

def format_stats(mean, std, is_best):
    stats = f'{mean:.1f} \\pm {std:.1f}'
    if is_best:
        return f'$\\mathbf{{{stats}}}$'
    return f'${stats}$'

def format_header(species, modality):
    species_label = species_labels.get(species, species.title())
    modality_parts = [part.upper().title() if part == 'rhizo' else part.upper()
                      for part in modality.split('-')]
    modality_label = r'\\'.join(modality_parts)
    return rf'\shortstack{{{species_label}\\{modality_label}}}'

groups = load_groups()

# compute accuracy metrics
rows = []
for backbone, patch_size, model_name in models:
    for tile_size in tile_sizes:
        metrics_file = os.path.join(
            '..', config.METRICS_PATH.format(model_name, tile_size, 'generalization'))
        df = pd.read_csv(metrics_file)
        stats = [compute_overall_stats(df)]
        stats += [compute_stats(df, species, modality) for species, modality in groups]
        rows.append((backbone, patch_size, tile_size, stats))

# bold all models tied for the best displayed mean
rounded_means = np.array([[round(mean, 1) for mean, _ in stats] for _, _, _, stats in rows])
best_means = rounded_means.max(axis=0)

# build LaTeX table
metric_header = ['Overall']
metric_header += [format_header(species, modality) for species, modality in groups]
first_species = groups[0][0]
split_at = 1 + sum(species == first_species for species, _ in groups)
metric_sections = [
    (0, metric_header[:split_at]),
    (split_at, metric_header[split_at:]),
]
lines = [
    r'\begin{table*}[t]',
    r'    \centering',
    r'    \caption{\textbf{Generalization accuracy by species and modality.} Image-level Dice scores (mean $\pm$ standard deviation, percentage points) are reported for each U-Net and Vision Transformer (ViT) configuration evaluated on the generalization split. Supplemental results are shown for every observed species-modality pair. Higher Dice values indicate greater overlap between predicted and manually annotated segmentation masks. Best-performing models for each column are shown in bold.}',
    r'    \renewcommand{\arraystretch}{1.35}',
]
for section_index, (offset, section_header) in enumerate(metric_sections):
    if section_index > 0:
        lines.append(r'    \vspace{0.75em}')
    column_spec = 'l' + 'c' * (2 + len(section_header))
    header = ['Backbone', 'Patch', 'Tile'] + section_header
    lines += [
        r'    \rowcolors{3}{gray!25}{white}',
        rf'    \begin{{tabular}}{{{column_spec}}}',
        f'        {" & ".join(header)} \\\\',
        r'        \hline',
    ]
    for backbone, patch_size, tile_size, stats in rows:
        values = []
        for i, (mean, std) in enumerate(stats[offset:offset + len(section_header)]):
            stat_index = offset + i
            is_best = np.isclose(round(mean, 1), best_means[stat_index])
            values.append(format_stats(mean, std, is_best))
        row = ' & '.join([backbone, patch_size, str(tile_size)] + values)
        lines.append(f'        {row} \\\\')
    lines.append(r'    \end{tabular}')
lines += [
    r'    \label{tab:ood_quantitative_comparison_by_modality}',
    r'\end{table*}',
]

print('\n'.join(lines))
