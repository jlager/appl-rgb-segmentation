"""Shared data loading, statistics, and formatting for main and supplemental tables."""
from functools import lru_cache

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config

# =============================================================================
# Configuration
# =============================================================================

BACKBONES = {
    'U-Net': ['resnet34', 'resnet50', 'resnet101', 'resnet152'],
    'ViT': ['vit_small_patch16_224', 'vit_small_patch8_224',
            'vit_base_patch16_224', 'vit_base_patch8_224'],
}
CATEGORIES = [
    ('Threshold', '--', None),
    ('U-Net', 'Scratch', 'scratch'),
    ('U-Net', 'Pretrained', 'pretrained'),
    ('ViT', 'Scratch', 'scratch'),
    ('ViT', 'Pretrained', 'pretrained'),
]
MAIN_SPLITS = ['val', 'test', 'generalization']
BACKBONE_LABELS = {
    'threshold': '--',
    'resnet34': 'ResNet-34',
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'resnet152': 'ResNet-152',
    'vit_small_patch16_224': 'ViT-Small/16',
    'vit_small_patch8_224': 'ViT-Small/8',
    'vit_base_patch16_224': 'ViT-Base/16',
    'vit_base_patch8_224': 'ViT-Base/8',
}

# =============================================================================
# Main table metrics
# =============================================================================

def load_result(run, tile, mode, split):
    """Load and summarize Dice scores for one model run."""
    path = ROOT / config.METRICS_PATH.format(run, tile, split)
    path = path.with_name(f'{path.stem}_inference-{mode}{path.suffix}')
    dice = pd.read_csv(path)['Dice'].to_numpy(dtype=float)
    if dice.size == 0 or not np.isfinite(dice).all():
        raise ValueError(f'Empty or non-finite Dice values in {path}')
    return {
        'tile': tile if run != 'threshold' else '--',
        'backbone': BACKBONE_LABELS[run.rsplit('_', 1)[0] if run != 'threshold' else run],
        'mean': 100 * dice.mean(),
        'std': 100 * dice.std(ddof=0),
        'source': path.name,
        'path': path,
    }

def selected_results(model, initialization):
    """Select by validation Dice and report the same run on each dataset."""
    if model == 'Threshold':
        candidates = [('threshold', 448, 'none')]
    else:
        candidates = [
            (f'{backbone}_{initialization}', tile, 'hann')
            for backbone in BACKBONES[model]
            for tile in [224, 448]
        ]
    # Select by unrounded mean; exact ties retain candidate order.
    selected = max(candidates, key=lambda candidate: load_result(*candidate, 'val')['mean'])
    return [load_result(*selected, split) for split in MAIN_SPLITS]


KEYS = ['Species', 'Modality', 'File Name']
SPLITS = ['train', 'val', 'test', 'generalization']
SPECIES = ['sorghum', 'soybean', 'eucalyptus', 'arabidopsis', 'pennycress']
MODALITIES = ['rgb1', 'rgb1-rhizo', 'rgb2', 'rgb2-rhizo']
STAT_NOTE = (r'Each image has equal weight; values are mean $\pm$ population '
             r'standard deviation ($ddof=0$). $n$ denotes image count. ')
SELECTION_NOTE = (r'$\dagger$ indicates the four configurations selected by highest '
                  r'overall validation Dice for the main text, using Hann inference. '
                  r'Test and generalization scores are not used for selection. ')

# =============================================================================
# Supplemental table metrics
# =============================================================================

def configurations(family=None):
    """Yield the model, backbone, tile size, and initialization combinations."""
    for model, backbones in BACKBONES.items():
        if family is not None and model != family:
            continue
        for backbone in backbones:
            for tile in [224, 448]:
                for initialization in ['scratch', 'pretrained']:
                    yield model, backbone, tile, initialization

def metrics_path(configuration, split, mode='hann'):
    """Return the metrics CSV path for one configuration and dataset."""
    _, backbone, tile, initialization = configuration
    run = backbone if backbone == 'threshold' else f'{backbone}_{initialization}'
    path = ROOT / config.METRICS_PATH.format(run, tile, split)
    return path.with_name(f'{path.stem}_inference-{mode}{path.suffix}')

def validate_frame(df, context):
    """Validate metric values and index rows by unique image identifiers."""
    df = df.copy()
    required = KEYS + ['Dice', 'Time (s)']
    if df.empty or not set(required).issubset(df.columns):
        raise ValueError(f'Empty metrics or missing columns: {context}')
    if df[KEYS].isna().any().any():
        raise ValueError(f'Missing image identifiers: {context}')
    for column in ['Species', 'Modality']:
        df[column] = df[column].str.lower()
    if df.duplicated(KEYS).any():
        raise ValueError(f'Duplicate image identifiers: {context}')
    values = df[['Dice', 'Time (s)']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values[:, 0] > 1).any():
        raise ValueError(f'Invalid Dice or timing values: {context}')
    return df.set_index(KEYS).sort_index()

@lru_cache(maxsize=None)
def metadata_index(split):
    """Load the expected image identifiers for a dataset split."""
    df = pd.read_csv(ROOT / config.SPLIT_DIR / f'{split}.csv')
    for column in ['Species', 'Modality']:
        df[column] = df[column].str.lower()
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
        raise ValueError(f'Missing or duplicate image identifiers in {split} metadata')
    return df.set_index(KEYS).sort_index().index

@lru_cache(maxsize=None)
def load(configuration, split, mode='hann'):
    """Load metrics and check that their images match the split metadata."""
    path = metrics_path(configuration, split, mode)
    df = validate_frame(pd.read_csv(path), path)
    if not df.index.equals(metadata_index(split)):
        raise ValueError(f'Image set does not match {split} metadata: {path}')
    return df

def paired_difference(positive, negative):
    """Match images and compute Dice differences in percentage points."""
    if not positive.index.is_unique or not negative.index.is_unique:
        raise ValueError('Paired comparison contains duplicate image identifiers')
    if len(positive) != len(negative) or not positive.index.isin(negative.index).all():
        raise ValueError('Paired comparison requires identical image sets')
    aligned = negative.reindex(positive.index)
    return 100 * (positive['Dice'] - aligned['Dice'])

def stats(values):
    """Return image count, mean, and population standard deviation."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Cannot summarize empty or non-finite values')
    return len(values), values.mean(), values.std(ddof=0)

# =============================================================================
# Table formatting
# =============================================================================

def cell(values, signed=False, decimals=1):
    """Format the mean and standard deviation as a LaTeX table cell."""
    _, mean, std = stats(values)
    mean_text = f'{mean:+.{decimals}f}' if signed else f'{mean:.{decimals}f}'
    return f'${mean_text} \\pm {std:.{decimals}f}$'

def subset(df, species=None, modality=None):
    """Select images for a species, modality, or both."""
    mask = np.ones(len(df), dtype=bool)
    if species is not None:
        mask &= df.index.get_level_values('Species') == species
    if modality is not None:
        mask &= df.index.get_level_values('Modality') == modality
    return df.loc[mask]

@lru_cache(maxsize=1)
def selected_configurations():
    """Find the best validation configuration per family and initialization."""
    winners = set()
    for family in BACKBONES:
        for initialization in ['scratch', 'pretrained']:
            candidates = [c for c in configurations(family) if c[3] == initialization]
            winners.add(max(candidates, key=lambda c: load(c, 'val')['Dice'].mean()))
    return winners

def identifiers(configuration, mark=False):
    """Format model labels and optionally mark validation-selected configurations."""
    model, backbone, tile, initialization = configuration
    label = BACKBONE_LABELS[backbone]
    if mark and configuration in selected_configurations():
        label += r'$^{\dagger}$'
    return [model, label, str(tile), initialization.capitalize()]

def score_header(label, count):
    """Format a column heading with its image count."""
    return rf'\shortstack{{{label}\\$n={count}$}}'

def panel(title, caption, headers, rows, label):
    """Collect the caption, headers, and rows for a table panel."""
    return dict(title=title, caption=caption, headers=headers, rows=rows,
                label=label)

def tabular_lines(section, omit_columns):
    """Format panel rows as LaTeX with alternating row shading."""
    headers = section['headers']
    keep = [i for i in range(len(headers)) if i not in omit_columns]
    columns = ''.join('c' if headers[i].startswith(r'\shortstack') else 'l' for i in keep)
    lines = [r'    \rowcolors{2}{gray!15}{white}',
             rf'    \begin{{tabular}}{{{columns}}}',
             '        ' + ' & '.join(headers[i] for i in keep) + r' \\',
             r'        \hline']
    for row in section['rows']:
        if len(row) != len(headers):
            raise ValueError(f"Column count mismatch in {section['label']}")
        lines.append('        ' + ' & '.join(row[i] for i in keep) + r' \\')
    return lines + [r'    \end{tabular}\par', rf"    \label{{{section['label']}}}"]

def print_tables(panels, layout='stacked', pages=None, row_labels=None, omit_columns=()):
    """Render the manuscript's portrait, two-column, or shared-heading grid layout."""
    if layout not in ('stacked', 'columns', 'grid'):
        raise ValueError(f'Unknown table layout: {layout}')
    pages = pages if pages is not None else [list(range(len(panels)))]
    if sorted(i for page in pages for i in page) != list(range(len(panels))):
        raise ValueError('Each panel must appear exactly once')
    first = panels[0]
    title = first['title'].split(':', 1)[0].split('. ', 1)[1]
    caption = first['caption']
    if any(section['caption'] != caption for section in panels):
        raise ValueError('Panels must share the same caption')
    label = first['label'].split('_', 1)[0]
    lines = []
    if layout != 'stacked':
        lines += [r'\savegeometry{supplementportrait}', r'\newgeometry{margin=0.65in}',
                  r'\begin{landscape}']
    for page_index, indices in enumerate(pages):
        lines += [r'\begin{table}[!ht]', r'    \centering']
        if page_index:
            lines += [r'    \ContinuedFloat',
                      rf'    \caption{{\textbf{{{title} (continued).}} Definitions are given on the first page of this table.}}']
        else:
            lines += [rf'    \caption{{\textbf{{{title}.}} {caption}}}', rf'    \label{{{label}}}']
        stretch = '1.05' if layout == 'grid' else '1.25'
        lines += [r'    \small', r'    \setlength{\tabcolsep}{3pt}',
                  rf'    \renewcommand{{\arraystretch}}{{{stretch}}}', r'    \vspace{0.5em}']
        if layout == 'grid':
            if len(indices) != 2 * len(row_labels[page_index]):
                raise ValueError('Grid needs two model panels per row label')
            lines += [r'    \noindent\makebox[0.06\linewidth]{}%',
                      r'    \makebox[0.46\linewidth]{\textbf{U-Net}}\hfill',
                      r'    \makebox[0.46\linewidth]{\textbf{ViT}}\par\smallskip']
        for position, index in enumerate(indices):
            section = panels[index]
            heading = section['title'].split(':', 1)[1].strip().rstrip('.')
            if layout == 'grid':
                if position % 2 == 0:
                    if position:
                        lines.append(r'    \vspace{1em}')
                    case = row_labels[page_index][position // 2]
                    lines += [r'    \noindent\begin{minipage}[c]{0.06\linewidth}',
                              r'    \centering\rotatebox{90}{\textbf{' + case + '}}',
                              r'    \end{minipage}%']
                lines += [r'    \begin{minipage}[c]{0.46\linewidth}', r'    \centering']
            elif layout == 'columns':
                lines += [r'    \begin{minipage}[t]{0.485\linewidth}', r'    \centering',
                          rf'    \textbf{{{heading}}}\par\smallskip']
            else:
                if position:
                    lines.append(r'    \vspace{1em}')
                lines.append(rf'    \textbf{{{heading}}}\par\smallskip')
            lines += tabular_lines(section, omit_columns)
            if layout != 'stacked':
                lines.append(r'    \end{minipage}' + (r'\hfill' if position % 2 == 0 else r'\par'))
        lines += [r'\end{table}', r'\clearpage']
    if layout != 'stacked':
        lines += [r'\end{landscape}', r'\loadgeometry{supplementportrait}']
    print('\n'.join(lines))

def family_slug(family):
    """Format a model family name for a LaTeX label."""
    return family.lower().replace('-', '')
