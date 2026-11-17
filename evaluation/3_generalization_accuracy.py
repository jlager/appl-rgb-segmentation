import numpy as np
import pandas as pd

import helper_functions as helpers

# =============================================================================
# Configuration
# =============================================================================

SPLIT = 'generalization'
GROUPS = [
    ('Overall', None, None),
    ('Sorghum', 'sorghum', None),
    ('Soybean', 'soybean', None),
    ('Eucalyptus', 'eucalyptus', None),
    ('Arabidopsis', 'arabidopsis', None),
    ('Pennycress', 'pennycress', None),
]

# =============================================================================
# Metrics
# =============================================================================

def build_rows():
    """Collect validation-selected scores for each model and initialization."""
    rows = []
    for model, label, initialization in helpers.CATEGORIES:
        results = helpers.selected_results(model, initialization)
        result = results[helpers.MAIN_SPLITS.index(SPLIT)]
        validation = results[helpers.MAIN_SPLITS.index('val')]
        df = pd.read_csv(result['path'])
        stats = []
        for _, species, modality in GROUPS:
            subset = df
            if species is not None:
                subset = subset[subset['Species'].str.lower() == species]
            if modality is not None:
                subset = subset[subset['Modality'].str.lower() == modality]
            dice = subset['Dice'].to_numpy(dtype=float)
            if dice.size == 0 or not np.isfinite(dice).all():
                raise ValueError(f"Empty or non-finite Dice for {species}, {modality} in {result['path']}")
            stats.append((100 * dice.mean(), 100 * dice.std(ddof=0)))
        rows.append((model, label, validation, result, stats))
    return rows

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    rows = build_rows()
    best_means = [max(round(stats[i][0], 1) for *_, stats in rows)
                  for i in range(len(GROUPS))]
    lines = [
        r'\begin{table}[ht]',
        r'    \centering',
        r'    \caption{\textbf{Generalization accuracy.} Image-level Dice scores '
        r'(\%, mean $\pm$ population standard deviation) are reported for the threshold '
        r'baseline and each model and initialization category on the multi-species generalization dataset. '
        r'Results are shown overall and separately for each species. '
        r'All trained models use Hann inference. Each row uses the configuration '
        r'selected by highest overall mean validation Dice in '
        r'Table~\ref{tab:strategy_comparison}, which lists backbones and tile sizes. '
        r'The same configuration is used for every column in a row. '
        r'Higher Dice is better; best displayed Dice means for each column are bold.}',
        r'    \renewcommand{\arraystretch}{1.5}',
        r'    \rowcolors{2}{gray!25}{white}',
        r'    \setlength{\tabcolsep}{4pt}',
        r'    \begin{tabular}{ll' + 'c' * len(GROUPS) + '}',
        '        Model & Initialization & '
        + ' & '.join(group[0] for group in GROUPS) + r' \\',
        r'        \hline',
    ]
    for model, label, validation, result, stats in rows:
        values = []
        for i, (mean, std) in enumerate(stats):
            value = f'{mean:.1f} \\pm {std:.1f}'
            if round(mean, 1) == best_means[i]:
                value = rf'\mathbf{{{value}}}'
            values.append(f'${value}$')
        lines.append('        ' + ' & '.join(
            [model, label] + values) + r' \\')
    lines += [
        r'    \end{tabular}',
        r'    \label{tab:ood_quantitaive_comparison}',
        r'\end{table}',
    ]
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
