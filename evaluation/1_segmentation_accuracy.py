from helper_functions import CATEGORIES, MAIN_SPLITS, selected_results

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    rows = [
        (model, label, selected_results(model, init))
        for model, label, init in CATEGORIES
    ]
    best_means = [max(round(results[i]['mean'], 1) for _, _, results in rows)
                  for i in range(len(MAIN_SPLITS))]
    lines = [
        r'\begin{table}[ht]',
        r'    \centering',
        r'    \caption{\textbf{Segmentation accuracy.} Image-level Dice scores '
        r'(\%, mean $\pm$ population standard deviation) are '
        r'reported for the threshold baseline and each model and initialization '
        r'category. All trained models use Hann inference. '
        r'ViT backbone suffixes denote patch sizes. '
        r'Within each trained category, the configuration with the highest '
        r'mean validation Dice is selected across backbones and tile sizes. '
        r'The same configuration is evaluated on validation, test, and '
        r'generalization datasets. Higher Dice is better; '
        r'best displayed Dice means for each dataset are bold.}',
        r'    \renewcommand{\arraystretch}{1.5}',
        r'    \rowcolors{2}{gray!25}{white}',
        r'    \begin{tabular}{llllccc}',
        r'        Model & Initialization & Backbone & Tile & Validation Dice & Test Dice & Generalization Dice \\',
        r'        \hline',
    ]
    for model, label, results in rows:
        values = []
        for i, result in enumerate(results):
            value = f"{result['mean']:.1f} \\pm {result['std']:.1f}"
            if round(result['mean'], 1) == best_means[i]:
                value = rf'\mathbf{{{value}}}'
            values.append(f'${value}$')
        lines.append('        ' + ' & '.join(
            [model, label, results[0]['backbone'], str(results[0]['tile'])] + values) + r' \\')
    lines += [
        r'    \end{tabular}',
        r'    \label{tab:strategy_comparison}',
        r'\end{table}',
    ]
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
