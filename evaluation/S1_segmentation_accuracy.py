"""S1: all Hann configurations; optionally export all splits and methods to CSV."""
import argparse
from pathlib import Path

import pandas as pd

import helper_functions as helpers

# =============================================================================
# Export results
# =============================================================================

def export_results(path):
    """Export scores and timing statistics for all configurations and splits."""
    records = []
    configs = list(helpers.configurations()) + [('Threshold', 'threshold', 448, 'none')]
    for c in configs:
        for split in helpers.SPLITS:
            for mode in (['none'] if c[0] == 'Threshold' else ['classical', 'hann']):
                df = helpers.load(c, split, mode)
                groups = [(None, None)] + list(df.index.droplevel('File Name').unique())
                for species, modality in groups:
                    group = helpers.subset(df, species, modality)
                    n, mean, std = helpers.stats(100 * group['Dice'])
                    _, time_mean, time_std = helpers.stats(group['Time (s)'])
                    records.append(dict(
                        model=c[0], backbone=c[1], tile=c[2], initialization=c[3],
                        split=split, inference=mode, species=species or 'overall',
                        modality=modality or 'all', n_images=n, dice_mean_pct=mean,
                        dice_sd_pp=std, time_mean_s=time_mean, time_sd_s=time_std,
                        sd_ddof=0, selected_by_validation_hann=c in helpers.selected_configurations(),
                        source=str(helpers.metrics_path(c, split, mode).relative_to(helpers.ROOT))))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-csv', type=Path, help='Export all configurations, splits, inference methods, and species/modality groups.')
    args = parser.parse_args()
    caption = (
        r'Image-level Dice scores (\%) are reported for every trained configuration, using Hann inference. '
        r'Each image has equal weight; values are mean $\pm$ population standard deviation. '
        r'$n$ denotes image count. Bold rows indicate the four configurations selected by '
        r'highest overall validation Dice for the main text, using Hann inference. '
        r'Test and generalization scores are not used for selection. '
        r'ViT backbone suffixes denote patch sizes. '
        r'Higher Dice values indicate greater overlap between predicted and manually annotated segmentation masks.'
    )
    headers = ['Model', 'Backbone', 'Tile', 'Initialization'] + [
        helpers.score_header(label, len(helpers.metadata_index(split)))
        for label, split in [('Validation Dice', 'val'), ('Test Dice', 'test'),
                            ('Generalization Dice', 'generalization')]]
    panels = []
    for family in helpers.BACKBONES:
        rows = []
        for c in helpers.configurations(family):
            identifiers = helpers.identifiers(c)
            scores = [helpers.cell(100 * helpers.load(c, split)['Dice']) for split in helpers.MAIN_SPLITS]
            if c in helpers.selected_configurations():
                identifiers = [rf'\textbf{{{value}}}' for value in identifiers]
                scores = [rf'$\mathbf{{{value[1:-1]}}}$' for value in scores]
            rows.append(identifiers + scores)
        panels.append(helpers.panel(f'S1. Segmentation accuracy: {family}.',
                              caption, headers, rows, f'tab:s1_{helpers.family_slug(family)}'))
    helpers.print_tables(panels, layout='columns', omit_columns=(0,))
    if args.results_csv:
        export_results(args.results_csv)


if __name__ == '__main__':
    main()
