"""S6: actual per-image timing measurements on the generalization split."""
import helper_functions as helpers

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    panels = []
    for family in helpers.BACKBONES:
        for initialization in ['scratch', 'pretrained']:
            rows = []
            for c in helpers.configurations(family):
                if c[3] != initialization:
                    continue
                for mode in ['classical', 'hann']:
                    df = helpers.load(c, 'generalization', mode)
                    groups = [df] + [df.loc[df.index.get_level_values('Modality').isin(modalities)]
                                     for modalities in [('rgb1', 'rgb1-rhizo'), ('rgb2', 'rgb2-rhizo')]]
                    rows.append(helpers.identifiers(c) + [mode.capitalize()] + [
                        helpers.cell(group['Time (s)'], decimals=2) for group in groups])
            headers = ['Model', 'Backbone', 'Tile', 'Initialization', 'Inference'] + [
                helpers.score_header(label, len(group)) for label, group in zip(['Overall', 'RGB1', 'RGB2'], groups)]
            panels.append(helpers.panel(f'S6. Inference time: {family}, {initialization}.',
                          r'Per-image inference times (seconds) are reported on the generalization split. '
                          r'Each row reports one measured run, without averaging across configurations or initializations. '
                          r'Initialization identifies the measured run, not a speed-related factor. '
                          r'RGB1 includes RGB1-rhizo; RGB2 includes RGB2-rhizo. '
                          + helpers.STAT_NOTE + 'ViT backbone suffixes denote patch sizes. Lower values indicate faster inference.',
                          headers, rows, f'tab:s6_{helpers.family_slug(family)}_{initialization}'))

    helpers.print_tables(panels, layout='grid', pages=[[0, 2, 1, 3]],
                   row_labels=[['Scratch', 'Pretrained']], omit_columns=(0, 3))


if __name__ == '__main__':
    main()
