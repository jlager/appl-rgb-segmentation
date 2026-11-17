"""S5: Hann minus classical inference for each fixed trained configuration."""
import helper_functions as helpers

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    panels = []
    for family in helpers.BACKBONES:
        rows = []
        for c in helpers.configurations(family):
            values = []
            for split in ['test', 'generalization']:
                differences = helpers.paired_difference(helpers.load(c, split, 'hann'), helpers.load(c, split, 'classical'))
                values.append(helpers.cell(differences, signed=True))
            rows.append(helpers.identifiers(c) + values)
        headers = ['Model', 'Backbone', 'Tile', 'Initialization'] + [
            helpers.score_header(label, len(helpers.metadata_index(split)))
            for label, split in [(r'Test $\Delta$', 'test'), (r'Generalization $\Delta$', 'generalization')]]
        panels.append(helpers.panel(f'S5. Effect of overlapping inference: {family}.',
                      r'Paired image-level Dice differences (Hann minus classical non-overlapping inference, '
                      r'percentage points) are reported for the same trained configuration and identical images. '
                      r'SD is computed from paired image-level differences. Positive values favor Hann. '
                      + helpers.STAT_NOTE + 'ViT backbone suffixes denote patch sizes.',
                      headers, rows, f'tab:s5_{helpers.family_slug(family)}'))

    helpers.print_tables(panels)


if __name__ == '__main__':
    main()
