"""S4: pretrained minus scratch, paired on identical images and configurations."""
import helper_functions as helpers

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    panels = []
    for family in helpers.BACKBONES:
        rows = []
        for scratch in helpers.configurations(family):
            if scratch[3] != 'scratch':
                continue
            pretrained = (*scratch[:3], 'pretrained')
            values = []
            for split in ['test', 'generalization']:
                differences = helpers.paired_difference(helpers.load(pretrained, split), helpers.load(scratch, split))
                values.append(helpers.cell(differences, signed=True))
            rows.append(helpers.identifiers(scratch)[:3] + values)
        headers = ['Model', 'Backbone', 'Tile'] + [
            helpers.score_header(label, len(helpers.metadata_index(split)))
            for label, split in [(r'Test $\Delta$', 'test'), (r'Generalization $\Delta$', 'generalization')]]
        panels.append(helpers.panel(f'S4. Effect of pretraining: {family}.',
                      r'Paired image-level Dice differences (pretrained minus scratch, percentage points) '
                      r'are reported for identical backbone, patch size, tile size, and Hann inference. '
                      r'SD is computed from paired image-level differences, not from differences of SDs. '
                      r'Positive values favor pretraining. ' + helpers.STAT_NOTE
                      + 'ViT backbone suffixes denote patch sizes.',
                      headers, rows, f'tab:s4_{helpers.family_slug(family)}'))

    helpers.print_tables(panels)


if __name__ == '__main__':
    main()
