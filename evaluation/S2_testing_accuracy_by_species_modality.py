"""S2: test subgroup scores for all configurations, with adjacent initializations."""
import helper_functions as helpers

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    panels = []
    for family in helpers.BACKBONES:
        for modality in ['rgb1', 'rgb2']:
            panels.append(build_panel(family, modality))

    helpers.print_tables(panels, layout='grid', pages=[[0, 2, 1, 3]],
                   row_labels=[['RGB1', 'RGB2']], omit_columns=(0,))

# =============================================================================
# Table panels
# =============================================================================

def build_panel(family, modality):
    """Build the species and modality panel for one model family."""
    groups = [('Overall', None, None)] + [
        (rf'{species.title()}\\{modality.upper()}', species, modality)
        for species in ['poplar', 'switchgrass']]
    rows = []
    for c in helpers.configurations(family):
        df = helpers.load(c, 'test')
        rows.append(helpers.identifiers(c, mark=True) + [
            helpers.cell(100 * helpers.subset(df, species, modality)['Dice']) for _, species, modality in groups])
    headers = ['Model', 'Backbone', 'Tile', 'Initialization'] + [
        helpers.score_header(label, len(helpers.subset(df, species, modality))) for label, species, modality in groups]
    return helpers.panel(f'S2. Test accuracy by species and modality: {family}, {modality.upper()}.',
                  r'Image-level Dice scores (\%) are reported for all configurations using Hann inference. '
                  r'RGB1 is side-view and RGB2 is top-view. Overall includes all test images and is repeated across modality panels. '
                  + helpers.STAT_NOTE + helpers.SELECTION_NOTE
                  + 'ViT backbone suffixes denote patch sizes. '
                  + 'Higher Dice values indicate greater overlap between predicted and manually annotated segmentation masks.',
                  headers, rows, f'tab:s2_{helpers.family_slug(family)}_{modality}')


if __name__ == '__main__':
    main()
