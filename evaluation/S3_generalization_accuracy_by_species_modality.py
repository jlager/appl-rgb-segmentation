"""S3: family/species panels expose every observed generalization modality."""
import helper_functions as helpers

# =============================================================================
# Table
# =============================================================================

def main() -> None:
    """Print the table as LaTeX."""
    panels = []
    for family in helpers.BACKBONES:
        for species in helpers.SPECIES:
            reference = helpers.load(next(helpers.configurations(family)), 'generalization')
            modalities = [m for m in helpers.MODALITIES if len(helpers.subset(reference, species, m))]
            for start in range(0, len(modalities), 2):
                panels.append(build_panel(family, species, reference, modalities[start:start + 2], start // 2 + 1))

    # Each page pairs U-Net and ViT for two species/modality cases.
    helpers.print_tables(panels, layout='grid',
                   pages=[[0, 6, 1, 7], [2, 8, 3, 9], [4, 10, 5, 11]],
                   row_labels=[['Sorghum, RGB1', 'Sorghum, RGB2'],
                               ['Soybean', 'Eucalyptus'], ['Arabidopsis', 'Pennycress']],
                   omit_columns=(0,))

# =============================================================================
# Table panels
# =============================================================================

def build_panel(family, species, reference, modalities, part):
    """Build the species and modality panel for one model family."""
    rows = []
    for c in helpers.configurations(family):
        df = helpers.load(c, 'generalization')
        rows.append(helpers.identifiers(c, mark=True) + [helpers.cell(100 * df['Dice'])] + [
            helpers.cell(100 * helpers.subset(df, species, m)['Dice']) for m in modalities])
    headers = ['Model', 'Backbone', 'Tile', 'Initialization', helpers.score_header('Overall', len(reference))] + [
        helpers.score_header(m.upper(), len(helpers.subset(reference, species, m))) for m in modalities]
    return helpers.panel(f'S3. Generalization accuracy by species and modality: {family}, {species.title()}, panel {part}.',
                  r'Image-level Dice scores (\%) are reported for all configurations using Hann inference. '
                  r'Overall includes all generalization images and is repeated across species panels. '
                  r'Arabidopsis and pennycress have only RGB2; eucalyptus and soybean have RGB1 and RGB2; '
                  r'sorghum additionally has RGB1-rhizo and RGB2-rhizo. ' + helpers.STAT_NOTE + helpers.SELECTION_NOTE
                  + 'ViT backbone suffixes denote patch sizes. '
                  + 'Higher Dice values indicate greater overlap between predicted and manually annotated segmentation masks.',
                  headers, rows, f'tab:s3_{helpers.family_slug(family)}_{species}_{part}')


if __name__ == '__main__':
    main()
