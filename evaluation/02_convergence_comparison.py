import os
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append('../')
import config

# options
resnets = ['resnet34', 'resnet50', 'resnet101', 'resnet152']
vits = ['vit_small_patch16_224', 'vit_small_patch8_224', 
        'vit_base_patch16_224', 'vit_base_patch8_224']
tile_sizes = [224, 448]
model_groups = [
    ('U-Net variants (from scratch)', resnets, False, 'Blues'),
    ('ViT variants (pre-trained)', vits, True, 'Reds'),
]
variant_labels = {
    'resnet34': 'ResNet-34',
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'resnet152': 'ResNet-152',
    'vit_small_patch16_224': 'ViT-small/16',
    'vit_small_patch8_224': 'ViT-small/8',
    'vit_base_patch16_224': 'ViT-base/16',
    'vit_base_patch8_224': 'ViT-base/8',
}
tile_linestyles = {
    224: '-',
    448: '--',
}

# initialize plot
fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey='row')
variant_handles_by_col = []

for col, (group_label, backbones, pretrained, colormap) in enumerate(model_groups):
    loss_ax = axs[0, col]
    dice_ax = axs[1, col]
    loss_ax.set_title(group_label)
    variant_handles = []

    for i, backbone in enumerate(backbones):
        color = matplotlib.colormaps[colormap](0.35 + 0.5 * i / (len(backbones) - 1))
        legend_label = variant_labels[backbone]
        if backbone.startswith('resnet'):
            legend_label = f'U-Net {legend_label}'
        variant_handles.append(Line2D(
            [0], [0],
            color=color,
            linewidth=2,
            label=legend_label,
        ))

        for j, tile_size in enumerate(tile_sizes):

            # read log file
            run_name = f"{backbone}_{'pretrained' if pretrained else 'scratch'}"
            log_file = os.path.join('..', config.LOG_PATH.format(run_name, tile_size))
            df = pd.read_csv(log_file)

            # unpack values
            epochs = df['epoch']
            val_loss = df['val_loss']
            val_dice = df['val_dice']

            linestyle = tile_linestyles[tile_size]

            # loss
            loss_ax.semilogy(epochs, val_loss,
                linestyle=linestyle, linewidth=2, color=color)

            # dice
            dice_ax.plot(epochs, val_dice,
                linestyle=linestyle, linewidth=2, color=color)

    variant_handles_by_col.append(variant_handles)

for ax in axs.flat:
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)

axs[0, 0].set_ylabel('Validation loss')
axs[1, 0].set_ylabel('Validation Dice')
axs[1, 0].set_xlabel('Epoch')
axs[1, 1].set_xlabel('Epoch')
axs[1, 0].set_ylim(0.0, 1.0)

tile_handles = [
    Line2D([0], [0], color='black', linestyle=linestyle, linewidth=2,
           label=f'{tile_size} px tile')
    for tile_size, linestyle in tile_linestyles.items()
]
for col, variant_handles in enumerate(variant_handles_by_col):
    axs[1, col].legend(
        handles=variant_handles + tile_handles,
        loc='center',
        frameon=True,
        fontsize='small',
    )
fig.tight_layout()

figure_dir = os.path.join('..', 'outputs', 'figures')
os.makedirs(figure_dir, exist_ok=True)
figure_stem = os.path.join(
    figure_dir,
    'convergence_comparison_unet-scratch_vit-pretrained')
for ext in ['png', 'pdf']:
    figure_file = f'{figure_stem}.{ext}'
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    print(f'Saved {figure_file}')
