import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.append('../')
import config

# options
resnets = ['resnet34', 'resnet50', 'resnet101', 'resnet152']
vits = ['vit_small_patch16_224', 'vit_small_patch8_224', 
        'vit_base_patch16_224', 'vit_base_patch8_224']
tile_sizes = [224, 448]
pretrained = True

# initialize plot
plt.figure(figsize=(8, 8))

for i, backbone in enumerate(vits):# + vits):
    for j, tile_size in enumerate(tile_sizes):

        # read log file
        run_name = f"{backbone}_{'pretrained' if pretrained else 'scratch'}"
        log_file = os.path.join('..', config.LOG_PATH.format(run_name, tile_size))
        df = pd.read_csv(log_file)
        columns = df.columns

        # unpack values
        best_epoch = np.argmax(df['val_dice'].astype(float)) + 1
        epochs = df['epoch']
        val_loss = df['val_loss']
        val_dice = df['val_dice']

        color = ['Blues', 'Reds']['res' in backbone]
        color = ['viridis', 'plasma']['vit' in backbone]
        color = matplotlib.colormaps[color](((i+1)%5)/4)
        linestyle = ['-', '--'][j]

        # loss
        ax = plt.subplot(2, 1, 1)
        ax.semilogy(epochs, val_loss, 
            label=f'{backbone.replace('_224', '')} x {tile_size}', 
            linestyle=linestyle, linewidth=2, color=color)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylim(top=4e-1, bottom=2e-1)
        ax.minorticks_on()
        ax.grid()

        # dice
        ax = plt.subplot(2, 1, 2)
        ax.plot(epochs, val_dice, 
            label=f'{backbone.replace('_224', '')} x {tile_size}', 
            linestyle=linestyle, linewidth=2, color=color)
        ax.set_ylabel('Dice')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0.88, 0.98)
        ax.legend(loc='upper right')
        ax.minorticks_on()
        ax.grid()

plt.tight_layout()
plt.show()
