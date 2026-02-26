
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('../')
import config

# options
backbone = 'vit_base_patch8_224' # 'resnet34' or 'vit_small_patch8_224'
tile_size = 448
log_file = os.path.join('..', config.LOG_PATH.format(backbone, tile_size))

# read log file
df = pd.read_csv(log_file)
columns = df.columns

# unpack values
best_epoch = np.argmax(df['val_dice'].astype(float)) + 1
x = df['epoch']

# initialize plot
plt.figure(figsize=(10, 12))

# learning rate(s)
plt.subplot(3, 1, 1)
plt.plot(x, df['encoder_lr'], label='Encoder LR', linewidth=2)
if not np.isnan(df['decoder_lr']).any().item():
    plt.plot(x, df['decoder_lr'], label='Decoder LR', linewidth=2, linestyle='--')
plt.axvline(best_epoch, color='gray', linestyle='--', label='Best Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate')
plt.legend(loc='upper left')
plt.grid()

# train/val losses
plt.subplot(3, 1, 2)
for i, label in enumerate(['train_loss', 'train_ce_loss', 'train_dice_loss']):
    y, ls, lw = df[label], ['-', '--', ':'][i], 2 if i == 0 else 1
    label = label.replace('_', ' ').title().replace('Ce', 'CE')
    y = y / config.GRADIENT_ACCUMULATION_STEPS
    plt.semilogy(x, y, label=label, linestyle=ls, color='tab:blue', linewidth=lw)
for i, label in enumerate(['val_loss', 'val_ce_loss', 'val_dice_loss']):
    y, ls, lw = df[label], ['-', '--', ':'][i], 2 if i == 0 else 1
    label = label.replace('_', ' ').title().replace('Ce', 'CE')
    plt.semilogy(x, y, label=label, linestyle=ls, color='tab:orange', linewidth=lw)
plt.axvline(best_epoch, color='gray', linestyle='--', label='Best Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(loc='lower left')
plt.grid()

# train/val dice scores
plt.subplot(3, 1, 3)
plt.plot(x, df['train_dice'], label='Train Dice', linestyle='-', color='tab:blue', linewidth=2)
plt.plot(x, df['val_dice'], label='Val Dice', linestyle='-', color='tab:orange', linewidth=2)
plt.axvline(best_epoch, color='gray', linestyle='--', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.title(f'Dice Score (Best: {df["val_dice"][best_epoch-1]:.4f} at epoch {best_epoch})')
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()