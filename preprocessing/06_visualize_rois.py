import glob
import os

import matplotlib.pyplot as plt
import numpy as np


TILE_SIZE = 448  # 224, 448
RGB1_HEIGHT = 6556
RGB1_WIDTH = 4104
RGB2_HEIGHT = 3006
RGB2_WIDTH = 4104
DTYPE = "bool"
MASK_PATH = os.path.join(os.getcwd(), "data", "masks")
MASK_EXT = ".memmap"
ROI_PATH = os.path.join(os.getcwd(), "data", f"rois_{TILE_SIZE}")
ROI_EXT = ".npz"


def main() -> None:
    fig, axs = plt.subplots(2, 2, figsize=(5, 7.5))
    axs = axs.flatten()

    # Keep notebook behavior: show first 4 subdirectories.
    for i, sub_dir in enumerate(sorted(os.listdir(MASK_PATH))[:4]):
        if "rgb1" in sub_dir:
            w, h = RGB1_WIDTH, RGB1_HEIGHT
        if "rgb2" in sub_dir:
            w, h = RGB2_WIDTH, RGB2_HEIGHT
        paths = sorted(glob.glob(os.path.join(MASK_PATH, sub_dir, "*" + MASK_EXT)))
        memmap = np.memmap(paths[0], dtype=DTYPE, mode="r", shape=(h, w))
        mask = np.array(memmap)
        del memmap

        roi_paths = sorted(glob.glob(os.path.join(ROI_PATH, sub_dir, "*" + ROI_EXT)))
        roi = np.load(roi_paths[0])
        f_idx = roi["f_idx"]
        b_idx = roi["b_idx"]

        axs[i].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axs[i].scatter(f_idx % w, f_idx // w, s=0.1, c="red", label="foreground", alpha=0.1)
        axs[i].scatter(b_idx % w, b_idx // w, s=0.1, c="blue", label="background", alpha=1)
        axs[i].set_title(f"{sub_dir}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
