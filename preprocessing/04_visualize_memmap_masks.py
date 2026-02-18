import glob
import os

import matplotlib.pyplot as plt
import numpy as np


RGB1_HEIGHT = 6556
RGB1_WIDTH = 4104
RGB2_HEIGHT = 3006
RGB2_WIDTH = 4104
DTYPE = "bool"
MEMMAP_PATH = os.path.join(os.getcwd(), "data", "masks")
MEMMAP_EXT = ".memmap"


def main() -> None:
    for sub_dir in sorted(os.listdir(MEMMAP_PATH)):
        if "rgb1" in sub_dir:
            w, h = RGB1_WIDTH, RGB1_HEIGHT
        if "rgb2" in sub_dir:
            w, h = RGB2_WIDTH, RGB2_HEIGHT

        paths = sorted(glob.glob(os.path.join(MEMMAP_PATH, sub_dir, "*" + MEMMAP_EXT)))
        memmap = np.memmap(paths[0], dtype=DTYPE, mode="r", shape=(h, w))
        mask = np.array(memmap)
        del memmap

        fig, ax = plt.subplots(1, 1)
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{sub_dir}")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
