import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


MASK_PATH = os.path.join(os.getcwd(), "data", "raw", "masks")
MASK_EXT = ".png"
SAVE_PATH = os.path.join(os.getcwd(), "data", "masks")
SAVE_EXT = ".memmap"
PROCESSES = 32
DTYPE = "bool"


def mask_to_memmap(mask_path: str) -> None:
    """Convert one RGB mask image to a boolean memmap file."""
    name = os.path.basename(mask_path)
    sub_dir = os.path.dirname(mask_path).split(os.sep)[-2]
    save_dir = os.path.join(SAVE_PATH, sub_dir)
    save_name = os.path.join(save_dir, name.replace(MASK_EXT, SAVE_EXT))

    if os.path.exists(save_name):
        return

    mask = np.array(Image.open(mask_path).convert("RGB"))
    mask = mask.max(axis=2) > 128

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    memmap = np.memmap(save_name, dtype=DTYPE, mode="w+", shape=mask.shape)
    memmap[:] = mask[:]
    del memmap


def main() -> None:
    mask_paths = sorted(
        glob.glob(
            os.path.join(MASK_PATH, "**", "SegmentationClass", "*" + MASK_EXT),
            recursive=True,
        )
    )
    with Pool(PROCESSES) as pool:
        list(
            tqdm(
                pool.imap(mask_to_memmap, mask_paths),
                total=len(mask_paths),
                desc="Converting masks to memmaps",
            )
        )


if __name__ == "__main__":
    main()
