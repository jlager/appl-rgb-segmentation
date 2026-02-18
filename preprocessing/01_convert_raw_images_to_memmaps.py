import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


IMAGE_PATH = os.path.join(os.getcwd(), "data", "raw", "images")
IMAGE_EXT = ".png"
SAVE_PATH = os.path.join(os.getcwd(), "data", "images")
SAVE_EXT = ".memmap"
PROCESSES = 32
DTYPE = "uint8"


def image_to_memmap(image_path: str) -> None:
    """Convert one RGB image to a memmap file."""
    name = os.path.basename(image_path)
    sub_dir = os.path.dirname(image_path).split(os.sep)[-1]
    save_dir = os.path.join(SAVE_PATH, sub_dir)
    save_name = os.path.join(save_dir, name.replace(IMAGE_EXT, SAVE_EXT))

    if os.path.exists(save_name):
        return

    image = np.array(Image.open(image_path).convert("RGB"))
    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    memmap = np.memmap(save_name, dtype=DTYPE, mode="w+", shape=image.shape)
    memmap[:] = image[:]
    del memmap


def main() -> None:
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_PATH, "**", "*" + IMAGE_EXT), recursive=True)
    )
    with Pool(PROCESSES) as pool:
        list(
            tqdm(
                pool.imap(image_to_memmap, image_paths),
                total=len(image_paths),
                desc="Converting images to memmaps",
            )
        )


if __name__ == "__main__":
    main()
