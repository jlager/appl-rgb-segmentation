import glob
import os
from multiprocessing import Pool

import numpy as np
from scipy import ndimage
from tqdm.auto import tqdm


TILE_SIZE = 448  # 224, 448
N_POINTS = 10000
SEED = 0
PROCESSES = 8
RGB1_HEIGHT = 6556
RGB1_WIDTH = 4104
RGB2_HEIGHT = 3006
RGB2_WIDTH = 4104
DTYPE = "bool"
MASK_PATH = os.path.join(os.getcwd(), "data", "masks")
MASK_EXT = ".memmap"
SAVE_PATH = os.path.join(os.getcwd(), "data", f"rois_{TILE_SIZE}")
SAVE_EXT = ".npz"


def mask_to_roi(mask_path: str) -> None:
    """Convert one mask memmap into sampled foreground/background ROI indices."""
    np.random.seed(SEED)

    name = os.path.basename(mask_path)
    sub_dir = os.path.dirname(mask_path).split(os.sep)[-1]
    save_dir = os.path.join(SAVE_PATH, sub_dir)
    save_name = os.path.join(save_dir, name.replace(MASK_EXT, SAVE_EXT))

    os.makedirs(SAVE_PATH, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if "rgb1" in sub_dir:
        w, h = RGB1_WIDTH, RGB1_HEIGHT
    if "rgb2" in sub_dir:
        w, h = RGB2_WIDTH, RGB2_HEIGHT
    memmap = np.memmap(mask_path, dtype=DTYPE, mode="r", shape=(h, w))
    mask = np.array(memmap)
    del memmap

    edt = ndimage.distance_transform_edt(~mask)
    edt[edt < TILE_SIZE] = 1
    edt[edt >= TILE_SIZE] = 0
    edt[: TILE_SIZE // 2, :] = -1
    edt[-TILE_SIZE // 2 :, :] = -1
    edt[:, : TILE_SIZE // 2] = -1
    edt[:, -TILE_SIZE // 2 :] = -1

    foreground_idx = np.flatnonzero(edt == 1).astype(np.int32)
    background_idx = np.flatnonzero(edt == 0).astype(np.int32)
    foreground_idx = np.random.choice(
        foreground_idx, size=min(N_POINTS, len(foreground_idx)), replace=False
    )
    background_idx = np.random.choice(
        background_idx, size=min(N_POINTS, len(background_idx)), replace=False
    )

    np.savez_compressed(save_name, f_idx=foreground_idx, b_idx=background_idx)


def main() -> None:
    mask_paths = sorted(
        glob.glob(os.path.join(MASK_PATH, "**", "*" + MASK_EXT), recursive=True)
    )
    with Pool(PROCESSES) as pool:
        list(
            tqdm(
                pool.imap(mask_to_roi, mask_paths),
                total=len(mask_paths),
                desc="Converting masks to rois",
            )
        )


if __name__ == "__main__":
    main()
