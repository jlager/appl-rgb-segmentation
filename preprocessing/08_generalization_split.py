import os

import pandas as pd

import config


IMAGE_DIR = os.path.join(os.getcwd(), config.IMAGE_DIR)
MASK_DIR = os.path.join(os.getcwd(), config.MASK_DIR)
SAVE_DIR = os.path.join(os.getcwd(), config.SPLIT_DIR)
SAVE_NAME = "generalization.csv"
EXCLUDED_SUBDIRS = {
    "rgb1-poplar",
    "rgb1-switchgrass",
    "rgb2-poplar",
    "rgb2-switchgrass",
}


def main() -> None:
    image_subdirs = {
        d for d in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, d))
    }
    mask_subdirs = {
        d for d in os.listdir(MASK_DIR) if os.path.isdir(os.path.join(MASK_DIR, d))
    }

    candidate_subdirs = sorted((image_subdirs & mask_subdirs) - EXCLUDED_SUBDIRS)
    rows = []
    missing_masks = 0

    for subdir in candidate_subdirs:
        modality, species = subdir.split("-", 1)
        image_subdir = os.path.join(IMAGE_DIR, subdir)
        mask_subdir = os.path.join(MASK_DIR, subdir)

        image_names = sorted(
            [n for n in os.listdir(image_subdir) if n.endswith(config.DATA_EXT)]
        )

        for image_name in image_names:
            mask_path = os.path.join(mask_subdir, image_name)
            if not os.path.isfile(mask_path):
                missing_masks += 1
                continue

            rows.append(
                {
                    "Species": species,
                    "Modality": modality,
                    "File Name": image_name.replace(config.DATA_EXT, config.IMAGE_EXT),
                }
            )

    df_new = pd.DataFrame(rows, columns=["Species", "Modality", "File Name"])
    df_new = df_new.sort_values(["Modality", "Species", "File Name"]).reset_index(drop=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, SAVE_NAME)
    df_new.to_csv(save_path, index=False)

    print(f"Saved {len(df_new)} image/mask pairs to {save_path}")
    print(f"Included folders ({len(candidate_subdirs)}): {candidate_subdirs}")
    print(f"Images skipped due to missing mask files: {missing_masks}")


if __name__ == "__main__":
    main()
