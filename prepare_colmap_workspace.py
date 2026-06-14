############################################################
## Colmap Workspace Preparation Script for 7Scenes Dataset##
############################################################
#  This script prepares a Colmap workspace for 7Scenes     #
#  by splitting the images into two overlapping datasets.  #
############################################################
# Feel free to use, modify the source and workspace paths below as needed. :)
# Some explanations of the script's functionality are provided in the comments.
# Also if you found this useful, please consider starring the repo on GitHub! It would me a lot. Thanks!

# ------------------------------------------------------
# import necessary libraries
# ------------------------------------------------------

import os
import shutil
from pathlib import Path


# ----------------------------------------------------------------------------------------------------------------
# Find images in the dataset
# Use 7Scenes naming convention (*.color.png)
# Split based on user-defined ranges for Dataset 1 and Dataset 2
# Determine overlapping images based on the defined ranges
# Copies images to a new workspace structure for Colmap processing without modifying the original dataset.
# ----------------------------------------------------------------------------------------------------------------

def prepare_7scenes_workspace(source_dir, workspace_dir):
    source = Path(source_dir).resolve()
    workspace = Path(workspace_dir).resolve()
    images = sorted(list(source.glob("*.color.png")))
    
    if not images:
        images = sorted(
            [p for p in source.glob("*")
             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
    total_images = len(images)

    if total_images == 0:
        print(f"Theres no images found in {source}")
        return

    print(f"Found {total_images} images in total.")
    print("--------------------------------------------------")
    print("Define your overlapping split ranges (0 to {total_images - 1}):")
    try:
        ds1_start = int(input(f"Dataset 1 - Start Index (default 0): ") or 0)
        ds1_end = int(input(f"Dataset 1 - End Index: "))

        ds2_start = int(input(f"Dataset 2 - Start Index: "))
        ds2_end = int(input(f"Dataset 2 - End Index (default {total_images}): ") or total_images)

    except ValueError:
        print("Only integer values are allowed.")
        return
    ds1_indices = range(max(0, ds1_start), min(total_images, ds1_end))
    ds2_indices = range(max(0, ds2_start), min(total_images, ds2_end))
    overlap = set(ds1_indices).intersection(set(ds2_indices))

    print(f"\nSplit Summary:")
    print(f"  Dataset 1: {len(ds1_indices)} images (Indices {min(ds1_indices)} to {max(ds1_indices)})")
    print(f"  Dataset 2: {len(ds2_indices)} images (Indices {min(ds2_indices)} to {max(ds2_indices)})")
    print(f"  Overlap: {len(overlap)} images in common.")

    ds1_img_dir = workspace / "dataset1" / "images"
    ds2_img_dir = workspace / "dataset2" / "images"

    if workspace.exists():
        shutil.rmtree(workspace)

    ds1_img_dir.mkdir(parents=True, exist_ok=True)
    ds2_img_dir.mkdir(parents=True, exist_ok=True)

    for idx in ds1_indices:
        shutil.copy(images[idx], ds1_img_dir / images[idx].name)
    for idx in ds2_indices:
        shutil.copy(images[idx], ds2_img_dir / images[idx].name)


    print("\n")
    print(f"export WORKSPACE_PATH=\"{workspace}\"")
    print(f"export DATASET1_PATH=\"{workspace / 'dataset1'}\"")
    print(f"export DATASET2_PATH=\"{workspace / 'dataset2'}\"")
    print(f"export MERGED_PATH=\"{workspace / 'merged'}\"\n")


# ----------------------------------------------------------
# THIS IS IMPORTANT
# ----------------------------------------------------------
# Original dataset should be located in ./dataset
# Workspace will be created in ./colmap_workspace
# ----------------------------------------------------------

if __name__ == "__main__":
    prepare_7scenes_workspace(
        source_dir="./dataset",
        workspace_dir="./colmap_workspace"
    )