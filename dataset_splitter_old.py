############################################################
## Colmap Workspace Preparation Script for 7Scenes Dataset##
############################################################
#  This script prepares multiple Colmap workspaces for     #
#  7Scenes by splitting a range of images into two datasets#
#  with varying degrees of symmetrical overlap.            #
############################################################

import os
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------
# 1. Ask for boundaries (start/end) and a central split point (e.g., 50%)
# 2. Ask for a comma-separated list of overlap percentages
# 3. For each overlap %, calculate how many images cross the center line
# 4. Generate unique workspace folders containing Dataset 1 & Dataset 2 for every overlap configuration
# ----------------------------------------------------------------------------------------------------------------

def prepare_7scenes_workspace(source_dir, base_workspace_dir):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    
    # Gather Images
    images = sorted(list(source.glob("*.color.png")))
    if not images:
        images = sorted(
            [p for p in source.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
        
    total_available = len(images)

    if total_available == 0:
        print(f"There are no images found in {source}")
        return

    print(f"Found {total_available} images in total.")
    print("-" * 60)
    
    # Get user configuration
    try:
        start_idx = int(input("Start Index (default 0): ") or 0)
        end_idx = int(input(f"End Index (default {total_available}): ") or total_available)
        split_center_pct = float(input("Split Center % (default 50): ") or 50.0)
        
        overlaps_input = input("Enter overlap percentages separated by commas (e.g., 10, 20, 30): ")
        overlaps = [float(x.strip()) for x in overlaps_input.split(",")]
    except ValueError:
        print("Invalid input. Please enter numbers appropriately.")
        return

    # Validate constraints
    start_idx = max(0, start_idx)
    end_idx = min(total_available, end_idx)
    total_range = end_idx - start_idx
    
    if total_range <= 0:
        print("Error: End index must be strictly greater than Start index.")
        return

    if split_center_pct < 0 or split_center_pct > 100:
        print("Error: Split center must be between 0 and 100.")
        return

    # Determine the absolute index of the "center" cut
    center_idx = start_idx + int(round(total_range * (split_center_pct / 100.0)))

    print("\n" + "=" * 60)
    print("                 BATCH SPLIT SUMMARY")
    print("=" * 60)
    print(f"Base Range : {total_range} images (Indices {start_idx} to {end_idx - 1})")
    print(f"Center Cut : Index {center_idx} ({split_center_pct}% mark)")

    # Process each requested overlap variation
    for overlap_pct in overlaps:
        # Calculate how many images make up this percentage
        overlap_count = int(round(total_range * (overlap_pct / 100.0)))
        
        # Split the overlap evenly across the center line
        left_overlap = overlap_count // 2
        right_overlap = overlap_count - left_overlap
        
        # Calculate array slices for Dataset 1
        ds1_start = start_idx
        ds1_end = min(end_idx, center_idx + right_overlap)
        
        # Calculate array slices for Dataset 2
        ds2_start = max(start_idx, center_idx - left_overlap)
        ds2_end = end_idx
        
        # Setup specific workspace folder
        workspace_name = f"overlap_{int(overlap_pct)}pct"
        workspace = base_workspace / workspace_name
        
        if workspace.exists():
            shutil.rmtree(workspace)
            
        ds1_img_dir = workspace / "dataset1" / "images"
        ds2_img_dir = workspace / "dataset2" / "images"
        ds1_img_dir.mkdir(parents=True, exist_ok=True)
        ds2_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Output info
        actual_overlap = max(0, ds1_end - ds2_start)
        print(f"\n--- Overlap: {overlap_pct}% ({workspace_name}) ---")
        print(f"  Dataset 1: {ds1_end - ds1_start} images (Indices {ds1_start} to {ds1_end - 1})")
        print(f"  Dataset 2: {ds2_end - ds2_start} images (Indices {ds2_start} to {ds2_end - 1})")
        print(f"  Overlap  : {actual_overlap} images shared in common")
        
        # Copy files
        for i in range(ds1_start, ds1_end):
            shutil.copy(images[i], ds1_img_dir / images[i].name)
        for i in range(ds2_start, ds2_end):
            shutil.copy(images[i], ds2_img_dir / images[i].name)

    print("\n" + "=" * 60)
    print(f"Successfully generated {len(overlaps)} workspaces inside '{base_workspace}'")

# ----------------------------------------------------------
# Execution
# ----------------------------------------------------------
if __name__ == "__main__":
    prepare_7scenes_workspace(
        source_dir="./dataset",
        base_workspace_dir="./colmap_workspace"
    )