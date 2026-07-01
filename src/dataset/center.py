#-------------------------------------------------------------------------------
# 7Scenes COLMAP Center-Frame Splitter Module
# Splits a dataset into separate sub-folders based on varying center frames
# with a configurable fixed overlap (either percentage or frame counts).
# Designed for standalone execution and multi-module integration.
#-------------------------------------------------------------------------------

import os
import shutil
from pathlib import Path

def split_by_center_frames(source_dir, base_workspace_dir, center_frames, overlap_value, overlap_mode="frames", start_frame=0, end_frame=None):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    
    images = sorted(list(source.glob("*.color.png")))
    if not images:
        images = sorted([p for p in source.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        
    total_available = len(images)
    if total_available == 0:
        print(f"Error: No images found in directory: {source}")
        return False

    if end_frame is None:
        end_frame = total_available

    start_idx = max(0, min(total_available - 1, start_frame))
    end_idx = max(start_idx + 1, min(total_available, end_frame))
    total_sequence_len = end_idx - start_idx

    if overlap_mode == "percentage":
        overlap_frames = int(round(total_sequence_len * (overlap_value / 100.0)))
    else:
        overlap_frames = int(overlap_value)

    print(f"\nProcessing configurations for {len(center_frames)} center frame variants...")

    for center_frame in center_frames:
        center_frame = max(start_idx, min(end_idx, center_frame))
        
        left_overlap = overlap_frames // 2
        right_overlap = overlap_frames - left_overlap
        
        ds1_start = start_idx
        ds1_end = min(end_idx, center_frame + right_overlap)
        
        ds2_start = max(start_idx, center_frame - left_overlap)
        ds2_end = end_idx
        
        ds1_indices = range(ds1_start, ds1_end)
        ds2_indices = range(ds2_start, ds2_end)
        
        if not ds1_indices or not ds2_indices:
            print(f"Warning: Invalid frame range generated for center frame {center_frame}. Skipping.")
            continue

        if overlap_mode == "percentage":
            folder_suffix = f"center_{center_frame}_overlap_{int(overlap_value)}pct" if overlap_value.is_integer() else f"center_{center_frame}_overlap_{overlap_value}pct"
        else:
            folder_suffix = f"center_{center_frame}_overlap_{overlap_frames}f"
            
        workspace = base_workspace / folder_suffix
        
        if workspace.exists():
            shutil.rmtree(workspace)
            
        # Unified tracking directories structure
        unified_img_dir = workspace / "images"
        sparse1_dir = workspace / "sparse1"
        sparse2_dir = workspace / "sparse2"
        merged_dir = workspace / "merged"
        
        for directory in [unified_img_dir, sparse1_dir, sparse2_dir, merged_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        overlap_set = set(ds1_indices).intersection(set(ds2_indices))
        union_indices = sorted(list(set(ds1_indices).union(set(ds2_indices))))
        
        print(f" -> Generated: {folder_suffix}")
        print(f"    - Dataset 1 Range : {min(ds1_indices)} to {max(ds1_indices)} ({len(ds1_indices)} frames)")
        print(f"    - Dataset 2 Range : {min(ds2_indices)} to {max(ds2_indices)} ({len(ds2_indices)} frames)")
        print(f"    - Common Overlap  : {len(overlap_set)} frames total")
        
        # Copy union dataset to the unified images folder
        for idx in union_indices:
            shutil.copy(images[idx], unified_img_dir / images[idx].name)
            
        # Write specific image tracking lists
        ds1_list_path = workspace / "dataset1_list.txt"
        ds2_list_path = workspace / "dataset2_list.txt"
        
        with open(ds1_list_path, "w") as f:
            for idx in ds1_indices:
                f.write(f"{images[idx].name}\n")
                
        with open(ds2_list_path, "w") as f:
            for idx in ds2_indices:
                f.write(f"{images[idx].name}\n")

    print(f"\nSuccessfully generated workspaces inside: '{base_workspace}'")
    return True

if __name__ == "__main__":
    print("==================================================")
    print("         7SCENES CENTER-FRAME SPLITTER            ")
    print("==================================================")
    print("Select Overlap Specification Mode:")
    print("1. Define Fixed Overlap by Percentage")
    print("2. Define Fixed Overlap by Frame Count")
    print("--------------------------------------------------")
    
    overlap_choice = input("Enter choice (1-2): ").strip()
    if overlap_choice not in ["1", "2"]:
        print("Error: Invalid selection.")
        exit(1)
        
    print("--------------------------------------------------")
    source_input = input("Enter Dataset Source Directory (default: ./dataset): ").strip()
    source_dir = source_input if source_input else "./dataset"
    
    workspace_input = input("Enter Base Workspace Directory (default: ./colmap_workspace): ").strip()
    base_workspace_dir = workspace_input if workspace_input else "./colmap_workspace"
    print("--------------------------------------------------")

    source_path = Path(source_dir).resolve()
    images_list = sorted(list(source_path.glob("*.color.png")))
    if not images_list:
        images_list = sorted([p for p in source_path.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    total_imgs = len(images_list)
    
    if total_imgs == 0:
        print(f"Error: No images found in {source_path}")
        exit(1)
        
    print(f"Found {total_imgs} total images in sequence.")
    print("--------------------------------------------------")
    
    try:
        start_fr = int(input("Global Sequence Start Frame (default 0): ") or 0)
        end_fr = int(input(f"Global Sequence End Frame (default {total_imgs}): ") or total_imgs)
        print("--------------------------------------------------")
        
        if overlap_choice == "1":
            overlap_val = float(input("Enter fixed overlap size in percentage (e.g., 10): ") or 10.0)
            mode = "percentage"
        else:
            overlap_val = int(input("Enter fixed overlap size in frames (e.g., 20): ") or 20)
            mode = "frames"
            
        centers_str = input("Enter target center frame indices separated by commas (e.g., 50, 100, 150): ")
        center_vals = [int(c.strip()) for c in centers_str.split(",") if c.strip()]
        
    except ValueError:
        print("Error: Input formatting error. Please ensure numbers are typed correctly.")
        exit(1)
        
    split_by_center_frames(
        source_dir=source_dir,
        base_workspace_dir=base_workspace_dir,
        center_frames=center_vals,
        overlap_value=overlap_val,
        overlap_mode=mode,
        start_frame=start_fr,
        end_frame=end_fr
    )