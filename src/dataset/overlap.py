#-------------------------------------------------------------------------------
# COLMAP Dataset Overlap Variation Splitter Module
# Splits a dataset based on a fixed center frame with varying overlap 
# percentages or varying overlap frame counts.
# Designed for standalone execution and multi-module integration.
#-------------------------------------------------------------------------------

import os
import shutil
from pathlib import Path

def split_dataset_varying_overlap(source_dir, base_workspace_dir, split_mode="percentage", split_values=None, start_frame=0, end_frame=None, center_point=None):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    
    images = sorted(list(source.glob("*.color.png")))
    if not images:
        images = sorted([p for p in source.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        
    total_images = len(images)
    if total_images == 0:
        print(f"Error: No images found in directory: {source}")
        return False

    if end_frame is None:
        end_frame = total_images
    if center_point is None:
        center_point = int((start_frame + end_frame) / 2)

    start_frame = max(0, min(total_images - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_images, end_frame))
    center_point = max(start_frame, min(end_frame, center_point))
    total_sequence_len = end_frame - start_frame

    if not split_values:
        print("Error: No target split values provided.")
        return False

    print(f"\nProcessing tracking configurations for {len(split_values)} variants...")

    for val in split_values:
        if split_mode == "percentage":
            overlap_size = int(round(total_sequence_len * (val / 100.0)))
            folder_suffix = f"overlap_{int(val)}pct" if val.is_integer() else f"overlap_{val}pct"
        elif split_mode == "frames":
            overlap_size = int(val)
            folder_suffix = f"overlap_{overlap_size}f"
        else:
            print("Error: Unrecognized splitting mode selected.")
            return False

        half_overlap = overlap_size // 2
        
        ds1_start = start_frame
        ds1_end = min(end_frame, center_point + (overlap_size - half_overlap))
        
        ds2_start = max(start_frame, center_point - half_overlap)
        ds2_end = end_frame
        
        ds1_indices = range(ds1_start, ds1_end)
        ds2_indices = range(ds2_start, ds2_end)
        
        if not ds1_indices or not ds2_indices:
            print(f"Warning: Invalid frame range generated for value {val}. Skipping.")
            continue

        overlap_set = set(ds1_indices).intersection(set(ds2_indices))
        union_indices = sorted(list(set(ds1_indices).union(set(ds2_indices))))
        
        current_workspace = base_workspace / folder_suffix
        if current_workspace.exists():
            shutil.rmtree(current_workspace)
            
        unified_img_dir = current_workspace / "images"
        sparse1_dir = current_workspace / "sparse1"
        sparse2_dir = current_workspace / "sparse2"
        merged_dir = current_workspace / "merged"
        
        for directory in [unified_img_dir, sparse1_dir, sparse2_dir, merged_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        for idx in union_indices:
            shutil.copy(images[idx], unified_img_dir / images[idx].name)
            
        ds1_list_path = current_workspace / "dataset1_list.txt"
        ds2_list_path = current_workspace / "dataset2_list.txt"
        
        with open(ds1_list_path, "w") as f:
            for idx in ds1_indices:
                f.write(f"{images[idx].name}\n")
                
        with open(ds2_list_path, "w") as f:
            for idx in ds2_indices:
                f.write(f"{images[idx].name}\n")
                
        print(f" -> Generated: {folder_suffix}")
        print(f"    - Dataset 1 Range : {min(ds1_indices)} to {max(ds1_indices)} ({len(ds1_indices)} frames)")
        print(f"    - Dataset 2 Range : {min(ds2_indices)} to {max(ds2_indices)} ({len(ds2_indices)} frames)")
        print(f"    - Common Overlap  : {len(overlap_set)} frames total")
        
    print(f"\nSuccessfully generated workspaces inside: '{base_workspace}'")
    return True

if __name__ == "__main__":
    print("==================================================")
    print("       7SCENES OVERLAP PERCENTAGE SPLITTER        ")
    print("==================================================")
    print("Select Overlap Specification Mode:")
    print("1. Define Varying Overlap by Percentage")
    print("2. Define Varying Overlap by Frame Count")
    print("--------------------------------------------------")
    
    choice = input("Enter choice (1-2): ").strip()
    if choice not in ["1", "2"]:
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
        
        default_center = int((start_fr + end_fr) / 2)
        center_pt = int(input(f"Enter fixed center frame index (e.g., {default_center}): ") or default_center)
        
        if choice == "1":
            vals_str = input("Enter varying overlap percentages separated by commas (e.g., 10,20,30): ")
            split_vals = [float(v.strip()) for v in vals_str.split(",") if v.strip()]
            mode = "percentage"
        else:
            vals_str = input("Enter varying overlap frame counts separated by commas (e.g., 10,20,30): ")
            split_vals = [int(v.strip()) for v in vals_str.split(",") if v.strip()]
            mode = "frames"
            
    except ValueError:
        print("Error: Input formatting error. Please ensure numbers are typed correctly.")
        exit(1)
        
    split_dataset_varying_overlap(
        source_dir=source_dir,
        base_workspace_dir=base_workspace_dir,
        split_mode=mode,
        split_values=split_vals,
        start_frame=start_fr,
        end_frame=end_fr,
        center_point=center_pt
    )