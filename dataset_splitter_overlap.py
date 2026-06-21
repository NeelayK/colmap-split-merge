
#-----------------------------------------------
# Use this for splittng dataset based on diff.
# overlap percentages.
# note: keep all images with .pose files in
#       a folder ./dataset in root
#-----------------------------------------------

import os
import shutil
from pathlib import Path

def prepare_multioverlap_workspaces(source_dir, base_workspace_dir):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    
    images = sorted(list(source.glob("*.color.png")))
    if not images:
        images = sorted(
            [p for p in source.glob("*")
             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
    total_images = len(images)

    if total_images == 0:
        print(f"no images found in {source}")
        return

    print(f"found {total_images} images in total.")
    print("--------------------------------------------------")
    print(f"Define your subset splitting boundary windows (Indices 0 to {total_images - 1}):")
    
    try:
        start_frame = int(input("Global Sequence Start Frame (default 0): ") or 0)
        end_frame = int(input(f"Global Sequence End Frame (default {total_images}): ") or total_images)
        center_point = int(input(f"Center Split Boundary Index (e.g., {int((start_frame + end_frame)/2)}): "))
        
        percentages_str = input("Enter target overlap percentages separated by commas (10.5, 15.75, 21): ")
        percentages = [float(p.strip()) for p in percentages_str.split(",") if p.strip()]
        
    except ValueError:
        print("Error: Please enter integer indices and comma-separated percentages.")
        return

    start_frame = max(0, min(total_images - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_images, end_frame))
    center_point = max(start_frame, min(end_frame, center_point))
    total_sequence_len = end_frame - start_frame
    print(f"\nProcessing tracking configurations for {len(percentages)} target overlap variants...")

    for pct in percentages:
        overlap_size = int(round(total_sequence_len * (pct / 100.0)))
        half_overlap = overlap_size // 2
        
        ds1_start = start_frame
        ds1_end = min(end_frame, center_point + (overlap_size - half_overlap))
        
        ds2_start = max(start_frame, center_point - half_overlap)
        ds2_end = end_frame
        
        ds1_indices = range(ds1_start, ds1_end)
        ds2_indices = range(ds2_start, ds2_end)
        
        overlap_set = set(ds1_indices).intersection(set(ds2_indices))
        union_indices = sorted(list(set(ds1_indices).union(set(ds2_indices))))
        
        folder_suffix = f"overlap_{int(pct)}" if pct.is_integer() else f"overlap_{pct}"
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

    print(f"\nBase workspace: {base_workspace}")

if __name__ == "__main__":
    prepare_multioverlap_workspaces(
        source_dir="./dataset",
        base_workspace_dir="./colmap_workspace"
    )