import os
import shutil
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------
# Find images in the dataset and generate configurations based on SPLITTING FRAMES.
# Takes user-defined global range, a single fixed overlap percentage, and multiple split locations.
# Saves separate standalone COLMAP workspace structures inside the target parent folder.
# ----------------------------------------------------------------------------------------------------------------

def prepare_multisplit_workspaces(source_dir, base_workspace_dir):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    
    # 1. Discover sequence image tracks
    images = sorted(list(source.glob("*.color.png")))
    if not images:
        images = sorted(
            [p for p in source.glob("*")
             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        )
    total_images = len(images)

    if total_images == 0:
        print(f"There are no images found in {source}")
        return

    print(f"Found {total_images} images in total.")
    print("--------------------------------------------------")
    print(f"Define your subset splitting boundary windows (Indices 0 to {total_images - 1}):")
    
    try:
        start_frame = int(input("1. Global Sequence Start Frame (ex. 0): ") or 0)
        end_frame = int(input(f"2. Global Sequence End Frame (ex. {total_images}): ") or total_images)
        
        pct_str = input("3. Enter fixed overlap % (ex. 3): ").replace('%', '')
        overlap_pct = float(pct_str.strip())
        
        splits_str = input("4. Enter splitting frames separated by commas (ex. 100, 200, 300): ")
        splitting_frames = [int(s.strip()) for s in splits_str.split(",") if s.strip()]
        
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return

    # Clamp bounds to ensure memory safety
    start_frame = max(0, min(total_images - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_images, end_frame))
    total_sequence_len = end_frame - start_frame

    # Calculate the fixed overlap size based on the percentage
    overlap_size = int(round(total_sequence_len * (overlap_pct / 100.0)))
    half_overlap = overlap_size // 2

    print(f"\nProcessing tracking configurations for {len(splitting_frames)} split locations...")
    print(f"Fixed overlap calculated as {overlap_size} frames ({overlap_pct}% of {total_sequence_len}).\n")

    # 2. Iterate and build out workspaces for each splitting frame
    for center_point in splitting_frames:
        # Clamp the center point so it doesn't fall outside the sequence
        center_point = max(start_frame, min(end_frame, center_point))
        
        # Calculate dataset spans symmetric around the anchor center point
        ds1_start = start_frame
        ds1_end = min(end_frame, center_point + (overlap_size - half_overlap))
        
        ds2_start = max(start_frame, center_point - half_overlap)
        ds2_end = end_frame
        
        ds1_indices = range(ds1_start, ds1_end)
        ds2_indices = range(ds2_start, ds2_end)
        
        overlap_set = set(ds1_indices).intersection(set(ds2_indices))
        union_indices = sorted(list(set(ds1_indices).union(set(ds2_indices))))
        
        # Construct distinct folders per split configuration
        folder_suffix = f"split_{center_point}_overlap_{int(overlap_pct) if overlap_pct.is_integer() else overlap_pct}"
        current_workspace = base_workspace / folder_suffix
        
        if current_workspace.exists():
            shutil.rmtree(current_workspace)
            
        unified_img_dir = current_workspace / "images"
        sparse1_dir = current_workspace / "sparse1"
        sparse2_dir = current_workspace / "sparse2"
        merged_dir = current_workspace / "merged"
        
        for directory in [unified_img_dir, sparse1_dir, sparse2_dir, merged_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # 3. Copy image sets matching the local subset union block
        for idx in union_indices:
            shutil.copy(images[idx], unified_img_dir / images[idx].name)
            
        # 4. Generate the corresponding explicit image mapping lists
        ds1_list_path = current_workspace / "dataset1_list.txt"
        ds2_list_path = current_workspace / "dataset2_list.txt"
        
        with open(ds1_list_path, "w") as f:
            for idx in ds1_indices:
                f.write(f"{images[idx].name}\n")
                
        with open(ds2_list_path, "w") as f:
            for idx in ds2_indices:
                f.write(f"{images[idx].name}\n")
                
        print(f" -> Generated: {folder_suffix}")
        print(f"    - Split Point     : Frame {center_point}")
        print(f"    - Dataset 1 Range : {min(ds1_indices)} to {max(ds1_indices)} ({len(ds1_indices)} frames)")
        print(f"    - Dataset 2 Range : {min(ds2_indices)} to {max(ds2_indices)} ({len(ds2_indices)} frames)")
        print(f"    - Common Overlap  : {len(overlap_set)} frames total\n")

    print(f"All datasets generated successfully inside: {base_workspace}")

if __name__ == "__main__":
    prepare_multisplit_workspaces(
        source_dir="./dataset",
        base_workspace_dir="./colmap_workspace"
    )