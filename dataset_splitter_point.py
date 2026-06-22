# ----------------------------------------------------------------------------------------------------------------
# Find images in the dataset by comparing against a pre-computed density profile of 3D points per image
# Takes user-defined global range, a dividing center point, and multiple mean point count targets to sample from.
# Saves separate standalone COLMAP workspace structures inside the target parent folder.
# ----------------------------------------------------------------------------------------------------------------

import os
import shutil
import json
import random
import math
from pathlib import Path

SOURCE_IMG_DIR = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")
BASE_WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace")
JSON_METRICS_PATH = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\global_reconstruction\analysis\scene_density_profile.json")

def prepare_probabilistic_workspaces():
    source = Path(SOURCE_IMG_DIR).resolve()
    base_workspace = Path(BASE_WORKSPACE).resolve()
    json_path = Path(JSON_METRICS_PATH).resolve()

    if not json_path.exists():
        print(f"[!] Error: Density profile not found at {json_path}.\nPlease run your analysis script first.")
        return

    with open(json_path, "r") as f:
        density_profile = json.load(f)

    all_frames = sorted(list(density_profile.keys()))
    total_images = len(all_frames)

    if total_images == 0:
        print("[!] Error: Density profile contains no image entries.")
        return

    print(f"Loaded density profile containing {total_images} frames.")
    print("--------------------------------------------------")
    print(f"Define your sequence constraints (Indices 0 to {total_images - 1}):")

    try:
        start_frame = int(input("Global Sequence Start Frame (default 0): ") or 0)
        end_frame = int(input(f"Global Sequence End Frame (default {total_images}): ") or total_images)
        
        num_common = int(input("Number of target common/overlap images (e.g., 30): "))
        window_size = int(input("Window radius around selected anchors (e.g., 2 means center ± 2 frames): "))
        
        means_str = input("Enter target 3D point means separated by commas (e.g., 200, 350, 600): ")
        target_means = [float(m.strip()) for m in means_str.split(",") if m.strip()]
        
    except ValueError:
        print("[!] Error: Please enter valid integers and numeric comma-separated means.")
        return

    start_frame = max(0, min(total_images - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_images, end_frame))
    
    range_frames = all_frames[start_frame:end_frame]
    print(f"\nProcessing active window from index {start_frame} to {end_frame} ({len(range_frames)} frames available).")

    for target_mean in target_means:
        print(f"\n[*] Generating workspace for target mean: {int(target_mean)} 3D points...")

        sigma = 75.0
        weights = []
        for frame in range_frames:
            pts = density_profile[frame]
            distance = float(pts) - target_mean
            weight = math.exp(-(distance ** 2) / (2 * (sigma ** 2)))
            weights.append(weight + 1e-6)

        common_set = set()
        available_anchors = list(range_frames)
        available_weights = list(weights)

        while len(common_set) < num_common and available_anchors:
            sampled_anchor = random.choices(available_anchors, weights=available_weights, k=1)[0]
            
            idx_in_avail = available_anchors.index(sampled_anchor)
            available_anchors.pop(idx_in_avail)
            available_weights.pop(idx_in_avail)

            global_idx = all_frames.index(sampled_anchor)
            
            w_start = max(start_frame, global_idx - window_size)
            w_end = min(end_frame, global_idx + window_size + 1)
            
            for i in range(w_start, w_end):
                common_set.add(all_frames[i])
                if len(common_set) == num_common:
                    break

        common_images = sorted(list(common_set))

        remaining_images = [f for f in range_frames if f not in common_set]
        
        mid_split = len(remaining_images) // 2
        ds1_remainder = remaining_images[:mid_split]
        ds2_remainder = remaining_images[mid_split:]

        ds1_final = sorted(ds1_remainder + common_images)
        ds2_final = sorted(ds2_remainder + common_images)
        union_all = sorted(list(set(ds1_final).union(set(ds2_final))))

        folder_name = f"mean_{int(target_mean)}_common_{num_common}"
        current_workspace = base_workspace / folder_name
        
        if current_workspace.exists():
            shutil.rmtree(current_workspace)
            
        unified_img_dir = current_workspace / "images"
        sparse1_dir = current_workspace / "sparse1"
        sparse2_dir = current_workspace / "sparse2"
        merged_dir = current_workspace / "merged"
        
        for directory in [unified_img_dir, sparse1_dir, sparse2_dir, merged_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        for frame in union_all:
            shutil.copy(source / frame, unified_img_dir / frame)

        ds1_list_path = current_workspace / "dataset1_list.txt"
        ds2_list_path = current_workspace / "dataset2_list.txt"
        
        with open(ds1_list_path, "w") as f:
            for frame in ds1_final:
                f.write(f"{frame}\n")

        with open(ds2_list_path, "w") as f:
            for frame in ds2_final:
                f.write(f"{frame}\n")

        avg_pts_in_common = sum(density_profile[f] for f in common_images) / max(1, len(common_images))
        print(f" -> Generated Folder: {folder_name}")
        print(f"    - Common Images Sampled: {len(common_images)} frames")
        print(f"    - Observed Mean Points of Overlap: {avg_pts_in_common:.1f} 3D Points")
        print(f"    - Dataset 1 Total Size   : {len(ds1_final)} frames")
        print(f"    - Dataset 2 Total Size   : {len(ds2_final)} frames")

    print(f"\nAll configurations deployed successfully into: {base_workspace}")

if __name__ == "__main__":
    prepare_probabilistic_workspaces()