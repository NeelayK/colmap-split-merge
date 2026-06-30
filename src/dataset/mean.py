#-------------------------------------------------------------------------------
# 7Scenes COLMAP Probabilistic Density Splitter Module
# Samples overlapping frames based on a pre-computed global reconstruction 
# 3D points density profile using a Gaussian distribution across target means.
# Designed for standalone execution and multi-module integration.
#-------------------------------------------------------------------------------

import os
import shutil
import json
import random
import math
from pathlib import Path

def split_dataset_probabilistic(source_dir, base_workspace_dir, json_profile_path, target_means, num_common, window_size, start_frame=0, end_frame=None):
    source = Path(source_dir).resolve()
    base_workspace = Path(base_workspace_dir).resolve()
    json_path = Path(json_profile_path).resolve()

    if not json_path.exists():
        print(f"Error: Density profile configuration not found at: {json_path}")
        return False

    with open(json_path, "r") as f:
        density_profile = json.load(f)

    all_frames = sorted(list(density_profile.keys()))
    total_images = len(all_frames)

    if total_images == 0:
        print("Error: Target profile JSON contains no valid image sequence keys.")
        return False

    if end_frame is None:
        end_frame = total_images

    start_idx = max(0, min(total_images - 1, start_frame))
    end_idx = max(start_idx + 1, min(total_images, end_frame))
    range_frames = all_frames[start_idx:end_idx]

    print(f"\nProcessing configurations for {len(target_means)} structural mean variants...")

    for target_mean in target_means:
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
            w_start = max(start_idx, global_idx - window_size)
            w_end = min(end_idx, global_idx + window_size + 1)
            
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
        print(f" -> Generated: {folder_name}")
        print(f"    - Common Overlap Size  : {len(common_images)} frames")
        print(f"    - Tracked Average Pts  : {avg_pts_in_common:.1f} 3D Points")
        print(f"    - Dataset 1 Total Size : {len(ds1_final)} frames")
        print(f"    - Dataset 2 Total Size : {len(ds2_final)} frames")

    print(f"\nSuccessfully generated workspaces inside: '{base_workspace}'")
    return True

if __name__ == "__main__":
    print("==================================================")
    print("       7SCENES PROBABILISTIC MEAN SPLITTER        ")
    print("==================================================")
    
    source_input = input("Enter Dataset Source Directory (default: ./dataset): ").strip()
    source_dir = source_input if source_input else "./dataset"
    
    workspace_input = input("Enter Base Workspace Directory (default: ./colmap_workspace): ").strip()
    base_workspace_dir = workspace_input if workspace_input else "./colmap_workspace"
    
    json_input = input("Enter Density Profile JSON Path (default: ./scene_density_profile.json): ").strip()
    json_profile_path = json_input if json_input else "./scene_density_profile.json"
    print("--------------------------------------------------")

    json_path = Path(json_profile_path).resolve()
    if not json_path.exists():
        print(f"Error: Reconstruction file profile sequence tracking not found at {json_path}")
        exit(1)

    with open(json_path, "r") as f:
        loaded_profile = json.load(f)
    total_imgs = len(loaded_profile.keys())
    
    if total_imgs == 0:
        print("Error: Pre-computed analysis profile contains zero reference points.")
        exit(1)
        
    print(f"Found {total_imgs} total tracking frames inside profile.")
    print("--------------------------------------------------")
    
    try:
        start_fr = int(input("Global Sequence Start Frame (default 0): ") or 0)
        end_fr = int(input(f"Global Sequence End Frame (default {total_imgs}): ") or total_imgs)
        print("--------------------------------------------------")
        
        num_common_val = int(input("Enter number of target common overlap frames (e.g., 30): ") or 30)
        window_radius = int(input("Enter anchor selection window radius size (e.g., 2): ") or 2)
        
        means_str = input("Enter target 3D point means separated by commas (e.g., 200,350,600): ")
        target_vals = [float(m.strip()) for m in means_str.split(",") if m.strip()]
        
    except ValueError:
        print("Error: Input formatting error. Please ensure numbers are typed correctly.")
        exit(1)
        
    split_dataset_probabilistic(
        source_dir=source_dir,
        base_workspace_dir=base_workspace_dir,
        json_profile_path=json_profile_path,
        target_means=target_vals,
        num_common=num_common_val,
        window_size=window_radius,
        start_frame=start_fr,
        end_frame=end_fr
    )