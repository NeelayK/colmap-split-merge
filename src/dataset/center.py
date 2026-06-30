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
        
        if ds1_end <= ds1_start or ds2_end <= ds2_start:
            print(f"Warning: Invalid frame range generated for center frame {center_frame}. Skipping.")
            continue

        if overlap_mode == "percentage":
            folder_suffix = f"center_{center_frame}_overlap_{int(overlap_value)}pct" if overlap_value.is_integer() else f"center_{center_frame}_overlap_{overlap_value}pct"
        else:
            folder_suffix = f"center_{center_frame}_overlap_{overlap_frames}f"
            
        workspace = base_workspace / folder_suffix
        
        if workspace.exists():
            shutil.rmtree(workspace)
            
        ds1_img_dir = workspace / "dataset1" / "images"
        ds2_img_dir = workspace / "dataset2" / "images"
        
        ds1_img_dir.mkdir(parents=True, exist_ok=True)
        ds2_img_dir.mkdir(parents=True, exist_ok=True)
        
        actual_overlap = max(0, ds1_end - ds2_start)
        print(f"\n--- Center Frame: {center_frame} ({folder_suffix}) ---")
        print(f"  Dataset 1: {ds1_end - ds1_start} images (Indices {ds1_start} to {ds1_end - 1})")
        print(f"  Dataset 2: {ds2_end - ds2_start} images (Indices {ds2_start} to {ds2_end - 1})")
        print(f"  Overlap  : {actual_overlap} images shared in common")
        
        for i in range(ds1_start, ds1_end):
            shutil.copy(images[i], ds1_img_dir / images[i].name)
        for i in range(ds2_start, ds2_end):
            shutil.copy(images[i], ds2_img_dir / images[i].name)

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