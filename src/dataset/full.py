#-------------------------------------------------------------------------------
# 7Scenes COLMAP Full Reconstruction Dataset Builder Module
# Consolidates a specific global sequence range of frames into a single, unified 
# workspace folder for a complete full-sequence reconstruction pass.
# Designed for standalone execution and multi-module integration.
#-------------------------------------------------------------------------------

import os
import shutil
from pathlib import Path

def prepare_full_reconstruction_workspace(source_dir, base_workspace_dir, start_frame=0, end_frame=None):
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

    start_idx = max(0, min(total_images - 1, start_frame))
    end_idx = max(start_idx + 1, min(total_images, end_frame))
    selected_images = images[start_idx:end_idx]

    print("\nProcessing configuration for full sequence reconstruction...")

    folder_suffix = f"full_sequence_{start_idx}_to_{end_idx}"
    current_workspace = base_workspace / folder_suffix
    
    if current_workspace.exists():
        shutil.rmtree(current_workspace)
        
    unified_img_dir = current_workspace / "images"
    sparse_dir = current_workspace / "sparse"
    
    for directory in [unified_img_dir, sparse_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        
    for img in selected_images:
        shutil.copy(img, unified_img_dir / img.name)
        
    sequence_list_path = current_workspace / "reconstruction_frame_list.txt"
    with open(sequence_list_path, "w") as f:
        for img in selected_images:
            f.write(f"{img.name}\n")

    print(f" -> Generated: {folder_suffix}")
    print(f"    - Frame Index Range  : {start_idx} to {end_idx - 1}")
    print(f"    - Total Copied Frames: {len(selected_images)} images")
    
    print(f"\nSuccessfully generated workspaces inside: '{base_workspace}'")
    return True

if __name__ == "__main__":
    print("==================================================")
    print("       7SCENES FULL RECONSTRUCTION BUILDER        ")
    print("==================================================")
    
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
    except ValueError:
        print("Error: Input formatting error. Please ensure numbers are typed correctly.")
        exit(1)
        
    prepare_full_reconstruction_workspace(
        source_dir=source_dir,
        base_workspace_dir=base_workspace_dir,
        start_frame=start_fr,
        end_frame=end_fr
    )