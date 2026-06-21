import subprocess
import os
import shutil
from pathlib import Path

SOURCE_IMAGE_DIR = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")
GLOBAL_WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\global_reconstruction")

def get_completed_steps(workspace_dir):
    checkpoint_file = workspace_dir / ".pipeline_checkpoint.txt"
    if not checkpoint_file.exists():
        return set()
    with open(checkpoint_file, "r") as f:
        return set(line.strip() for line in f if line.strip())

def mark_step_completed(workspace_dir, step_name):
    checkpoint_file = workspace_dir / ".pipeline_checkpoint.txt"
    with open(checkpoint_file, "a") as f:
        f.write(f"{step_name}\n")

def clean_directory_contents(dir_path):
    """Safely clears everything inside a directory without deleting the directory itself."""
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

def run_command(command_list, step_name):
    print(f"\n--- Running: {step_name} ---")
    print("Command:", " ".join(command_list))
    
    try:
        result = subprocess.run(
            command_list,
            check=True,
            text=True,
            shell=True,
            capture_output=False 
        )
        print(f"--> [SUCCESS] {step_name} completed.")
    except subprocess.CalledProcessError as e:
        print(f"--> [ERROR] {step_name} failed with return code {e.returncode}.")
        raise e

def run_global_pipeline():
    src_img_path = Path(SOURCE_IMAGE_DIR).resolve()
    workspace_path = Path(GLOBAL_WORKSPACE).resolve()
    
    if not src_img_path.exists():
        print(f"Error: Source image directory '{src_img_path}' does not exist.")
        return

    db_path = workspace_path / "database.db"
    sparse_path = workspace_path / "sparse"
    img_list_path = workspace_path / "color_images_list.txt"
    checkpoint_file = workspace_path / ".pipeline_checkpoint.txt"

    print("="*70)
    print(" COLMAP GLOBAL RECONSTRUCTION PIPELINE ")
    print("="*70)
    print(f"Source Images : {src_img_path}")
    print(f"Workspace     : {workspace_path}\n")
    print("Select execution mode:")
    print(" [1] RESUME : Skip already completed steps (Safe choice if code crashed/stopped).")
    print(" [2] FRESH  : Wipe all previous progress, database, and models to start completely over.")
    
    mode_input = input("Enter choice [1 or 2] (Default 1): ").strip()
    start_fresh = (mode_input == "2")

    if start_fresh:
        print(f"\n[*] Clearing old files for a fresh start...")
        if db_path.exists(): db_path.unlink()
        if checkpoint_file.exists(): checkpoint_file.unlink()
        if img_list_path.exists(): img_list_path.unlink()
        clean_directory_contents(sparse_path)
    
    workspace_path.mkdir(parents=True, exist_ok=True)
    sparse_path.mkdir(parents=True, exist_ok=True)

    print("\n[*] Scanning dataset directory for color images...")
    color_images = sorted([f.name for f in src_img_path.glob("*.color.png")])
    
    if not color_images:
        print(f"[!] Error: No files matching '*.color.png' found in {src_img_path}.")
        return
        
    with open(img_list_path, "w") as f:
        f.write("\n".join(color_images))
    print(f"--> [SUCCESS] Created image list targeting exactly {len(color_images)} color frames.")

    completed = get_completed_steps(workspace_path)

    cmd_extract = [
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(src_img_path),
        "--image_list_path", str(img_list_path),
        "--FeatureExtraction.type", "ALIKED_N16ROT",
        "--AlikedExtraction.max_num_features", "2048",
        "--FeatureExtraction.use_gpu", "1"
    ]
    
    cmd_match = [
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
        "--FeatureMatching.use_gpu", "1"
    ]
    
    cmd_map = [
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(src_img_path),
        "--Mapper.image_list_path", str(img_list_path),
        "--output_path", str(sparse_path)
    ]

    cmd_ba = [
        "colmap", "bundle_adjuster",
        "--input_path", str(sparse_path / "0"),
        "--output_path", str(sparse_path / "0")
    ]

    try:
        # Step 1: Feature Extraction
        if "EXTRACT" in completed:
            print("--> [SKIP] Feature Extraction already completed.")
        else:
            if db_path.exists(): db_path.unlink() 
            run_command(cmd_extract, "1/4 Global Feature Extraction (GPU Filtered)")
            mark_step_completed(workspace_path, "EXTRACT")

        # Step 2: Exhaustive Matching
        if "MATCH" in completed:
            print("--> [SKIP] Exhaustive Matching already completed.")
        else:
            run_command(cmd_match, "2/4 Global Exhaustive Matching (GPU)")
            mark_step_completed(workspace_path, "MATCH")

        # Step 3: Sparse Mapping
        if "MAP" in completed:
            print("--> [SKIP] Global Mapping already completed.")
        else:
            clean_directory_contents(sparse_path)
            run_command(cmd_map, "3/4 Global Sparse Mapping")
            mark_step_completed(workspace_path, "MAP")
        
        # Step 4: Global Bundle Adjustment
        if "BA" in completed:
            print("--> [SKIP] Global Bundle Adjustment already completed.")
        else:
            if (sparse_path / "0").exists():
                run_command(cmd_ba, "4/4 Global Bundle Adjustment")
                mark_step_completed(workspace_path, "BA")
            else:
                print("\n[!] Error: 'sparse/0' directory not found. The mapper might have failed to reconstruct a core model.")

    except subprocess.CalledProcessError:
        print("\n[!] Pipeline halted due to an engine error execution failure.")
        return

    print("\n" + "="*70)
    print(" GLOBAL RECONSTRUCTION COMPLETE ")
    print("="*70)
    print(f"Your final sparse model is located at: {sparse_path / '0'}")

if __name__ == "__main__":
    run_global_pipeline()