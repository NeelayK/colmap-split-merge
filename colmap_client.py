import subprocess
import os
from pathlib import Path

# ------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------
# Point this to the base workspace where the previous script generated the overlap folders
BASE_WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace")

# ------------------------------------------------------
# 2. Execution Helper
# ------------------------------------------------------
def run_command(command_list, step_name):
    print(f"\n--- Running: {step_name} ---")
    print("Command:", " ".join(command_list))
    
    try:
        # shell=True allows Windows to find the 'colmap' executable in PATH
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
        print("Stopping execution for this configuration to prevent cascading errors.")
        raise e

# ------------------------------------------------------
# 3. Main Automation Pipeline
# ------------------------------------------------------
def automate_colmap_pipeline(base_dir):
    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        print(f"Error: Base workspace directory '{base_path}' does not exist.")
        return

    # Find all overlap directories inside the base workspace
    overlap_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("overlap_")]
    
    if not overlap_dirs:
        print(f"No 'overlap_' directories found in {base_path}. Run the split script first.")
        return

    print(f"Found {len(overlap_dirs)} overlap configurations. Starting batch processing...")

    # Iterate through each overlap configuration
    for overlap_dir in sorted(overlap_dirs):
        print("\n" + "="*70)
        print(f" PROCESSING WORKSPACE: {overlap_dir.name} ")
        print("="*70)

        # Setup Core Paths for this specific overlap configuration
        db_path = overlap_dir / "database.db"
        img_path = overlap_dir / "images"
        
        list1_path = overlap_dir / "dataset1_list.txt"
        list2_path = overlap_dir / "dataset2_list.txt"
        
        sparse1_path = overlap_dir / "sparse1"
        sparse2_path = overlap_dir / "sparse2"
        merged_path = overlap_dir / "merged"
        
        # Ensure output directories exist
        sparse1_path.mkdir(parents=True, exist_ok=True)
        sparse2_path.mkdir(parents=True, exist_ok=True)
        merged_path.mkdir(parents=True, exist_ok=True)

        # --- Define Pipeline Commands ---
        
        # 1. Feature Extraction
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features", "2048",
            "--FeatureExtraction.use_gpu", "0"
        ]
        
        # 2. Exhaustive Matching
        cmd_match = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--FeatureMatching.use_gpu", "0"
        ]
        
        # 3. Mapper for Dataset 1
        cmd_map1 = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--Mapper.image_list_path", str(list1_path),
            "--output_path", str(sparse1_path)
        ]

        # 4. Mapper for Dataset 2
        cmd_map2 = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--Mapper.image_list_path", str(list2_path),
            "--output_path", str(sparse2_path)
        ]

        # 5. Model Merger (combines sparse1/0 and sparse2/0)
        cmd_merge = [
            "colmap", "model_merger",
            "--input_path1", str(sparse1_path / "0"),
            "--input_path2", str(sparse2_path / "1"),
            "--output_path", str(merged_path)
        ]

        # 6. Bundle Adjuster (refines the merged model)
        cmd_ba = [
            "colmap", "bundle_adjuster",
            "--input_path", str(merged_path),
            "--output_path", str(merged_path)
        ]

        # --- Execute Pipeline ---
        try:
            run_command(cmd_extract, f"1/6 Feature Extraction ({overlap_dir.name})")
            run_command(cmd_match, f"2/6 Exhaustive Matching ({overlap_dir.name})")
            run_command(cmd_map1, f"3/6 Mapping Dataset 1 ({overlap_dir.name})")
            run_command(cmd_map2, f"4/6 Mapping Dataset 2 ({overlap_dir.name})")
            run_command(cmd_merge, f"5/6 Model Merger ({overlap_dir.name})")
            run_command(cmd_ba, f"6/6 Global Bundle Adjustment ({overlap_dir.name})")
            
        except subprocess.CalledProcessError:
            print(f"\n[!] Pipeline halted for {overlap_dir.name} due to an error. Moving to next configuration.")
            continue 

    print("\n" + "="*70)
    print(" BATCH AUTOMATION COMPLETE ")
    print("="*70)

if __name__ == "__main__":
    automate_colmap_pipeline(BASE_WORKSPACE)