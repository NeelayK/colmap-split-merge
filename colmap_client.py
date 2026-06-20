import subprocess
import os
from pathlib import Path

# ------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------
BASE_WORKSPACE = Path(r"C:\Users\user\OneDrive\Documents\Neelay\colmap-split-merge\colmap_workspace")

# ------------------------------------------------------
# 2. Execution Helper
# ------------------------------------------------------
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

# ------------------------------------------------------
# 3. Merger Fallback Helper
# ------------------------------------------------------
def attempt_smart_merge(sparse1_path, sparse2_path, merged_path, step_name):
    print(f"\n--- Running: {step_name} (with Fallbacks) ---")
    
    # Priority order for trying sub-models
    combinations = [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")]
    
    for mod1, mod2 in combinations:
        p1 = sparse1_path / mod1
        p2 = sparse2_path / mod2
        
        # Skip if the sub-model doesn't exist
        if not p1.exists() or not p2.exists():
            continue
            
        cmd_merge = [
            "colmap", "model_merger",
            "--input_path1", str(p1),
            "--input_path2", str(p2),
            "--output_path", str(merged_path)
        ]
        
        print(f"[*] Attempting merge with Dataset 1 Model '{mod1}' and Dataset 2 Model '{mod2}'...")
        try:
            subprocess.run(
                cmd_merge,
                check=True,
                text=True,
                shell=True,
                capture_output=False
            )
            print(f"--> [SUCCESS] {step_name} completed using {mod1} and {mod2}.")
            return True # Exit function on first successful merge
            
        except subprocess.CalledProcessError:
            print(f"[!] Warning: Merge failed for combinations {mod1} and {mod2}. Trying next...")
            
    # If the loop finishes without returning True, all combinations failed
    print(f"--> [ERROR] All merge combinations failed for {step_name}.")
    raise subprocess.CalledProcessError(1, "colmap model_merger (all fallbacks failed)")

# ------------------------------------------------------
# 4. Main Automation Pipeline
# ------------------------------------------------------
def automate_colmap_pipeline(base_dir):
    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        print(f"Error: Base workspace directory '{base_path}' does not exist.")
        return

    # Update: Now looks for both "overlap_" and "split_" prefix directories
    workspace_dirs = [d for d in base_path.iterdir() if d.is_dir() and (d.name.startswith("overlap_") or d.name.startswith("split_"))]
    
    if not workspace_dirs:
        print(f"No valid workspace directories found in {base_path}. Run the split script first.")
        return

    print(f"Found {len(workspace_dirs)} tracking configurations. Starting batch processing...")

    for config_dir in sorted(workspace_dirs):
        print("\n" + "="*70)
        print(f" PROCESSING WORKSPACE: {config_dir.name} ")
        print("="*70)

        db_path = config_dir / "database.db"
        img_path = config_dir / "images"
        
        list1_path = config_dir / "dataset1_list.txt"
        list2_path = config_dir / "dataset2_list.txt"
        
        sparse1_path = config_dir / "sparse1"
        sparse2_path = config_dir / "sparse2"
        merged_path = config_dir / "merged"
        
        sparse1_path.mkdir(parents=True, exist_ok=True)
        sparse2_path.mkdir(parents=True, exist_ok=True)
        merged_path.mkdir(parents=True, exist_ok=True)

        # --- Define Pipeline Commands ---
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features", "2048",
            "--FeatureExtraction.use_gpu", "0"
        ]
        
        cmd_match = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--FeatureMatching.use_gpu", "0"
        ]
        
        cmd_map1 = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--Mapper.image_list_path", str(list1_path),
            "--output_path", str(sparse1_path)
        ]

        cmd_map2 = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--Mapper.image_list_path", str(list2_path),
            "--output_path", str(sparse2_path)
        ]

        cmd_ba = [
            "colmap", "bundle_adjuster",
            "--input_path", str(merged_path),
            "--output_path", str(merged_path)
        ]

        # --- Execute Pipeline ---
        try:
            run_command(cmd_extract, f"1/6 Feature Extraction ({config_dir.name})")
            run_command(cmd_match, f"2/6 Exhaustive Matching ({config_dir.name})")
            run_command(cmd_map1, f"3/6 Mapping Dataset 1 ({config_dir.name})")
            run_command(cmd_map2, f"4/6 Mapping Dataset 2 ({config_dir.name})")
            
            # Use the new smart merger here
            attempt_smart_merge(sparse1_path, sparse2_path, merged_path, f"5/6 Model Merger ({config_dir.name})")
            
            run_command(cmd_ba, f"6/6 Global Bundle Adjustment ({config_dir.name})")
            
        except subprocess.CalledProcessError:
            print(f"\n[!] Pipeline halted for {config_dir.name} due to an error. Moving to next configuration.")
            continue 

    print("\n" + "="*70)
    print(" BATCH AUTOMATION COMPLETE ")
    print("="*70)

if __name__ == "__main__":
    automate_colmap_pipeline(BASE_WORKSPACE)