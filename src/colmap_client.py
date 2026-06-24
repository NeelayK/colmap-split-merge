import subprocess
import os
from pathlib import Path

BASE_WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace")

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
        print("Stopping execution for this configuration to prevent cascading errors.")
        raise e

def automate_colmap_pipeline(base_dir):
    base_path = Path(base_dir).resolve()
    
    if not base_path.exists():
        print(f"Error: Base workspace directory '{base_path}' does not exist.")
        return

    dataset_dirs = [
        d for d in base_path.iterdir() 
        if d.is_dir() and (d.name.startswith("mean_") or d.name.startswith("overlap_"))
    ]
    
    if not dataset_dirs:
        print(f"No valid experiment directories ('mean_' or 'overlap_') found in {base_path}.")
        return

    print(f"Found {len(dataset_dirs)} configurations. Starting batch processing...")

    for workspace_dir in sorted(dataset_dirs):
        print("\n" + "="*70)
        print(f" PROCESSING WORKSPACE: {workspace_dir.name} ")
        print("="*70)

        db_path = workspace_dir / "database.db"
        img_path = workspace_dir / "images"
        
        list1_path = workspace_dir / "dataset1_list.txt"
        list2_path = workspace_dir / "dataset2_list.txt"
        
        sparse1_path = workspace_dir / "sparse1"
        sparse2_path = workspace_dir / "sparse2"
        merged_path = workspace_dir / "merged"
        
        sparse1_path.mkdir(parents=True, exist_ok=True)
        sparse2_path.mkdir(parents=True, exist_ok=True)
        merged_path.mkdir(parents=True, exist_ok=True)

        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
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

        cmd_merge = [
            "colmap", "model_merger",
            "--input_path1", str(sparse1_path / "0"),
            "--input_path2", str(sparse2_path / "0"),
            "--output_path", str(merged_path)
        ]

        cmd_ba = [
            "colmap", "bundle_adjuster",
            "--input_path", str(merged_path),
            "--output_path", str(merged_path)
        ]

        try:
            run_command(cmd_extract, f"1/6 Feature Extraction ({workspace_dir.name})")
            run_command(cmd_match, f"2/6 Exhaustive Matching ({workspace_dir.name})")
            run_command(cmd_map1, f"3/6 Mapping Dataset 1 ({workspace_dir.name})")
            run_command(cmd_map2, f"4/6 Mapping Dataset 2 ({workspace_dir.name})")
            run_command(cmd_merge, f"5/6 Model Merger ({workspace_dir.name})")
            run_command(cmd_ba, f"6/6 Global Bundle Adjustment ({workspace_dir.name})")
            
        except subprocess.CalledProcessError:
            print(f"\n[!] Pipeline halted for {workspace_dir.name} due to an error. Skipping to next configuration.")
            continue 

    print("\n" + "="*70)
    print(" BATCH AUTOMATION COMPLETE ")
    print("="*70)

if __name__ == "__main__":
    automate_colmap_pipeline(BASE_WORKSPACE)