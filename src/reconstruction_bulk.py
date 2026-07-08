#-------------------------------------------------------------------------------
# 7Scenes COLMAP Adaptive Batch Reconstruction Runner
# Automatically identifies experiment workspace structures and selects the 
# appropriate pipeline path: split-mapping with sub-model merging or full-pass 
# single sequence tracking reconstruction. Tracks execution duration per step 
# and exports telemetry reports to JSON format within each workspace.
# Designed for Windows environments running COLMAP 4.1.0.
# Supports multi-workspace bulk processing.
#-------------------------------------------------------------------------------

import subprocess
import os
import time
import json
from pathlib import Path

USE_GPU = "0"

def run_command(command_list, step_name):
    print(f"\n--- Running: {step_name} ---")
    print("Command:", " ".join(command_list))
    
    start_time = time.time()
    try:
        subprocess.run(
            command_list,
            check=True,
            text=True,
            shell=True,
            capture_output=False 
        )
        elapsed_time = time.time() - start_time
        print(f"--> [SUCCESS] {step_name} completed in {elapsed_time:.2f} seconds.")
        return elapsed_time
    except subprocess.CalledProcessError as e:
        print(f"--> [ERROR] {step_name} failed with return code {e.returncode}.")
        print("Stopping execution for this step configuration to prevent cascading errors.")
        raise e

def detect_scenario_type(folder_name):
    name_lower = folder_name.lower()
    if "full_sequence" in name_lower or "full" in name_lower:
        return "Full Reconstruction"
    elif "mean_" in name_lower:
        return "Varied Mean Split"
    elif "center" in name_lower:
        return "Varied Frame Center Split"
    elif "overlap" in name_lower:
        return "Varied Overlap Split"
    else:
        return "Generic/Unknown Dataset"

def execute_adaptive_batch_reconstruction(base_dir):
    base_path = Path(base_dir).resolve()
    
    print("\n" + "#"*80)
    print(f" ENTERING ROOT WORKSPACE: {base_path} ")
    print("#"*80)
    
    if not base_path.exists():
        print(f"[!] Error: Base workspace directory '{base_path}' does not exist. Skipping.")
        return

    dataset_dirs = [
        d for d in base_path.iterdir() 
        if d.is_dir() and any(k in d.name.lower() for k in ["mean_", "overlap_", "center", "full"])
    ]
    
    if not dataset_dirs:
        print(f"[!] No valid experiment directories found inside {base_path}. Skipping.")
        return

    print(f"Found {len(dataset_dirs)} workspace configurations inside {base_path.name}. Starting batch processing...")

    for workspace_dir in sorted(dataset_dirs):
        scenario_type = detect_scenario_type(workspace_dir.name)
        is_split_pipeline = (scenario_type != "Full Reconstruction")

        print("\n" + "="*70)
        print(f" PROCESSING SUB-WORKSPACE: {workspace_dir.name} ")
        print(f" PIPELINE STRATEGY:       {scenario_type} ")
        print("="*70)

        db_path = workspace_dir / "database.db"
        img_path = workspace_dir / "images"
        
        cmd_extract = [
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--FeatureExtraction.type", "ALIKED_N16ROT",
            "--AlikedExtraction.max_num_features", "2048",
            "--FeatureExtraction.use_gpu", USE_GPU
        ]
        
        cmd_match = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--FeatureMatching.type", "ALIKED_LIGHTGLUE",
            "--FeatureMatching.use_gpu", USE_GPU
        ]

        timing_data = {}

        try:
            timing_data["feature_extraction"] = run_command(cmd_extract, f"1/X Feature Extraction ({workspace_dir.name})")
            timing_data["exhaustive_matching"] = run_command(cmd_match, f"2/X Exhaustive Matching ({workspace_dir.name})")

            if is_split_pipeline:
                list1_path = workspace_dir / "dataset1_list.txt"
                list2_path = workspace_dir / "dataset2_list.txt"
                sparse1_path = workspace_dir / "sparse1"
                sparse2_path = workspace_dir / "sparse2"
                merged_path = workspace_dir / "merged"

                sparse1_path.mkdir(parents=True, exist_ok=True)
                sparse2_path.mkdir(parents=True, exist_ok=True)
                merged_path.mkdir(parents=True, exist_ok=True)

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

                timing_data["mapping_sub_dataset_1"] = run_command(cmd_map1, f"3/6 Mapping Sub-Dataset 1 ({workspace_dir.name})")
                timing_data["mapping_sub_dataset_2"] = run_command(cmd_map2, f"4/6 Mapping Sub-Dataset 2 ({workspace_dir.name})")

                if (sparse1_path / "0").exists() and (sparse2_path / "0").exists():
                    timing_data["model_merger"] = run_command(cmd_merge, f"5/6 Model Merger Pass ({workspace_dir.name})")
                    timing_data["global_bundle_adjustment"] = run_command(cmd_ba, f"6/6 Global Bundle Adjustment ({workspace_dir.name})")
                else:
                    print(f"\nWarning: One or both sub-models ('sparse1/0', 'sparse2/0') failed to generate.")
                    print(f"Skipping merging and bundle adjustment steps for {workspace_dir.name}.")

            else:
                sparse_path = workspace_dir / "sparse"
                full_list_path = workspace_dir / "reconstruction_frame_list.txt"
                sparse_path.mkdir(parents=True, exist_ok=True)

                cmd_map_full = [
                    "colmap", "mapper",
                    "--database_path", str(db_path),
                    "--image_path", str(img_path),
                    "--output_path", str(sparse_path)
                ]
                
                if full_list_path.exists():
                    cmd_map_full.extend(["--Mapper.image_list_path", str(full_list_path)])

                cmd_ba_full = [
                    "colmap", "bundle_adjuster",
                    "--input_path", str(sparse_path / "0"),
                    "--output_path", str(sparse_path / "0")
                ]

                timing_data["full_sequence_sparse_mapping"] = run_command(cmd_map_full, f"3/4 Full Sequence Sparse Mapping ({workspace_dir.name})")
                
                if (sparse_path / "0").exists():
                    timing_data["full_model_bundle_adjustment"] = run_command(cmd_ba_full, f"4/4 Full Model Bundle Adjustment ({workspace_dir.name})")
                else:
                    print(f"\nWarning: 'sparse/0' model index folder was not generated for {workspace_dir.name}.")

        except subprocess.CalledProcessError:
            print(f"\nPipeline execution failure on sub-workspace context {workspace_dir.name}. Skipping to next configuration.")
            continue 
            
        finally:
            if timing_data:
                timing_data["total_pipeline_time"] = sum(v for v in timing_data.values() if isinstance(v, (int, float)))
                json_out_path = workspace_dir / "reconstruction_timing.json"
                with open(json_out_path, "w") as f:
                    json.dump(timing_data, f, indent=4)
                print(f"--> Performance metric profile saved to: {json_out_path}")

if __name__ == "__main__":
    print("==================================================")
    print("     7SCENES ADAPTIVE RECONSTRUCTION RUNNER       ")
    print("==================================================")
    
    print("Example: ./colmap-workspace-1, C:\\workspaces\\set-2, ./test")
    workspace_input = input("Enter Workspace Directories (default: ./colmap_workspace) [Use \",\" for multiple]: ").strip()
    
    if not workspace_input:
        target_workspaces = ["./colmap_workspace"]
    else:
        target_workspaces = [w.strip().strip('"').strip("'") for w in workspace_input.split(",") if w.strip()]
        
    print("--------------------------------------------------")
    print(f"Queued {len(target_workspaces)} root workspace targets for sequential execution.")
    
    start_bulk_time = time.time()
    
    for base_dir in target_workspaces:
        execute_adaptive_batch_reconstruction(base_dir)

    total_bulk_duration = time.time() - start_bulk_time
    print("\n" + "="*70)
    print(f" ALL BULK WORKSPACE RECONSTRUCTIONS COMPLETE ")
    print(f" Total Bulk Queue Runtime: {total_bulk_duration:.2f} seconds")
    print("="*70)