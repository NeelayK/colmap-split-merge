import subprocess
import time
from pathlib import Path
import sys

def run_command(command_list, step_name):
    """Executes a subprocess command and handles error outputs gracefully."""
    print(f"\n--- Running: {step_name} ---")
    start_time = time.time()
    try:
        subprocess.run(command_list, check=True, text=True, shell=True)
        elapsed = time.time() - start_time
        print(f"--> [SUCCESS] {step_name} completed in {elapsed:.1f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ [ERROR] {step_name} failed with return code {e.returncode}.")
        print("Stopping execution for this configuration to prevent cascading errors.")
        sys.exit(1)

def run_pipeline(workspace_path, context, use_gpu):
    """Orchestrates the COLMAP pipeline dynamically based on dataset configuration."""
    workspace = Path(workspace_path).resolve()
    gpu_flag = "1" if use_gpu else "0"
    
    # Paths setup
    db_path = workspace / "database.db"
    img_path = workspace / "images"
    merged_path = workspace / "merged"
    
    merged_path.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1 & 2: GLOBAL FEATURE EXTRACTION & MATCHING
    # -------------------------------------------------------------------------
    cmd_extract = [
        "colmap", "feature_extractor", 
        "--database_path", str(db_path), 
        "--image_path", str(img_path), 
        "--FeatureExtraction.type", "ALIKED_N16ROT", 
        "--AlikedExtraction.max_num_features", "2048", 
        "--FeatureExtraction.use_gpu", gpu_flag
    ]
    
    cmd_match = [
        "colmap", "exhaustive_matcher", 
        "--database_path", str(db_path), 
        "--FeatureMatching.type", "ALIKED_LIGHTGLUE", 
        "--FeatureMatching.use_gpu", gpu_flag
    ]

    run_command(cmd_extract, f"Step 1: Feature Extraction")
    run_command(cmd_match, f"Step 2: Exhaustive Matching")

    # -------------------------------------------------------------------------
    # ROUTE A: FULL RECONSTRUCTION (No Split/Merge)
    # -------------------------------------------------------------------------
    if context.get("generation_mode") == "full_no_split":
        print("\n⚡ [MODE DETECTED]: Linear Full Reconstruction. Bypassing merger.")
        cmd_map = [
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(img_path),
            "--output_path", str(merged_path)
        ]
        run_command(cmd_map, "Step 3: Global Sparse Mapping")
        
        # In a standard mapper run, output is saved into '0'
        mapped_0_dir = merged_path / "0"
        if mapped_0_dir.exists():
            cmd_ba = [
                "colmap", "bundle_adjuster",
                "--input_path", str(mapped_0_dir),
                "--output_path", str(merged_path)
            ]
            run_command(cmd_ba, "Step 4: Global Bundle Adjustment")
        else:
            print("❌ [ERROR]: Mapper failed to produce a valid model in '0' directory.")

    # -------------------------------------------------------------------------
    # ROUTE B: SPLIT-MERGE RECONSTRUCTION
    # -------------------------------------------------------------------------
    else:
        print("\n⚡ [MODE DETECTED]: Split-Merge Sequence Workflow.")
        sparse1_path = workspace / "sparse1"
        sparse2_path = workspace / "sparse2"
        list1_path = workspace / "dataset1_list.txt"
        list2_path = workspace / "dataset2_list.txt"

        sparse1_path.mkdir(exist_ok=True)
        sparse2_path.mkdir(exist_ok=True)

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
        
        run_command(cmd_map1, "Step 3: Mapping Sub-Model 1")
        run_command(cmd_map2, "Step 4: Mapping Sub-Model 2")

        cmd_merge = [
            "colmap", "model_merger",
            "--input_path1", str(sparse1_path / "0"),
            "--input_path2", str(sparse2_path / "0"),
            "--output_path", str(merged_path)
        ]
        run_command(cmd_merge, "Step 5: Model Merger")

        cmd_ba = [
            "colmap", "bundle_adjuster",
            "--input_path", str(merged_path),
            "--output_path", str(merged_path)
        ]
        run_command(cmd_ba, "Step 6: Global Bundle Adjustment")

    print("\n✅ Reconstruction pipeline completed successfully!")