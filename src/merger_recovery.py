#-------------------------------------------------------------------------------
# COLMAP Split-and-Merge Recovery Strategy Runner
# Target Use Case: Merging failures or post-processing adjustments.
# Features:
#   - Automatic target directory contents purging.
#   - Dynamic pycolmap sub-model evaluation matching via highest 3D point count.
#   - Automated reconstruction runtime telemetry JSON rewriting.
#-------------------------------------------------------------------------------

import subprocess
import os
import sys
import time
import json
import shutil
from pathlib import Path
import pycolmap

BASE_PROJECT_DIR = Path(r"D:\Neelay\colmap-split-merge")

cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_bin_path):
    os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ["PATH"]
    if sys.version_info >= (3, 8):
        try:
            os.add_dll_directory(cuda_bin_path)
            print(f"--> [SYSTEM] Windows DLL kernel link established with CUDA 12.6")
        except Exception as e:
            print(f"--> [SYSTEM] Note: Custom DLL directory handling: {e}")

TARGET_SUB_WORKSPACES = [
    r"output\red\overlap_60f"
]


def run_command(command_list, step_name):
    print(f"\n--- Running Recovery Step: {step_name} ---")
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
        raise e


def find_best_submodel(sparse_base_path):
    """
    Scans all available subdirectories inside a sparse mapping folder
    and uses pycolmap to choose the sub-model with the highest 3D points.
    """
    if not sparse_base_path.exists():
        print(f"    [!] Error: Mapping base folder does not exist: {sparse_base_path.name}")
        return None

    best_sub_dir = None
    max_points3d_count = -1

    # Look through subdirectories (usually folders named '0', '1', etc.)
    for sub_dir in sparse_base_path.iterdir():
        if sub_dir.is_dir():
            try:
                # Attempt to parse layout configuration files via pycolmap core binding
                reconstruction = pycolmap.Reconstruction(sub_dir)
                points_count = len(reconstruction.points3D)
                cameras_count = len(reconstruction.images)
                print(f"    -> Found submodel model folder '{sparse_base_path.name}/{sub_dir.name}' | Points3D: {points_count}, Registered Images: {cameras_count}")

                if points_count > max_points3d_count:
                    max_points3d_count = points_count
                    best_sub_dir = sub_dir
            except Exception:
                # Not a valid COLMAP reconstruction folder or directory is empty
                continue

    if best_sub_dir:
        print(f"    [*] Selected optimal index: '{sparse_base_path.name}/{best_sub_dir.name}' ({max_points3d_count} points)")
    return best_sub_dir


def execute_merger_recovery_pipeline(sub_workspace_str):
    # Resolve the full workspace context path safely
    workspace_path = (BASE_PROJECT_DIR / sub_workspace_str).resolve()

    print("\n" + "#" * 80)
    print(f" RECOVERY INITIALIZATION: {workspace_path.name.upper()} ")
    print("#" * 80)

    if not workspace_path.exists():
        print(f"[!] Error: Target path directory layout target not found: {workspace_path}. Skipping.")
        return

    sparse1_path = workspace_path / "sparse1"
    sparse2_path = workspace_path / "sparse2"
    merged_path = workspace_path / "merged"

    # Step 1: Deep structural cleanup of the destination target 'merged' path
    if merged_path.exists():
        print(f"--> [Purge] Cleary old workspace artifacts inside: {merged_path}")
        shutil.rmtree(merged_path)
    merged_path.mkdir(parents=True, exist_ok=True)

    # Step 2: Dynamically isolate the largest models inside both components
    print("--> Evaluating optimal reconstruction candidates inside sub-datasets...")
    best_sparse1 = find_best_submodel(sparse1_path)
    best_sparse2 = find_best_submodel(sparse2_path)

    if not best_sparse1 or not best_sparse2:
        print(f"[!] Critical Error: Unable to determine valid sub-models inside sub-dataset maps.")
        print(f"    Skipping execution tracking loop for context: {workspace_path.name}")
        return

    # Initialize tracking container for update loops
    recovery_timings = {}

    try:
        # Step 3: Run Model Merger
        cmd_merge = [
            "colmap", "model_merger",
            "--input_path1", str(best_sparse1),
            "--input_path2", str(best_sparse2),
            "--output_path", str(merged_path)
        ]
        recovery_timings["model_merger"] = run_command(cmd_merge, f"Model Merger Pass ({workspace_path.name})")

        # Step 4: Run Global Bundle Adjustment
        cmd_ba = [
            "colmap", "bundle_adjuster",
            "--input_path", str(merged_path),
            "--output_path", str(merged_path)
        ]
        recovery_timings["global_bundle_adjustment"] = run_command(cmd_ba, f"Global Bundle Adjustment ({workspace_path.name})")

        # Step 5: Read, rewrite, and correct the existing reconstruction timings profile JSON
        timing_file = workspace_path / "reconstruction_timing.json"
        existing_timings = {}
        if timing_file.exists():
            try:
                with open(timing_file, "r") as f:
                    existing_timings = json.load(f)
            except Exception as e:
                print(f"--> [Warning] Could not parse existing telemetry JSON: {e}")

        # Inject update parameters
        existing_timings["model_merger"] = recovery_timings["model_merger"]
        existing_timings["global_bundle_adjustment"] = recovery_timings["global_bundle_adjustment"]

        # Safely remove total pipeline time field before summing to avoid dirty additions
        existing_timings.pop("total_pipeline_time", None)
        existing_timings["total_pipeline_time"] = sum(v for v in existing_timings.values() if isinstance(v, (int, float)))

        with open(timing_file, "w") as f:
            json.dump(existing_timings, f, indent=4)
        print(f"--> [Telemetry] Updated performance profile successfully saved to: {timing_file}")

    except Exception as error_context:
        print(f"[!] Pipeline failure encountered during system recovery sub-execution step: {error_context}")


def main():
    print("======================================================================")
    print("      COLMAP RECONSTRUCTION PIPELINE SUB-MODEL MERGE RECOVERY ENGINE  ")
    print("======================================================================")
    print(f"Loaded {len(TARGET_SUB_WORKSPACES)} sub-workspace configurations into recovery queue.")

    start_time = time.time()
    for sub_workspace in TARGET_SUB_WORKSPACES:
        execute_merger_recovery_pipeline(sub_workspace)

    print("\n" + "=" * 70)
    print(f" ALL QUEUED SUB-MODEL RE-MERGING PROCESSES COMPLETE")
    print(f" Total Recovery Engine Execution Time: {time.time() - start_time:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
