
#-----------------------------------------------
# Use this for merging sparse reconstruction
# with GT poses 
#-----------------------------------------------



import time
import json
import subprocess
from pathlib import Path
import numpy as np
import pycolmap

WORKSPACE_DIR="./colmap_workspace"
DATASET_DIR="./dataset"
BASE_WORKSPACE = Path(WORKSPACE_DIR).resolve()
DIR_GT = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")
LOG_OUTPUT_FILE = BASE_WORKSPACE / "batch_evaluation_summary.json"

def umeyama_alignment(src, dst):
    num_pts, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    H = (dst_centered.T @ src_centered) / num_pts
    U, S, Vt = np.linalg.svd(H)
    S_mat = np.eye(dim)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[dim-1, dim-1] = -1
    Rotation = U @ S_mat @ Vt
    Scale = np.trace(np.diag(S) @ S_mat) / np.var(src_centered, axis=0).sum()
    Translation = dst_mean - Scale * Rotation @ src_mean
    T = np.eye(4); T[:3, :3] = Scale * Rotation; T[:3, 3] = Translation
    return T, Scale, Rotation, Translation

def evaluate_reconstruction(merged_sparse_dir, dir_gt):
    recon = pycolmap.Reconstruction(merged_sparse_dir)
    reg_images = {im.name: im for im_id, im in recon.images.items() if recon.exists_image(im_id)}
    
    recon_points, gt_points = [], []
    for name in sorted(list(reg_images.keys())):
        base_name = Path(name).name.split('.')[0]
        pose_file = dir_gt / f"{base_name}.pose.txt"
        if pose_file.exists():
            gt_points.append(np.loadtxt(pose_file)[:3, 3])
            recon_points.append(reg_images[name].projection_center())

    if len(gt_points) < 3:
        return {"status": "Failed", "error": "Insufficient overlapping GT poses matched."}

    X_recon = np.array(recon_points)
    Y_gt = np.array(gt_points)
    
    T, scale, R, t = umeyama_alignment(X_recon, Y_gt)
    aligned_to_gt = (scale * (X_recon @ T[:3, :3].T / scale)) + T[:3, 3]
    
    ate_errors = np.linalg.norm(Y_gt - aligned_to_gt, axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_errors**2)))
    mean_ate = float(np.mean(ate_errors))
    max_ate = float(np.max(ate_errors))
    
    return {
        "status": "Success",
        "registered_frames_count": len(reg_images),
        "aligned_frames_count": len(gt_points),
        "scale_factor": float(scale),
        "ate_rmse_meters": ate_rmse,
        "ate_rmse_mm": ate_rmse * 1000.0,
        "mean_error_meters": mean_ate,
        "max_error_meters": max_ate,
        "rotation_matrix_trace": float(np.trace(R)),
        "translation_vector": t.tolist()
    }

def run_command(command_list, step_name):
    print(f"Executing: {step_name}...")
    start_time = time.time()
    try:
        subprocess.run(command_list, check=True, text=True, shell=True, capture_output=True)
        return time.time() - start_time
    except subprocess.CalledProcessError as e:
        print(f"--> [ERROR] {step_name} failed. Return code: {e.returncode}")
        if e.stderr:
            print(f"Details:\n{e.stderr[-500:]}") 
        raise e

def batch_process_and_log():
    base_path = Path(BASE_WORKSPACE).resolve()
    overlap_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("overlap_")]
    
    if not overlap_dirs:
        print(f"No target folders located inside {base_path}.")
        return

    master_log = {}

    for overlap_dir in sorted(overlap_dirs):
        print("\n" + "="*80)
        print(f" RUNNING BENCHMARK SEQUENCE FOR: {overlap_dir.name} ")
        print("="*80)

        db_path = overlap_dir / "database.db"
        img_path = overlap_dir / "images"
        list1_path = overlap_dir / "dataset1_list.txt"
        list2_path = overlap_dir / "dataset2_list.txt"
        sparse1_path = overlap_dir / "sparse1"
        sparse2_path = overlap_dir / "sparse2"
        merged_path = overlap_dir / "merged"

        for p in [sparse1_path, sparse2_path, merged_path]: p.mkdir(parents=True, exist_ok=True)

        master_log[overlap_dir.name] = {
            "timings_seconds": {},
            "total_pipeline_time_seconds": 0,
            "pipeline_status": "Incomplete",
            "metrics": {}
        }

        cmd_extract = ["colmap", "feature_extractor", "--database_path", str(db_path), "--image_path", str(img_path), "--FeatureExtraction.type", "ALIKED_N16ROT", "--AlikedExtraction.max_num_features", "2048", "--FeatureExtraction.use_gpu", "0"]
        cmd_match = ["colmap", "exhaustive_matcher", "--database_path", str(db_path), "--FeatureMatching.type", "ALIKED_LIGHTGLUE", "--FeatureMatching.use_gpu", "0"]
        cmd_map1 = ["colmap", "mapper", "--database_path", str(db_path), "--image_path", str(img_path), "--Mapper.image_list_path", str(list1_path), "--output_path", str(sparse1_path)]
        cmd_map2 = ["colmap", "mapper", "--database_path", str(db_path), "--image_path", str(img_path), "--Mapper.image_list_path", str(list2_path), "--output_path", str(sparse2_path)]
        cmd_merge = ["colmap", "model_merger", "--input_path1", str(sparse1_path / "0"), "--input_path2", str(sparse2_path / "0"), "--output_path", str(merged_path)]
        cmd_ba = ["colmap", "bundle_adjuster", "--input_path", str(merged_path), "--output_path", str(merged_path)]

        start_total_time = time.time()
        
        try:
            master_log[overlap_dir.name]["timings_seconds"]["feature_extraction"] = run_command(cmd_extract, "Step 1/6: Feature Extraction")
            master_log[overlap_dir.name]["timings_seconds"]["exhaustive_matching"] = run_command(cmd_match, "Step 2/6: Feature Matching")
            master_log[overlap_dir.name]["timings_seconds"]["mapping_dataset1"] = run_command(cmd_map1, "Step 3/6: Mapping Sub-Model 1")
            master_log[overlap_dir.name]["timings_seconds"]["mapping_dataset2"] = run_command(cmd_map2, "Step 4/6: Mapping Sub-Model 2")
            master_log[overlap_dir.name]["timings_seconds"]["model_merging"] = run_command(cmd_merge, "Step 5/6: Model Merger")
            master_log[overlap_dir.name]["timings_seconds"]["bundle_adjustment"] = run_command(cmd_ba, "Step 6/6: Global Bundle Adjustment")
            
            master_log[overlap_dir.name]["total_pipeline_time_seconds"] = time.time() - start_total_time
            master_log[overlap_dir.name]["pipeline_status"] = "Completed Successfully"
            
            print("Calculating alignment accuracy configurations against Ground Truth...")
            eval_metrics = evaluate_reconstruction(merged_path, DIR_GT)
            master_log[overlap_dir.name]["metrics"] = eval_metrics
            
            print(f"--> Done. ATE RMSE: {eval_metrics.get('ate_rmse_mm', 0.0):.2f} mm")

        except Exception as e:
            master_log[overlap_dir.name]["pipeline_status"] = f"Failed during pipeline processing. Context: {str(e)}"
            master_log[overlap_dir.name]["total_pipeline_time_seconds"] = time.time() - start_total_time
            print(f"[!] Moving to next configuration folder due to error in sequence.")
            continue

    with open(LOG_OUTPUT_FILE, "w") as out_file:
        json.dump(master_log, out_file, indent=4)
        
    print(f"\n======================================================================\n"
          f"BATCH RUN FINISHED. Report summary saved to:\n{LOG_OUTPUT_FILE}\n"
          f"======================================================================")

if __name__ == "__main__":
    batch_process_and_log()