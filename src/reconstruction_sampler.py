import json
import shutil
from pathlib import Path

def create_split_experiment(base_workspace, source_img_dir, density_json_path, target_frame_name, total_window_size=100, overlap_size=12, experiment_prefix="split"):
    base_path = Path(base_workspace).resolve()
    src_dir = Path(source_img_dir).resolve()
    
    with open(density_json_path, "r") as f:
        density_profile = json.load(f)
        
    all_frames = list(density_profile.keys())
    
    if target_frame_name not in density_profile:
        raise ValueError(f"Target frame '{target_frame_name}' not found in density metrics file.")
        
    target_pts = density_profile[target_frame_name]
    target_idx = all_frames.index(target_frame_name)
    half_window = total_window_size // 2
    start_idx = max(0, target_idx - half_window)
    end_idx = min(len(all_frames), start_idx + total_window_size)
    
    # Readjust start index if we hit the sequence tail bounds
    if end_idx - start_idx < total_window_size:
        start_idx = max(0, end_idx - total_window_size)
        
    window_frames = all_frames[start_idx:end_idx]
    
    # 3. Create Dataset 1 and Dataset 2 sequence tracking files with accurate central overlap
    local_center = len(window_frames) // 2
    half_overlap = overlap_size // 2
    
    # Slice the chronological window
    ds1_frames = window_frames[:local_center + half_overlap]
    ds2_frames = window_frames[local_center - half_overlap:]
    
    # 4. Generate unique experiment destination path
    folder_name = f"{experiment_prefix}_pts{target_pts}_frame{target_idx}"
    exp_workspace = base_path / folder_name
    exp_img_dir = exp_workspace / "images"
    
    exp_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Export frame trackers (.txt lists) matching batch script format
    with open(exp_workspace / "dataset1_list.txt", "w") as f1:
        f1.write("\n".join(ds1_frames))
        
    with open(exp_workspace / "dataset2_list.txt", "w") as f2:
        f2.write("\n".join(ds2_frames))
        
    # 6. Populate sub-workspace image directory (copies only window sequence images)
    print(f"\n[*] Generating workspace: {folder_name}")
    print(f"    - Target Frame: {target_frame_name} ({target_pts} points)")
    print(f"    - Total Windows: {len(window_frames)} frames | Overlap Window: {overlap_size} frames")
    
    for frame in window_frames:
        src_file = src_dir / frame
        dest_file = exp_img_dir / frame
        if src_file.exists() and not dest_file.exists():
            shutil.copy(src_file, dest_file)
            
    print(f"--> [SUCCESS] Experiment package structured cleanly at: {exp_workspace}")
    return exp_workspace

# Usage Example inside this file for testing validation
if __name__ == "__main__":
    # Test paths
    WORKSPACE = r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace"
    IMAGES = r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset"
    JSON_DATA = r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\analysis\scene_density_profile.json"
    
    # Example test run if executed directly
    if Path(JSON_DATA).exists():
        # Modify these values depending on your histogram valleys/peaks outputs
        create_split_experiment(WORKSPACE, IMAGES, JSON_DATA, "frame-000150.color.png", total_window_size=100, overlap_size=12, experiment_prefix="split_low")