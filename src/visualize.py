import os
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap

WORKSPACE_DIR = Path(r"C:\Users\neela\OneDrive\Documents\Github\colmap-split-merge\colmap-workspace-2")
DATASET_DIR = Path(r"C:\Users\neela\OneDrive\Documents\Github\colmap-split-merge\dataset")

POINT_SIZE = 1.5
CAMERA_FRUSTUM_SIZE = 0.005
LINE_WIDTH = 2.0

# Centralized Color Palettes (RGB Normalized 0.0 - 1.0)
COLOR_GROUND_TRUTH_TRAJECTORY = [0.0, 1.0, 0.0]  # Green
COLOR_PREDICTED_TRAJECTORY = [1.0, 0.0, 0.0]     # Red
COLOR_DATASET_1_CAMERAS = [1.0, 0.5, 0.5]        # Light Red
COLOR_DATASET_2_CAMERAS = [0.5, 0.5, 1.0]        # Light Blue
COLOR_OVERLAP_CAMERAS = [0.2, 1.0, 1.0]          # Cyan
COLOR_FULL_RECON_CAMERAS = [1.0, 0.8, 0.0]       # Orange
COLOR_UNASSIGNED_CAMERAS = [1.0, 0.1, 1.0]       # Purple
COLOR_FULL_RECON_POINTS = [0.7, 0.0, 1.0]        # Purple-ish blue


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
    
    T = np.eye(4)
    T[:3, :3] = Scale * Rotation
    T[:3, 3] = Translation
    return T, Scale

def make_rigid(matrix_4x4):
    T_rigid = np.eye(4)
    R = matrix_4x4[:3, :3]
    scale = np.linalg.norm(R, axis=0)
    T_rigid[:3, :3] = R / scale
    T_rigid[:3, 3] = matrix_4x4[:3, 3]
    return T_rigid

def camera_to_world_pose(image):
    if hasattr(image, "cam_from_world"):
        cfw = image.cam_from_world
        rigid3d = cfw() if callable(cfw) else cfw
        R_w2c = rigid3d.rotation.matrix() if hasattr(rigid3d.rotation, "matrix") else rigid3d.rotation
        t_w2c = rigid3d.translation if hasattr(rigid3d, "translation") else rigid3d.tvec
    elif hasattr(image, "rotation_matrix"):
        R_w2c = image.rotation_matrix
        t_w2c = image.tvec
    else:
        raise AttributeError("Unsupported pycolmap version pose layout structure.")
    
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c() if callable(R_w2c) else R_w2c
    T_w2c[:3, 3] = t_w2c() if callable(t_w2c) else t_w2c
    return np.linalg.inv(T_w2c)

def load_ground_truth_poses(dataset_path):
    dataset_p = Path(dataset_path).resolve()
    gt_poses = {}
    
    for pose_file in dataset_p.rglob("*.pose.txt"):
        base_stem = pose_file.name.split('.')[0].lower()
        try:
            matrix = np.loadtxt(pose_file)
            if matrix.shape == (4, 4):
                gt_poses[base_stem] = matrix
            elif matrix.shape == (3, 4):
                full_matrix = np.eye(4)
                full_matrix[:3, :] = matrix
                gt_poses[base_stem] = full_matrix
        except Exception:
            continue
    return gt_poses

def create_camera_frustum(T_pose, color, size=0.03):
    local_pts = np.array([
        [0, 0, 0], 
        [-size, -size, size * 2], 
        [size, -size, size * 2], 
        [size, size, size * 2], 
        [-size, size, size * 2]
    ])
    world_pts = (T_pose @ np.hstack([local_pts, np.ones((5, 1))]).T).T[:, :3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

def evaluate_and_render_workspace(workspace_path, gt_poses):
    w_path = Path(workspace_path).resolve()
    
    reconstruction = None
    model_dir = None
    candidate_subdirs = ["merged", "sparse/0", "sparse"]
    
    for sub_dir in candidate_subdirs:
        target_path = w_path / sub_dir
        if target_path.exists():
            try:
                recon_test = pycolmap.Reconstruction(target_path)
                if len(recon_test.images) > 0:
                    reconstruction = recon_test
                    model_dir = target_path
                    break
            except Exception:
                continue

    if reconstruction is None:
        print(f"Skipping {w_path.name}: Could not find a valid sparse reconstruction containing registered cameras.")
        return

    list1_path = w_path / "dataset1_list.txt"
    list2_path = w_path / "dataset2_list.txt"
    is_split_pipeline = list1_path.exists() or list2_path.exists() or model_dir.name == "merged"

    set_ds1, set_ds2 = set(), set()
    if list1_path.exists():
        with open(list1_path, "r") as f:
            set_ds1 = set(line.strip() for line in f if line.strip())
    if list2_path.exists():
        with open(list2_path, "r") as f:
            set_ds2 = set(line.strip() for line in f if line.strip())
    set_overlap = set_ds1.intersection(set_ds2)

    reg_images = reconstruction.images
    img_id_to_name = {im_id: im.name for im_id, im in reg_images.items()}
    
    matched_keys = []
    predicted_centers = []
    true_centers = []

    for im_id in sorted(list(reg_images.keys())):
        im = reg_images[im_id]
        name = im.name
        base_stem = Path(name).name.split('.')[0].lower()
        
        if base_stem in gt_poses:
            matched_keys.append((im_id, base_stem))
            predicted_centers.append(im.projection_center())
            true_centers.append(gt_poses[base_stem][:3, 3])

    if len(matched_keys) < 3:
        print(f"Skipping {w_path.name}: Insufficient overlapping frames matched with ground truth ({len(matched_keys)} found).")
        return

    predicted_centers = np.array(predicted_centers)
    true_centers = np.array(true_centers)
    
    transform_matrix, scale_factor = umeyama_alignment(predicted_centers, true_centers)

    aligned_predicted_centers = []
    errors_mm = []
    for i, (im_id, base_stem) in enumerate(matched_keys):
        aligned_c = (scale_factor * (predicted_centers[i] @ transform_matrix[:3, :3].T / scale_factor)) + transform_matrix[:3, 3]
        aligned_predicted_centers.append(aligned_c)
        err = np.linalg.norm(aligned_c - true_centers[i]) * 1000.0
        errors_mm.append(err)

    errors_mm = np.array(errors_mm)
    ate_rmse = np.sqrt(np.mean(errors_mm ** 2))
    ate_mean = np.mean(errors_mm)
    ate_max = np.max(errors_mm)

    xyz_list, rgb_list = [], []
    for point3D_id, point3D in reconstruction.points3D.items():
        xyz_list.append(point3D.xyz)
        
        if not is_split_pipeline:
            pt_color = COLOR_FULL_RECON_POINTS
        else:
            observing_images = [img_id_to_name[te.image_id] for te in point3D.track.elements if te.image_id in img_id_to_name]
            seen_ds1 = any(i in set_ds1 for i in observing_images)
            seen_ds2 = any(i in set_ds2 for i in observing_images)
            
            if seen_ds1 and seen_ds2:
                pt_color = COLOR_OVERLAP_CAMERAS
            elif seen_ds1:
                pt_color = COLOR_DATASET_1_CAMERAS
            elif seen_ds2:
                pt_color = COLOR_DATASET_2_CAMERAS
            else:
                pt_color = [0.3, 0.3, 0.3]
                
        rgb_list.append(pt_color)

    xyz_arr = np.array(xyz_list)
    rgb_arr = np.array(rgb_list)

    print("\n" + "="*60)
    print(f" METRIC SUMMARY REPORT: {w_path.name.upper()} ")
    print("="*60)
    print(f"Pipeline Profile          : {'Split and Merge' if is_split_pipeline else 'Full Reconstruction Sequence'}")
    print(f"Model Data Path Source    : {model_dir.relative_to(w_path)}")
    print(f"Total Reconstructed Points: {len(xyz_arr)}")
    print(f"Registered Map Cameras    : {len(reg_images)}")
    print(f"Evaluation Pose Anchors   : {len(matched_keys)}")
    print(f"Calculated Scale Factor   : {scale_factor:.5f}")
    print(f"Trajectory ATE RMSE       : {ate_rmse:.2f} mm")
    print(f"Trajectory ATE Mean       : {ate_mean:.2f} mm")
    print(f"Trajectory ATE Max        : {ate_max:.2f} mm")
    print("="*60)

    geometries = []

    if len(xyz_arr) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_arr)
        pcd.colors = o3d.utility.Vector3dVector(rgb_arr)
        pcd.transform(transform_matrix)
        
        _, inliers = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        pcd_filtered = pcd.select_by_index(inliers)
        geometries.append(pcd_filtered)

    active_gt_stems = sorted([stem for _, stem in matched_keys])
    gt_trajectory_points = [gt_poses[stem][:3, 3] for stem in active_gt_stems]
    gt_trajectory_lines = [[i, i+1] for i in range(len(gt_trajectory_points)-1)]
    
    gt_line_set = o3d.geometry.LineSet()
    gt_line_set.points = o3d.utility.Vector3dVector(gt_trajectory_points)
    gt_line_set.lines = o3d.utility.Vector2iVector(gt_trajectory_lines)
    gt_line_set.paint_uniform_color(COLOR_GROUND_TRUTH_TRAJECTORY)
    geometries.append(gt_line_set)

    pred_trajectory_lines = [[i, i+1] for i in range(len(aligned_predicted_centers)-1)]
    pred_line_set = o3d.geometry.LineSet()
    pred_line_set.points = o3d.utility.Vector3dVector(aligned_predicted_centers)
    pred_line_set.lines = o3d.utility.Vector2iVector(pred_trajectory_lines)
    pred_line_set.paint_uniform_color(COLOR_PREDICTED_TRAJECTORY)
    geometries.append(pred_line_set)

    for im_id in reg_images.keys():
        im = reg_images[im_id]
        name = im.name
        c2w_orig = camera_to_world_pose(im)
        c2w_aligned = make_rigid(transform_matrix @ c2w_orig)
        
        if not is_split_pipeline:
            cam_color = COLOR_FULL_RECON_CAMERAS
        elif name in set_overlap:
            cam_color = COLOR_OVERLAP_CAMERAS
        elif name in set_ds1:
            cam_color = COLOR_DATASET_1_CAMERAS
        elif name in set_ds2:
            cam_color = COLOR_DATASET_2_CAMERAS
        else:
            cam_color = COLOR_UNASSIGNED_CAMERAS
            
        frustum = create_camera_frustum(c2w_aligned, cam_color, CAMERA_FRUSTUM_SIZE)
        geometries.append(frustum)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Metrics Engine - {w_path.name}", width=1280, height=720)
    
    render_options = vis.get_render_option()
    render_options.point_size = POINT_SIZE
    render_options.line_width = LINE_WIDTH
    render_options.background_color = np.array([0.05, 0.05, 0.05])
    
    for geom in geometries:
        vis.add_geometry(geom)
        
    vis.run()
    vis.destroy_window()

def main():
    if not DATASET_DIR.exists():
        print(f"Error: Target data path directory does not exist: {DATASET_DIR}")
        return

    print("Pre-indexing structural ground truth dataset pose configurations...")
    gt_poses = load_ground_truth_poses(DATASET_DIR)
    if not gt_poses:
        print(f"Error: No valid *.pose.txt anchors could be found inside: {DATASET_DIR}")
        return
    print(f"Successfully tracked and loaded {len(gt_poses)} ground truth spatial points.")

    if not WORKSPACE_DIR.exists():
        print(f"Error: Active execution workspace layout cannot be found: {WORKSPACE_DIR}")
        return

    valid_workspaces = []
    
    workspace_name_lower = WORKSPACE_DIR.name.lower()
    if any(k in workspace_name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
        valid_workspaces.append(WORKSPACE_DIR)
    else:
        for directory in sorted(list(WORKSPACE_DIR.iterdir())):
            if directory.is_dir():
                name_lower = directory.name.lower()
                if any(k in name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
                    valid_workspaces.append(directory)

    if not valid_workspaces:
        print("No active reconstruction pipeline configurations detected inside the tracked workspace root.")
        return

    print(f"Discovered {len(valid_workspaces)} reconstruction structures for processing. Running rendering canvas loop...")
    for path in valid_workspaces:
        evaluate_and_render_workspace(path, gt_poses)

if __name__ == "__main__":
    main()