import os
from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap

WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\mean_400_common_30")
dir_merged = WORKSPACE / "merged"
dir_gt = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")

print(f"Evaluating Workspace: {WORKSPACE.name}")

set_ds1, set_ds2 = set(), set()
list1_path = WORKSPACE / "dataset1_list.txt"
list2_path = WORKSPACE / "dataset2_list.txt"

if list1_path.exists():
    with open(list1_path, "r") as f:
        set_ds1 = set(line.strip() for line in f if line.strip())
if list2_path.exists():
    with open(list2_path, "r") as f:
        set_ds2 = set(line.strip() for line in f if line.strip())

set_overlap_imgs = set_ds1.intersection(set_ds2)

COLOR_DS1 = [1.0, 0.5, 0.5]       # Red
COLOR_DS2 = [0.5, 0.5, 1.0]       # Blue
COLOR_OVERLAP = [0.2, 1.0, 1.0]   # Cyan
COLOR_GT = [0.0, 1.0, 0.0]        # Green Ground Truth Trajectory

print(f"Recursively indexing ground truth directory: {dir_gt}")
gt_pose_map = {}
if dir_gt.exists():
    for p in dir_gt.rglob("*.txt"):
        base_stem = p.name.lower().split('.')[0]
        gt_pose_map[base_stem] = p

print(f"Indexed {len(gt_pose_map)} total ground truth pose matrices available.")

try:
    recon = pycolmap.Reconstruction(dir_merged)
except Exception as e:
    raise RuntimeError(f"Could not load sparse reconstruction at {dir_merged}. Check your path. Error: {e}")

reg_images = {im.name: im for im_id, im in recon.images.items() if recon.exists_image(im_id)}
img_id_to_name = {im_id: im.name for im_id, im in recon.images.items() if recon.exists_image(im_id)}
print(f"Loaded merged model containing {len(reg_images)} registered frames.")

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

def make_rigid(T):
    T_rigid = np.eye(4)
    R = T[:3, :3]
    scale = np.linalg.norm(R, axis=0)
    T_rigid[:3, :3] = R / scale
    T_rigid[:3, 3] = T[:3, 3]
    return T_rigid

def create_camera_frustum(T_pose, color, size=0.03):
    local_pts = np.array([[0,0,0], [-size,-size,size*2], [size,-size,size*2], [size,size,size*2], [-size,size,size*2]])
    world_pts = (T_pose @ np.hstack([local_pts, np.ones((5,1))]).T).T[:, :3]
    lines = [[0,1], [0,2], [0,3], [0,4], [1,2], [2,3], [3,4], [4,1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(world_pts)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set

recon_points, gt_points, valid_names = [], [], []

for name in sorted(list(reg_images.keys())):
    base_stem = Path(name).stem.lower().split('.')[0]
    
    if base_stem in gt_pose_map:
        pose_file = gt_pose_map[base_stem]
        try:
            gt_mat = np.loadtxt(pose_file)
            if gt_mat.shape[0] >= 3 and gt_mat.shape[1] >= 4:
                gt_points.append(gt_mat[:3, 3])
                recon_points.append(reg_images[name].projection_center())
                valid_names.append(name)
        except Exception:
            continue

if len(gt_points) < 3:
    print(f"\n[ERROR] Match Verification Failed. Found {len(gt_points)} matched files.")
    print(f"Sample registered image stems: {list(Path(n).stem.lower().split('.')[0] for n in list(reg_images.keys())[:3])}")
    print(f"Sample indexed GT stems: {list(gt_pose_map.keys())[:3]}")
    raise ValueError("Insufficient overlapping tracking nodes found to execute Umeyama trajectory fitting.")

X_recon, Y_gt = np.array(recon_points), np.array(gt_points)
transform_recon_to_gt, scale_recon_to_gt = umeyama_alignment(X_recon, Y_gt)

aligned_to_gt = (scale_recon_to_gt * (X_recon @ transform_recon_to_gt[:3, :3].T / scale_recon_to_gt)) + transform_recon_to_gt[:3, 3]
ate_errors = np.linalg.norm(Y_gt - aligned_to_gt, axis=1)
ate_rmse = np.sqrt(np.mean(ate_errors**2))

print("\n" + "="*50)
print(f"  METRIC PERFORMANCE REPORT: {WORKSPACE.name}")
print("="*50)
print(f"Total Model Registered Images : {len(reg_images)}")
print(f"Aligned Sequence Positions    : {len(gt_points)}")
print(f"Calculated Scale Factor       : {scale_recon_to_gt:.5f}")
print(f"ATE RMSE                      : {ate_rmse * 1000.0:.2f} mm ({ate_rmse:.4f} meters)")
print(f"Mean Spatial Distance Error   : {np.mean(ate_errors) * 1000.0:.2f} mm")
print(f"Max Spatial Distance Error    : {np.max(ate_errors) * 1000.0:.2f} mm")
print("="*50 + "\n")

xyz_list, rgb_list = [], []
for point3D_id, pt3D in recon.points3D.items():
    observing_images = [img_id_to_name[te.image_id] for te in pt3D.track.elements if te.image_id in img_id_to_name]
    seen_ds1 = any(i in set_ds1 for i in observing_images)
    seen_ds2 = any(i in set_ds2 for i in observing_images)
    
    pt_color = COLOR_OVERLAP if (seen_ds1 and seen_ds2) else (COLOR_DS1 if seen_ds1 else (COLOR_DS2 if seen_ds2 else [0.3, 0.3, 0.3]))
    xyz_list.append(pt3D.xyz)
    rgb_list.append(pt_color)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(np.array(xyz_list))
pcd_raw.colors = o3d.utility.Vector3dVector(np.array(rgb_list))
pcd_raw.transform(transform_recon_to_gt)

cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
pcd_colored = pcd_raw.select_by_index(ind)

geometries = [pcd_colored]

traj_gt = o3d.geometry.LineSet()
traj_gt.points = o3d.utility.Vector3dVector(Y_gt)
traj_gt.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(Y_gt)-1)])
traj_gt.paint_uniform_color(COLOR_GT)
geometries.append(traj_gt)

scene_extent = np.max(Y_gt, axis=0) - np.min(Y_gt, axis=0)
frustum_size = max(np.linalg.norm(scene_extent) * 0.0024, 0.0006)

for name in sorted(list(reg_images.keys())):
    cam_color = COLOR_OVERLAP if name in set_overlap_imgs else (COLOR_DS1 if name in set_ds1 else (COLOR_DS2 if name in set_ds2 else [0.5, 0.5, 0.5]))
    T_final = make_rigid(transform_recon_to_gt @ camera_to_world_pose(reg_images[name]))
    geometries.append(create_camera_frustum(T_final, cam_color, frustum_size))

vis = o3d.visualization.Visualizer()
vis.create_window(window_name=f"Reconstruction Accuracy Profiler - {WORKSPACE.name}", width=1600, height=900)
for geom in geometries: 
    vis.add_geometry(geom)

ctr = vis.get_view_control()
ctr.set_lookat(np.mean(np.asarray(pcd_colored.points), axis=0))
ctr.set_zoom(0.6)

opt = vis.get_render_option()
opt.point_size = 1.2
opt.background_color = np.asarray([0.05, 0.05, 0.05])

print("Launching Open3D rendering canvas viewport...")
vis.run()
vis.destroy_window()