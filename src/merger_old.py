import os
from pathlib import Path
import numpy as np
import open3d as o3d
import pycolmap

WORKSPACE = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\overlap_3")
dir_merged = WORKSPACE / "merged"
dir_gt = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")

# Load lists
with open(WORKSPACE / "dataset1_list.txt", "r") as f:
    set_ds1 = set(line.strip() for line in f if line.strip())
with open(WORKSPACE / "dataset2_list.txt", "r") as f:
    set_ds2 = set(line.strip() for line in f if line.strip())

set_overlap_imgs = set_ds1.intersection(set_ds2)

COLOR_DS1 = [1.0, 0.2, 0.2]        
COLOR_DS2 = [0.2, 0.2, 1.0]        
COLOR_OVERLAP = [0.2, 1.0, 1.0] 
COLOR_GT = [0.0, 1.0, 0.0]    

recon = pycolmap.Reconstruction(dir_merged)
reg_images = {im.name: im for im_id, im in recon.images.items() if recon.exists_image(im_id)}
img_id_to_name = {im_id: im.name for im_id, im in recon.images.items() if recon.exists_image(im_id)}

print(f"Loaded merged model containing {len(reg_images)} registered frames.")

# --- Helper Functions ---
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
        raise AttributeError("Unsupported pycolmap version pose layout.")
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
    if np.linalg.det(U @ Vt) < 0: S_mat[dim-1, dim-1] = -1
    Rotation = U @ S_mat @ Vt
    Scale = np.trace(np.diag(S) @ S_mat) / np.var(src_centered, axis=0).sum()
    Translation = dst_mean - Scale * Rotation @ src_mean
    T = np.eye(4); T[:3, :3] = Scale * Rotation; T[:3, 3] = Translation
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

# Alignment
recon_points, gt_points, valid_names = [], [], []
for name in sorted(list(reg_images.keys())):
    base_name = Path(name).name.split('.')[0]
    pose_file = dir_gt / f"{base_name}.pose.txt"
    if pose_file.exists():
        gt_points.append(np.loadtxt(pose_file)[:3, 3])
        recon_points.append(reg_images[name].projection_center())
        valid_names.append(name)

X_recon, Y_gt = np.array(recon_points), np.array(gt_points)
transform_recon_to_gt, scale_recon_to_gt = umeyama_alignment(X_recon, Y_gt)

# Prepare Point Cloud with Statistical Outlier Removal
xyz_list, rgb_list = [], []
for point3D_id, pt3D in recon.points3D.items():
    observing_images = [img_id_to_name[te.image_id] for te in pt3D.track.elements if te.image_id in img_id_to_name]
    seen_ds1, seen_ds2 = any(i in set_ds1 for i in observing_images), any(i in set_ds2 for i in observing_images)
    pt_color = COLOR_OVERLAP if (seen_ds1 and seen_ds2) else (COLOR_DS1 if seen_ds1 else (COLOR_DS2 if seen_ds2 else [0.3, 0.3, 0.3]))
    xyz_list.append(pt3D.xyz)
    rgb_list.append(pt_color)

pcd_raw = o3d.geometry.PointCloud()
pcd_raw.points = o3d.utility.Vector3dVector(np.array(xyz_list))
pcd_raw.colors = o3d.utility.Vector3dVector(np.array(rgb_list))
pcd_raw.transform(transform_recon_to_gt)

# Clean outliers
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
pcd_colored = pcd_raw.select_by_index(ind)

# Build Geometries
geometries = [pcd_colored]
traj_gt = o3d.geometry.LineSet()
traj_gt.points = o3d.utility.Vector3dVector(Y_gt)
traj_gt.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(Y_gt)-1)])
traj_gt.paint_uniform_color(COLOR_GT)
geometries.append(traj_gt)

scene_extent = np.max(Y_gt, axis=0) - np.min(Y_gt, axis=0)
frustum_size = max(np.linalg.norm(scene_extent) * 0.0012, 0.0003)

for name in sorted(list(reg_images.keys())):
    cam_color = COLOR_OVERLAP if name in set_overlap_imgs else (COLOR_DS1 if name in set_ds1 else (COLOR_DS2 if name in set_ds2 else [0.5, 0.5, 0.5]))
    T_final = make_rigid(transform_recon_to_gt @ camera_to_world_pose(reg_images[name]))
    geometries.append(create_camera_frustum(T_final, cam_color, frustum_size))

# Visualization
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Color-Coded Model Merge Verification (7-Scenes)", width=1600, height=900)
for geom in geometries: vis.add_geometry(geom)

ctr = vis.get_view_control()
ctr.set_lookat(np.mean(np.asarray(pcd_colored.points), axis=0))
ctr.set_zoom(0.5) # Zoom value < 1.0 zooms in, > 1.0 zooms out

opt = vis.get_render_option()
opt.point_size = 0.5
opt.background_color = np.asarray([0.08, 0.08, 0.08])

vis.run()
vis.destroy_window()