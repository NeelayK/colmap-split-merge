############################################################
## Colmap Merger Script for 7Scenes Dataset               ##
############################################################
#  This script merges two Colmap datasets for 7Scenes     #
#  by aligning their sparse reconstructions.              #
############################################################
# Feel free to use, modify the source and workspace paths below as needed. :)
# Some explanations of the script's functionality are provided in the comments.
# Also if you found this useful, please consider starring the repo on GitHub! It would me a lot. Thanks!

# ------------------------------------------------------
# import necessary libraries
# ------------------------------------------------------

import pycolmap
from pathlib import Path
import numpy as np
import open3d as o3d

# Change these paths to your actual Colmap workspace locations for Dataset 1 and Dataset 2
dir_a = Path(r"C:\Users\...\colmap\colmap_workspace\dataset1\sparse\0")
dir_b = Path(r"C:\Users\...\colmap\colmap_workspace\dataset2\sparse\0")

recon_a = pycolmap.Reconstruction(dir_a)
recon_b = pycolmap.Reconstruction(dir_b)
reg_images_a = {im.name: im for im_id, im in recon_a.images.items() if recon_a.exists_image(im_id)}
reg_images_b = {im.name: im for im_id, im in recon_b.images.items() if recon_b.exists_image(im_id)}

common_names = sorted(list(set(reg_images_a.keys()) & set(reg_images_b.keys())))
print(f"Found {len(common_names)} registered overlapping images.")

if len(common_names) < 3:
    raise ValueError("Not enough overlapping registered images to reliably calculate 3D scale alignment.")

points_a = []
points_b = []

for name in common_names:
    im_a = reg_images_a[name]
    im_b = reg_images_b[name]
    points_a.append(im_a.projection_center())
    points_b.append(im_b.projection_center())

X = np.array(points_b)
Y = np.array(points_a)

def umeyama_alignment(src, dst):
    num_pts, dim = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    H = (dst_centered.T @ src_centered) / num_pts
    
    U, S, Vt = np.linalg.svd(H)
    
    d = np.linalg.det(U @ Vt)
    S_mat = np.eye(dim)
    if d < 0:
        S_mat[dim-1, dim-1] = -1
        
    Rotation = U @ S_mat @ Vt
    src_var = np.var(src_centered, axis=0).sum()
    Scale = np.trace(np.diag(S) @ S_mat) / src_var
    
    Translation = dst_mean - Scale * Rotation @ src_mean
    
    T = np.eye(4)
    T[:3, :3] = Scale * Rotation
    T[:3, 3] = Translation
    return T, Scale

transformation, scale_factor = umeyama_alignment(X, Y)
aligned_X = (scale_factor * (X @ transformation[:3, :3].T / scale_factor)) + transformation[:3, 3]
errors = np.linalg.norm(Y - aligned_X, axis=1)
rmse = np.sqrt(np.mean(errors**2))

print(f"\n\n")
print(f"Scale Factor (Model B -> Model A): {scale_factor:.5f}")
print(f"Trajectory RMSE Error: {rmse:.4f} units")
if rmse > 1.0: 
    print("High RMSE detected! This may indicate a poor alignment.")


pcd_a = o3d.geometry.PointCloud()
pcd_b = o3d.geometry.PointCloud()
pcd_a.points = o3d.utility.Vector3dVector(np.array([p.xyz for p in recon_a.points3D.values()]))
pcd_b.points = o3d.utility.Vector3dVector(np.array([p.xyz for p in recon_b.points3D.values()]))
pcd_a.paint_uniform_color([0.8, 0.2, 0.2])
pcd_b.paint_uniform_color([0.2, 0.4, 0.8])

pcd_b.transform(transformation)
traj_a = o3d.geometry.LineSet()
traj_a.points = o3d.utility.Vector3dVector(Y)
traj_a.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(Y)-1)])
traj_a.paint_uniform_color([1, 0.8, 0])

traj_b = o3d.geometry.LineSet()
traj_b.points = o3d.utility.Vector3dVector(X)
traj_b.transform(transformation)
traj_b.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(X)-1)])
traj_b.paint_uniform_color([0, 1, 1])


scene_extent = np.max(Y, axis=0) - np.min(Y, axis=0)
scene_size = np.linalg.norm(scene_extent)
cam_radius = max(scene_size * 0.005, 0.001)

geometries = [pcd_a, pcd_b, traj_a, traj_b]

for pt in Y:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cam_radius)
    sphere.translate(pt)
    sphere.paint_uniform_color([1, 0.8, 0]) 
    geometries.append(sphere)

for pt in np.asarray(traj_b.points):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cam_radius)
    sphere.translate(pt)
    sphere.paint_uniform_color([0, 1, 1])
    geometries.append(sphere)

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Precision Aligned Scale-Drift Visualizer", width=1600, height=900)

for geom in geometries:
    vis.add_geometry(geom)

opt = vis.get_render_option()
opt.point_size = 1.0
opt.background_color = np.asarray([0.1, 0.1, 0.1])


vis.run()
vis.destroy_window()