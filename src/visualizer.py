import sys
from pathlib import Path
import numpy as np

# Sniff for visualization engines safely
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False



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
    return T, Scale, Rotation


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
        raise AttributeError("Unsupported or unrecognized pycolmap image pose layout structure.")
    
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_w2c() if callable(R_w2c) else R_w2c
    T_w2c[:3, 3] = t_w2c() if callable(t_w2c) else t_w2c
    return np.linalg.inv(T_w2c)


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


def parse_color(color_str):
    return [float(x.strip()) for x in color_str.split(",")]

def run_visualization(args):
    if not PYCOLMAP_AVAILABLE:
        raise ImportError("pycolmap is missing. Run environment setup or install dependencies first.")
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D wrapper engine missing. Please install open3d to visualize.")

    workspace_path = Path(args.workspace).resolve()
    gt_path = Path(args.gt).resolve()
    dir_merged = workspace_path / "merged"

    if not dir_merged.exists():
        raise FileNotFoundError(f"Sparse merged model directory missing at target location: {dir_merged}")

    set_ds1, set_ds2 = set(), set()
    list1_path = workspace_path / "dataset1_list.txt"
    list2_path = workspace_path / "dataset2_list.txt"

    if list1_path.exists():
        with open(list1_path, "r") as f:
            set_ds1 = set(line.strip() for line in f if line.strip())
    if list2_path.exists():
        with open(list2_path, "r") as f:
            set_ds2 = set(line.strip() for line in f if line.strip())

    set_overlap_imgs = set_ds1.intersection(set_ds2)

    COLOR_DS1 = parse_color(args.color_ds1)
    COLOR_DS2 = parse_color(args.color_ds2)
    COLOR_OVERLAP = parse_color(args.color_overlap)
    COLOR_GT = parse_color(args.color_gt)

    gt_pose_map = {}
    if gt_path.exists():
        for p in gt_path.rglob("*.txt"):
            base_stem = p.name.lower().split('.')[0]
            gt_pose_map[base_stem] = p

    print(f"-> Loading model reconstruction from: {dir_merged.name}")
    recon = pycolmap.Reconstruction(dir_merged)
    reg_images = {im.name: im for im_id, im in recon.images.items() if recon.exists_image(im_id)}
    img_id_to_name = {im_id: im.name for im_id, im in recon.images.items() if recon.exists_image(im_id)}

    recon_points, gt_points, valid_names = [], [], []
    for name in sorted(list(reg_images.keys())):
        base_stem = Path(name).stem.lower().split('.')[0]
        if base_stem in gt_pose_map:
            try:
                gt_mat = np.loadtxt(gt_pose_map[base_stem])
                if gt_mat.shape[0] >= 3 and gt_mat.shape[1] >= 4:
                    gt_points.append(gt_mat[:3, 3])
                    recon_points.append(reg_images[name].projection_center())
                    valid_names.append(name)
            except Exception:
                continue

    if len(gt_points) < 3:
        print(f"\n[ALIGNMENT ERROR]: Matrix evaluation skipped. Found only {len(gt_points)} matching nodes.")
        sys.exit(1)

    X_recon, Y_gt = np.array(recon_points), np.array(gt_points)
    T_align, scale_factor, _ = umeyama_alignment(X_recon, Y_gt)

    aligned_to_gt = (scale_factor * (X_recon @ T_align[:3, :3].T / scale_factor)) + T_align[:3, 3]
    ate_errors = np.linalg.norm(Y_gt - aligned_to_gt, axis=1)
    ate_rmse = np.sqrt(np.mean(ate_errors**2))
    ate_mm = ate_rmse * 1000.0

    print("\n" + "="*60)
    print(f"TRAJECTORY SPATIAL METRIC PERFORMANCE REPORT")
    print("="*60)
    print(f" Total Active Workspace Nodes   : {len(reg_images)}")
    print(f" Paired Trajectory Nodes        : {len(gt_points)}")
    print(f" Calculated Absolute Scale      : {scale_factor:.6f}")
    print(f" Absolute Trajectory Error RMSE : {ate_mm:.2f} mm")
    print(f" Mean Translation Offset Error  : {np.mean(ate_errors) * 1000.0:.2f} mm")
    print(f" Maximum Peak Geometric Deviation: {np.max(ate_errors) * 1000.0:.2f} mm")
    print("="*60 + "\n")

    print("-> Preparing Open3D scene graph layer...")
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
    pcd_raw.transform(T_align)

    cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd_colored = pcd_raw.select_by_index(ind)

    geometries = [pcd_colored]

    traj_gt = o3d.geometry.LineSet()
    traj_gt.points = o3d.utility.Vector3dVector(Y_gt)
    traj_gt.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(Y_gt)-1)])
    traj_gt.paint_uniform_color(COLOR_GT)
    geometries.append(traj_gt)

    scene_extent = np.max(Y_gt, axis=0) - np.min(Y_gt, axis=0)
    base_frustum_size = max(np.linalg.norm(scene_extent) * 0.0018, 0.0004)
    frustum_size = base_frustum_size * args.camera_scale

    for name in sorted(list(reg_images.keys())):
        cam_color = COLOR_OVERLAP if name in set_overlap_imgs else (COLOR_DS1 if name in set_ds1 else (COLOR_DS2 if name in set_ds2 else [0.5, 0.5, 0.5]))
        T_final = make_rigid(T_align @ camera_to_world_pose(reg_images[name]))
        geometries.append(create_camera_frustum(T_final, cam_color, frustum_size))

    vis = o3d.visualization.Visualizer()
    
    window_title = f"SfM Accuracy Profiler: {workspace_path.name} | ATE RMSE: {ate_mm:.2f} mm"
    vis.create_window(window_name=window_title, width=1600, height=900)
    
    for geom in geometries: 
        vis.add_geometry(geom)

    ctr = vis.get_view_control()
    if len(pcd_colored.points) > 0:
        ctr.set_lookat(np.mean(np.asarray(pcd_colored.points), axis=0))
    ctr.set_zoom(0.6)

    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.asarray([0.06, 0.06, 0.06])

    print("Launching Open3D interactive canvas window...")
    vis.run()
    vis.destroy_window()