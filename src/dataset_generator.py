import os
import shutil
import json
import random
import math
from pathlib import Path

try:
    import pycolmap
    PYCOLMAP_AVAILABLE = True
except ImportError:
    PYCOLMAP_AVAILABLE = False


def _setup_workspace_folders(target_dir, images_map, ds1_names, ds2_names, metadata):
    """Core helper to build the COLMAP folder structure, copy files, and write lists."""
    if target_dir.exists():
        shutil.rmtree(target_dir)

    unified_img_dir = target_dir / "images"
    sparse1_dir = target_dir / "sparse1"
    sparse2_dir = target_dir / "sparse2"
    merged_dir = target_dir / "merged"

    for directory in [unified_img_dir, sparse1_dir, sparse2_dir, merged_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    union_names = sorted(list(set(ds1_names).union(set(ds2_names))))

    for name in union_names:
        shutil.copy(images_map[name], unified_img_dir / name)

    with open(target_dir / "dataset1_list.txt", "w") as f:
        for name in sorted(ds1_names):
            f.write(f"{name}\n")

    with open(target_dir / "dataset2_list.txt", "w") as f:
        for name in sorted(ds2_names):
            f.write(f"{name}\n")

    with open(target_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return len(union_names)


def generate_datasets(args):
    """Main routing function called by pipeline.py"""
    source_dir = Path(args.dataset).resolve()
    output_root = Path(args.output_dir).resolve() / args.project

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory missing: {source_dir}")

    images = sorted(list(source_dir.glob("*.color.png")))
    if not images:
        images = sorted([p for p in source_dir.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    if not images:
        raise FileNotFoundError(f"No valid images found in {source_dir}")

    images_map = {img.name: img for img in images}
    img_names = sorted(list(images_map.keys()))
    total_images = len(img_names)

    requested_start = args.start_frame if args.start_frame is not None else 0
    requested_end = args.end_frame if args.end_frame is not None else total_images

    start_frame = max(0, min(total_images - 1, requested_start))
    end_frame = max(start_frame + 1, min(total_images, requested_end))
    
    seq_len = end_frame - start_frame
    active_names = img_names[start_frame:end_frame]

    print(f"\nBase Project Directory: {output_root}")
    print(f"Sequence Range: Frames {start_frame} to {end_frame} ({seq_len} images active)")

    base_metadata = {
        "project_name": args.project,
        "source_dataset_path": str(source_dir),
        "sequence_range": [start_frame, end_frame]
    }

    if args.type == "full":
        target_dir = output_root / "full_reconstruction"
        meta = base_metadata.copy()
        meta.update({"generation_mode": "full_no_split"})
        
        _setup_workspace_folders(target_dir, images_map, active_names, active_names, meta)
        print(f"   -> Created Full Baseline Workspace: {target_dir.name}")


    elif args.type == "overlap":
        raw_center_str = str(args.center_frame).split(",")[0].strip()
        center = max(start_frame, min(end_frame, int(raw_center_str)))
        
        percentages = [float(p.strip()) for p in str(args.overlap_pct).split(",")]

        for pct in percentages:
            overlap_size = int(round(seq_len * (pct / 100.0)))
            half_overlap = overlap_size // 2

            ds1_start = start_frame
            ds1_end = min(end_frame, center + (overlap_size - half_overlap))
            ds2_start = max(start_frame, center - half_overlap)
            ds2_end = end_frame

            ds1_names = img_names[ds1_start:ds1_end]
            ds2_names = img_names[ds2_start:ds2_end]
            
            folder_name = f"overlap_{int(pct) if pct.is_integer() else pct}_pct"
            target_dir = output_root / folder_name
            
            meta = base_metadata.copy()
            meta.update({"generation_mode": "overlap_percentage", "value": pct, "center_frame": center})
            
            _setup_workspace_folders(target_dir, images_map, ds1_names, ds2_names, meta)
            print(f"   -> Created {folder_name} (Shared: {len(set(ds1_names).intersection(set(ds2_names)))})")

    elif args.type == "center":
        overlap_pct = float(str(args.overlap_pct).split(",")[0]) 
        centers = [int(c.strip()) for c in str(args.center_frame).split(",")]
        
        overlap_size = int(round(seq_len * (overlap_pct / 100.0)))
        half_overlap = overlap_size // 2

        for center in centers:
            center_clamped = max(start_frame, min(end_frame, center))
            
            ds1_start = start_frame
            ds1_end = min(end_frame, center_clamped + (overlap_size - half_overlap))
            ds2_start = max(start_frame, center_clamped - half_overlap)
            ds2_end = end_frame

            ds1_names = img_names[ds1_start:ds1_end]
            ds2_names = img_names[ds2_start:ds2_end]
            
            folder_name = f"split_{center_clamped}_overlap_{int(overlap_pct) if overlap_pct.is_integer() else overlap_pct}"
            target_dir = output_root / folder_name
            
            meta = base_metadata.copy()
            meta.update({"generation_mode": "center_frame_pivot", "value": center_clamped, "overlap_pct": overlap_pct})
            
            _setup_workspace_folders(target_dir, images_map, ds1_names, ds2_names, meta)
            print(f"   -> Created {folder_name} (Shared: {len(set(ds1_names).intersection(set(ds2_names)))})")

    elif args.type == "3d_points":
        if not PYCOLMAP_AVAILABLE:
            raise ImportError("pycolmap is required for 3D point extraction. Run 'pipeline.py setup' first.")
        if not args.full_recon_dir:
            raise ValueError("Must provide --full_recon_dir to sample 3D points.")

        recon_path = Path(args.full_recon_dir).resolve() / "merged"
        recon = pycolmap.Reconstruction(recon_path)
        
        density_profile = {}
        for img_id, img in recon.images.items():
            if recon.exists_image(img_id):
                valid_pts = sum(1 for p2d in img.points2D if p2d.has_point3D())
                density_profile[img.name] = valid_pts

        targets = [float(m.strip()) for m in str(args.target_means).split(",")]
        num_common = args.num_common
        window_size = args.window_size
        sigma = 75.0

        for target_mean in targets:
            weights = []
            valid_active_names = [n for n in active_names if n in density_profile]
            
            for name in valid_active_names:
                pts = density_profile[name]
                distance = float(pts) - target_mean
                weights.append(math.exp(-(distance ** 2) / (2 * (sigma ** 2))) + 1e-6)

            common_set = set()
            avail_anchors = list(valid_active_names)
            avail_weights = list(weights)

            while len(common_set) < num_common and avail_anchors:
                sampled = random.choices(avail_anchors, weights=avail_weights, k=1)[0]
                idx = avail_anchors.index(sampled)
                avail_anchors.pop(idx)
                avail_weights.pop(idx)

                global_idx = img_names.index(sampled)
                w_start = max(start_frame, global_idx - window_size)
                w_end = min(end_frame, global_idx + window_size + 1)
                
                for i in range(w_start, w_end):
                    if img_names[i] in valid_active_names:
                        common_set.add(img_names[i])
                    if len(common_set) == num_common:
                        break

            common_list = sorted(list(common_set))
            remain_list = [f for f in active_names if f not in common_set]
            
            mid = len(remain_list) // 2
            ds1_names = sorted(remain_list[:mid] + common_list)
            ds2_names = sorted(remain_list[mid:] + common_list)

            folder_name = f"mean_{int(target_mean)}_common_{num_common}"
            target_dir = output_root / folder_name
            
            meta = base_metadata.copy()
            meta.update({
                "generation_mode": "random_normal_3d_sampling", 
                "target_mean": target_mean,
                "common_frames_requested": num_common
            })
            
            _setup_workspace_folders(target_dir, images_map, ds1_names, ds2_names, meta)
            actual_mean = sum(density_profile[f] for f in common_list) / max(1, len(common_list))
            print(f"   -> Created {folder_name} (Actual Mean Overlap Pts: {actual_mean:.1f})")

    print("\nDataset generation complete.")