import sys
import json
from pathlib import Path

try:
    import pycolmap
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("❌ [ERROR]: Required libraries (pycolmap, matplotlib, numpy) missing.")
    sys.exit(1)

def analyze_model(workspace_path):
    workspace = Path(workspace_path).resolve()
    merged_dir = workspace / "merged"
    
    if not merged_dir.exists():
        print(f"❌ [ERROR]: Model directory missing at {merged_dir}")
        sys.exit(1)

    print(f"\n📊 Analyzing Sparse Reconstruction Model: {workspace.name}")
    try:
        recon = pycolmap.Reconstruction(merged_dir)
    except Exception as e:
        print(f"❌ [ERROR]: Could not load model. Details: {e}")
        sys.exit(1)

    # 1. Parse Densities
    density_profile = {}
    counts = []
    
    for img_id, img in recon.images.items():
        if recon.exists_image(img_id):
            valid_pts = sum(1 for p2d in img.points2D if p2d.has_point3D())
            density_profile[img.name] = valid_pts
            counts.append(valid_pts)

    if not counts:
        print("❌ [ERROR]: No valid 3D points found in model.")
        sys.exit(1)

    # 2. Extract Statistics
    mean_pts = np.mean(counts)
    low_threshold = max(100, int(mean_pts * 0.2))
    
    low_frames = [img for img, count in density_profile.items() if count < low_threshold]
    high_frames = [img for img, count in density_profile.items() if count >= int(mean_pts * 1.5)]

    print("\n" + "="*50)
    print(" RECONSTRUCTION DENSITY REPORT ")
    print("="*50)
    print(f" Total Registered Frames       : {len(density_profile)}")
    print(f" Mean 3D Points per Image      : {mean_pts:.1f}")
    print(f" Frames in Danger Zone (<{low_threshold})  : {len(low_frames)}")
    print(f" Frames in Anchor Zone (>{int(mean_pts*1.5)}): {len(high_frames)}")
    print("="*50)

    # 3. Export JSON Profile
    out_json = workspace / "scene_density_profile.json"
    with open(out_json, "w") as f:
        json.dump(density_profile, f, indent=4)
    print(f"-> Saved density profile to: {out_json.name}")

    # 4. Render Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=40, color='royalblue', edgecolor='darkblue', alpha=0.8)
    plt.axvline(mean_pts, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_pts:.1f})')
    plt.title("Distribution of 3D Points per Image", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Visible 3D Points", fontsize=12)
    plt.ylabel("Frequency (Number of Images)", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    print("-> Rendering histogram visualization...")
    plt.show()