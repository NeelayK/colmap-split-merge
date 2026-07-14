import re
from pathlib import Path
import matplotlib.pyplot as plt
import pycolmap

# ==============================================================================
# TARGET RECONSTRUCTION WORKSPACE LAYOUT
# ==============================================================================
SCENE_WORKSPACES = [
    Path(r"D:\Neelay\colmap-split-merge\output\chess\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\heads\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\office\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\red\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\fire\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\pumpkin\full_sequence_0_to_300"),
    Path(r"D:\Neelay\colmap-split-merge\output\stairs\full_sequence_0_to_300"),
]


def extract_frame_number(image_name):
    """
    Extracts the sequential integer index from a 7Scenes image filename.
    Example: 'frame-000123.color.png' -> 123
    """
    match = re.search(r"\d+", image_name)
    return int(match.group()) if match else None


def plot_3d_points_per_frame(workspace_path):
    w_path = Path(workspace_path).resolve()

    # Locate the valid sparse reconstruction directory inside the workspace
    reconstruction = None
    candidate_subdirs = ["sparse/0", "sparse", "merged", ""]

    for sub_dir in candidate_subdirs:
        target_path = w_path / sub_dir
        if target_path.exists():
            try:
                recon_test = pycolmap.Reconstruction(target_path)
                if len(recon_test.images) > 0:
                    reconstruction = recon_test
                    break
            except Exception:
                continue

    if reconstruction is None:
        print(
            f"[!] Skipping {w_path.parent.name}: No valid sparse reconstruction map found."
        )
        return

    frames = []
    point_counts = []

    # Iterate over all registered cameras in the tracking graph
    for im_id, im in reconstruction.images.items():
        frame_idx = extract_frame_number(im.name)

        # Fallback to internal COLMAP image ID if filename parsing fails
        if frame_idx is None:
            frame_idx = im_id

        # Safely extract visible 3D tracking count across different pycolmap versions
        if hasattr(im, "num_points3D"):
            num_pts = (
                im.num_points3D() if callable(im.num_points3D) else im.num_points3D
            )
        else:
            num_pts = sum(1 for p in im.points2D if p.has_point3D())

        frames.append(frame_idx)
        point_counts.append(num_pts)

    if not frames:
        print(f"[!] No registered frame telemetry detected inside {w_path.parent.name}.")
        return

    # Sort data by frame number so the X-axis sequence displays correctly (0 to 300)
    sorted_data = sorted(zip(frames, point_counts))
    sorted_frames, sorted_counts = zip(*sorted_data)

    # Clean the directory name for the plot header (e.g., 'chess')
    scene_title = w_path.parent.name.upper()

    print(
        f"--> Rendering tracking histogram for scene: {scene_title} ({len(sorted_frames)} frames tracked)..."
    )

    # Generate the sequential frame distribution plot
    plt.figure(figsize=(12, 6))

    # Using a bar plot with full width creates a clean histogram-like layout per-frame
    plt.bar(
        sorted_frames,
        sorted_counts,
        width=1.0,
        color="#1f77b4",
        edgecolor="#124569",
        alpha=0.85,
        label="Visible 3D Points",
    )

    # Chart formatting setup
    plt.title(
        f"COLMAP Reconstruction Analysis: 3D Point Observations per Frame\nScene Focus Target: {scene_title}",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Sequential Video Frame Index (X)", fontsize=11, labelpad=8)
    plt.ylabel("Number of Registered 3D Points (Y)", fontsize=11, labelpad=8)

    plt.xlim(min(sorted_frames) - 5, max(sorted_frames) + 5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Display plot. This acts as a blocking call—close the active window to continue to the next scene.
    plt.show()
    plt.close()


def main():
    print("======================================================================")
    print("      7SCENES SEQUENTIAL FRAME 3D POINT DISTRIBUTION METRIC ENGINE    ")
    print("======================================================================")
    print(
        f"Processing queue contains {len(SCENE_WORKSPACES)} scene target configurations."
    )
    print("[Note: Close each canvas visualization window to load the next plot]\n")

    for path in SCENE_WORKSPACES:
        plot_3d_points_per_frame(path)

    print("\n[SUCCESS] Completed sequence track execution loops for all 7 scenes.")


if __name__ == "__main__":
    main()