import re
from pathlib import Path
import matplotlib.pyplot as plt
import pycolmap

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
    match = re.search(r"\d+", image_name)
    return int(match.group()) if match else None


def plot_3d_points_per_frame(workspace_path):
    w_path = Path(workspace_path).resolve()
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
            f"Skipping {w_path.parent.name}: No valid sparse reconstruction map found."
        )
        return

    frames = []
    point_counts = []

    for im_id, im in reconstruction.images.items():
        frame_idx = extract_frame_number(im.name)

        if frame_idx is None:
            frame_idx = im_id

        if hasattr(im, "num_points3D"):
            num_pts = (
                im.num_points3D() if callable(im.num_points3D) else im.num_points3D
            )
        else:
            num_pts = sum(1 for p in im.points2D if p.has_point3D())

        frames.append(frame_idx)
        point_counts.append(num_pts)

    if not frames:
        print(f"Skipping {w_path.parent.name}: No registered frame telemetry detected.")
        return

    sorted_data = sorted(zip(frames, point_counts))
    sorted_frames, sorted_counts = zip(*sorted_data)

    scene_title = w_path.parent.name.upper()

    print(
        f"--> Rendering tracking histogram for scene: {scene_title} ({len(sorted_frames)} frames tracked)..."
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        sorted_frames,
        sorted_counts,
        width=1.0,
        color="#1f77b4",
        edgecolor="#124569",
        alpha=0.85,
        label="Visible 3D Points",
    )

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

    plt.show()
    plt.close()


def main():
    print("======================================================================")
    print("      7SCENES SEQUENTIAL FRAME 3D POINT DISTRIBUTION METRIC ENGINE    ")
    print("======================================================================")
    print(
        f"Processing queue contains {len(SCENE_WORKSPACES)} scene target configurations."
    )

    for path in SCENE_WORKSPACES:
        plot_3d_points_per_frame(path)



if __name__ == "__main__":
    main()