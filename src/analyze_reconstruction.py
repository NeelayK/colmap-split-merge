import subprocess
import os
import re
import json
from pathlib import Path
import matplotlib.pyplot as plt

GLOBAL_SPARSE_DIR = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\global_reconstruction\sparse\0")
SOURCE_IMAGE_DIR = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\dataset")
OUTPUT_ANALYSIS_DIR = Path(r"C:\Users\chand\OneDrive\Documents\Neelay\colmap\colmap_workspace\analysis")

def analyze_scene_density():
    OUTPUT_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    txt_model_dir = OUTPUT_ANALYSIS_DIR / "sparse_txt"
    txt_model_dir.mkdir(exist_ok=True)
    
    print("[*] Converting sparse binary model to text files...")
    convert_cmd = [
        "colmap", "model_converter",
        "--input_path", str(GLOBAL_SPARSE_DIR),
        "--output_path", str(txt_model_dir),
        "--output_type", "TXT"
    ]
    
    try:
        subprocess.run(convert_cmd, check=True, shell=True, capture_output=True)
        print("--> [SUCCESS] Model converted to text.")
    except subprocess.CalledProcessError as e:
        print(f"--> [ERROR] Failed to convert model: {e.stderr.decode()}")
        return

    images_txt_path = txt_model_dir / "images.txt"
    if not images_txt_path.exists():
        print(f"[!] Error: {images_txt_path} was not generated.")
        return

    print("[*] Parsing images.txt to calculate 3D point metrics...")
    point_counts = {}
    
    with open(images_txt_path, "r") as f:
        lines = f.readlines()
        
    iterator = iter(lines)
    for line in iterator:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
            
        parts = line.split()
        image_name = parts[-1]
        
        try:
            points_line = next(iterator).strip()
            point_elements = points_line.split()
            
            valid_3d_points = 0
            for i in range(2, len(point_elements), 3):
                if point_elements[i] != "-1":
                    valid_3d_points += 1
            
            point_counts[image_name] = valid_3d_points
        except StopIteration:
            break

    all_color_images = sorted([f.name for f in SOURCE_IMAGE_DIR.glob("*.color.png")])
    for img in all_color_images:
        if img not in point_counts:
            point_counts[img] = 0

    chronological_data = {img: point_counts[img] for img in all_color_images}

    json_path = OUTPUT_ANALYSIS_DIR / "scene_density_profile.json"
    with open(json_path, "w") as jf:
        json.dump(chronological_data, jf, indent=4)
    print(f"--> [SUCCESS] Saved point data profiles to: {json_path}")

    counts = list(chronological_data.values())
    low_threshold = 200
    low_frames = [img for img, count in chronological_data.items() if count < low_threshold]
    high_frames = [img for img, count in chronological_data.items() if count >= 1000]

    print("\n" + "="*50)
    print(" RECONSTRUCTION DENSITY REPORT ")
    print("="*50)
    print(f"Total Sequence Frames Analyzed : {len(chronological_data)}")
    print(f"Frames in Low Region (< {low_threshold} pts) : {len(low_frames)}")
    print(f"Frames in High Region (>= 1000 pts): {len(high_frames)}")
    
    print("\n[*] Sample Chronological Frame Benchmarks (Use these to test splits):")
    step = max(1, len(all_color_images) // 10)
    for i in range(0, len(all_color_images), step):
        img_name = all_color_images[i]
        print(f"  - Frame Index {str(i).zfill(3)} ({img_name}): {chronological_data[img_name]} 3D Points")
    print("="*50)

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=40, color='royalblue', edgecolor='royalblue', alpha=0.8)
    plt.title("Distribution of 3D Points per Image", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Visible 3D Points", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    
    plot_path = OUTPUT_ANALYSIS_DIR / "3d_points_histogram.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"--> [SUCCESS] Histogram graphic generated and saved to: {plot_path}")

if __name__ == "__main__":
    analyze_scene_density()