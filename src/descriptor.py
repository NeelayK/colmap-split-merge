#-------------------------------------------------------------------------------
# 7Scenes COLMAP Global Image Descriptor
# Uses a pre-trained VGG16 model to extract global image descriptors
# for all images in a dataset.
# Computes pairwise cosine similarity and exports the top-k most similar image pairs
# Use mainly for simple non-temporal image matching and retrieval tasks.
# Simple version of a global descriptor engine for COLMAP datasets.
#-------------------------------------------------------------------------------

import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

WORKSPACE_DIRS = [
    Path(r"D:\Neelay\colmap-split-merge\output\red"),
    Path(r"D:\Neelay\colmap-split-merge\output\chess"),
    Path(r"D:\Neelay\colmap-split-merge\output\fire"),
    Path(r"D:\Neelay\colmap-split-merge\output\heads"),
    Path(r"D:\Neelay\colmap-split-merge\output\office"),
    Path(r"D:\Neelay\colmap-split-merge\output\pumpkin"),
    Path(r"D:\Neelay\colmap-split-merge\output\stairs"),
]

TOP_K = 20

class GlobalDescriptorExtractor:
    def __init__(self):
        self.device = "cpu"
        
        base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = base_model.features.to(self.device)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_descriptor(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            features = self.feature_extractor(tensor)
            
            global_vector = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            global_vector = global_vector.flatten().cpu().numpy()
            
            norm = np.linalg.norm(global_vector)
            if norm > 0:
                global_vector = global_vector / norm
            return global_vector
        except Exception as e:
            print(f"Warning: Failed to process descriptor for image {image_path.name}. Error: {e}")
            return None

def compute_top_k_pairs(image_names, descriptors, top_k):
    num_images = len(descriptors)
    if num_images == 0:
        return []
    
    similarity_matrix = np.dot(descriptors, descriptors.T)
    
    pairs = []
    actual_k = min(top_k, num_images - 1)
    
    for i in range(num_images):
        similarity_matrix[i, i] = -1.0

        top_indices = np.argsort(similarity_matrix[i])[::-1][:actual_k]
        
        for idx in top_indices:
            pairs.append((image_names[i], image_names[idx]))
            
    return pairs

def save_colmap_pairs_file(output_path, image_pairs):
    with open(output_path, "w") as f:
        for img1, img2 in image_pairs:
            f.write(f"{img1} {img2}\n")
    print(f" -> Successfully exported {len(image_pairs)} matching links to: {output_path.name}")

def run_global_descriptor_pipeline(workspace_path, extractor, top_k):
    print("\n" + "="*60)
    print(f" PROCESSING WORKSPACE: {workspace_path.name.upper()} ")
    print("="*60)
    
    images_dir = workspace_path / "images"
    if not images_dir.exists():
        print(f"Skipping: 'images' subdirectory does not exist in this workspace.")
        return

    list1_path = workspace_path / "dataset1_list.txt"
    list2_path = workspace_path / "dataset2_list.txt"
    is_split_pipeline = list1_path.exists() and list2_path.exists()
    
    if is_split_pipeline:
        
        with open(list1_path, "r") as f:
            ds1_names = [line.strip() for line in f if line.strip()]
        with open(list2_path, "r") as f:
            ds2_names = [line.strip() for line in f if line.strip()]
            
        print(f"Processing Global Descriptors for Sequence 1 ({len(ds1_names)} images)...")
        ds1_descriptors, ds1_valid_names = [], []
        for name in ds1_names:
            img_path = images_dir / name
            if img_path.exists():
                desc = extractor.extract_descriptor(img_path)
                if desc is not None:
                    ds1_descriptors.append(desc)
                    ds1_valid_names.append(name)
                    
        if ds1_valid_names:
            ds1_pairs = compute_top_k_pairs(ds1_valid_names, np.array(ds1_descriptors), top_k)
            save_colmap_pairs_file(workspace_path / "image_pairs_dataset1.txt", ds1_pairs)
            
        print(f"Processing Global Descriptors for Sequence 2 ({len(ds2_names)} images)...")
        ds2_descriptors, ds2_valid_names = [], []
        for name in ds2_names:
            img_path = images_dir / name
            if img_path.exists():
                desc = extractor.extract_descriptor(img_path)
                if desc is not None:
                    ds2_descriptors.append(desc)
                    ds2_valid_names.append(name)
                    
        if ds2_valid_names:
            ds2_pairs = compute_top_k_pairs(ds2_valid_names, np.array(ds2_descriptors), top_k)
            save_colmap_pairs_file(workspace_path / "image_pairs_dataset2.txt", ds2_pairs)

    else:
        
        all_images = sorted([
            p.name for p in images_dir.iterdir() 
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])
        
        if not all_images:
            print(f"Skipping: No images found inside {images_dir}")
            return
            
        print(f"Processing Global Descriptors for complete sequence ({len(all_images)} images)...")
        descriptors, valid_names = [], []
        for name in all_images:
            img_path = images_dir / name
            desc = extractor.extract_descriptor(img_path)
            if desc is not None:
                descriptors.append(desc)
                valid_names.append(name)
                
        if valid_names:
            global_pairs = compute_top_k_pairs(valid_names, np.array(descriptors), top_k)
            save_colmap_pairs_file(workspace_path / "image_pairs_global.txt", global_pairs)

def main():
    extractor = None

    for workspace_root in WORKSPACE_DIRS:
        if not workspace_root.exists():
            print(f"Skipping missing directory: {workspace_root}")
            continue

        print("\n" + "#" * 70)
        print(f"DATASET: {workspace_root.name.upper()}")
        print("#" * 70)

        valid_workspaces = []

        workspace_name_lower = workspace_root.name.lower()
        if any(k in workspace_name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
            valid_workspaces.append(workspace_root)
        else:
            for directory in sorted(workspace_root.iterdir()):
                if directory.is_dir():
                    name_lower = directory.name.lower()
                    if any(k in name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
                        valid_workspaces.append(directory)

        if not valid_workspaces:
            print("No active reconstruction pipeline configurations found.")
            continue

        print(f"Discovered {len(valid_workspaces)} pipeline configurations.")

        # Build the network only once
        if extractor is None:
            print("Building neural descriptor network environment...")
            extractor = GlobalDescriptorExtractor()

        for path in valid_workspaces:
            run_global_descriptor_pipeline(path, extractor, TOP_K)
            
if __name__ == "__main__":
    main()