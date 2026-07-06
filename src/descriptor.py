import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

WORKSPACE_DIR = Path(r"C:\Users\neela\OneDrive\Documents\Github\colmap-split-merge\test2")
TOP_K = 20

class GlobalDescriptorExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Global Descriptor Engine on backend device: {self.device}")
        
        base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = base_model.features.to(self.device)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)), # Scale to uniform shape for fast dense descriptor extraction
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
        print("[INFO] Split pipeline profile identified. Processing sequences separately.")
        
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
        print("[INFO] Full unified sequence profile identified. Processing all directory contents together.")
        
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
    if not WORKSPACE_DIR.exists():
        print(f"Error: Active execution workspace layout cannot be found: {WORKSPACE_DIR}")
        return

    valid_workspaces = []
    
    workspace_name_lower = WORKSPACE_DIR.name.lower()
    if any(k in workspace_name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
        valid_workspaces.append(WORKSPACE_DIR)
    else:
        for directory in sorted(list(WORKSPACE_DIR.iterdir())):
            if directory.is_dir():
                name_lower = directory.name.lower()
                if any(k in name_lower for k in ["mean_", "overlap_", "center", "full", "global"]):
                    valid_workspaces.append(directory)

    if not valid_workspaces:
        print("No active reconstruction pipeline configurations detected inside the tracked workspace root.")
        return

    print(f"Discovered {len(valid_workspaces)} pipeline configurations. Building neural descriptor network environment...")
    
    extractor = GlobalDescriptorExtractor()
    
    for path in valid_workspaces:
        run_global_descriptor_pipeline(path, extractor, TOP_K)

if __name__ == "__main__":
    main()