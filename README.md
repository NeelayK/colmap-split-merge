
# COLMAP Pipeline (ALIKED + LightGlue)

This repository contains an end-to-end pipeline designed to split image datasets , execute independent spatial reconstructions via the COLMAP CLI using **ALIKED** and **LightGlue**, and mathematically align the resulting trajectories to compute scale drift.

By rebuilding a shared scene from separate subsets, you can accurately isolate, measure, and visualize scale variations and trajectory accumulation errors using the **Umeyama Absolute Orientation** algorithm.


### Prerequisites

* **Python 3.11 or 3.12 ** (Recommended for Open3D and PyCOLMAP stability)
* **COLMAP Executable**: Ensure `colmap` is added to your system's environmental `PATH` variables so it can be called globally from the terminal.

### Environment Setup

Clone this repository, initialize a Python virtual environment, and install the tracking dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Upgrade pip and install libraries
python -m pip install --upgrade pip
pip install -r requirements.txt

```




## Workflow

#### Split the Datase and paste all your images in one folder.

#### Run prepare_colmap_workspace

#### Run Independent COLMAP Pipelines

#### Dataset Reconstruction (Run Seperately for each Dataset, Don't forget to change the paths )

```powershell
# Extract features using ALIKED
colmap feature_extractor `
    --database_path " ...\database.db" `
    --image_path " ...\images" `
    --FeatureExtraction.type ALIKED_N16ROT `
    --AlikedExtraction.max_num_features 2048 `
    --FeatureExtraction.use_gpu 0 

# Match features using LightGlue
colmap exhaustive_matcher `
    --database_path " ...\database.db" `
    --FeatureMatching.type ALIKED_LIGHTGLUE `
    --FeatureMatching.use_gpu 0

# Triangulate points and build the sparse map
colmap mapper `
    --database_path " ...\database.db" `
    --image_path " ...\images" `
    --output_path " ...\sparse"

```

#### Once Finished run merger.py
