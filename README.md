# Unified SfM Reconstruction & Evaluation Pipeline

This repository provides a framework to automate dataset preparation, perform partitioned or unified sparse reconstructions using COLMAP, and comparing alignment with ground truth camera poses. This code has been tested on 7-Scenes Chess Dataset.

---

## Architectural Workflow Overview


```
                            +--------------------------+
                            |   Raw Sequence Images    |
                            +--------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |  python pipeline.py dataset   |
                         +-------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         | (Splitting Modes)             | (Unified Mode)                | (Probabilistic Mode)
         v                               v                               v
+------------------+            +------------------+            +------------------+
| Overlap / Center |            |  Full Sequence   |            |  3D Point Density|
+------------------+            +------------------+            +------------------+
         |                               |                               |
         | [Generates ds1/ds2 lists]     | [Generates single model]      | [Windowed boundaries]
         v                               v                               v
+----------------------------------------------------------------------------------+
|                          Workspace Directory Generation                          |
|         (Structured with images/, sparse1/, sparse2/, merged/, metadata.json)    |
+----------------------------------------------------------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |     python pipeline.py run    |
                         +-------------------------------+
                                         |
         +-------------------------------+-------------------------------+
         |                                                               |
         v [IF split_mode DETECTED]                                      v [IF full_no_split DETECTED]
+----------------------------------+                            +----------------------------------+
| 1. Extract Features (ALIKED)    |                            | 1. Extract Features (ALIKED)    |
| 2. Match Features (LightGlue)    |                            | 2. Match Features (LightGlue)    |
| 3. Map Sub-Model 1 (Image List1) |                            | 3. Global Sparse Mapping (Full)  |
| 4. Map Sub-Model 2 (Image List2) |                            | 4. Global Bundle Adjustment      |
| 5. COLMAP Model Merger           |                            +----------------------------------+
| 6. Global Bundle Adjustment      |                                             |
+----------------------------------+                                             v
         |                                                      +----------------------------------+
         v                                                      |  python pipeline.py analyze      |
+----------------------------------+                            +----------------------------------+
|   python pipeline.py visualize   |                                             |
+----------------------------------+                                             v
         |                                                      +----------------------------------+
         v                                                      | Generates JSON Profile & Plots   |
| Interactive Open3D Canvas Engine |                            | Matplotlib Density Histogram     |
| Real-Time Window Title ATE RMSE  |                            +----------------------------------+
+----------------------------------+

```

---

## Prerequisites & Installation

### Core Requirements

* **Python 3.11 or 3.12** (Mandatory for `open3d` and `pycolmap`).
* **COLMAP Binary Executable**: The `colmap` command must be mapped to your system's environmental `PATH` variables. Verify your installation by executing:
```bash
colmap -h

```



### Localized Bootstrap Deployment

Initialize your virtual environment, isolate configuration parameters, and trigger the programmatic environment setup wrapper:

```bash
# Initialize native virtual environment
python -m venv venv

# Activate the isolated environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS (Bash):
source venv/bin/activate

# Execute automated module bootstrap setup
python pipeline.py setup --requirements requirements.txt

```

---

## Workspace Directory Design Layout

When generating datasets, the system creates structured tracking environments within your defined workspace. This preserves data integrity across batch experiments:

```
[output_parent_directory]/
└── [project_name]/
    └── [experiment_configuration_folder]/
        ├── metadata.json              # State tracking file
        ├── database.db                # SQLite database for keypoints and matches
        ├── dataset1_list.txt          # Image filenames isolated for sub-model 1
        ├── dataset2_list.txt          # Image filenames isolated for sub-model 2
        ├── images/                    # Local system symlinks or physical file copies
        ├── sparse1/                   # Local workspace for sub-model 1 mapping
        │   └── 0/                     # Binary model outputs (cameras, images, points3D)
        ├── sparse2/                   # Local workspace for sub-model 2 mapping
        │   └── 0/                     # Binary model outputs (cameras, images, points3D)
        └── merged/                    # Combined trajectory tracking directory

```

---

## Command-Line Interface (CLI) Reference Manual

### 1. `setup`

Initializes virtual environment packages and handles dependencies.

| Option Flag | Argument Type | Default Value | Functional Definition |
| --- | --- | --- | --- |
| `-r`, `--requirements` | `string` | `"requirements.txt"` | Explicit path to the pip installation dependency configuration file. |

#### Code Execution Context

```bash
python pipeline.py setup --requirements requirements.txt

```

---

### 2. `dataset`

Generates isolated tracking configurations based on temporal frame overlap, center partition boundaries, or structural 3D point density profiles.

| Option Flag | Argument Type | Default Value | Functional Definition |
| --- | --- | --- | --- |
| `-s`, `--source` | `string` | *Required* | High-resolution path pointing directly to your raw sequence images directory. |
| `-o`, `--output` | `string` | `"./output"` | Parent directory destination path where project sequences are written. |
| `-p`, `--project` | `string` | *Required* | Unique name constraint identifier for sorting distinct sequences. |
| `-t`, `--type` | `choice` | *Required* | Selection strategy formatting mode. Options: `overlap`, `center`, `full`, `3d_points`. |
| `--start_frame` | `integer` | `0` | Sequence start slice window index offset boundary. |
| `--end_frame` | `integer` | `None` | Sequence end slice window index offset boundary (defaults to complete tracking array). |
| `--center_point` | `integer` | `None` | Sequence frame index defining the cutting horizon separating Model 1 and Model 2. |
| `--overlap_pcts` | `string` | `"10,20,30"` | Comma-delimited list of slice windows evaluated for overlapping bounds. |
| `--target_means` | `string` | `"200,300"` | Comma-delimited target mean numbers of 3D points required for window sampling. |
| `--common_frames` | `integer` | `30` | Quantitative window sizing constraints for shared tracking frames in 3D splits. |
| `--window_size` | `integer` | `50` | Symmetrical neighborhood search parameters utilized by point density configurations. |
| `--full_recon_dir` | `string` | `None` | Path targeting a completed full reconstruction model containing `scene_density_profile.json`. |
| `--custom_dir` | `string` | `None` | Directory override argument naming convention flag bypassing automated naming strings. |

#### Code Execution Context

* **Generate Symmetrical Overlap Progressions:**
```bash
python pipeline.py dataset -s ./dataset -p bridge_sequence -t overlap --overlap_pcts "10,25,50" --center_point 120 --start_frame 0 --end_frame 240

```


* **Isolate Explicit Center Cutting Planes:**
```bash
python pipeline.py dataset -s ./dataset -p factory_run -t center --center_point 150 --overlap_pcts "20" --custom_dir "center_split_frame_150"

```


* **Unified Pipeline Sequence Mapping Baseline:**
```bash
python pipeline.py dataset -s ./dataset -p city_block -t full

```


* **Probabilistic 3D Point Density Partitioning:**
```bash
python pipeline.py dataset -s ./dataset -p city_block -t 3d_points --full_recon_dir ./output/city_block/full_reconstruction --target_means "300,500" --common_frames 40 --window_size 60

```



---

### 3. `run`

Executes localized feature extractions, LightGlue matching chains, mapping routines, model merging, and bundle adjustments.

| Option Flag | Argument Type | Default Value | Functional Definition |
| --- | --- | --- | --- |
| `-w`, `--workspace` | `string` | *Required* | High-resolution tracking path pointing directly to the generated destination subfolder. |
| `--use_gpu` | `boolean` | `False` | Toggles CUDA-bound hardware acceleration pipelines for extraction/matching. |

#### Code Execution Context

```bash
python pipeline.py run -w ./output/bridge_sequence/overlap_25pct --use_gpu

```

---

### 4. `visualizer`

Launches an integrated Open3D canvas visualization window. This system explicitly strips Matplotlib fallbacks from the 3D visual workspace to optimize processing efficiency. It performs Umeyama alignment against a Ground Truth tracking log file, rendering spatial error tracking variables directly into the application title window frame in real time.

| Option Flag | Argument Type | Default Value | Functional Definition |
| --- | --- | --- | --- |
| `-w`, `--workspace` | `string` | *Required* | High-resolution path targeting the workspace folder containing the compiled `merged/` folder. |
| `-g`, `--gt` | `string` | `"./dataset"` | Storage file path pointing to ground truth positional text logs tracking transformations. |
| `--color_ds1` | `string` | `"1.0, 0.4, 0.4"` | Comma-separated float values defining RGB rendering for Sub-Model 1 (Light Red). |
| `--color_ds2` | `string` | `"0.4, 0.4, 1.0"` | Comma-separated float values defining RGB rendering for Sub-Model 2 (Light Blue). |
| `--color_overlap` | `string` | `"0.2, 1.0, 1.0"` | Comma-separated float values defining RGB rendering for overlapping points/frustums. |
| `--color_gt` | `string` | `"0.0, 1.0, 0.0"` | Comma-separated float values defining RGB rendering for the Ground Truth trajectory (Green). |
| `--point_size` | `float` | `1.0` | Scalar adjustments altering pixel point cloud cluster rendering weight profiles. |
| `--camera_scale` | `float` | `1.0` | Sizing parameter altering wireframe geometric dimensions of camera tracking frustums. |

#### Code Execution Context

```bash
python pipeline.py visualize -w ./output/bridge_sequence/overlap_25pct --point_size 2.0 --camera_scale 1.5 --color_overlap "0.8, 0.2, 0.8"

```

---

### 5. `analyze`

Performs structure density evaluation. This utility processes complete global models via PyCOLMAP and visualizes point cloud features. **Matplotlib utilization is strictly isolated to this subcommand** to render point cloud density profile tracking histograms.

| Option Flag | Argument Type | Default Value | Functional Definition |
| --- | --- | --- | --- |
| `-w`, `--workspace` | `string` | *Required* | Workspace directory tracking target location hosting standard global reconstruction data. |

#### Code Execution Context

```bash
python pipeline.py analyze -w ./output/city_block/full_reconstruction

```

---

## Detailed Step-by-Step Production Guide

Follow this sequential workflow pattern to configure, execute, and analyze spatial experiments:

### Step 1: Initialize the Target Project Baseline

Execute a unified sequence mapping operation to establish geometric tracking baselines:

```bash
python pipeline.py dataset -s /path/to/raw/images -p architectural_test -t full

```

### Step 2: Build the Global Sparse Baseline

Reconstruct the master reference model across the complete sequence timeline:

```bash
python pipeline.py run -w ./output/architectural_test/full_reconstruction --use_gpu

```

### Step 3: Extract Structure Point Density Profiles

Analyze tracking visibility profiles across the global sequence model to identify structural point layout statistics:

```bash
python pipeline.py analyze -w ./output/architectural_test/full_reconstruction

```

*This command generates `scene_density_profile.json` inside the workspace folder and displays a Matplotlib distribution histogram detailing structural feature depth visibility.*

### Step 4: Perform Targeted Pipeline Experiments

Generate localized sequence splits using specific tracking options (e.g., assessing scale drift configurations across 20% shared image visibility horizons):

```bash
python pipeline.py dataset -s /path/to/raw/images -p architectural_test -t overlap --overlap_pcts "20" --center_point 150

```

### Step 5: Execute the Split-Merge Reconstruction Engine

Trigger the automated execution chain on the partitioned tracking folders. The system automatically reads the workspace configuration tags, routing data processing through the divided workspace pipeline architecture:

```bash
python pipeline.py run -w ./output/architectural_test/overlap_20pct --use_gpu

```

#### Under the Hood: Split-Merge Engine Sequence

```
[Database Initialization]
          │
          ▼
[Step 1: ALIKED Feature Extractor] ──► Feature Extraction Matrix (ALIKED_N16ROT)
          │
          ▼
[Step 2: LightGlue Exhaustive Matcher] ──► Feature Matching Engine (ALIKED_LIGHTGLUE)
          │
          ├──► [Route: Split-Merge Sequence Workflow]
          │             │
          │             ├──► [Step 3] Map Sub-Model 1 (Using dataset1_list.txt) ──► sparse1/0/
          │             ├──► [Step 4] Map Sub-Model 2 (Using dataset2_list.txt) ──► sparse2/0/
          │             │
          │             ▼
          │       [Step 5] COLMAP Model Merger (Combines sparse1 and sparse2)
          │             │
          │             ▼
          │       [Step 6] Global Bundle Adjustment (Refines merged structural metrics) ──► merged/
          │
          └──► [Route: Linear Baseline Mapping]
                        │
                        ├──► [Step 3] Global Sparse Mapping (Maps entire image directory)
                        └──► [Step 4] Global Bundle Adjustment (Refines full sequence) ──► merged/

```

### Step 6: Render Absolute Alignment Metrics

Launch the Open3D visualization tool to calculate spatial transformations and evaluate trajectory drift. The console displays detailed metrics (Absolute Trajectory Error RMSE, Translation Offset, and Peak Geometric Deviation) while rendering interactive 3D structures:

```bash
python pipeline.py visualize -w ./output/architectural_test/overlap_20pct -g /path/to/ground_truth/poses --point_size 1.2 --camera_scale 0.8

```