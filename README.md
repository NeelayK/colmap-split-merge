# Unified SfM Reconstruction & Evaluation Pipeline

This repository provides a framework to automate dataset preparation, perform partitioned or unified sparse reconstructions using COLMAP, and comparing alignment with ground truth camera poses. This code has been tested on 7-Scenes Chess Dataset.

---


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