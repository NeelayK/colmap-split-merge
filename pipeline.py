import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.dataset_generator import generate_datasets

def setup_environment(args):
    """Automates localized virtual environment creation and handles pip dependency chains."""
    root_dir = Path(__file__).resolve().parent
    venv_dir = root_dir / "venv"
    req_file = root_dir / args.requirements
    
    print("\n" + "="*80)
    print(f"ENVIRONMENT BOOTSTRAP INITIALIZATION")
    print(f"   Target Virtual Environment Directory : {venv_dir}")
    print(f"   Host Python Version Interpreted      : {sys.version.split()[0]}")
    print("="*80)
    
    if not req_file.exists():
        print(f"'{req_file.name}' not found. Creating a default fallback target...")
        default_dependencies = "numpy>=1.24.0\npycolmap>=0.3.0\nopen3d>=0.17.0\n"
        req_file.write_text(default_dependencies)
        print(f"   -> Drop-file written: {req_file}")

    # 2. Native programmatic deployment of venv
    if venv_dir.exists():
        print(f"Target environment signature already allocated at: {venv_dir}")
    else:
        print(f"Provisioning isolated virtual ecosystem space...")
        try:
            import venv
            # Matches your 'include-system-site-packages = false' profile default
            venv.create(venv_dir, system_site_packages=False, with_pip=True)
            print("   -> Virtual environment successfully built.")
        except Exception as e:
            print(f"[CRITICAL ERROR]: venv construction failed natively. Details: {str(e)}")
            sys.exit(1)

    # 3. Resolve platform execution paths safely (Windows vs Unix/macOS)
    is_windows = sys.platform == "win32"
    python_binary = venv_dir / "Scripts" / "python.exe" if is_windows else venv_dir / "bin" / "python"
    pip_binary = venv_dir / "Scripts" / "pip.exe" if is_windows else venv_dir / "bin" / "pip"

    if not python_binary.exists():
        print(f"[CRITICAL ERROR]: Subprocess router failed to locate environment binaries at: {python_binary}")
        sys.exit(1)

    # 4. Run setup optimization layers
    try:
        print(f"Core-upgrading internal package manager (pip)...")
        subprocess.run([str(python_binary), "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True)
        
        print(f"Mapping package arrays from {req_file.name} to environment...")
        subprocess.run([str(pip_binary), "install", "-r", str(req_file)], check=True)
        
        print("\n" + "="*80)
        print("ENVIRONMENT CONFIGURATION SUCCESSFULLY RESOLVED!")
        if is_windows:
            print(f"Run the following execution command to engage your virtual wrapper:\n\n"
                  f"   .\\venv\\Scripts\\activate\n")
        else:
            print(f"Run the following execution command to engage your virtual wrapper:\n\n"
                  f"   source venv/bin/activate\n")
        print("="*80)

    except subprocess.CalledProcessError as err:
        print(f"\n[INSTALLATION FAILURE]: Dependency pip engine exited with structural failure flag.")
        if err.stderr:
            print(f"Error Diagnostic:\n{err.stderr.decode()}")
        sys.exit(1)


def generate_dataset_layout(args):
    """Routes the CLI arguments to the unified dataset generator."""
    try:
        generate_datasets(args)
    except Exception as e:
        print(f"[DATASET GENERATOR ERROR]: {str(e)}")
        sys.exit(1)


def execute_reconstruction_pipeline(args):
    """Reads context markers from target workspace and executes localized COLMAP processes."""
    workspace_path = Path(args.workspace).resolve()
    meta_file = workspace_path / "metadata.json"
    
    if not meta_file.exists():
        print(f"[CRITICAL ERROR]: Could not identify a valid metadata tracking signature inside:\n   -> {workspace_path}")
        sys.exit(1)
        
    with open(meta_file, "r") as f:
        context = json.load(f)
        
    print("\n" + "="*80)
    print(f"RUNNING SFM PIPELINE FOR PROJECT: {context['project_name'].upper()}")
    print(f"   Context Profile Detected : {context['generation_mode']}")
    print(f"   Hardware Optimization    : {'GPU Accelerated' if args.use_gpu else 'CPU Bound (Default)'}")
    print("="*80)


def launch_visualization_interface(args):
    """Aligns existing sparse trajectories to Ground Truth and triggers localized Open3D or Plot3D profiles."""
    workspace_path = Path(args.workspace).resolve()
    gt_path = Path(args.gt).resolve()
    
    if not gt_path.exists():
        print(f"[CRITICAL ERROR]: Ground truth directory path not found: {gt_path}")
        sys.exit(1)

    print("\n" + "="*80)
    print(f"COMMENCING GEOMETRIC EVALUATION ANALYSIS")
    print(f"   Selected Interface View: {args.mode.upper()}")
    print("=" * 80)

# -----------------------------------------------------------------------------
# 3. CLI CONTROL INTERFACE ARGPARSE DEF DEFINITIONS
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified Split-Merge Structure from Motion (SfM) Control Wrapper Interface Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline functional subcommands")

    # Command 0: INITIALIZATION SERVICE SETUP
    p_setup = subparsers.add_parser("setup", help="Automatically generate python venv instance and fetch project requirements targets.")
    p_setup.add_argument("-r", "--requirements", type=str, default="requirements.txt", help="Target filename matching dependencies parameters")
    p_setup.set_defaults(func=setup_environment)

    # Command 1: DATASET GENERATION
    # Command 1: DATASET GENERATION
    p_data = subparsers.add_parser("dataset", help="Generate custom split variants compatible with COLMAP model configurations.")
    p_data.add_argument("-p", "--project", type=str, required=True, help="Unique name tag folder for this experiment sequence run")
    p_data.add_argument("-d", "--dataset", type=str, default="./dataset", help="Input absolute or relative path to raw source image folder")
    p_data.add_argument("-o", "--output_dir", type=str, default="./output", help="Base directory path where projects are stored")
    
    p_data.add_argument("-t", "--type", type=str, choices=["overlap", "center", "full", "3d_points"], required=True, help="Splitting strategy execution matrix")
    
    # Optional Range Bound Args
    p_data.add_argument("--start_frame", type=int, default=None, help="Global Sequence Start Frame Index (Defaults to 0)")
    p_data.add_argument("--end_frame", type=int, default=None, help="Global Sequence End Frame Index (Defaults to last available frame)")
    
    # Sub-parameters (Configured to accept comma-separated strings for batch generation)
    p_data.add_argument("--overlap_pct", type=str, default="50", help="Comma-separated list of overlaps (e.g., '10, 20, 35')")
    p_data.add_argument("--center_frame", type=str, default="500", help="Comma-separated list of center frames (e.g., '100, 300, 500')")
    
    # 3D Point Specific Parameters
    p_data.add_argument("--full_recon_dir", type=str, default=None, help="Reference directory containing complete baseline model data")
    p_data.add_argument("--target_means", type=str, default="300", help="Comma-separated list of target 3D point means (e.g., '200, 350, 600')")
    p_data.add_argument("--num_common", type=int, default=30, help="Number of target common/overlap images for 3D point sampling")
    p_data.add_argument("--window_size", type=int, default=2, help="Window radius around selected anchors (e.g., 2 means center ± 2 frames)")
    
    p_data.set_defaults(func=generate_dataset_layout)

    # Command 2: RECONSTRUCTION ENGINE RUNNER
    p_run = subparsers.add_parser("run", help="Auto-detect context and execute full processing sequence loops via COLMAP.")
    p_run.add_argument("-w", "--workspace", type=str, required=True, help="Path directly targeting the generated split experiment folder")
    p_run.add_argument("--use_gpu", action="store_true", help="Toggle flag to switch pipeline engine execution matrix to GPU acceleration.")
    p_run.set_defaults(func=execute_reconstruction_pipeline)

    # Command 3: VISUALIZATION PROFILE METRICS
    p_eval = subparsers.add_parser("visualize", help="Execute geometric trajectory comparisons, error logging, and rendering profiles.")
    p_eval.add_argument("-w", "--workspace", type=str, required=True, help="Path directly targeting the processed workspace directory")
    p_eval.add_argument("-g", "--gt", type=str, default="./dataset", help="Path directory location containing corresponding ground truth pose profile text sequences")
    p_eval.add_argument("-m", "--mode", type=str, choices=["open3d", "plot3d"], default="open3d", help="Interface rendering profile selection engine format")
    p_eval.set_defaults(func=launch_visualization_interface)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARNING]: Process forcefully terminated via manual keyboard interrupt command sequence.")
        sys.exit(1)