import os
import sys
import subprocess
from pathlib import Path
import argparse


def create_venv(venv_path: Path) -> bool:
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print("Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        return False


def get_pip_executable(venv_path: Path = None) -> str:
    if venv_path and venv_path.exists():
        if sys.platform == "win32":
            return str(venv_path / "Scripts" / "pip.exe")
        else:
            return str(venv_path / "bin" / "pip")
    return "pip"


def install_dependencies(pip_executable: str, requirements_path: Path) -> bool:
    print(f"Installing dependencies from {requirements_path}...")
    try:
        subprocess.run([pip_executable, "install", "-r", str(requirements_path)], check=True)
        print("Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def create_directories() -> bool:
    print("Creating directory structure...")
    project_root = Path(__file__).parent
    directories = [
        project_root / "data" / "rdd2022" / "train" / "images",
        project_root / "data" / "rdd2022" / "train" / "labels",
        project_root / "data" / "rdd2022" / "val" / "images",
        project_root / "data" / "rdd2022" / "val" / "labels",
        project_root / "data" / "rdd2022" / "test" / "images",
        project_root / "outputs" / "runs",
        project_root / "outputs" / "predictions",
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("Directory structure created")
    return True


def verify_installation() -> dict:
    print("\nVerifying installation...")
    results = {
        'torch': False,
        'ultralytics': False,
        'cv2': False,
        'albumentations': False,
        'cuda': False,
        'gpu_name': None
    }

    try:
        import torch
        results['torch'] = True
        results['cuda'] = torch.cuda.is_available()
        if results['cuda']:
            results['gpu_name'] = torch.cuda.get_device_name(0)
        print(f"  PyTorch {torch.__version__}")
        if results['cuda']:
            print(f"    GPU: {results['gpu_name']}")
        else:
            print("    No GPU available (CPU mode)")
    except ImportError:
        print("  PyTorch not installed")

    try:
        from ultralytics import YOLO
        import ultralytics
        results['ultralytics'] = True
        print(f"  Ultralytics {ultralytics.__version__}")
    except ImportError:
        print("  Ultralytics not installed")

    try:
        import cv2
        results['cv2'] = True
        print(f"  OpenCV {cv2.__version__}")
    except ImportError:
        print("  OpenCV not installed")

    try:
        import albumentations
        results['albumentations'] = True
        print(f"  Albumentations {albumentations.__version__}")
    except ImportError:
        print("  Albumentations not installed (optional)")

    try:
        import gdown
        print(f"  gdown {gdown.__version__}")
    except ImportError:
        print("  gdown not installed (needed for dataset download)")

    core_ok = results['torch'] and results['ultralytics'] and results['cv2']

    print()
    if core_ok:
        print("Core dependencies installed successfully!")
        if results['cuda']:
            print(f"GPU acceleration available: {results['gpu_name']}")
        else:
            print("No GPU detected - training will be slow")
    else:
        print("Some core dependencies are missing")

    return results


def main():
    parser = argparse.ArgumentParser(description="Setup Crackathon environment")
    parser.add_argument('--venv', action='store_true',
                        help='Create virtual environment')
    parser.add_argument('--verify', action='store_true',
                        help='Verify installation only')
    parser.add_argument('--venv-path', type=str, default='venv',
                        help='Path for virtual environment')

    args = parser.parse_args()

    project_root = Path(__file__).parent
    requirements_path = project_root / "requirements.txt"
    venv_path = project_root / args.venv_path

    print("=" * 60)
    print("CRACKATHON ENVIRONMENT SETUP")
    print("=" * 60)

    if args.verify:
        verify_installation()
        return

    if args.venv:
        if not create_venv(venv_path):
            return
        pip_executable = get_pip_executable(venv_path)
        print(f"\nActivate the environment with:")
        if sys.platform == "win32":
            print(f"  {venv_path}\\Scripts\\activate")
        else:
            print(f"  source {venv_path}/bin/activate")
    else:
        pip_executable = get_pip_executable()

    if requirements_path.exists():
        if not install_dependencies(pip_executable, requirements_path):
            return
    else:
        print(f"requirements.txt not found at {requirements_path}")

    create_directories()

    if not args.venv:
        verify_installation()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    if args.venv:
        if sys.platform == "win32":
            print(f"  1. Activate: {venv_path}\\Scripts\\activate")
        else:
            print(f"  1. Activate: source {venv_path}/bin/activate")
        print("  2. Run: python main.py --full")
    else:
        print("  Run: python main.py --full")


if __name__ == "__main__":
    main()
