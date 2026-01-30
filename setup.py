"""
Setup script to create virtual environment and install dependencies.
Run: python setup.py
"""
import subprocess
import sys
import os

def run_command(command, check=True):
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("ATS Resume Analyzer - Environment Setup")
    print("=" * 60)
    
    # Check if virtual environment exists
    venv_path = "venv"
    if not os.path.exists(venv_path):
        print("\nCreating virtual environment...")
        if sys.platform == "win32":
            run_command(f"{sys.executable} -m venv {venv_path}")
            pip_cmd = f"{venv_path}\\Scripts\\pip"
            python_cmd = f"{venv_path}\\Scripts\\python"
        else:
            run_command(f"{sys.executable} -m venv {venv_path}")
            pip_cmd = f"{venv_path}/bin/pip"
            python_cmd = f"{venv_path}/bin/python"
    else:
        print("\nVirtual environment already exists.")
        if sys.platform == "win32":
            pip_cmd = f"{venv_path}\\Scripts\\pip"
            python_cmd = f"{venv_path}\\Scripts\\python"
        else:
            pip_cmd = f"{venv_path}/bin/pip"
            python_cmd = f"{venv_path}/bin/python"
    
    # Upgrade pip
    print("\nUpgrading pip...")
    run_command(f"{pip_cmd} install --upgrade pip", check=False)
    
    # Install PyTorch with CUDA support detection
    print("\nDetecting CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print("CUDA is available. Installing PyTorch with CUDA support...")
            run_command(f"{pip_cmd} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118", check=False)
        else:
            print("CUDA not available. Installing CPU-only PyTorch...")
            run_command(f"{pip_cmd} install torch torchvision torchaudio", check=False)
    except ImportError:
        print("PyTorch not found. Installing CPU version (CUDA can be added later)...")
        run_command(f"{pip_cmd} install torch torchvision torchaudio", check=False)
    
    # Install other requirements
    print("\nInstalling other dependencies...")
    run_command(f"{pip_cmd} install -r requirements.txt")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("\nTo run the application:")
    print(f"  {python_cmd} app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

