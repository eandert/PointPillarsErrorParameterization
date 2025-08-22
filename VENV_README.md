# PointPillars Virtual Environment Setup

This directory contains scripts to set up and manage the PointPillars virtual environment.

## Quick Start

### 1. Initial Setup (First time only)
```bash
./setup_env.sh
```
This script will:
- Install required system packages (python3.10-venv)
- Create a virtual environment with Python 3.10
- Install PyTorch with CUDA 12.1 support
- Install all dependencies from `requirements_updated.txt`
- Build CUDA extensions
- Install PointPillars in development mode
- Verify the installation

### 2. Activate Environment (Every time you work)
```bash
./activate.sh
```
This script will:
- Activate the virtual environment
- Show environment information
- Display available commands

### 3. Deactivate Environment
```bash
deactivate
```

## Manual Commands

If you prefer to run commands manually:

```bash
# Create and setup environment
python3 -m venv pointpillars_env
source pointpillars_env/bin/activate
pip install --upgrade pip
pip install wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_updated.txt
python setup.py build_ext --inplace
pip install -e .

# Activate environment
source pointpillars_env/bin/activate

# Deactivate environment
deactivate
```

## Environment Details

- **Python Version**: 3.10.12
- **PyTorch Version**: 2.5.1+cu121
- **CUDA Version**: 12.1 (compatible with system CUDA 12.6)
- **Virtual Environment**: `pointpillars_env/`

## Troubleshooting

### CUDA Version Mismatch
If you see CUDA version errors, ensure you're using CUDA 12.1 PyTorch:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Missing python3-venv
If virtual environment creation fails:
```bash
sudo apt update && sudo apt install -y python3.10-venv
```

### Rebuilding CUDA Extensions
If you need to rebuild the CUDA extensions:
```bash
source pointpillars_env/bin/activate
python setup.py build_ext --inplace
```

## Available Scripts

- `setup_env.sh` - Complete environment setup
- `activate.sh` - Quick environment activation
- `requirements_updated.txt` - Updated dependencies for Python 3.10 
