# PointPillars Scripts Overview

This document provides an overview of all available scripts for the PointPillars project.

## 📁 Available Scripts

### 🛠️ Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_env.sh` | Complete environment setup | `./setup_env.sh` |
| `activate.sh` | Activate virtual environment | `./activate.sh` |
| `setup_data.sh` | Download and setup KITTI dataset | `./setup_data.sh [data_path] [download_path]` |

### 🚀 Training Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `train.sh` | Complete training pipeline | `./train.sh [data_path] [logs_path] [batch_size] [num_workers] [max_epoch] [learning_rate]` |

### 🧪 Testing Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_setup.sh` | Test complete setup | `./test_setup.sh` |

## 🚀 Quick Start Commands

### 1. Initial Setup (First time only)
```bash
# Setup environment
./setup_env.sh

# Test setup
./test_setup.sh

# Download dataset (optional - 46GB)
./setup_data.sh

# Start training
./train.sh
```

### 2. Daily Usage
```bash
# Activate environment
./activate.sh

# Start training (if dataset is ready)
./train.sh
```

## 📋 Script Details

### `setup_env.sh`
**Purpose**: Complete environment setup with CUDA support
**What it does**:
- Installs `python3.10-venv` if missing
- Creates virtual environment
- Installs PyTorch with CUDA 12.1
- Installs all dependencies
- Builds CUDA extensions
- Verifies installation

**Usage**: `./setup_env.sh`

### `activate.sh`
**Purpose**: Quick environment activation
**What it does**:
- Activates virtual environment
- Shows environment information
- Displays available commands

**Usage**: `./activate.sh`

### `setup_data.sh`
**Purpose**: Download and organize KITTI dataset
**What it downloads**:
- Point clouds (29GB)
- Images (12GB)
- Calibration files (16MB)
- Labels (5MB)

**Usage**: `./setup_data.sh [data_path] [download_path]`
**Example**: `./setup_data.sh ./data/kitti ./downloads`

### `train.sh`
**Purpose**: Complete training pipeline
**What it does**:
- Activates environment
- Checks CUDA availability
- Runs pre-processing if needed
- Starts training
- Saves checkpoints and logs

**Usage**: `./train.sh [data_path] [logs_path] [batch_size] [num_workers] [max_epoch] [learning_rate]`
**Example**: `./train.sh ./data/kitti ./pillar_logs 6 4 160 0.00025`

### `test_setup.sh`
**Purpose**: Test complete setup
**What it tests**:
- Virtual environment
- Python imports
- CUDA availability
- Script availability
- GPU status
- Disk space
- Network connectivity

**Usage**: `./test_setup.sh`

## 📊 Parameter Reference

### setup_data.sh Parameters
- `data_path`: Dataset storage location (default: `./data/kitti`)
- `download_path`: Download storage location (default: `./downloads`)

### train.sh Parameters
- `data_path`: KITTI dataset path (default: `./data/kitti`)
- `logs_path`: Training logs path (default: `./pillar_logs`)
- `batch_size`: Training batch size (default: `6`)
- `num_workers`: Data loading workers (default: `4`)
- `max_epoch`: Training epochs (default: `160`)
- `learning_rate`: Initial learning rate (default: `0.00025`)

## 🔧 Manual Commands

If you prefer manual commands:

```bash
# Environment setup
python3 -m venv pointpillars_env
source pointpillars_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_updated.txt
python setup.py build_ext --inplace
pip install -e .

# Dataset pre-processing
python pre_process_kitti.py --data_root ./data/kitti

# Training
python train.py --data_root ./data/kitti --saved_path ./pillar_logs
```

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir ./pillar_logs/summary
```

### GPU Monitoring
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

### Common Issues

1. **Environment not found**
   ```bash
   ./setup_env.sh
   ```

2. **CUDA version mismatch**
   ```bash
   source pointpillars_env/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Out of GPU memory**
   ```bash
   ./train.sh ./data/kitti ./pillar_logs 4 4 160 0.00025
   ```

4. **Dataset not found**
   ```bash
   ./setup_data.sh
   ```

## 📁 File Structure

```
PointPillars/
├── setup_env.sh      # Environment setup
├── activate.sh             # Environment activation
├── setup_data.sh           # Dataset setup
├── train.sh               # Training script
├── test_setup.sh          # Setup test
├── pointpillars_env/      # Virtual environment
├── data/                  # Dataset (after setup)
├── pillar_logs/           # Training logs (after training)
├── downloads/             # Downloaded files (after setup)
├── VENV_README.md         # Virtual environment guide
├── TRAINING_README.md     # Training guide
└── README_SCRIPTS.md      # This file
```

## 📚 Additional Documentation

- `VENV_README.md` - Virtual environment setup guide
- `TRAINING_README.md` - Complete training guide
- `requirements_updated.txt` - Updated dependencies

## 🎯 Next Steps

After setup:
1. **Test setup**: `./test_setup.sh`
2. **Download dataset**: `./setup_data.sh` (optional)
3. **Start training**: `./train.sh`
4. **Monitor training**: `tensorboard --logdir ./pillar_logs/summary`
5. **Evaluate model**: `python evaluate.py --data_root ./data/kitti --ckpt ./pillar_logs/checkpoints/epoch_160.pth` 
