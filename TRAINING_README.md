# PointPillars Training Guide

This guide covers the complete training pipeline for PointPillars on the KITTI dataset.

## Quick Start

### 1. Setup Environment
```bash
./setup_env.sh
```

### 2. Download and Setup Dataset
```bash
./setup_data.sh [data_path] [download_path]
# Example: ./setup_data.sh ./data/kitti ./downloads
```

### 3. Start Training
```bash
./train.sh [data_path] [logs_path] [batch_size] [num_workers] [max_epoch] [learning_rate]
# Example: ./train.sh ./data/kitti ./pillar_logs 6 4 160 0.00025
```

## Detailed Steps

### Step 1: Environment Setup
The environment setup script installs all dependencies and builds CUDA extensions:
```bash
./setup_env.sh
```

**What it does:**
- Creates virtual environment with Python 3.10
- Installs PyTorch with CUDA 12.1 support
- Installs all required packages
- Builds CUDA extensions
- Verifies installation

### Step 2: Dataset Setup
The dataset setup script downloads and organizes the KITTI dataset:
```bash
./setup_data.sh [data_path] [download_path]
```

**What it downloads:**
- Point clouds (29GB) - Velodyne data
- Images (12GB) - Camera images
- Calibration files (16MB) - Camera calibration
- Labels (5MB) - Ground truth annotations

**Dataset structure created:**
```
kitti/
├── training/
│   ├── calib/          # 7481 .txt files
│   ├── image_2/        # 7481 .png files
│   ├── label_2/        # 7481 .txt files
│   └── velodyne/       # 7481 .bin files
└── testing/
    ├── calib/          # 7518 .txt files
    ├── image_2/        # 7518 .png files
    └── velodyne/       # 7518 .bin files
```

### Step 3: Training
The training script handles the complete training pipeline:
```bash
./train.sh [data_path] [logs_path] [batch_size] [num_workers] [max_epoch] [learning_rate]
```

**What it does:**
- Activates virtual environment
- Checks CUDA availability
- Runs pre-processing if needed
- Starts training with specified parameters
- Saves checkpoints and logs

## Script Parameters

### setup_data.sh
- `data_path`: Where to store the organized dataset (default: `./data/kitti`)
- `download_path`: Where to store downloaded files (default: `./downloads`)

### train.sh
- `data_path`: Path to KITTI dataset (default: `./data/kitti`)
- `logs_path`: Path to save training logs (default: `./pillar_logs`)
- `batch_size`: Training batch size (default: `6`)
- `num_workers`: Number of data loading workers (default: `4`)
- `max_epoch`: Maximum training epochs (default: `160`)
- `learning_rate`: Initial learning rate (default: `0.00025`)

## Training Configuration

### Default Training Parameters
- **Batch Size**: 6 (adjust based on GPU memory)
- **Learning Rate**: 0.00025
- **Epochs**: 160
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Weight Decay**: 0.01
- **Classes**: 3 (Car, Pedestrian, Cyclist)

### GPU Memory Requirements
- **Minimum**: 8GB GPU memory
- **Recommended**: 16GB+ GPU memory
- **Batch size 6**: ~12GB GPU memory
- **Batch size 4**: ~8GB GPU memory

### Performance Tips
1. **Reduce batch size** if you run out of GPU memory
2. **Increase num_workers** if you have many CPU cores
3. **Use SSD storage** for faster data loading
4. **Monitor GPU memory** with `nvidia-smi`

## Monitoring Training

### TensorBoard
Monitor training progress with TensorBoard:
```bash
tensorboard --logdir ./pillar_logs/summary
```

### Checkpoints
Checkpoints are saved every 20 epochs:
- Location: `./pillar_logs/checkpoints/`
- Format: `epoch_X.pth`

### Logs
Training logs are saved to:
- TensorBoard logs: `./pillar_logs/summary/`
- Checkpoints: `./pillar_logs/checkpoints/`

## Evaluation

After training, evaluate your model:
```bash
python evaluate.py --data_root ./data/kitti --ckpt ./pillar_logs/checkpoints/epoch_160.pth
```

## Troubleshooting

### Common Issues

#### 1. Out of GPU Memory
**Symptoms**: CUDA out of memory error
**Solution**: Reduce batch size
```bash
./train.sh ./data/kitti ./pillar_logs 4 4 160 0.00025
```

#### 2. Dataset Not Found
**Symptoms**: "Dataset directory not found" error
**Solution**: Run dataset setup first
```bash
./setup_data.sh
```

#### 3. CUDA Version Mismatch
**Symptoms**: CUDA version errors during training
**Solution**: Ensure PyTorch CUDA version matches system
```bash
source pointpillars_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Pre-processing Errors
**Symptoms**: Errors during pre-processing
**Solution**: Check dataset structure and file permissions
```bash
ls -la ./data/kitti/training/
```

### Performance Optimization

#### For Jetson Orin:
- Use batch size 2-4
- Set num_workers to 2-4
- Monitor GPU temperature
- Use SSD storage for dataset

#### For Desktop GPUs:
- Use batch size 6-8
- Set num_workers to 4-8
- Use multiple GPUs if available

## Expected Training Time

### Hardware Estimates:
- **RTX 4090 (24GB)**: ~8-12 hours
- **RTX 3080 (10GB)**: ~12-16 hours
- **Jetson Orin (8GB)**: ~24-48 hours
- **CPU only**: ~1-2 weeks

### Training Progress:
- **Epoch 1-20**: Rapid loss decrease
- **Epoch 20-80**: Steady improvement
- **Epoch 80-160**: Fine-tuning, diminishing returns

## File Structure After Training

```
PointPillars/
├── data/
│   └── kitti/                    # Dataset
├── pillar_logs/                  # Training logs
│   ├── checkpoints/              # Model checkpoints
│   └── summary/                  # TensorBoard logs
├── downloads/                    # Downloaded files
├── pointpillars_env/            # Virtual environment
└── [training scripts]
```

## Next Steps

After successful training:
1. **Evaluate** the model on test set
2. **Visualize** results with evaluation tools
3. **Deploy** the model for inference
4. **Fine-tune** hyperparameters if needed

## Support

For issues:
1. Check the troubleshooting section above
2. Verify CUDA and PyTorch installation
3. Check GPU memory availability
4. Review training logs for specific errors 
