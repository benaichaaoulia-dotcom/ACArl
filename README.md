# WeakMedSAM with ACArL

Weakly-supervised medical image segmentation using SAM (Segment Anything Model) enhanced with **ACArL** - Adaptive Class-Conditional Affinity Refinement with Learnable Sub-class Exploration.

## Overview

WeakMedSAM leverages SAM's zero-shot segmentation capabilities to generate pseudo-labels from image-level annotations. ACArL improves upon this by introducing:

- **Class-Conditional Affinity Learning**: Learnable affinity matrices per class for capturing class-specific structural patterns
- **Anisotropic Diffusion**: Graph-based CAM refinement using learned affinity coefficients
- **Uncertainty Quantification**: Confidence-weighted pseudo-labels for improved training

## Key Components

```
├── train.py                    # ACArL training on SAM features
├── lab_gen_acarl.py           # Pseudo-label generation with ACArL
├── train_unet_acarl.py        # U-Net training with confidence weighting
├── eval.py                    # Evaluation metrics
├── cluster.py                 # K-means pre-clustering
│
├── samus/                     # SAM-based model
│   └── modeling/acarl.py      # ACArL module implementation
├── unet/                      # U-Net segmentation model
├── brats/, kits/, lits/      # Dataset preprocessing modules
└── utils/                     # Utility functions
```

## Installation

### Requirements
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 24GB+ GPU memory (for SAM encoder)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd WeakMedSAM

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint
mkdir -p checkpoints
wget -O checkpoints/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Quick Start

### 1. Prepare Your Dataset

Place raw dataset (3D NIfTI files) in a directory:
```
dataset/
├── Patient_001/
│   ├── image.nii.gz
│   └── label.nii.gz
└── Patient_002/
    ├── image.nii.gz
    └── label.nii.gz
```

### 2. Run Full Pipeline

```bash
# BraTS example
python train.py \
    --data_path ./BRATS_Dataset \
    --data_module brats \
    --vit_name vit_b \
    --sam_ckpt checkpoints/sam_vit_b_01ec64.pth \
    --batch_size 8 \
    --lr 1e-4 \
    --max_epochs 10 \
    --parent_classes 1 \
    --child_classes 8 \
    --cluster_file logs/clusters.bin \
    --logdir logs \
    --seed 42 \
    --gpus 0
```

### 3. Generate Pseudo-Labels

```bash
python lab_gen_acarl.py \
    --data-path ./BRATS_Dataset \
    --data-module brats \
    --samus-ckpt logs/model_checkpoint.pth \
    --acarl-ckpt logs/acarl_checkpoint.pth \
    --sam-ckpt checkpoints/sam_vit_b_01ec64.pth \
    --save-path ./pseudo_labels \
    --save-confidence \
    --conf-path ./confidence_maps \
    --parent-classes 1 \
    --child-classes 8
```

### 4. Train Segmentation Network

```bash
python train_unet_acarl.py \
    --data_path ./BRATS_Dataset \
    --data_module brats \
    --lab_path ./pseudo_labels \
    --conf_path ./confidence_maps \
    --use_confidence \
    --batch_size 128 \
    --lr 1e-4 \
    --max_epochs 10 \
    --num_classes 2 \
    --seed 42 \
    --gpus 0
```

### 5. Evaluate

```bash
python eval.py \
    --data_path ./BRATS_Dataset \
    --data_module brats \
    --ckpt logs/unet_checkpoint.pth \
    --num_classes 2 \
    --gpus 0
```

## Supported Datasets

The framework includes preprocessing modules for:

- **BraTS** (`brats/`): Brain tumor segmentation
- **KITS** (`kits/`): Kidney tumor segmentation  
- **LITS** (`lits/`): Liver tumor segmentation

Each dataset module provides:
- `dataset.py`: PyTorch dataset loaders
- `preprocess.py`: NIfTI to 2D slice conversion

## Architecture

### ACArL Module

```python
from samus.modeling.acarl import ACArLModule

acarl = ACArLModule(
    encoder_dim=768,           # ViT-B encoder dimension
    num_classes=1,             # Number of target classes
    num_sub_classes=8,         # Number of sub-classes for exploration
    diffusion_steps=3,         # Anisotropic diffusion steps
    disable_affinity_learning=False,  # Enable Block A
    disable_diffusion=False            # Enable Block B
)

# Forward pass
output = acarl(features, cam_raw, y_primary)
# Returns: refined CAMs, confidence maps, affinity matrices
```

### Ablation Study Flags

Control ACArL components:
```bash
# No Class-Conditional Affinity (Block A)
--disable_affinity_learning

# No Anisotropic Diffusion (Block B)
--disable_diffusion

# Both disabled (baseline sub-class exploration only)
--disable_affinity_learning --disable_diffusion
```

## Hyperparameters

Key parameters for ACArL:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_s` | 1.0 | Sub-class exploration loss weight |
| `lambda_a` | 0.2 | Affinity learning loss weight |
| `lambda_conf` | 1.0 | Confidence regularization weight |
| `acarl_weight` | 0.3 | Overall ACArL contribution |
| `diffusion_steps` | 3 | Number of diffusion iterations |
| `conf_threshold` | 0.3 | Confidence threshold for pseudo-labels |

## Output Structure

```
logs/
├── train.log                  # ACArL training log
├── model_checkpoint.pth       # Trained SAMUS model
├── acarl_checkpoint.pth       # Trained ACArL module
├── pseudo_labels/             # Generated pseudo-label masks
├── confidence_maps/           # Confidence score maps
├── unet_train.log             # U-Net training log
├── unet_checkpoint.pth        # Trained U-Net model
└── eval.log                   # Evaluation results
```

## Configuration

Main training parameters in command-line arguments:

```python
parser.add_argument("--data_path", help="Dataset directory")
parser.add_argument("--data_module", choices=["brats", "kits", "lits"])
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--parent_classes", type=int, help="Number of target classes")
parser.add_argument("--child_classes", type=int, default=8, help="Sub-classes")
parser.add_argument("--acarl_weight", type=float, default=0.3)
parser.add_argument("--lambda_s", type=float, default=1.0)
parser.add_argument("--lambda_a", type=float, default=0.2)
parser.add_argument("--lambda_conf", type=float, default=1.0)
```

## Visualization

Plot training convergence curves:

```bash
python plot_single_exp.py logs/experiment_dir
```

Generates convergence plots for train/val loss and accuracy metrics.

## Troubleshooting

### Out of Memory

- Reduce `batch_size`
- Reduce `max_epochs` 
- Enable gradient checkpointing (automatic in code)

### Missing Data Files

Ensure dataset directory structure matches expected format. See dataset preprocessing modules for details.

### Tensorboard Event Files Not Found

Check that training completed successfully. Event files are created during training in the log directory.

## References

- **SAM**: Segment Anything Model (Meta AI)
- **WeakMedSAM**: Weakly-supervised medical segmentation
- **ACArL**: Class-conditional affinity refinement approach

## License

See LICENSE file for details.

## Citation

If you use this implementation, please cite:

```bibtex
@article{acarl2026,
  title={ACArL: Anisotropic Class-conditional Affinity Refinement Learning for Weakly Supervised Medical Image Segmentation},
  author={O. Benaicha, B. Khaldi and O. Aiadi},
  year={2026}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
