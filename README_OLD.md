# WeakMedSAM with ACArL Enhancement

This repository contains the implementation of WeakMedSAM enhanced with **ACArL** (Adaptive Class-Conditional Affinity Refinement with Learnable Sub-class Exploration), a mathematically principled improvement that replaces fixed, class-agnostic affinity refinement with class-conditional, learnable affinity modeling.

## What's New in ACArL

ACArL introduces several key improvements:

1. **Learnable Sub-class Posterior**: Replaces fixed K-means clustering with a continuous, learnable posterior distribution over sub-class assignments using variational inference.

2. **Class-Conditional Affinity Learning**: Each class learns its own affinity matrix via parametric MLPs, capturing class-specific structural patterns (e.g., tumor edges vs. organ boundaries).

3. **Anisotropic Diffusion**: Refines CAMs using learned, class-conditional affinity matrices as diffusion coefficients, providing more accurate boundary refinement.

4. **Uncertainty-Aware Pseudo-Labels**: Generates confidence scores for each pixel, enabling confidence-weighted U-Net training that reduces error cascading.

## Mathematical Framework

ACArL optimizes:
```
L_ACArL = L_p + λ_s * L_s(z) + λ_a * L_a(z, A_c)
```

Where:
- **L_s**: Sub-class exploration loss (KL divergence + entropy regularization)
- **L_a**: Class-conditional affinity learning loss (contrastive)
- **A_c**: Learned class-specific affinity matrices

See `improvement.md` for complete mathematical formulation.

## Supported Datasets

This implementation supports:
- **BraTS 2019**: Brain tumor segmentation dataset
- **MICCAI Medical Decathlon Task02_Heart**: Cardiac MRI segmentation dataset

---

# Quick Start with Pipeline Script

The easiest way to run the complete pipeline is using the provided script:

```bash
# For BraTS dataset
bash run_pipeline.sh --data-path /path/to/brats/dataset --gpus 0

# For MCD Heart dataset
bash run_pipeline.sh --data-path /path/to/Task02_Heart --gpus 0

# Run specific stage (e.g., only preprocessing)
bash run_pipeline.sh --data-path /path/to/dataset --stage 1 --gpus 0
```

The pipeline automatically:
- Detects dataset type (BraTS vs MCD)
- Handles preprocessing and split generation
- Runs all stages: preprocessing → clustering → ACArL training → pseudo-label generation → U-Net training → evaluation
- Skips completed stages on re-runs
- Organizes outputs by timestamp/index

**Pipeline Stages:**
0. Environment setup
1. Data preprocessing (NIfTI → 2D slices)
2. Pre-clustering (K-means on ResNet features)
3. ACArL training (learnable affinity refinement)
4. Pseudo-label generation with confidence maps
5. U-Net training with confidence weighting
6. Evaluation

---

# Manual Setup (Advanced Users)

## Environment Setup

```bash
# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Data Preparation

### BraTS 2019
1. Download from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/brain-tumor-segmentation-brats-2019)
2. Preprocess:
```bash
python brats/preprocess.py --input-path /path/to/brats --output-path /path/to/brats/preprocessed
```

### MICCAI Medical Decathlon Task02_Heart
1. Download Task02_Heart from [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
2. Preprocess:
```bash
python mcd/preprocess.py --input-path /path/to/Task02_Heart --output-path /path/to/Task02_Heart/preprocessed
```

## Pre-clustering

```bash
# BraTS
python cluster.py --batch_size 256 --data_path /path/to/brats/preprocessed --data_module brats --parent_classes 1 --child_classes 8 --save_path ./clusters --gpus 0

# MCD
python cluster.py --batch_size 256 --data_path /path/to/Task02_Heart/preprocessed --data_module mcd --parent_classes 1 --child_classes 8 --save_path ./clusters --gpus 0
```

## Training WeakMedSAM with ACArL

1. Download SAM ViT-b checkpoint from [metaAI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

2. Train:
```bash
# BraTS
python train.py --seed 42 --sam_ckpt ./checkpoints/sam_vit_b_01ec64.pth --lr 1e-4 --batch_size 24 --max_epochs 10 --val_iters 3000 --index run1 --data_path /path/to/brats/preprocessed --data_module brats --parent_classes 1 --child_classes 8 --child_weight 0.5 --acarl_weight 0.3 --cluster_file ./clusters/brats-8.bin --logdir ./logs --gpus 0

# MCD
python train.py --seed 42 --sam_ckpt ./checkpoints/sam_vit_b_01ec64.pth --lr 1e-4 --batch_size 24 --max_epochs 10 --val_iters 3000 --index run1 --data_path /path/to/Task02_Heart/preprocessed --data_module mcd --parent_classes 1 --child_classes 8 --child_weight 0.5 --acarl_weight 0.3 --cluster_file ./clusters/Task02_Heart-8.bin --logdir ./logs --gpus 0
```

**New arguments**:
- `--acarl_weight`: Weight for ACArL loss (default: 0.3)
- `--acarl_ckpt`: Path to pre-trained ACArL module (optional)

This saves two checkpoints:
- `./logs/run1/run1.pth`: Main SAMUS model
- `./logs/run1/run1_acarl.pth`: ACArL module

## Generating Pseudo Labels with ACArL

```bash
# BraTS
python lab_gen_acarl.py --batch-size 24 --data-path /path/to/brats/preprocessed --save-path ./pseudo_labels --data-module brats --parent-classes 1 --child-classes 8 --samus-ckpt ./logs/run1/run1.pth --acarl-ckpt ./logs/run1/run1_acarl.pth --sam-ckpt ./checkpoints/sam_vit_b_01ec64.pth --diffusion-steps 3 --diffusion-step-size 0.1 --threshold 0.5 --save-confidence --conf-path ./confidence --save-soft-labels --soft-path ./soft_labels --gpus 0

# MCD
python lab_gen_acarl.py --batch-size 24 --data-path /path/to/Task02_Heart/preprocessed --save-path ./pseudo_labels --data-module mcd --parent-classes 1 --child-classes 8 --samus-ckpt ./logs/run1/run1.pth --acarl-ckpt ./logs/run1/run1_acarl.pth --sam-ckpt ./checkpoints/sam_vit_b_01ec64.pth --diffusion-steps 3 --diffusion-step-size 0.1 --threshold 0.5 --save-confidence --conf-path ./confidence --save-soft-labels --soft-path ./soft_labels --gpus 0
```

**New arguments**:
- `--acarl-ckpt`: Path to trained ACArL module (required)
- `--diffusion-steps`: Number of anisotropic diffusion steps (default: 3)
- `--diffusion-step-size`: Step size for diffusion (default: 0.1)
- `--save-confidence`: Save pixel-wise confidence maps
- `--conf-path`: Directory to save confidence maps
- `--save-soft-labels`: Save continuous CAM values for confidence weighting
- `--soft-path`: Directory to save soft pseudo-labels

**Legacy method** (without ACArL):
```bash
python lab_gen.py --batch-size 24 --data-path /path/to/dataset/preprocessed --save-path ./pseudo_labels --data-module brats --parent-classes 1 --child-classes 8 --samus-ckpt ./logs/run1/run1.pth --sam-ckpt ./checkpoints/sam_vit_b_01ec64.pth --t 4 --beta 4 --threshold 0.5 --gpus 0
```

## Training Segmentation Network with Confidence Weighting

**ACArL-enhanced** (with confidence weighting):
```bash
# BraTS
python train_unet_acarl.py --seed 42 --lr 1e-4 --batch_size 128 --max_epochs 50 --val_iters 500 --index run1_unet --data_path /path/to/brats/preprocessed --lab_path ./pseudo_labels --conf_path ./confidence --use_confidence --conf_threshold 0.3 --loss_type combined --data_module brats --num_classes 2 --logdir ./logs --gpus 0

# MCD
python train_unet_acarl.py --seed 42 --lr 1e-4 --batch_size 128 --max_epochs 50 --val_iters 500 --index run1_unet --data_path /path/to/Task02_Heart/preprocessed --lab_path ./pseudo_labels --conf_path ./confidence --use_confidence --conf_threshold 0.3 --loss_type combined --data_module mcd --num_classes 2 --logdir ./logs --gpus 0
```

**New arguments**:
- `--use_confidence`: Enable confidence-weighted loss
- `--conf_path`: Path to confidence maps
- `--conf_threshold`: Minimum confidence threshold (default: 0.3)
- `--loss_type`: Loss function: `ce` (cross-entropy), `dice`, or `combined` (default: ce)

**Legacy method** (without confidence weighting):
```bash
python train_unet.py --seed 42 --lr 1e-4 --batch_size 128 --max_epochs 50 --val_iters 500 --index run1_unet --data_path /path/to/dataset/preprocessed --lab_path ./pseudo_labels --data_module brats --num_classes 2 --logdir ./logs --gpus 0
```

## Evaluation

```bash
# BraTS
python eval.py --data_path /path/to/brats/preprocessed --data_module brats --batch_size 128 --num_classes 2 --ckpt ./logs/run1_unet/run1_unet.pth --gpus 0

# MCD
python eval.py --data_path /path/to/Task02_Heart/preprocessed --data_module mcd --batch_size 128 --num_classes 2 --ckpt ./logs/run1_unet/run1_unet.pth --gpus 0
```

**Expected output:**
```
dice:           79.69
jaccard:        74.06
assd:           5.57
hd95:           28.34
```

---

# Project Structure

```
WeakMedSAM/
├── brats/                 # BraTS dataset utilities
│   ├── dataset.py         # BraTS data loading
│   └── preprocess.py      # BraTS preprocessing
├── mcd/                   # MCD dataset utilities
│   ├── dataset.py         # MCD data loading
│   └── preprocess.py      # MCD preprocessing
├── samus/                 # SAMUS model components
├── unet/                  # U-Net model
├── utils/                 # Utilities and metrics
├── checkpoints/           # Pre-trained checkpoints
├── logs/                  # Training logs and outputs
├── run_pipeline.sh        # Main pipeline script
├── train.py               # ACArL training
├── lab_gen_acarl.py       # Pseudo-label generation
├── train_unet_acarl.py    # U-Net training with confidence
├── eval.py                # Evaluation
├── cluster.py             # Pre-clustering
└── README.md
```

---

# Citation

If you use this code, please cite:

```bibtex
@article{weakmedsam2024,
  title={WeakMedSAM with ACArL Enhancement},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```