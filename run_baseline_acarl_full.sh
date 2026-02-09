#!/bin/bash

################################################################################
# Baseline + ACArL Pseudo-Labels Pipeline for Flat BRATS Dataset
# 
# Handles both:
# - Standard BRATS structure (HGG/LGG subdirectories)
# - Flat structure (all patient directories at root)
#
# Usage:
#   bash run_baseline_acarl_full.sh --data-path /path/to/dataset [--gpus 0]
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
GPUS=${GPUS:-"0"}
SEED=42
DATASET_PATH=""

# Configuration
BATCH_SIZE=8
LR=1e-4
MAX_EPOCHS=10
VAL_ITERS=1000
PARENT_CLASSES=1
CHILD_CLASSES=8
CHILD_WEIGHT=0.5
ACARL_WEIGHT=0.3
UNET_BATCH_SIZE=128
UNET_MAX_EPOCHS=50
UNET_VAL_ITERS=500
UNET_LR=1e-4
DIFFUSION_STEPS=3
DIFFUSION_STEP_SIZE=0.1
THRESHOLD=0.5

SAM_CHECKPOINT="./checkpoints/sam_vit_b_01ec64.pth"

print_header() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

# Parse arguments
START_STAGE=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --start-stage)
            START_STAGE="$2"
            shift 2
            ;;
        *)
            print_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate dataset path
if [ -z "$DATASET_PATH" ]; then
    print_error "Dataset path not specified!"
    print_info "Usage: bash run_baseline_acarl_full.sh --data-path /path/to/dataset [--gpus 0]"
    exit 1
fi

# Resolve absolute path
DATASET_PATH=$(cd "$DATASET_PATH" 2>/dev/null && pwd) || {
    print_error "Dataset path does not exist: $DATASET_PATH"
    exit 1
}

DATASET_NAME=$(basename "$DATASET_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
INDEX="baseline_acarl_${TIMESTAMP}"

# Setup paths
LOGDIR="./logs_baseline_acarl"
DATA_MODULE="brats"
if [[ "$DATASET_NAME" == *"mcd"* ]] || [[ "$DATASET_NAME" == *"Task02_Heart"* ]]; then
    DATA_MODULE="mcd"
fi

PREPROCESS_OUTPUT="$LOGDIR/$INDEX/preprocessed_data"
CLUSTER_FILE_PATH="./data/clusters_$DATASET_NAME"
PSEUDO_LABEL_PATH="$LOGDIR/$INDEX/pseudo_labels"
CONFIDENCE_PATH="$LOGDIR/$INDEX/confidence_maps"
SOFT_LABEL_PATH="$LOGDIR/$INDEX/soft_labels"

mkdir -p "$LOGDIR/$INDEX"
mkdir -p "$CLUSTER_FILE_PATH"

print_header "Baseline + ACArL Pseudo-Labels Pipeline"
print_info "Dataset: $DATASET_PATH"
print_info "Data Module: $DATA_MODULE"
print_info "Logs: $LOGDIR/$INDEX"
print_info "GPU: $GPUS"

# ============================================================================
# STAGE 1: Preprocessing
# ============================================================================
if [ "$START_STAGE" -gt 1 ]; then
    print_info "Skipping STAGE 1 (START_STAGE=$START_STAGE)"
    # Infer DATA_PATH_USED from dataset
    if [ -d "$DATASET_PATH/preprocessed_data" ]; then
        DATA_PATH_USED="$DATASET_PATH/preprocessed_data"
    else
        DATA_PATH_USED="$DATASET_PATH"
    fi
else
    print_header "STAGE 1: Data Preprocessing"
fi

if [ "$START_STAGE" -le 1 ]; then

if [ -d "$PREPROCESS_OUTPUT" ] && [ "$(find "$PREPROCESS_OUTPUT" -type f -name "*.jpg" 2>/dev/null | wc -l)" -gt 0 ]; then
    print_success "Data already preprocessed, skipping..."
    DATA_PATH_USED="$PREPROCESS_OUTPUT"
elif [ "$(find "$DATASET_PATH" -type f -name "*.jpg" 2>/dev/null | wc -l)" -gt 0 ]; then
    print_success "Dataset is already in preprocessed 2D format, using directly..."
    
    # Create splits directory and files if they don't exist
    if [ ! -d "$DATASET_PATH/splits" ]; then
        print_info "Creating splits directory..."
        mkdir -p "$DATASET_PATH/splits"
    fi
    
    if [ ! -f "$DATASET_PATH/splits/train.txt" ] || [ ! -f "$DATASET_PATH/splits/val.txt" ]; then
        print_info "Generating train/val splits..."
        # Find all patient directories
        patient_dirs=$(find "$DATASET_PATH" -maxdepth 1 -type d -name "BraTS*" | sort)
        total=$(echo "$patient_dirs" | wc -l)
        n_train=$((total * 80 / 100))
        
        echo "$patient_dirs" | head -n $n_train | xargs -I {} basename {} > "$DATASET_PATH/splits/train.txt"
        echo "$patient_dirs" | tail -n +$((n_train+1)) | xargs -I {} basename {} > "$DATASET_PATH/splits/val.txt"
        touch "$DATASET_PATH/splits/test.txt"
        
        print_success "Generated splits: $n_train train, $((total-n_train)) val"
    fi
    
    DATA_PATH_USED="$DATASET_PATH"
else
    print_info "Preprocessing raw NIfTI data to slices..."
    
    # Use flat preprocessing wrapper if available
    if [ -f "preprocess_flat_brats.py" ]; then
        PREPROCESS_SCRIPT="preprocess_flat_brats.py"
        print_info "Using flat BRATS preprocessor (handles both structured and flat layouts)"
    else
        PREPROCESS_SCRIPT="brats/preprocess.py"
        print_info "Using standard BRATS preprocessor"
    fi
    
    if [ ! -f "$PREPROCESS_SCRIPT" ]; then
        print_error "Preprocessing script not found: $PREPROCESS_SCRIPT"
        exit 1
    fi
    
    uv run python "$PREPROCESS_SCRIPT" \
        --input-path "$DATASET_PATH" \
        --output-path "$PREPROCESS_OUTPUT" \
        --workers 4
    
    DATA_PATH_USED="$PREPROCESS_OUTPUT"
    print_success "Data preprocessing completed"
fi
fi

# ============================================================================
# STAGE 2: Clustering
# ============================================================================
if [ "$START_STAGE" -gt 2 ]; then
    print_info "Skipping STAGE 2 (START_STAGE=$START_STAGE)"
else
    print_header "STAGE 2: Pre-clustering"
fi

if [ "$START_STAGE" -le 2 ]; then

CLUSTER_FILE="$CLUSTER_FILE_PATH/$DATASET_NAME-$CHILD_CLASSES.bin"
if [ -f "$CLUSTER_FILE" ]; then
    print_success "Clustering already done, skipping..."
else
    print_info "Running K-means clustering..."
    uv run python cluster.py \
        --batch_size 256 \
        --data_path "$DATA_PATH_USED" \
        --data_module "$DATA_MODULE" \
        --parent_classes "$PARENT_CLASSES" \
        --child_classes "$CHILD_CLASSES" \
        --save_path "$CLUSTER_FILE_PATH" \
        --cluster_name "$DATASET_NAME" \
        --gpus "$GPUS"
    
    print_success "Pre-clustering completed"
fi
fi

# ============================================================================
# STAGE 3: Download SAM
# ============================================================================
if [ "$START_STAGE" -gt 3 ]; then
    print_info "Skipping STAGE 3 (START_STAGE=$START_STAGE)"
else
    print_header "STAGE 3: Download SAM Checkpoint"
fi

if [ "$START_STAGE" -le 3 ]; then

if [ -f "$SAM_CHECKPOINT" ]; then
    size=$(du -b "$SAM_CHECKPOINT" | cut -f1)
    if [ "$size" -gt 350000000 ]; then
        print_success "SAM checkpoint already exists: $SAM_CHECKPOINT"
    else
        print_info "Downloading SAM ViT-B checkpoint..."
        mkdir -p "$(dirname "$SAM_CHECKPOINT")"
        wget -O "$SAM_CHECKPOINT" \
            https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        print_success "SAM checkpoint downloaded"
    fi
else
    print_info "Downloading SAM ViT-B checkpoint..."
    mkdir -p "$(dirname "$SAM_CHECKPOINT")"
    wget -O "$SAM_CHECKPOINT" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    print_success "SAM checkpoint downloaded"
fi
fi

# ============================================================================
# STAGE 4: ACArL Training
# ============================================================================
if [ "$START_STAGE" -gt 4 ]; then
    print_info "Skipping STAGE 4 (START_STAGE=$START_STAGE)"
else
    print_header "STAGE 4: ACArL Training (SAMUS with ACArL)"
fi

if [ "$START_STAGE" -le 4 ]; then

SAMUS_CKPT="$LOGDIR/$INDEX/${INDEX}.pth"
ACARL_CKPT="$LOGDIR/$INDEX/${INDEX}_acarl.pth"

# Check if we have a pre-trained checkpoint to copy
PRETRAINED_CKPT="./logs_brats_full/20251212_153905/checkpoint.pt"

if [ -f "$SAMUS_CKPT" ] && [ -f "$ACARL_CKPT" ]; then
    print_success "ACArL training already completed, skipping..."
elif [ -f "$PRETRAINED_CKPT" ]; then
    print_success "Using pre-trained checkpoint from: $PRETRAINED_CKPT"
    print_info "Extracting SAMUS and ACArL checkpoints from pre-trained model..."
    
    # Create output directory
    mkdir -p "$LOGDIR/$INDEX"
    
    # Extract individual model checkpoints using Python
    uv run python << EXTRACT_CKPT
import torch
import sys

output_dir = "$LOGDIR/$INDEX"
index = "$INDEX"
full_ckpt_path = "$PRETRAINED_CKPT"

try:
    # Load full checkpoint
    full_ckpt = torch.load(full_ckpt_path, map_location='cpu')
    
    # Extract and save model state
    model_state = full_ckpt['model']
    torch.save(model_state, f"{output_dir}/{index}.pth")
    print(f"✓ Saved SAMUS checkpoint: {output_dir}/{index}.pth")
    
    # Extract and save ACArL module state
    acarl_state = full_ckpt['acarl_module']
    torch.save(acarl_state, f"{output_dir}/{index}_acarl.pth")
    print(f"✓ Saved ACArL checkpoint: {output_dir}/{index}_acarl.pth")
except Exception as e:
    print(f"Error extracting checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EXTRACT_CKPT
    
    print_success "ACArL training checkpoint loaded"
else
    pkill -9 -f "train.py" 2>/dev/null || true
    sleep 2
    
    print_info "Training SAMUS with ACArL module..."
    uv run python train.py \
        --seed "$SEED" \
        --sam_ckpt "$SAM_CHECKPOINT" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --max_epochs "$MAX_EPOCHS" \
        --val_iters "$VAL_ITERS" \
        --index "$INDEX" \
        --data_path "$DATA_PATH_USED" \
        --data_module "$DATA_MODULE" \
        --parent_classes "$PARENT_CLASSES" \
        --child_classes "$CHILD_CLASSES" \
        --child_weight "$CHILD_WEIGHT" \
        --acarl_weight "$ACARL_WEIGHT" \
        --cluster_file "$CLUSTER_FILE" \
        --logdir "$LOGDIR" \
        --gpus "$GPUS"
    
    print_success "ACArL training completed"
fi
fi

# ============================================================================
# STAGE 5: Pseudo-Label Generation (ACArL)
# ============================================================================
print_header "STAGE 5: Pseudo-Label Generation with ACArL"

if [ "$START_STAGE" -le 5 ]; then
    if [ -d "$PSEUDO_LABEL_PATH" ] && [ "$(ls -A "$PSEUDO_LABEL_PATH" 2>/dev/null | wc -l)" -gt 0 ]; then
        print_success "Pseudo-labels already generated, skipping..."
    else
        print_info "Generating pseudo-labels with ACArL refinement..."
        mkdir -p "$PSEUDO_LABEL_PATH"
        mkdir -p "$CONFIDENCE_PATH"
        mkdir -p "$SOFT_LABEL_PATH"
        
        uv run python lab_gen_acarl.py \
            --batch-size 64 \
            --data-path "$DATA_PATH_USED" \
            --save-path "$PSEUDO_LABEL_PATH" \
            --data-module "$DATA_MODULE" \
            --parent-classes "$PARENT_CLASSES" \
            --child-classes "$CHILD_CLASSES" \
            --samus-ckpt "$SAMUS_CKPT" \
            --acarl-ckpt "$ACARL_CKPT" \
            --sam-ckpt "$SAM_CHECKPOINT" \
            --diffusion-steps "$DIFFUSION_STEPS" \
            --diffusion-step-size "$DIFFUSION_STEP_SIZE" \
            --threshold "$THRESHOLD" \
            --save-confidence \
            --conf-path "$CONFIDENCE_PATH" \
            --save-soft-labels \
            --soft-path "$SOFT_LABEL_PATH" \
            --gpus "$GPUS"
        
        print_success "Pseudo-label generation completed"
    fi
fi

# ============================================================================
# STAGE 6: U-Net Training (VANILLA - NO confidence weighting)
# ============================================================================
if [ "$START_STAGE" -gt 6 ]; then
    print_info "Skipping STAGE 6 (START_STAGE=$START_STAGE)"
else
    print_header "STAGE 6: U-Net Training (Vanilla - Baseline Only)"
fi

if [ "$START_STAGE" -le 6 ]; then
    UNET_CKPT="$LOGDIR/$INDEX/${INDEX}_unet.pth"
    
    if [ -f "$UNET_CKPT" ]; then
        print_success "U-Net training already completed, skipping..."
    else
        pkill -9 -f "train_unet" 2>/dev/null || true
        sleep 2
        
        print_info "Training vanilla U-Net with ACArL pseudo-labels (no confidence weighting)..."
        print_info "This tests if pseudo-label quality alone drives performance"
        
        uv run python train_unet.py \
            --seed "$SEED" \
            --lr "$UNET_LR" \
            --batch_size "$UNET_BATCH_SIZE" \
            --max_epochs "$UNET_MAX_EPOCHS" \
            --val_iters "$UNET_VAL_ITERS" \
            --index "${INDEX}_unet" \
            --data_path "$DATA_PATH_USED" \
            --lab_path "$PSEUDO_LABEL_PATH" \
            --data_module "$DATA_MODULE" \
            --num_classes 2 \
            --logdir "$LOGDIR" \
            --gpus "$GPUS"
        
        print_success "U-Net training completed"
    fi
fi

# ============================================================================
# STAGE 7: Evaluation
# ============================================================================
if [ "$START_STAGE" -gt 7 ]; then
    print_info "Skipping STAGE 7 (START_STAGE=$START_STAGE)"
else
    print_header "STAGE 7: Evaluation"
fi

if [ "$START_STAGE" -le 7 ]; then
    EVAL_RESULTS="$LOGDIR/$INDEX/${INDEX}_eval_results.txt"
    
    if [ -f "$EVAL_RESULTS" ]; then
        print_success "Evaluation already completed, skipping..."
        cat "$EVAL_RESULTS"
    else
        print_info "Evaluating model on validation set..."
        
        uv run python eval.py \
            --data_path "$DATA_PATH_USED" \
            --data_module "$DATA_MODULE" \
            --batch_size 128 \
            --num_classes 2 \
            --ckpt "$UNET_CKPT" \
            --gpus "$GPUS" | tee "$EVAL_RESULTS"
        
        print_success "Evaluation completed"
    fi
fi

# ============================================================================
# Summary
# ============================================================================
print_header "Pipeline Summary"

print_info "Configuration: Baseline SAM + ACArL Pseudo-Labels"
print_info "Dataset: $DATASET_NAME"
print_info "Logs: $LOGDIR/$INDEX"
print_info ""
print_info "Key files:"
print_info "  - SAMUS checkpoint: $SAMUS_CKPT"
print_info "  - ACArL checkpoint: $ACARL_CKPT"
print_info "  - Pseudo-labels: $PSEUDO_LABEL_PATH"
print_info "  - U-Net checkpoint: $UNET_CKPT"
print_info "  - Evaluation results: $EVAL_RESULTS"
print_info ""
print_info "To view results:"
print_info "  cat $EVAL_RESULTS"

print_success "Pipeline completed successfully!"
