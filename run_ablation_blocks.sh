#!/bin/bash
set -e

# ACArL Block Ablation Study
# This script runs 4 experiments to evaluate the impact of individual ACArL blocks:
# Block A: Class-Conditional Affinity Learning
# Block B: Anisotropic Diffusion
#
# Experiments:
# exp1 - No Block A (disable ClassConditionalAffinity, keep AnisotropicDiffusion)
# exp2 - No Block B (disable AnisotropicDiffusion, keep ClassConditionalAffinity)
# exp3 - No Both Blocks (baseline: only sub-class exploration)
# exp4 - Full ACArL (both Block A and Block B enabled)

# Parse command-line arguments
EXPERIMENT="${1:-all}"  # Default to 'all' if not specified

# Validate experiment argument
case "$EXPERIMENT" in
    exp1|exp2|exp3|exp4|all)
        ;;
    *)
        echo "Usage: $0 [exp1|exp2|exp3|exp4|all]"
        echo ""
        echo "Available experiments:"
        echo "  exp1 - No Block A (disable ClassConditionalAffinity)"
        echo "  exp2 - No Block B (disable AnisotropicDiffusion)"
        echo "  exp3 - No Both Blocks (only sub-class exploration)"
        echo "  exp4 - Full ACArL (both blocks enabled)"
        echo "  all  - Run all experiments (default)"
        exit 1
        ;;
esac

# Common parameters
DATA_PATH="/home/belal/brats_dataset/10_percent"
DATA_MODULE="brats"
LOG_PREFIX="logs_ablation_blocks"
VIT_NAME="vit_b"
SAM_CKPT="checkpoints/sam_vit_b_01ec64.pth"
BATCH_SIZE=8
LR=1e-4
MAX_EPOCHS=10
PARENT_CLASSES=1
CHILD_CLASSES=8
CHILD_WEIGHT=0.5
ACARL_WEIGHT=0.3
GPUS="0"
SEED=42

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo ""
}

print_info() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to run command with logging
run_with_log() {
    local exp_name="$1"
    local log_file="$2"
    shift 2
    
    mkdir -p "$(dirname "$log_file")"
    
    # Check if log file exists and last line contains SUCCESS
    if [ -f "$log_file" ] && tail -1 "$log_file" 2>/dev/null | grep -q "SUCCESS:"; then
        echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPING: $exp_name (already completed)${NC}"
        echo -e "${YELLOW}  Log: $log_file${NC}"
        return 0
    fi
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp_name${NC}" | tee -a "$log_file"
    echo "Command: $*" | tee -a "$log_file"
    echo "Log file: $log_file" | tee -a "$log_file"
    echo "----------------------------------------" | tee -a "$log_file"
    
    if "$@" 2>&1 | tee -a "$log_file"; then
        echo "----------------------------------------" | tee -a "$log_file"
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $exp_name${NC}" | tee -a "$log_file"
        return 0
    else
        echo "----------------------------------------" | tee -a "$log_file"
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $exp_name${NC}" | tee -a "$log_file"
        return 1
    fi
}

print_header "PRE-STAGE: Setup and Preprocessing for BRATS"

# 1. Environment and Directories
print_info "Checking environment and creating directories..."
mkdir -p "$LOG_PREFIX"
mkdir -p checkpoints
mkdir -p "$DATA_PATH"

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    print_error "Data path $DATA_PATH does not exist"
    echo "Please ensure the BRATS dataset is available at: $DATA_PATH"
    exit 1
fi

# Count preprocessed data
data_count=$(find "$DATA_PATH" -mindepth 1 -type f -name "*.jpg" 2>/dev/null | wc -l)
if [ "$data_count" -eq 0 ]; then
    print_error "No preprocessed data found in $DATA_PATH"
    echo "Data should be in JPEG format with shape labels"
    exit 1
fi
print_success "Found $data_count preprocessed data files in $DATA_PATH"

DATA_PATH_USED="$DATA_PATH"

# 2. Pre-clustering
print_info "Checking pre-clustering..."
CLUSTER_FILE="$LOG_PREFIX/clusters_brats_blocks.bin"
if [ -f "$CLUSTER_FILE" ]; then
    print_success "Clusters already exist at $CLUSTER_FILE"
else
    print_info "Running K-means clustering..."
    uv run python cluster.py \
        --batch_size 256 \
        --data_path "$DATA_PATH_USED" \
        --data_module "$DATA_MODULE" \
        --parent_classes "$PARENT_CLASSES" \
        --child_classes "$CHILD_CLASSES" \
        --save_path "$LOG_PREFIX" \
        --cluster_name "brats_blocks" \
        --gpus "$GPUS"
    # Handle cluster file naming
    if [ ! -f "$CLUSTER_FILE" ]; then
        mv "$LOG_PREFIX/brats_blocks-${CHILD_CLASSES}.bin" "$CLUSTER_FILE" 2>/dev/null || true
    fi
fi

# 3. Download SAM
print_info "Checking SAM checkpoint..."
if [ ! -f "$SAM_CKPT" ]; then
    print_info "Downloading SAM checkpoint..."
    mkdir -p checkpoints
    wget -O "$SAM_CKPT" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi
print_success "SAM checkpoint ready."

print_header "Block Ablation Study on BRATS (10% subset)"
print_info "Running experiments: $EXPERIMENT"

# Define a function to run experiments conditionally
should_run_exp() {
    local exp=$1
    if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "$exp" ]; then
        return 0
    fi
    return 1
}

# Experiment 1: No Block A (disable ClassConditionalAffinity)
if should_run_exp "exp1"; then
    print_header "Experiment 1: No Block A (Disable ClassConditionalAffinity)"
    print_info "Running training without Class-Conditional Affinity Learning..."
    
run_with_log "Exp1_Train" "$LOG_PREFIX/exp1_no_block_a/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --index "brats_blocks_exp1_no_a" \
    --max_epochs "$MAX_EPOCHS" \
    --val_iters 500 \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.0 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --seed "$SEED" \
    --gpus "$GPUS" \
    --disable_affinity_learning

run_with_log "Exp1_PseudoLabels" "$LOG_PREFIX/exp1_no_block_a/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/brats_blocks_exp1_no_a/brats_blocks_exp1_no_a.pth" \
    --acarl-ckpt "$LOG_PREFIX/brats_blocks_exp1_no_a/brats_blocks_exp1_no_a_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/exp1_no_block_a/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/exp1_no_block_a/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5 \
    --disable-affinity-learning

run_with_log "Exp1_UnetTrain" "$LOG_PREFIX/exp1_no_block_a/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/exp1_no_block_a/pseudo_labels" \
    --conf_path "$LOG_PREFIX/exp1_no_block_a/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "brats_blocks_exp1_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp1_Eval" "$LOG_PREFIX/exp1_no_block_a/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/brats_blocks_exp1_unet/brats_blocks_exp1_unet.pth" \
    --gpus "$GPUS"

    if [ $? -eq 0 ]; then
        print_success "Experiment 1 completed successfully"
    else
        print_error "Experiment 1 failed"
    fi
fi

# Experiment 2: No Block B (disable AnisotropicDiffusion)
if should_run_exp "exp2"; then
    print_header "Experiment 2: No Block B (Disable AnisotropicDiffusion)"
    print_info "Running training without Anisotropic Diffusion..."
    
run_with_log "Exp2_Train" "$LOG_PREFIX/exp2_no_block_b/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --index "brats_blocks_exp2_no_b" \
    --max_epochs "$MAX_EPOCHS" \
    --val_iters 500 \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.2 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --seed "$SEED" \
    --gpus "$GPUS" \
    --disable_diffusion

run_with_log "Exp2_PseudoLabels" "$LOG_PREFIX/exp2_no_block_b/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/brats_blocks_exp2_no_b/brats_blocks_exp2_no_b.pth" \
    --acarl-ckpt "$LOG_PREFIX/brats_blocks_exp2_no_b/brats_blocks_exp2_no_b_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/exp2_no_block_b/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/exp2_no_block_b/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5 \
    --disable-diffusion

run_with_log "Exp2_UnetTrain" "$LOG_PREFIX/exp2_no_block_b/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/exp2_no_block_b/pseudo_labels" \
    --conf_path "$LOG_PREFIX/exp2_no_block_b/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "brats_blocks_exp2_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp2_Eval" "$LOG_PREFIX/exp2_no_block_b/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/brats_blocks_exp2_unet/brats_blocks_exp2_unet.pth" \
    --gpus "$GPUS"

    if [ $? -eq 0 ]; then
        print_success "Experiment 2 completed successfully"
    else
        print_error "Experiment 2 failed"
    fi
fi

# Experiment 3: No Both Blocks (only sub-class exploration)
if should_run_exp "exp3"; then
    print_header "Experiment 3: No Both Blocks (Only Sub-class Exploration)"
    print_info "Running training with only sub-class exploration (no affinity or diffusion)..."
    
run_with_log "Exp3_Train" "$LOG_PREFIX/exp3_no_blocks/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --index "brats_blocks_exp3_no_blocks" \
    --max_epochs "$MAX_EPOCHS" \
    --val_iters 500 \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.0 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --seed "$SEED" \
    --gpus "$GPUS" \
    --disable_affinity_learning \
    --disable_diffusion

run_with_log "Exp3_PseudoLabels" "$LOG_PREFIX/exp3_no_blocks/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/brats_blocks_exp3_no_blocks/brats_blocks_exp3_no_blocks.pth" \
    --acarl-ckpt "$LOG_PREFIX/brats_blocks_exp3_no_blocks/brats_blocks_exp3_no_blocks_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/exp3_no_blocks/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/exp3_no_blocks/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5 \
    --disable-affinity-learning \
    --disable-diffusion

run_with_log "Exp3_UnetTrain" "$LOG_PREFIX/exp3_no_blocks/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/exp3_no_blocks/pseudo_labels" \
    --conf_path "$LOG_PREFIX/exp3_no_blocks/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "brats_blocks_exp3_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp3_Eval" "$LOG_PREFIX/exp3_no_blocks/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/brats_blocks_exp3_unet/brats_blocks_exp3_unet.pth" \
    --gpus "$GPUS"

    if [ $? -eq 0 ]; then
        print_success "Experiment 3 completed successfully"
    else
        print_error "Experiment 3 failed"
    fi
fi

# Experiment 4: Full ACArL (both blocks enabled)
if should_run_exp "exp4"; then
    print_header "Experiment 4: Full ACArL (Both Blocks Enabled)"
    print_info "Running training with full ACArL (affinity learning + anisotropic diffusion)..."
    
run_with_log "Exp4_Train" "$LOG_PREFIX/exp4_full_blocks/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --index "brats_blocks_exp4_full" \
    --max_epochs "$MAX_EPOCHS" \
    --val_iters 500 \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.2 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp4_PseudoLabels" "$LOG_PREFIX/exp4_full_blocks/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/brats_blocks_exp4_full/brats_blocks_exp4_full.pth" \
    --acarl-ckpt "$LOG_PREFIX/brats_blocks_exp4_full/brats_blocks_exp4_full_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/exp4_full_blocks/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/exp4_full_blocks/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5

run_with_log "Exp4_UnetTrain" "$LOG_PREFIX/exp4_full_blocks/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/exp4_full_blocks/pseudo_labels" \
    --conf_path "$LOG_PREFIX/exp4_full_blocks/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "brats_blocks_exp4_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp4_Eval" "$LOG_PREFIX/exp4_full_blocks/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/brats_blocks_exp4_unet/brats_blocks_exp4_unet.pth" \
    --gpus "$GPUS"

    if [ $? -eq 0 ]; then
        print_success "Experiment 4 completed successfully"
    else
        print_error "Experiment 4 failed"
    fi
fi

print_header "Block Ablation Study Summary"
echo -e "${GREEN}All experiments completed!${NC}"
echo ""
echo "Results location: $LOG_PREFIX/"
echo ""
echo "Experiment results:"
echo "  exp1 - No Block A: $LOG_PREFIX/exp1_no_block_a/"
echo "  exp2 - No Block B: $LOG_PREFIX/exp2_no_block_b/"
echo "  exp3 - No Both:    $LOG_PREFIX/exp3_no_blocks/"
echo "  exp4 - Full ACArL: $LOG_PREFIX/exp4_full_blocks/"
echo ""
