#!/bin/bash
set -e

# WeakMedSAM ACArL Ablation Study
# This script runs 4 experiments to validate the contributions of ACArL components.
# Includes preceding stages (Setup, Preprocessing, Clustering) as in run_pipeline.sh.

# Parse command-line arguments
EXPERIMENT="${1:-all}"  # Default to 'all' if not specified
DATASET="${2:-mcd}"     # Default to 'mcd' if not specified

# Validate dataset argument first
case "$DATASET" in
    mcd|brats|lits|kits)
        ;;
    *)
        echo "Usage: $0 [exp1|exp2|exp3|exp4|all] [mcd|brats|lits|kits]"
        echo ""
        echo "Available experiments:"
        echo "  exp1 - Baseline (WeakMedSAM)"
        echo "  exp2 - ACArL with Standard Pseudo-labels"
        echo "  exp3 - ACArL with Uncertainty Weighting"
        echo "  exp4 - ACArL Full (All Components)"
        echo "  all  - Run all experiments (default)"
        echo ""
        echo "Available datasets:"
        echo "  mcd   - Myocardial Contusion Dataset (default)"
        echo "  brats - Brain Tumor Segmentation Dataset"
        echo "  lits  - Liver Tumor Segmentation Dataset"
        echo "  kits  - Kidney Tumor Segmentation Dataset"
        exit 1
        ;;
esac

# Validate experiment argument
case "$EXPERIMENT" in
    exp1|exp2|exp3|exp4|all)
        ;;
    *)
        echo "Usage: $0 [exp1|exp2|exp3|exp4|all] [mcd|brats|lits|kits]"
        echo ""
        echo "Available experiments:"
        echo "  exp1 - Baseline (WeakMedSAM)"
        echo "  exp2 - ACArL with Standard Pseudo-labels"
        echo "  exp3 - ACArL with Uncertainty Weighting"
        echo "  exp4 - ACArL Full (All Components)"
        echo "  all  - Run all experiments (default)"
        echo ""
        echo "Available datasets:"
        echo "  mcd   - Myocardial Contusion Dataset (default)"
        echo "  brats - Brain Tumor Segmentation Dataset"
        echo "  lits  - Liver Tumor Segmentation Dataset"
        echo "  kits  - Kidney Tumor Segmentation Dataset"
        exit 1
        ;;
esac

# Common parameters - set based on dataset
case "$DATASET" in
    mcd)
        DATA_PATH="./Task02_Heart"
        DATA_MODULE="mcd"
        LOG_PREFIX="logs_ablation"
        ;;
    brats)
        DATA_PATH="./BRATS_Dataset"
        DATA_MODULE="brats"
        LOG_PREFIX="logs_ablation_brats"
        ;;
    lits)
        DATA_PATH="./lits_kaggle_preprocessed"
        DATA_MODULE="lits"
        LOG_PREFIX="logs_ablation_lits_kaggle"
        ;;
    kits)
        DATA_PATH="./kits_preprocessed"
        DATA_MODULE="kits"
        LOG_PREFIX="logs_ablation_kits"
        ;;
esac

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
    if [ -f "$log_file" ] && tail -1 "$log_file" | grep -q "SUCCESS:"; then
        echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPING: $exp_name (already completed)${NC}"
        return 0
    fi
    
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $exp_name${NC}" | tee -a "$log_file"
    echo "Command: $*" | tee -a "$log_file"
    echo "----------------------------------------" | tee -a "$log_file"
    
    if "$@" 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $exp_name${NC}" | tee -a "$log_file"
        return 0
    else
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $exp_name${NC}" | tee -a "$log_file"
        return 1
    fi
}

print_header "PRE-STAGE: Setup and Preprocessing for $DATASET"

# 1. Environment and Directories
print_info "Checking environment and creating directories..."
mkdir -p "$LOG_PREFIX"
mkdir -p checkpoints
mkdir -p "$DATA_PATH"

# 2. Data Preprocessing (Stage 1 logic from run_pipeline.sh)
print_info "Checking data preprocessing status..."
PREPROCESS_OUTPUT="$LOG_PREFIX/preprocessed_data"

# Check if data is already preprocessed in the output directory
preprocessed_count=$(find "$PREPROCESS_OUTPUT" -mindepth 2 -type f -name "*.jpg" 2>/dev/null | wc -l)
if [ "$preprocessed_count" -gt 0 ]; then
    print_success "Data already preprocessed in $PREPROCESS_OUTPUT"
    DATA_PATH_USED="$PREPROCESS_OUTPUT"
# Check if data is already in native dataset format (e.g., LITS)
elif [ -d "$DATA_PATH" ] && [ "$(find "$DATA_PATH" -mindepth 2 -type f -name "*.jpg" 2>/dev/null | wc -l)" -gt 0 ]; then
    print_success "Data already preprocessed in $DATA_PATH"
    DATA_PATH_USED="$DATA_PATH"
elif [ -d "$DATA_PATH/imagesTr" ] && [ "$(find "$DATA_PATH/imagesTr" -name "*.jpg" | wc -l)" -gt 0 ]; then
    print_success "Data already preprocessed in $DATA_PATH"
    DATA_PATH_USED="$DATA_PATH"
else
    print_info "Preprocessing raw data..."
    
    # Check if raw data exists in imagesTr subdirectory
    RAW_DATA_PATH="$DATA_PATH/imagesTr"
    if [ ! -d "$RAW_DATA_PATH" ]; then
        # Check if raw NIfTI files exist directly in dataset path
        nifti_count=$(find "$DATA_PATH" -maxdepth 1 -type f -name "*.nii.gz" 2>/dev/null | wc -l)
        if [ "$nifti_count" -eq 0 ]; then
            echo -e "${RED}Error: No raw NIfTI files found in $DATA_PATH${NC}"
            echo -e "${YELLOW}Please ensure raw NIfTI files are present in $DATA_PATH or $DATA_PATH/imagesTr${NC}"
            exit 1
        fi
        RAW_DATA_PATH="$DATA_PATH"
    fi
    
    print_info "Input path: $RAW_DATA_PATH"
    print_info "Output path: $PREPROCESS_OUTPUT"
    
    uv run python "$DATA_MODULE/preprocess.py" \
        --input-path "$RAW_DATA_PATH" \
        --output-path "$PREPROCESS_OUTPUT" \
        --workers 4
    
    DATA_PATH_USED="$PREPROCESS_OUTPUT"
fi
print_info "Using data path: $DATA_PATH_USED"

# 3. Pre-clustering (Stage 2 logic)
print_info "Checking pre-clustering..."
CLUSTER_FILE="$LOG_PREFIX/clusters_${DATASET}.bin"
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
        --cluster_name "$DATASET" \
        --gpus "$GPUS"
    # Move/Rename if needed, cluster.py usually saves as {name}-{child_classes}.bin
    mv "$LOG_PREFIX/${DATASET}-${CHILD_CLASSES}.bin" "$CLUSTER_FILE" 2>/dev/null || true
fi

# 4. Download SAM (Stage 3 logic)
print_info "Checking SAM checkpoint..."
if [ ! -f "$SAM_CKPT" ]; then
    print_info "Downloading SAM checkpoint..."
    wget -O "$SAM_CKPT" https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
fi
print_success "SAM checkpoint ready."

print_header "ACArL Ablation Study on $DATASET"
print_info "Running experiments: $EXPERIMENT"

# Define a function to run experiments conditionally
should_run_exp() {
    local exp=$1
    if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "$exp" ]; then
        return 0
    fi
    return 1
}

# Experiment 1: Baseline (Original WeakMedSAM)
if should_run_exp "exp1"; then
    print_header "Experiment 1: Baseline (WeakMedSAM)"

run_with_log "Exp1_Train" "$LOG_PREFIX/exp1_baseline/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight 0.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --index "${DATASET}_exp1_baseline" \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp1_PseudoLabels" "$LOG_PREFIX/exp1_baseline/pseudo_labels.log" uv run python lab_gen.py \
    --data-path "$DATA_PATH_USED" \
    --save-path "$LOG_PREFIX/${DATASET}_exp1_baseline/pseudo_labels" \
    --data-module "$DATA_MODULE" \
    --vit-name "$VIT_NAME" \
    --sam-ckpt "$SAM_CKPT" \
    --batch-size "$BATCH_SIZE" \
    --samus-ckpt "$LOG_PREFIX/${DATASET}_exp1_baseline/${DATASET}_exp1_baseline.pth" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --t 4 \
    --beta 4 \
    --threshold 0.5 \
    --gpus "$GPUS"

run_with_log "Exp1_UnetTrain" "$LOG_PREFIX/exp1_baseline/unet_train.log" uv run python train_unet.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/${DATASET}_exp1_baseline/pseudo_labels" \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "${DATASET}_exp1_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp1_Eval" "$LOG_PREFIX/exp1_baseline/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/${DATASET}_exp1_unet/${DATASET}_exp1_unet.pth" \
    --gpus "$GPUS"

print_success "Experiment 1 completed"
fi


# Experiment 2: ACArL-NoAux (Refinement + Uncertainty, No Sub-class Reg)
if should_run_exp "exp2"; then
    print_header "Experiment 2: ACArL-NoAux (No Sub-class Regularization)"

run_with_log "Exp2_Train" "$LOG_PREFIX/exp2_noaux/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 0.0 \
    --lambda_a 0.2 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --index "${DATASET}_exp2_noaux" \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp2_PseudoLabels" "$LOG_PREFIX/exp2_noaux/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/${DATASET}_exp2_noaux/${DATASET}_exp2_noaux.pth" \
    --acarl-ckpt "$LOG_PREFIX/${DATASET}_exp2_noaux/${DATASET}_exp2_noaux_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/${DATASET}_exp2_noaux/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/${DATASET}_exp2_noaux/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5

run_with_log "Exp2_UnetTrain" "$LOG_PREFIX/exp2_noaux/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/${DATASET}_exp2_noaux/pseudo_labels" \
    --conf_path "$LOG_PREFIX/${DATASET}_exp2_noaux/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "${DATASET}_exp2_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp2_Eval" "$LOG_PREFIX/exp2_noaux/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/${DATASET}_exp2_unet/${DATASET}_exp2_unet.pth" \
    --gpus "$GPUS"

print_success "Experiment 2 completed"
fi


# Experiment 3: ACArL-NoUnc (Full Generation, Binary Training)
if should_run_exp "exp3"; then
    print_header "Experiment 3: ACArL-NoUnc (Full Gen, Binary Training)"

run_with_log "Exp3_Train" "$LOG_PREFIX/exp3_nounc/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.2 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --index "${DATASET}_exp3_nounc" \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp3_PseudoLabels" "$LOG_PREFIX/exp3_nounc/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/${DATASET}_exp3_nounc/${DATASET}_exp3_nounc.pth" \
    --acarl-ckpt "$LOG_PREFIX/${DATASET}_exp3_nounc/${DATASET}_exp3_nounc_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/${DATASET}_exp3_nounc/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/${DATASET}_exp3_nounc/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5

run_with_log "Exp3_UnetTrain" "$LOG_PREFIX/exp3_nounc/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/${DATASET}_exp3_nounc/pseudo_labels" \
    --conf_path "$LOG_PREFIX/${DATASET}_exp3_nounc/confidence_maps" \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "${DATASET}_exp3_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type ce \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp3_Eval" "$LOG_PREFIX/exp3_nounc/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/${DATASET}_exp3_unet/${DATASET}_exp3_unet.pth" \
    --gpus "$GPUS"

print_success "Experiment 3 completed"
fi


# Experiment 4: ACArL-Full (All Components)
if should_run_exp "exp4"; then
    print_header "Experiment 4: ACArL-Full (Proposed Method)"

run_with_log "Exp4_Train" "$LOG_PREFIX/exp4_full/train.log" uv run python train.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --vit_name "$VIT_NAME" \
    --sam_ckpt "$SAM_CKPT" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --max_epochs "$MAX_EPOCHS" \
    --parent_classes "$PARENT_CLASSES" \
    --child_classes "$CHILD_CLASSES" \
    --child_weight "$CHILD_WEIGHT" \
    --acarl_weight "$ACARL_WEIGHT" \
    --lambda_s 1.0 \
    --lambda_a 0.2 \
    --lambda_conf 1.0 \
    --cluster_file "$CLUSTER_FILE" \
    --logdir "$LOG_PREFIX" \
    --index "${DATASET}_exp4_full" \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp4_PseudoLabels" "$LOG_PREFIX/exp4_full/pseudo_labels.log" uv run python lab_gen_acarl.py \
    --data-path "$DATA_PATH_USED" \
    --data-module "$DATA_MODULE" \
    --samus-ckpt "$LOG_PREFIX/${DATASET}_exp4_full/${DATASET}_exp4_full.pth" \
    --acarl-ckpt "$LOG_PREFIX/${DATASET}_exp4_full/${DATASET}_exp4_full_acarl.pth" \
    --sam-ckpt "$SAM_CKPT" \
    --parent-classes "$PARENT_CLASSES" \
    --child-classes "$CHILD_CLASSES" \
    --save-path "$LOG_PREFIX/${DATASET}_exp4_full/pseudo_labels" \
    --save-confidence \
    --conf-path "$LOG_PREFIX/${DATASET}_exp4_full/confidence_maps" \
    --diffusion-steps 3 \
    --threshold 0.5

run_with_log "Exp4_UnetTrain" "$LOG_PREFIX/exp4_full/unet_train.log" uv run python train_unet_acarl.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --lab_path "$LOG_PREFIX/${DATASET}_exp4_full/pseudo_labels" \
    --conf_path "$LOG_PREFIX/${DATASET}_exp4_full/confidence_maps" \
    --use_confidence \
    --logdir "$LOG_PREFIX" \
    --batch_size 128 \
    --lr 1e-4 \
    --index "${DATASET}_exp4_unet" \
    --max_epochs 10 \
    --val_iters 500 \
    --loss_type combined \
    --num_classes $((PARENT_CLASSES + 1)) \
    --seed "$SEED" \
    --gpus "$GPUS"

run_with_log "Exp4_Eval" "$LOG_PREFIX/exp4_full/eval.log" uv run python eval.py \
    --data_path "$DATA_PATH_USED" \
    --data_module "$DATA_MODULE" \
    --batch_size "$BATCH_SIZE" \
    --num_classes $((PARENT_CLASSES + 1)) \
    --ckpt "$LOG_PREFIX/${DATASET}_exp4_unet/${DATASET}_exp4_unet.pth" \
    --gpus "$GPUS"

print_success "Experiment 4 completed"
fi

print_header "Ablation Study Completed"
print_success "Results saved in: $LOG_PREFIX/"
echo ""
echo "========================================================"
echo "To view results, check:"
echo "  - Experiment 1 (Baseline):      $LOG_PREFIX/exp1_baseline/eval.log"
echo "  - Experiment 2 (ACArL-NoAux):   $LOG_PREFIX/exp2_noaux/eval.log"
echo "  - Experiment 3 (ACArL-NoUnc):   $LOG_PREFIX/exp3_nounc/eval.log"
echo "  - Experiment 4 (ACArL-Full):    $LOG_PREFIX/exp4_full/eval.log"
echo "========================================================"
