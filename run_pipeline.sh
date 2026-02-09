#!/bin/bash

################################################################################
# WeakMedSAM + ACArL End-to-End Pipeline Script
# 
# This script executes the complete pipeline:
# 0. Environment Setup
# 1. Data Preprocessing (if needed)
# 2. Pre-clustering
# 3. ACArL Training
# 4. Pseudo-label Generation with ACArL
# 5. U-Net Training with Confidence Weighting
# 6. Evaluation
#
# Usage:
#   bash run_pipeline.sh --data-path /path/to/dataset [--stage STAGE] [--gpus GPUS]
#
# Environment Setup:
#   uv venv .venv
#   source .venv/bin/activate
#   uv pip install -r requirements.txt
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
STAGE=${STAGE:-"all"}
GPUS=${GPUS:-"0"}
SEED=42
RESUME_CHECKPOINT=""  # Optional checkpoint to resume from
DATASET_PATH=""       # Dataset directory path

# Configuration - these will be set based on DATASET_PATH
DATA_OUTPUT_PATH=""
SAM_CHECKPOINT="./checkpoints/sam_vit_b_01ec64.pth"
PSEUDO_LABEL_PATH=""
CONFIDENCE_PATH=""
SOFT_LABEL_PATH=""
CLUSTER_FILE_PATH=""
LOGDIR=""

# Hyperparameters
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
CONF_THRESHOLD=0.3

# Index for results (will be set from --index parameter or auto-generated)
INDEX="default"

################################################################################
# Helper Functions
################################################################################

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

check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        exit 1
    fi
    print_success "Found: $1"
}

check_dir() {
    if [ ! -d "$1" ]; then
        print_info "Creating directory: $1"
        mkdir -p "$1"
    fi
    print_success "Directory ready: $1"
}

init_paths() {
    # Initialize all paths based on dataset directory
    if [ -z "$DATASET_PATH" ]; then
        print_error "Dataset path not specified!"
        print_info "Usage: bash run_pipeline.sh --data-path /path/to/dataset [--stage STAGE] [--gpus GPUS]"
        exit 1
    fi
    
    # Auto-generate INDEX if not provided
    if [ -z "$INDEX" ]; then
        INDEX=$(date +%Y%m%d_%H%M%S)
    fi
    
    # Resolve absolute path
    DATASET_PATH=$(cd "$DATASET_PATH" 2>/dev/null && pwd) || {
        print_error "Dataset path does not exist: $DATASET_PATH"
        exit 1
    }
    
    # Get dataset name from path
    DATASET_NAME=$(basename "$DATASET_PATH")
    # Determine data module (brats, mcd, or lits)
    if [[ "$DATASET_NAME" == *"mcd"* ]] || [[ "$DATASET_NAME" == *"Task02_Heart"* ]]; then
        DATA_MODULE="mcd"
    elif [[ "$DATASET_NAME" == *"lits"* ]]; then
        DATA_MODULE="lits"
    else
        DATA_MODULE="brats"
    fi
    
    # Set all output paths based on dataset
    LOGDIR="./logs_${DATASET_NAME}"
    DATA_OUTPUT_PATH="$DATASET_PATH"
    # Runtime data path (either original data or preprocessed output)
    DATA_PATH_USED="$DATA_OUTPUT_PATH"
    
    # Create necessary directories
    mkdir -p "$LOGDIR"
    
    print_info "Paths initialized for dataset: $DATASET_NAME"
    print_info "  - Data: $DATA_OUTPUT_PATH"
    print_info "  - Logs: $LOGDIR"
}

init_index_paths() {
    # Initialize INDEX-specific output paths (all inside $LOGDIR/$INDEX)
    CLUSTER_FILE_PATH="$LOGDIR/$INDEX/clusters"
    PSEUDO_LABEL_PATH="$LOGDIR/$INDEX/pseudo_labels"
    CONFIDENCE_PATH="$LOGDIR/$INDEX/confidence"
    SOFT_LABEL_PATH="$LOGDIR/$INDEX/soft_labels"
    
    # Create necessary directories
    mkdir -p "$CLUSTER_FILE_PATH" "$PSEUDO_LABEL_PATH" "$CONFIDENCE_PATH" "$SOFT_LABEL_PATH"
    
    print_info "INDEX-specific paths initialized:"
    print_info "  - Clusters: $CLUSTER_FILE_PATH"
    print_info "  - Pseudo-labels: $PSEUDO_LABEL_PATH"
    print_info "  - Confidence: $CONFIDENCE_PATH"
    print_info "  - Soft labels: $SOFT_LABEL_PATH"

    # If preprocessing was already run for this index, use that preprocessed data by default
    if [ -d "$LOGDIR/$INDEX/preprocessed_data" ]; then
        DATA_PATH_USED="$LOGDIR/$INDEX/preprocessed_data"
        print_info "Using preprocessed data at: $DATA_PATH_USED"
    fi
}

################################################################################
# Stage Functions
################################################################################

stage_setup() {
    print_header "STAGE 0: Environment Setup"
    
    print_info "Checking Python environment..."
    uv run python --version
    
    print_info "Creating output directories..."
    check_dir "$DATA_OUTPUT_PATH"
    check_dir "$LOGDIR/$INDEX"
    check_dir "$CLUSTER_FILE_PATH"
    check_dir "$PSEUDO_LABEL_PATH"
    check_dir "$CONFIDENCE_PATH"
    check_dir "$SOFT_LABEL_PATH"
    
    print_success "Environment setup completed"
}

stage_preprocessing() {
    print_header "STAGE 1: Data Preprocessing"
    
    # Check if data is already preprocessed in DATA_OUTPUT_PATH
    # Determine which preprocessing script to use based on dataset name (needed even if skipping)
    PREPROCESS_SCRIPT="mcd/preprocess.py"
    if [[ "$DATASET_NAME" == *"brats"* ]]; then
        PREPROCESS_SCRIPT="brats/preprocess.py"
    fi

    # Where preprocessed output will be written for this run
    PREPROCESS_OUTPUT="$LOGDIR/$INDEX/preprocessed_data"

    # Check if data is already preprocessed in the index-specific output directory
    preprocessed_count=$(find "$PREPROCESS_OUTPUT" -mindepth 2 -type f -name "*.jpg" 2>/dev/null | wc -l)
    if [ "$preprocessed_count" -gt 0 ]; then
        print_success "Data already preprocessed for this run, skipping..."
        # Ensure splits exist by invoking preprocess script in generate-splits mode
        uv run python "$PREPROCESS_SCRIPT" --generate-splits --output-path "$PREPROCESS_OUTPUT"
        # use preprocessed data for this run
        DATA_PATH_USED="$PREPROCESS_OUTPUT"
        return 0
    fi

    # Check if data is already preprocessed in dataset directory
    preprocessed_count=$(find "$DATA_OUTPUT_PATH" -mindepth 2 -type f -name "*.jpg" 2>/dev/null | wc -l)
    if [ "$preprocessed_count" -gt 0 ]; then
        print_success "Data already preprocessed in dataset directory, skipping..."
        # Ensure splits exist by invoking preprocess script in generate-splits mode
        uv run python "$PREPROCESS_SCRIPT" --generate-splits --output-path "$DATA_OUTPUT_PATH"
        # use existing dataset as data path
        DATA_PATH_USED="$DATA_OUTPUT_PATH"
        return 0
    fi
    
    # Check if raw data exists in imagesTr subdirectory
    RAW_DATA_PATH="$DATA_OUTPUT_PATH/imagesTr"
    if [ ! -d "$RAW_DATA_PATH" ]; then
        # Check if raw NIfTI files exist directly in dataset path
        nifti_count=$(find "$DATA_OUTPUT_PATH" -maxdepth 1 -type f -name "*.nii.gz" 2>/dev/null | wc -l)
        if [ "$nifti_count" -eq 0 ]; then
            print_error "No raw data or preprocessed data found in $DATA_OUTPUT_PATH"
            print_info "Please ensure raw NIfTI files are present or data is preprocessed."
            exit 1
        fi
        RAW_DATA_PATH="$DATA_OUTPUT_PATH"
    fi
    
    # Preprocess raw NIfTI data
    print_info "Preprocessing raw NIfTI volumes to 2D slices..."
    
    # Determine which preprocessing script to use based on dataset name
    PREPROCESS_SCRIPT="mcd/preprocess.py"
    if [[ "$DATASET_NAME" == *"brats"* ]]; then
        PREPROCESS_SCRIPT="brats/preprocess.py"
    fi
    
    if [ ! -f "$PREPROCESS_SCRIPT" ]; then
        print_error "Preprocessing script not found: $PREPROCESS_SCRIPT"
        exit 1
    fi
    
    print_info "Input: $RAW_DATA_PATH"
    print_info "Output: $PREPROCESS_OUTPUT"
    
    uv run python "$PREPROCESS_SCRIPT" \
        --input-path "$RAW_DATA_PATH" \
        --output-path "$PREPROCESS_OUTPUT" \
        --workers 4
    
    # Keep original raw data intact. Use preprocessed output for downstream stages.
    DATA_PATH_USED="$PREPROCESS_OUTPUT"

    print_success "Data preprocessing completed"
    print_info "Preprocessed data available at: $PREPROCESS_OUTPUT"
    
    # splits are generated by the preprocessing script (no-op here)
}
    

stage_clustering() {
    print_header "STAGE 2: Pre-clustering"
    
    # Skip if already done
    if [ -f "$CLUSTER_FILE_PATH/$DATASET_NAME-$CHILD_CLASSES.bin" ]; then
        print_success "Clustering already done, skipping..."
        return 0
    fi
    
    print_info "Running K-means clustering..."
    print_info "Output: $CLUSTER_FILE_PATH/$DATASET_NAME-$CHILD_CLASSES.bin"
    
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
}

stage_download_sam() {
    print_header "STAGE 3: Download SAM Checkpoint"
    
    if [ -f "$SAM_CHECKPOINT" ]; then
        # Verify checkpoint size (should be ~358MB)
        size=$(du -b "$SAM_CHECKPOINT" | cut -f1)
        if [ "$size" -gt 350000000 ]; then
            print_success "SAM checkpoint already exists and valid: $SAM_CHECKPOINT"
            return 0
        fi
    fi
    
    print_info "Downloading SAM ViT-B checkpoint..."
    mkdir -p "$(dirname "$SAM_CHECKPOINT")"
    
    wget -O "$SAM_CHECKPOINT" \
        https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    
    print_success "SAM checkpoint downloaded"
}

stage_acarl_training() {
    print_header "STAGE 4: ACArL Training (WeakMedSAM Enhancement)"
    
    # Skip if ACArL checkpoint already exists for this INDEX
    if [ -f "$LOGDIR/$INDEX/${INDEX}_acarl.pth" ] && [ -f "$LOGDIR/$INDEX/${INDEX}.pth" ]; then
        print_success "ACArL training already completed, skipping..."
        TRAINING_INDEX="$INDEX"
        return 0
    fi
    
    # Kill any existing training processes to free GPU memory
    print_info "Cleaning up any existing training processes..."
    pkill -9 -f "train.py" 2>/dev/null || true
    sleep 2
    
    check_file "$SAM_CHECKPOINT"
    check_file "$CLUSTER_FILE_PATH/$DATASET_NAME-$CHILD_CLASSES.bin"
    
    # Determine which index to use (resume from checkpoint or create new)
    if [ -n "$RESUME_CHECKPOINT" ]; then
        # Extract the index from checkpoint path (e.g., logs/20251212_082453 -> 20251212_082453)
        TRAINING_INDEX=$(basename "$RESUME_CHECKPOINT")
        CHECKPOINT_PATH="$RESUME_CHECKPOINT/checkpoint.pt"
        
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            print_error "Checkpoint file not found: $CHECKPOINT_PATH"
            exit 1
        fi
        
        print_info "Resuming training from checkpoint: $CHECKPOINT_PATH"
        print_info "Checkpoint size: $(du -h $CHECKPOINT_PATH | cut -f1)"
    else
        TRAINING_INDEX="$INDEX"
        print_info "Starting fresh training with new index: $TRAINING_INDEX"
    fi
    
    print_info "Training SAMUS with ACArL module..."
    print_info "Configuration:"
    print_info "  - Parent Classes: $PARENT_CLASSES"
    print_info "  - Child Classes: $CHILD_CLASSES"
    print_info "  - ACArL Weight: $ACARL_WEIGHT"
    print_info "  - Learning Rate: $LR"
    print_info "  - Batch Size: $BATCH_SIZE"
    print_info "  - Max Epochs: $MAX_EPOCHS"
    print_info "  - Training Index: $TRAINING_INDEX"
    
    uv run python train.py \
        --seed "$SEED" \
        --sam_ckpt "$SAM_CHECKPOINT" \
        --lr "$LR" \
        --batch_size "$BATCH_SIZE" \
        --max_epochs "$MAX_EPOCHS" \
        --val_iters "$VAL_ITERS" \
        --index "$TRAINING_INDEX" \
        --data_path "$DATA_PATH_USED" \
        --data_module "$DATA_MODULE" \
        --parent_classes "$PARENT_CLASSES" \
        --child_classes "$CHILD_CLASSES" \
        --child_weight "$CHILD_WEIGHT" \
        --acarl_weight "$ACARL_WEIGHT" \
        --cluster_file "$CLUSTER_FILE_PATH/$DATASET_NAME-$CHILD_CLASSES.bin" \
        --logdir "$LOGDIR" \
        --gpus "$GPUS"
    
    print_success "ACArL training completed"
    print_info "Checkpoints saved:"
    print_info "  - SAMUS: $LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}.pth"
    print_info "  - ACArL: $LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}_acarl.pth"
}

stage_pseudo_labels() {
    print_header "STAGE 5: Pseudo-Label Generation with ACArL"
    
    # Skip if pseudo-labels already generated
    if [ -d "$PSEUDO_LABEL_PATH" ] && [ "$(ls -A $PSEUDO_LABEL_PATH 2>/dev/null | wc -l)" -gt 0 ]; then
        print_success "Pseudo-labels already generated, skipping..."
        return 0
    fi
    
    # Determine index to use (from resume or current)
    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAINING_INDEX=$(basename "$RESUME_CHECKPOINT")
    else
        TRAINING_INDEX="$INDEX"
    fi
    
    check_file "$SAM_CHECKPOINT"
    check_file "$LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}.pth"
    check_file "$LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}_acarl.pth"
    
    print_info "Generating pseudo-labels with ACArL refinement..."
    print_info "Configuration:"
    print_info "  - Training Index: $TRAINING_INDEX"
    print_info "  - Diffusion Steps: $DIFFUSION_STEPS"
    print_info "  - Diffusion Step Size: $DIFFUSION_STEP_SIZE"
    print_info "  - Threshold: $THRESHOLD"
    print_info "  - Save Confidence: true"
    
    uv run python lab_gen_acarl.py \
        --batch-size "$BATCH_SIZE" \
        --data-path "$DATA_PATH_USED" \
        --save-path "$PSEUDO_LABEL_PATH" \
        --data-module "$DATA_MODULE" \
        --parent-classes "$PARENT_CLASSES" \
        --child-classes "$CHILD_CLASSES" \
        --samus-ckpt "$LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}.pth" \
        --acarl-ckpt "$LOGDIR/$TRAINING_INDEX/${TRAINING_INDEX}_acarl.pth" \
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
    print_info "Outputs saved to:"
    print_info "  - Pseudo-labels: $PSEUDO_LABEL_PATH"
    print_info "  - Confidence maps: $CONFIDENCE_PATH"
    print_info "  - Soft labels: $SOFT_LABEL_PATH"
}

stage_unet_training() {
    print_header "STAGE 6: U-Net Training with Confidence Weighting"
    
    # Skip if U-Net already trained for this INDEX
    if [ -f "$LOGDIR/${INDEX}_unet/${INDEX}_unet.pth" ]; then
        print_success "U-Net training already completed, skipping..."
        return 0
    fi
    
    # Determine index to use (from resume or current)
    if [ -n "$RESUME_CHECKPOINT" ]; then
        TRAINING_INDEX=$(basename "$RESUME_CHECKPOINT")
    else
        TRAINING_INDEX="$INDEX"
    fi
    
    check_dir "$PSEUDO_LABEL_PATH"
    check_dir "$CONFIDENCE_PATH"
    
    print_info "Training U-Net with ACArL confidence-weighted pseudo-labels..."
    print_info "Configuration:"
    print_info "  - Training Index: $TRAINING_INDEX"
    print_info "  - Use Confidence: true"
    print_info "  - Confidence Threshold: $CONF_THRESHOLD"
    print_info "  - Loss Type: combined (Dice + CE)"
    print_info "  - Learning Rate: $UNET_LR"
    print_info "  - Batch Size: $UNET_BATCH_SIZE"
    print_info "  - Max Epochs: $UNET_MAX_EPOCHS"
    
    uv run python train_unet_acarl.py \
        --seed "$SEED" \
        --lr "$UNET_LR" \
        --batch_size "$UNET_BATCH_SIZE" \
        --max_epochs "$UNET_MAX_EPOCHS" \
        --val_iters "$UNET_VAL_ITERS" \
        --index "${TRAINING_INDEX}_unet" \
        --data_path "$DATA_PATH_USED" \
        --lab_path "$PSEUDO_LABEL_PATH" \
        --conf_path "$CONFIDENCE_PATH" \
        --use_confidence \
        --conf_threshold "$CONF_THRESHOLD" \
        --loss_type combined \
        --data_module "$DATA_MODULE" \
        --num_classes 2 \
        --logdir "$LOGDIR" \
        --gpus "$GPUS"
    
    print_success "U-Net training completed"
    print_info "U-Net checkpoint: $LOGDIR/$INDEX/${INDEX}_unet.pth"
}

stage_evaluation() {
    print_header "STAGE 7: Evaluation"
    
    # Skip evaluation if results already computed for this INDEX
    if [ -f "$LOGDIR/$INDEX/${INDEX}_eval_results.txt" ]; then
        print_success "Evaluation already completed, skipping..."
        return 0
    fi
    
    # Find U-Net checkpoint (it's saved with _unet suffix)
    UNET_INDEX="${INDEX}_unet"
    UNET_CKPT="$LOGDIR/$UNET_INDEX/${UNET_INDEX}.pth"
    check_file "$UNET_CKPT"
    
    print_info "Evaluating segmentation model..."
    
    uv run python eval.py \
        --data_path "$DATA_PATH_USED" \
        --data_module "$DATA_MODULE" \
        --batch_size 128 \
        --num_classes 2 \
        --ckpt "$UNET_CKPT" \
        --gpus "$GPUS"
    
    print_success "Evaluation completed"
}

stage_comparison() {
    print_header "STAGE 8: Legacy Method Comparison (Optional)"
    
    print_info "Running legacy WeakMedSAM (without ACArL) for comparison..."
    
    # Generate pseudo-labels with legacy method
    print_info "Generating pseudo-labels (legacy method without ACArL)..."
    
    LEGACY_PSEUDO_PATH="./outputs/pseudo_labels_legacy"
    check_dir "$LEGACY_PSEUDO_PATH"
    
    uv run python lab_gen.py \
        --batch-size "$BATCH_SIZE" \
        --data-path "$DATA_PATH_USED" \
        --save-path "$LEGACY_PSEUDO_PATH" \
        --data-module brats \
        --parent-classes "$PARENT_CLASSES" \
        --child-classes "$CHILD_CLASSES" \
        --samus-ckpt "$LOGDIR/$INDEX/$INDEX.pth" \
        --sam-ckpt "$SAM_CHECKPOINT" \
        --t 4 \
        --beta 4 \
        --threshold "$THRESHOLD" \
        --gpus "$GPUS"
    
    print_success "Legacy pseudo-labels generated"
    
    # Train legacy U-Net
    print_info "Training legacy U-Net (without confidence weighting)..."
    
    uv run python train_unet.py \
        --seed "$SEED" \
        --lr "$UNET_LR" \
        --batch_size "$UNET_BATCH_SIZE" \
        --max_epochs "$UNET_MAX_EPOCHS" \
        --val_iters "$UNET_VAL_ITERS" \
        --index "${INDEX}_legacy" \
        --data_path "$DATA_PATH_USED" \
        --lab_path "$LEGACY_PSEUDO_PATH" \
        --data_module brats \
        --num_classes 2 \
        --logdir "$LOGDIR" \
        --gpus "$GPUS"
    
    print_success "Legacy U-Net training completed"
    
    # Evaluate legacy model
    print_info "Evaluating legacy model..."
    
    uv run python eval.py \
        --data_path "$DATA_PATH_USED" \
        --data_module brats \
        --batch_size 128 \
        --num_classes 2 \
        --ckpt "$LOGDIR/${INDEX}_legacy/${INDEX}_legacy.pth" \
        --gpus "$GPUS"
    
    print_success "Legacy comparison completed"
}

################################################################################
# Main Pipeline Execution
################################################################################

main() {
    print_header "WeakMedSAM + ACArL End-to-End Pipeline"
    
    # Initialize paths based on dataset
    init_paths
    
    # Initialize INDEX-specific paths
    init_index_paths
    
    print_info "Pipeline Configuration:"
    print_info "  - Stage: $STAGE"
    print_info "  - GPU: $GPUS"
    print_info "  - Seed: $SEED"
    print_info "  - Index: $INDEX"
    print_info "  - Dataset: $DATA_OUTPUT_PATH"
    print_info "  - Log Directory: $LOGDIR/$INDEX"
    
    echo ""
    
    # Helper function to run a stage and log output
    run_stage() {
        local stage_num=$1
        local stage_name=$2
        local stage_func=$3
        
        STAGE_LOG="$LOGDIR/$INDEX/stage_${stage_num}_${stage_name}.log"
        mkdir -p "$(dirname "$STAGE_LOG")"
        
        print_info "Logging stage output to: $STAGE_LOG"
        $stage_func 2>&1 | tee -a "$STAGE_LOG"
    }
    
    case "$STAGE" in
        "all")
            run_stage 0 setup stage_setup
            run_stage 1 preprocessing stage_preprocessing
            run_stage 2 clustering stage_clustering
            run_stage 3 sam_download stage_download_sam
            run_stage 4 acarl_training stage_acarl_training
            run_stage 5 pseudo_labels stage_pseudo_labels
            run_stage 6 unet_training stage_unet_training
            run_stage 7 evaluation stage_evaluation
            print_header "Pipeline Completed Successfully!"
            ;;
        "1")
            run_stage 1 preprocessing stage_preprocessing
            ;;
        "2")
            run_stage 2 clustering stage_clustering
            ;;
        "3")
            run_stage 3 sam_download stage_download_sam
            ;;
        "4")
            run_stage 4 acarl_training stage_acarl_training
            ;;
        "5")
            run_stage 5 pseudo_labels stage_pseudo_labels
            ;;
        "6")
            run_stage 6 unet_training stage_unet_training
            ;;
        "7")
            run_stage 7 evaluation stage_evaluation
            ;;
        "8")
            run_stage 8 comparison stage_comparison
            ;;
        *)
            print_error "Unknown stage: $STAGE"
            print_info "Available stages: all, 1-8"
            exit 1
            ;;
    esac
    
    # Print summary
    print_header "Pipeline Summary"
    echo ""
    echo "Results saved in: $LOGDIR/$INDEX"
    echo ""
    echo "Output files:"
    echo "  - SAMUS Model: $LOGDIR/$INDEX/$INDEX.pth"
    echo "  - ACArL Module: $LOGDIR/$INDEX/${INDEX}_acarl.pth"
    echo "  - U-Net Model: $LOGDIR/$INDEX/${INDEX}_unet.pth"
    echo "  - Pseudo-labels: $PSEUDO_LABEL_PATH"
    echo "  - Confidence Maps: $CONFIDENCE_PATH"
    echo "  - Soft Labels: $SOFT_LABEL_PATH"
    echo ""
    echo "Next steps:"
    echo "  - Review logs: tail -f $LOGDIR/$INDEX/train.log"
    echo "  - Run evaluation: uv run python eval.py --ckpt $LOGDIR/$INDEX/${INDEX}_unet.pth"
    echo ""
    
    print_success "All operations completed successfully!"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --index)
            INDEX="$2"
            shift 2
            ;;
        --help)
            cat << EOF
WeakMedSAM + ACArL End-to-End Pipeline

Usage: bash run_pipeline.sh --data-path /path/to/dataset [OPTIONS]

Required:
  --data-path PATH       Path to raw or preprocessed dataset directory

Options:
  --stage STAGE          Pipeline stage to run: all, 1-8 (default: all)
  --gpus GPUS            GPU IDs to use (default: 0)
  --resume CHECKPOINT    Resume from existing checkpoint (path to logs/YYYYMMDD_HHMMSS)
  --seed SEED            Random seed (default: 42)
  --batch-size SIZE      Batch size for training (default: 8)
  --lr LR                Learning rate (default: 1e-4)
  --index INDEX          Output index/timestamp (default: auto-generated YYYYMMDD_HHMMSS)
  --help                 Show this help message

Stages:
  all  Run all stages
  1    Data Preprocessing (NIfTI to 2D slices, skipped if already preprocessed)
  2    Pre-clustering
  3    Download SAM Checkpoint
  4    ACArL Training
  5    Pseudo-label Generation
  6    U-Net Training
  7    Evaluation
  8    Legacy Method Comparison

Examples:
  bash run_pipeline.sh --data-path ~/datasets/mcd --stage all
  bash run_pipeline.sh --data-path ~/datasets/mcd --stage 1
  bash run_pipeline.sh --data-path ~/datasets/mcd --stage 4 --gpus 0
  bash run_pipeline.sh --data-path ~/datasets/mcd --stage 4 --resume ./logs_mcd/20251212_082453
EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main pipeline
main
