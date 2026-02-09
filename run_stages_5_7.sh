#!/bin/bash

set -e

# Configuration
LOGDIR="./logs_baseline_acarl"
INDEX="baseline_acarl_20251229_123211"
DATA_MODULE="brats"
DATA_PATH="/home/belal/brats_dataset/brats_full"
BATCH_SIZE=8
UNET_BATCH_SIZE=128
UNET_MAX_EPOCHS=50
UNET_VAL_ITERS=500
UNET_LR=1e-4
SEED=42
GPUS="0"

PSEUDO_LABEL_PATH="$LOGDIR/$INDEX/pseudo_labels"
CONFIDENCE_PATH="$LOGDIR/$INDEX/confidence_maps"
SOFT_LABEL_PATH="$LOGDIR/$INDEX/soft_labels"
SAMUS_CKPT="$LOGDIR/$INDEX/${INDEX}.pth"
ACARL_CKPT="$LOGDIR/$INDEX/${INDEX}_acarl.pth"
SAM_CHECKPOINT="./checkpoints/sam_vit_b_01ec64.pth"

DIFFUSION_STEPS=3
DIFFUSION_STEP_SIZE=0.1
THRESHOLD=0.5

# ========================================================================
# Copy pretrained checkpoints if not already present
# ========================================================================
echo "=========================================================================="
echo "Copying pretrained checkpoints..."
echo "=========================================================================="

mkdir -p "$LOGDIR/$INDEX"

if [ ! -f "$SAMUS_CKPT" ]; then
    echo "→ Copying SAMUS checkpoint..."
    cp logs_brats_full/20251212_153905/20251212_153905.pth "$SAMUS_CKPT"
    echo "✓ SAMUS checkpoint copied"
fi

if [ ! -f "$ACARL_CKPT" ]; then
    echo "→ Copying ACArL checkpoint..."
    cp logs_brats_full/20251212_153905/20251212_153905_acarl.pth "$ACARL_CKPT"
    echo "✓ ACArL checkpoint copied"
fi

echo ""
echo "=========================================================================="
echo "STAGE 5: Pseudo-Label Generation with ACArL"
echo "=========================================================================="

if [ -d "$PSEUDO_LABEL_PATH" ] && [ "$(ls -A "$PSEUDO_LABEL_PATH" 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "✓ Pseudo-labels already generated, skipping..."
else
    echo "→ Generating pseudo-labels with ACArL refinement..."
    mkdir -p "$PSEUDO_LABEL_PATH"
    mkdir -p "$CONFIDENCE_PATH"
    mkdir -p "$SOFT_LABEL_PATH"
    
    uv run python lab_gen_acarl.py \
        --batch-size 64 \
        --data-path "$DATA_PATH" \
        --save-path "$PSEUDO_LABEL_PATH" \
        --data-module "$DATA_MODULE" \
        --parent-classes 1 \
        --child-classes 8 \
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
    
    echo "✓ Pseudo-label generation completed"
fi

echo ""
echo "=========================================================================="
echo "STAGE 6: U-Net Training (ACArL - with Confidence Weighting)"
echo "=========================================================================="

UNET_CKPT="$LOGDIR/${INDEX}_unet/${INDEX}_unet.pth"

if [ -f "$UNET_CKPT" ]; then
    echo "✓ U-Net training already completed, skipping..."
else
    pkill -9 -f "train_unet" 2>/dev/null || true
    sleep 2
    
    echo "→ Training U-Net with ACArL pseudo-labels and confidence weighting..."
    
    uv run python train_unet_acarl.py \
        --seed "$SEED" \
        --lr "$UNET_LR" \
        --batch_size "$UNET_BATCH_SIZE" \
        --max_epochs "$UNET_MAX_EPOCHS" \
        --val_iters "$UNET_VAL_ITERS" \
        --index "${INDEX}_unet" \
        --data_path "$DATA_PATH" \
        --lab_path "$PSEUDO_LABEL_PATH" \
        --conf_path "$CONFIDENCE_PATH" \
        --data_module "$DATA_MODULE" \
        --num_classes 2 \
        --logdir "$LOGDIR" \
        --loss_type combined \
        --use_confidence \
        --gpus "$GPUS"
    
    echo "✓ U-Net training completed"
fi

echo ""
echo "=========================================================================="
echo "STAGE 7: Evaluation"
echo "=========================================================================="

EVAL_RESULTS="$LOGDIR/$INDEX/${INDEX}_eval_results.txt"

if [ -f "$EVAL_RESULTS" ]; then
    echo "✓ Evaluation already completed"
    cat "$EVAL_RESULTS"
else
    echo "→ Evaluating model on validation set..."
    
    uv run python eval.py \
        --data_path "$DATA_PATH" \
        --data_module "$DATA_MODULE" \
        --batch_size 128 \
        --num_classes 2 \
        --ckpt "$UNET_CKPT" \
        --gpus "$GPUS" | tee "$EVAL_RESULTS"
    
    echo "✓ Evaluation completed"
fi

echo ""
echo "=========================================================================="
echo "Results saved to: $EVAL_RESULTS"
echo "=========================================================================="
