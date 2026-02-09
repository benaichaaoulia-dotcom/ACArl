#!/usr/bin/env bash
# Simple test training script - directly run with appropriate Python path

cd /home/belal/Codes/WeakMedSAM

# Run training for just 1 step
uv run python - <<'EOF'
import os
import sys
import torch

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from samus.build_sam_us import samus_model_registry
from samus.modeling.acarl import ACArLModule
from brats.dataset import get_all_dataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("\n" + "="*80)
print("TEST TRAINING - SINGLE BATCH")
print("="*80 + "\n")

# Load dataset
print("Loading dataset...")
dataset = get_all_dataset("./data/preprocessed/brats", 8, "./data/clusters/brats-8.bin")

# Create small loader - just 3 batches
loader = DataLoader(dataset, batch_size=2, num_workers=0, pin_memory=True)

# Build model
print("Building SAM model...")
model = samus_model_registry["vit_b"](parent_classes=1, child_classes=8, checkpoint="./checkpoints/sam_vit_b_01ec64.pth")
model = nn.DataParallel(model)
model.to(device)

# Build ACArL module
print("Building ACArL module...")
acarl_module = ACArLModule(encoder_dim=768, num_classes=1, num_sub_classes=8)
acarl_module = nn.DataParallel(acarl_module)
acarl_module.to(device)

# Optimizer
optimizer = Adam(
    list(model.parameters()) + list(acarl_module.parameters()),
    lr=1e-4
)

# Train for 3 batches
print("\nTraining on 3 batches...\n")
for batch_idx, batch in enumerate(loader):
    if batch_idx >= 3:
        break
    
    print(f"Batch {batch_idx + 1}:")
    imgs = batch["img"].to(device)
    labels = batch["plab"].to(device)
    
    print(f"  Image shape: {imgs.shape}, Label shape: {labels.shape}")
    
    try:
        # Forward pass through SAMUS model
        parent_x, child_x, cam_raw = model(imgs)
        print(f"  Model output: parent {parent_x.shape}, child {child_x.shape}, cam {cam_raw.shape}")
        
        # Extract encoder features for ACArL
        cls_embedding, _ = model.module.image_encoder(imgs)
        print(f"  Encoder features: {cls_embedding.shape}")
        
        # ACArL forward
        acarl_out = acarl_module(features=cls_embedding, cam_raw=cam_raw, y_primary=labels)
        print(f"  ACArL output: confidence {acarl_out['confidence'].shape}")
        
        # Compute loss
        loss = acarl_out['confidence'].mean()
        print(f"  Loss: {loss.item():.6f}")
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(acarl_module.parameters()), 1.0
        )
        optimizer.step()
        print(f"  ✓ Backward pass complete\n")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Save models
print("\nSaving models...")
os.makedirs("./logs/test_small", exist_ok=True)
torch.save(model.module.state_dict(), "./logs/test_small/test_small.pth", _use_new_zipfile_serialization=False)
torch.save(acarl_module.module.state_dict(), "./logs/test_small/test_small_acarl.pth", _use_new_zipfile_serialization=False)

print("\n" + "="*80)
print("SUCCESS! Model training works and checkpoints saved")
print("  SAMUS: ./logs/test_small/test_small.pth")
print("  ACArL: ./logs/test_small/test_small_acarl.pth")
print("="*80 + "\n")
EOF
