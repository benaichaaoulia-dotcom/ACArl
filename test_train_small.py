#!/usr/bin/env python
"""Test training script with small dataset to verify pipeline"""

import os
import sys
import torch
import random
import numpy as np
from argparse import ArgumentParser

# Add paths
sys.path.insert(0, './samus')
sys.path.insert(0, '.')

def main():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="./data/preprocessed/brats")
    parser.add_argument("--sam_ckpt", type=str, default="./checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to train on")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--index", type=str, default="test_small")
    parser.add_argument("--cluster_file", type=str, default="./data/clusters/brats-8.bin")
    parser.add_argument("--child_classes", type=int, default=8)
    parser.add_argument("--parent_classes", type=int, default=1)
    parser.add_argument("--child_weight", type=float, default=0.5)
    parser.add_argument("--acarl_weight", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create log directory
    os.makedirs(os.path.join(args.logdir, args.index), exist_ok=True)

    # Import after setting device
    from brats.dataset import get_all_dataset
    from samus.build_sam_us import sam_model_registry
    from samus.modeling.acarl import ACArLModule
    from torch.utils.data import DataLoader
    from torch import nn
    from torch.optim import Adam
    import pickle

    print("\n" + "="*80)
    print("TEST TRAINING WITH SMALL DATASET")
    print("="*80 + "\n")

    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    full_dataset = get_all_dataset(args.data_path, args.child_classes, args.cluster_file)
    
    # Limit to num_samples using manual iteration
    print(f"Will train on first {args.num_samples} batches")

    train_loader = DataLoader(
        full_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Single worker for debugging
        pin_memory=True,
    )

    # Build model
    print("Building SAM model...")
    model = sam_model_registry["vit_b"](checkpoint=args.sam_ckpt)
    model = nn.DataParallel(model)
    model = model.to(device)

    # Build ACArL module
    print("Building ACArL module...")
    acarl_module = ACArLModule(
        encoder_dim=768,
        num_classes=args.parent_classes,
        num_sub_classes=args.child_classes,
        diffusion_steps=3,
        diffusion_step_size=0.1
    )
    acarl_module = nn.DataParallel(acarl_module)
    acarl_module = acarl_module.to(device)

    # Load cluster file
    print(f"Loading cluster file from {args.cluster_file}...")
    with open(args.cluster_file, "rb") as f:
        clusters = pickle.load(f)
    print(f"Loaded clusters with {len(clusters['centroids'])} centroids")

    # Optimizer
    optimizer = Adam(
        list(model.parameters()) + list(acarl_module.parameters()),
        lr=args.lr,
    )

    # Training loop
    print("\n" + "="*80)
    print(f"Starting training for {args.max_epochs} epochs")
    print("="*80 + "\n")

    for epoch in range(args.max_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.max_epochs}")
        print("="*80)
        
        model.train()
        acarl_module.train()
        
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            batch_count += 1
            
            # Limit to num_samples batches
            if batch_idx >= args.num_samples:
                break
            
            imgs = batch["img"].to(device)
            labels = batch["plab"].to(device)
            
            print(f"  Batch {batch_idx + 1}: Image shape {imgs.shape}, Label shape {labels.shape}")

            try:
                # Forward pass
                with torch.set_grad_enabled(True):
                    parent_x, child_x, cam_raw = model(imgs)
                    print(f"    ✓ Model forward: parent {parent_x.shape}, child {child_x.shape}")

                    # ACArL forward
                    acarl_out = acarl_module(parent_x, child_x, labels)
                    print(f"    ✓ ACArL forward: confidence {acarl_out['confidence'].shape}")

                    # Dummy loss (just for testing)
                    loss = acarl_out['confidence'].mean()
                    print(f"    ✓ Loss computed: {loss.item():.6f}")

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(acarl_module.parameters()),
                    1.0
                )
                optimizer.step()
                print(f"    ✓ Backward pass complete\n")

            except Exception as e:
                print(f"    ✗ Error in batch: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

        print(f"\nEpoch {epoch + 1} completed successfully!")

    # Save models
    print("\n" + "="*80)
    print("Saving models...")
    print("="*80)
    
    model_path = os.path.join(args.logdir, args.index, f"{args.index}.pth")
    acarl_path = os.path.join(args.logdir, args.index, f"{args.index}_acarl.pth")
    
    torch.save(model.module.state_dict(), model_path, _use_new_zipfile_serialization=False)
    print(f"✓ Saved model to {model_path}")
    
    torch.save(acarl_module.module.state_dict(), acarl_path, _use_new_zipfile_serialization=False)
    print(f"✓ Saved ACArL to {acarl_path}")

    print("\n" + "="*80)
    print("TEST TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
