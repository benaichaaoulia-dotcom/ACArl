"""
ACArL-enhanced Pseudo-Label Generation

This script generates confidence-aware pseudo-labels using:
1. Class-conditional affinity refinement
2. Anisotropic diffusion on CAMs
3. Uncertainty quantification for each pixel
"""

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
import argparse
import os
from samus.build_sam_us import samus_model_registry
from samus.modeling.acarl import ACArLModule
from tqdm import tqdm
import importlib
from utils.torchutils import max_norm
import numpy as np
from PIL import Image
import torch.multiprocessing as mp


def worker(rank, subsets, gpus, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[rank]
    subset = subsets[rank]
    sub_loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=4,
    )

    # Load trained model
    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )
    if args.samus_ckpt:
        checkpoint = torch.load(args.samus_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()
    
    # Initialize ACArL module
    acarl_module = ACArLModule(
        encoder_dim=768,  # ViT-B encoder dimension
        num_classes=args.parent_classes,
        num_sub_classes=args.child_classes,
        diffusion_steps=args.diffusion_steps,
        diffusion_step_size=args.diffusion_step_size,
        disable_affinity_learning=args.disable_affinity_learning,
        disable_diffusion=args.disable_diffusion
    ).cuda()
    
    # Load ACArL weights if available
    if args.acarl_ckpt:
        acarl_checkpoint = torch.load(args.acarl_ckpt, map_location='cpu')
        acarl_module.load_state_dict(acarl_checkpoint)
    acarl_module.eval()

    pbar = tqdm(enumerate(sub_loader), total=len(sub_loader), desc=f"Rank: {rank}")
    with torch.no_grad():
        for i, pack in pbar:
            imgs = pack["img"].cuda()
            idxs = pack["idx"]
            
            # Check which masks already exist
            existing_masks = []
            for idx in idxs:
                mask_path = os.path.join(args.save_path, f"{idx}.png")
                existing_masks.append(os.path.exists(mask_path))
            
            # Skip if all masks already exist
            if all(existing_masks):
                continue
            
            # Get encoder features and raw CAM
            x, _, cam = model(imgs)
            pred = (torch.sigmoid(x) > 0.5).float()
            
            # Get encoder features for ACArL
            cls_embedding, _ = model.image_encoder(imgs)
            
            # Apply ACArL refinement
            acarl_output = acarl_module(
                features=cls_embedding,
                cam_raw=cam,
                y_primary=pred
            )
            
            # Get refined CAM and confidence scores
            rw_cam = acarl_output['cam_refined']
            confidence = acarl_output['confidence']
            
            # Apply Gaussian blur for smoothness
            rw_cam = F.interpolate(
                rw_cam, (imgs.size(2), imgs.size(3)), mode="bilinear"
            )
            rw_cam = TF.gaussian_blur(rw_cam, kernel_size=21)
            
            # Mask by primary predictions
            rw_cam *= pred.view(pred.size(0), pred.size(1), 1, 1).expand_as(rw_cam)
            
            # Interpolate confidence to match CAM size
            confidence = F.interpolate(
                confidence, (imgs.size(2), imgs.size(3)), mode="bilinear"
            )

            # Generate pseudo-labels with confidence
            for batch_idx, (c, conf) in enumerate(zip(rw_cam, confidence)):
                # Skip if mask already exists
                mask_path = os.path.join(args.save_path, f"{idxs[batch_idx]}.png")
                if existing_masks[batch_idx]:
                    continue
                
                c = c.cpu().numpy()[0]  # (H, W)
                conf = conf.cpu().numpy()[0]  # (H, W)
                
                # Threshold with class-specific threshold
                binary_label = (c > args.threshold).astype(np.uint8) * 255
                
                # Save binary label
                Image.fromarray(binary_label, mode="L").save(mask_path)
                
                # Save confidence map as additional information
                if args.save_confidence:
                    confidence_scaled = (conf * 255).astype(np.uint8)
                    Image.fromarray(confidence_scaled, mode="L").save(
                        os.path.join(args.conf_path, f"{idxs[batch_idx]}_conf.png")
                    )
                
                # Save soft pseudo-labels (CAM values) for confidence-weighted training
                if args.save_soft_labels:
                    soft_label = (c * 255).astype(np.uint8)
                    Image.fromarray(soft_label, mode="L").save(
                        os.path.join(args.soft_path, f"{idxs[batch_idx]}_soft.png")
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--data-module", type=str, required=True)
    parser.add_argument("--vit-name", type=str, default="vit_b")
    parser.add_argument("--sam-ckpt", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samus-ckpt", type=str, required=True)
    parser.add_argument("--acarl-ckpt", type=str, default=None,
                        help="Path to trained ACArL module weights")
    parser.add_argument("--parent-classes", type=int, default=2)
    parser.add_argument("--child-classes", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=3,
                        help="Number of anisotropic diffusion steps")
    parser.add_argument("--diffusion-step-size", type=float, default=0.1,
                        help="Step size for diffusion")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary pseudo-labels")
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--save-confidence", action="store_true",
                        help="Save confidence maps")
    parser.add_argument("--conf-path", type=str, default=None,
                        help="Path to save confidence maps")
    parser.add_argument("--save-soft-labels", action="store_true",
                        help="Save soft pseudo-labels for confidence-weighted training")
    parser.add_argument("--soft-path", type=str, default=None,
                        help="Path to save soft pseudo-labels")
    parser.add_argument("--disable-affinity-learning", action="store_true",
                        help="Disable Block A: Class-Conditional Affinity Learning")
    parser.add_argument("--disable-diffusion", action="store_true",
                        help="Disable Block B: Anisotropic Diffusion")
    args = parser.parse_args()
    print(args)

    # Create output directories
    os.makedirs(args.save_path, exist_ok=True)
    if args.save_confidence and args.conf_path:
        os.makedirs(args.conf_path, exist_ok=True)
    if args.save_soft_labels and args.soft_path:
        os.makedirs(args.soft_path, exist_ok=True)

    # Load dataset
    data_module = importlib.import_module(f"{args.data_module}.dataset")
    dataset = data_module.get_all_dataset(args.data_path, 0, "")
    
    # Split dataset across GPUs
    gpus = args.gpus.split(",")
    subset_size = len(dataset) // len(gpus) + 1
    subsets = []
    start_idx = 0
    for i in range(len(gpus)):
        end_idx = min(start_idx + subset_size, len(dataset))
        subset_indices = list(range(start_idx, end_idx))
        subsets.append(torch.utils.data.Subset(dataset, subset_indices))
        start_idx = end_idx
    
    assert sum([len(subset) for subset in subsets]) == len(dataset)
    
    # Run multi-GPU pseudo-label generation
    mp.spawn(
        worker,
        args=(subsets, gpus, args),
        nprocs=len(gpus),
        join=True,
    )
