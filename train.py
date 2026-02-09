from torch.utils.data import DataLoader
import torch
import argparse
import os
from samus.build_sam_us import samus_model_registry
from samus.modeling.acarl import ACArLModule
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import importlib
from utils.pytuils import AverageMeter
import torch.nn.functional as F
from utils.metrics import dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_module", type=str)
    parser.add_argument("--vit_name", type=str)
    parser.add_argument("--sam_ckpt", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--index", type=str)
    parser.add_argument("--samus_ckpt", type=str)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--val_iters", type=int, default=3000)
    parser.add_argument("--parent_classes", type=int)
    parser.add_argument("--child_classes", type=int)
    parser.add_argument("--child_weight", type=float)
    parser.add_argument("--acarl_weight", type=float, default=0.3)
    parser.add_argument("--acarl_ckpt", type=str, default=None)
    parser.add_argument("--lambda_s", type=float, default=1.0, help="Weight for sub-class regularization loss")
    parser.add_argument("--lambda_a", type=float, default=0.2, help="Weight for affinity learning loss")
    parser.add_argument("--lambda_conf", type=float, default=1.0, help="Weight for confidence regularization loss")
    parser.add_argument("--cluster_file", type=str)
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", type=str)
    parser.add_argument("--disable_affinity_learning", action="store_true", help="Disable Block A: Class-Conditional Affinity Learning")
    parser.add_argument("--disable_diffusion", action="store_true", help="Disable Block B: Anisotropic Diffusion")
    args = parser.parse_args()
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    model = samus_model_registry["vit_b"](
        parent_classes=args.parent_classes,
        child_classes=args.child_classes,
        checkpoint=args.sam_ckpt,
    )
    model = torch.nn.DataParallel(model).cuda()
    
    # Enable gradient checkpointing to reduce memory and improve throughput
    if hasattr(model.module, 'image_encoder'):
        model.module.image_encoder.gradient_checkpointing = True
    
    if args.samus_ckpt:
        checkpoint = torch.load(args.samus_ckpt)
        model.load_state_dict(checkpoint)
    
    # Initialize ACArL module
    acarl_module = ACArLModule(
        encoder_dim=768,  # ViT-B encoder dimension
        num_classes=args.parent_classes,
        num_sub_classes=args.child_classes,
        diffusion_steps=3,
        diffusion_step_size=0.1,
        disable_affinity_learning=args.disable_affinity_learning,
        disable_diffusion=args.disable_diffusion
    )
    acarl_module = torch.nn.DataParallel(acarl_module).cuda()
    if args.acarl_ckpt:
        acarl_checkpoint = torch.load(args.acarl_ckpt)
        acarl_module.load_state_dict(acarl_checkpoint)

    data_module = importlib.import_module(f"{args.data_module}.dataset")
    train_dataset, val_dataset, _ = data_module.get_dataset(
        args.data_path, args.child_classes, args.cluster_file
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=16,  # Increased from 8 for better prefetching
        prefetch_factor=8,  # Increased from 4 for deeper queue
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=8,  # Increased from 4
        prefetch_factor=4,  # Increased from 2
        persistent_workers=True,
    )

    args.max_iters = args.max_epochs * len(train_loader)

    # Separate optimizers for model and ACArL
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(acarl_module.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=args.lr, total_steps=args.max_iters
    )

    writer = SummaryWriter(os.path.join(args.logdir, args.index))

    # Checkpoint path for resuming
    checkpoint_path = os.path.join(args.logdir, args.index, "checkpoint.pt")
    start_iter = 1
    
    # Try to resume from checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt["model"])
        acarl_module.load_state_dict(ckpt["acarl_module"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_iter = ckpt["n_iter"] + 1
        print(f"Resumed from iteration {start_iter}")

    pbar = tqdm(range(start_iter, args.max_iters + 1), ncols=100)
    train_loader_iter = iter(train_loader)
    for n_iter in pbar:
        model.train()
        acarl_module.train()
        optimizer.zero_grad()
        try:
            datapack = next(train_loader_iter)

        except:
            train_loader_iter = iter(train_loader)
            datapack = next(train_loader_iter)

        imgs = datapack["img"].cuda(non_blocking=True)
        parent_labs = datapack["plab"].cuda(non_blocking=True)
        child_labs = datapack["clab"].cuda(non_blocking=True)

        # Forward pass through SAMUS (cached with gradient computation)
        with torch.cuda.nvtx.range("samus_forward"):
            parent_x, child_x, cam_raw = model(imgs)
        
        # Primary classification losses
        parent_loss = F.binary_cross_entropy_with_logits(
            parent_x,
            parent_labs,
        )

        child_loss = F.binary_cross_entropy_with_logits(child_x, child_labs)
        
        # Forward pass through ACArL (conditionally executed)
        acarl_output = None
        acarl_losses = {
            'L_total': torch.tensor(0.0, device=imgs.device),
            'L_s': torch.tensor(0.0, device=imgs.device),
            'L_a': torch.tensor(0.0, device=imgs.device),
            'L_conf': torch.tensor(0.0, device=imgs.device)
        }
        
        if args.acarl_weight > 0.0:
            # Get encoder features for ACArL (only when needed)
            with torch.cuda.nvtx.range("encoder_forward"):
                cls_embedding, _ = model.module.image_encoder(imgs)
            
            with torch.cuda.nvtx.range("acarl_forward"):
                acarl_output = acarl_module(
                    features=cls_embedding,
                    cam_raw=cam_raw,
                    y_primary=parent_labs
                )
            
            # ACArL losses (batch compute)
            with torch.cuda.nvtx.range("acarl_loss"):
                acarl_losses = acarl_module.module.compute_loss(
                    z_mu=acarl_output['z_mu'],
                    z_logstd=acarl_output['z_logstd'],
                    affinity_matrices=acarl_output['affinity_matrices'],
                    y_primary=parent_labs,
                    cam_refined=acarl_output['cam_refined'],
                    cam_raw=cam_raw,
                    lambda_s=args.lambda_s,
                    lambda_a=args.lambda_a,
                    lambda_conf=args.lambda_conf
                )
        else:
            # Skip ACArL computation when weight is 0
            pass
        
        # Total loss (combined computation)
        loss = (parent_loss + args.child_weight * child_loss + 
                args.acarl_weight * acarl_losses['L_total'])

        # Backward pass
        with torch.cuda.nvtx.range("backward"):
            loss.backward()
            optimizer.step()
            scheduler.step()

        parent_pred = (torch.sigmoid(parent_x) > 0.5).float()
        parent_score = torch.eq(parent_pred, parent_labs).sum() / parent_labs.numel()

        child_pred = (torch.sigmoid(child_x) > 0.5).float()
        child_score = torch.eq(child_pred, child_labs).sum() / child_labs.numel()

        # Batch log scalar operations to reduce synchronization
        writer.add_scalar("train/train loss", loss.item(), n_iter)
        writer.add_scalar("train/parent loss", parent_loss.item(), n_iter)
        writer.add_scalar("train/child loss", child_loss.item(), n_iter)
        writer.add_scalar("train/acarl_L_s", acarl_losses['L_s'].item(), n_iter)
        writer.add_scalar("train/acarl_L_a", acarl_losses['L_a'].item(), n_iter)
        writer.add_scalar("train/acarl_L_conf", acarl_losses['L_conf'].item(), n_iter)
        writer.add_scalar("train/parent score", parent_score.item(), n_iter)
        writer.add_scalar("train/child score", child_score.item(), n_iter)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], n_iter)

        if n_iter % args.val_iters == 0:
            model.eval()
            val_loss = AverageMeter()
            val_score = AverageMeter()
            with torch.no_grad():
                for pack in val_loader:
                    imgs = pack["img"].cuda()
                    labs = pack["plab"].float().cuda()
                    x, _, _ = model(imgs)
                    val_loss.add(F.binary_cross_entropy_with_logits(x, labs).item())
                    pred = (torch.sigmoid(x) > 0.5).float()

                    val_score.add(torch.eq(pred, labs).sum().item(), labs.numel())

            model.train()
            writer.add_scalar("val/val loss", val_loss.get(), n_iter)
            writer.add_scalar("val/val score", val_score.get(), n_iter)

        pbar.set_postfix(
            {
                "tl": loss.item(),
                "ts": parent_score.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        
        # Save checkpoint every 500 iterations
        if n_iter % 500 == 0:
            checkpoint_data = {
                "n_iter": n_iter,
                "model": model.state_dict(),
                "acarl_module": acarl_module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(checkpoint_data, checkpoint_path)
            print(f"\nCheckpoint saved at iteration {n_iter}")

    torch.save(
        model.module.state_dict(),
        os.path.join(args.logdir, args.index, f"{args.index}.pth"),
        _use_new_zipfile_serialization=False,
    )
    
    # Save ACArL module
    torch.save(
        acarl_module.module.state_dict(),
        os.path.join(args.logdir, args.index, f"{args.index}_acarl.pth"),
        _use_new_zipfile_serialization=False,
    )


if __name__ == "__main__":
    main()
