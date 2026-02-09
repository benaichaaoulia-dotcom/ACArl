"""
Adaptive Class-Conditional Affinity Refinement with Learnable Sub-class Exploration (ACArL)

This module replaces the fixed, class-agnostic affinity refinement with:
1. Class-conditional, learnable affinity modeling
2. Online adaptive sub-class exploration
3. Energy-based variational inference for sub-class assignments
4. Anisotropic diffusion on CAMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class SubClassPosterior(nn.Module):
    """
    Learnable posterior network q_φ(z|f) for adaptive sub-class exploration.
    Uses reparameterization trick for differentiable sampling.
    """
    
    def __init__(self, encoder_dim: int, num_sub_classes: int):
        super().__init__()
        self.num_sub_classes = num_sub_classes
        
        # Network for mean parameter
        self.mu_net = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_sub_classes)
        )
        
        # Network for log standard deviation parameter
        self.logstd_net = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_sub_classes)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, D) pooled encoder features
        
        Returns:
            z_mu: (B, K) mean parameters
            z_logstd: (B, K) log std parameters
            z_samples: (B, K) sampled sub-class assignments
        """
        z_mu = self.mu_net(features)
        z_logstd = self.logstd_net(features)
        
        # Reparameterization trick
        z_std = torch.exp(z_logstd)
        eps = torch.randn_like(z_mu)
        z_samples = z_mu + z_std * eps
        
        return z_mu, z_logstd, z_samples


class ClassConditionalAffinity(nn.Module):
    """
    Learns class-specific affinity matrices using MLPs.
    Each class has its own parametric function for computing affinities.
    """
    
    def __init__(self, encoder_dim: int, num_classes: int, coord_embed_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.coord_embed_dim = coord_embed_dim
        
        # Class-conditional affinity MLPs
        self.affinity_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * encoder_dim + coord_embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ) for _ in range(num_classes)
        ])
        
        # Learnable coordinate embeddings for spatial locality
        self.coord_embedding = nn.Embedding(256 * 256, coord_embed_dim)
        
    def forward(
        self, 
        features: torch.Tensor, 
        class_idx: int,
        H: int,
        W: int,
        grid_size: int = 16
    ) -> torch.Tensor:
        """
        Compute class-conditional affinity matrix for a specific class.
        
        Args:
            features: (B, D, H, W) spatial features from encoder
            class_idx: which class to compute affinity for
            H, W: spatial dimensions
            grid_size: number of sampled points in each dimension
        
        Returns:
            A_c: (B, HW, HW) class-conditional affinity matrix
        """
        B = features.shape[0]
        device = features.device
        
        # Sample grid points for computational efficiency
        stride = max(1, H // grid_size)
        grid_indices = []
        
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                if i < H and j < W:
                    grid_indices.append((i, j))
        
        n_pts = len(grid_indices)
        
        # Build affinity matrix without in-place operations
        # Start with identity matrix
        A_c_base = torch.eye(H * W, device=device).unsqueeze(0).expand(B, -1, -1)
        
        # Create a mask for updates
        affinity_mask = torch.zeros(B, H * W, H * W, device=device)
        affinity_values = torch.zeros(B, H * W, H * W, device=device)
        
        mlp_c = self.affinity_mlps[class_idx]
        
        # Compute pairwise affinities for sampled grid points
        for ii, (i1, j1) in enumerate(grid_indices):
            idx1 = i1 * W + j1
            
            for jj, (i2, j2) in enumerate(grid_indices):
                if ii == jj:
                    continue
                
                idx2 = i2 * W + j2
                
                # Local radius constraint (γ=5 in the paper)
                dist = abs(i1 - i2) + abs(j1 - j2)
                if dist <= 5:
                    # Extract spatial features
                    f1 = features[:, :, i1, j1]  # (B, D)
                    f2 = features[:, :, i2, j2]  # (B, D)
                    
                    # Coordinate embedding
                    coord_emb_1 = self.coord_embedding(torch.tensor(idx1, device=device))
                    coord_emb_2 = self.coord_embedding(torch.tensor(idx2, device=device))
                    coord_emb = (coord_emb_1 + coord_emb_2) / 2
                    coord_emb = coord_emb.unsqueeze(0).repeat(B, 1)  # (B, coord_dim)
                    
                    # Concatenate features and predict affinity
                    feat_concat = torch.cat([f1, f2, coord_emb], dim=1)  # (B, 2D + coord_dim)
                    affinity_logit = mlp_c(feat_concat).squeeze(-1)  # (B,)
                    affinity_val = torch.sigmoid(affinity_logit)
                    
                    # Store in tensors (these are still created fresh, not in-place on gradients)
                    affinity_mask[:, idx1, idx2] = 1.0
                    affinity_values[:, idx1, idx2] = affinity_val
        
        # Combine base identity with computed affinities using where (non-inplace)
        A_c = torch.where(affinity_mask > 0, affinity_values, A_c_base)
        
        # Normalize to row-stochastic matrix
        A_c = A_c / (A_c.sum(dim=2, keepdim=True) + 1e-8)
        
        return A_c


class AnisotropicDiffusion(nn.Module):
    """
    Applies class-conditional anisotropic diffusion to refine CAMs.
    Uses learned affinity matrices as diffusion coefficients.
    """
    
    def __init__(self, num_steps: int = 3, step_size: float = 0.1):
        super().__init__()
        self.num_steps = num_steps
        self.step_size = step_size
    
    def forward(
        self,
        cam_raw: torch.Tensor,
        affinity_matrices: List[torch.Tensor],
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Apply anisotropic diffusion: M_c^(t+1) = M_c^(t) + dt * Div_c(A_c ∇M_c)
        
        Args:
            cam_raw: (B, C, H, W) raw class activation maps
            affinity_matrices: list of (B, HW, HW) affinity matrices per class
            H, W: spatial dimensions
        
        Returns:
            cam_refined: (B, C, H, W) refined CAMs
        """
        B, C, cam_H, cam_W = cam_raw.shape
        cam_refined = cam_raw.clone()
        
        # Verify dimensions match
        expected_hw = H * W
        actual_hw = cam_H * cam_W
        if expected_hw != actual_hw:
            raise RuntimeError(f"Dimension mismatch: CAM is {cam_H}x{cam_W}={actual_hw} but expected {H}x{W}={expected_hw}")
        
        for step in range(self.num_steps):
            # Process all classes and create new tensor (avoid in-place)
            updated_cams = []
            for c in range(C):
                A_c = affinity_matrices[c]  # (B, HW, HW)
                M_c_vec = cam_refined[:, c, :, :].reshape(B, -1)  # (B, HW)
                
                # Compute graph Laplacian: L = D - A
                D_diag = A_c.sum(dim=2)  # (B, HW)
                L_M = torch.bmm(A_c, M_c_vec.unsqueeze(2)).squeeze(2) - D_diag * M_c_vec
                
                # Update via graph diffusion
                M_c_updated = M_c_vec + self.step_size * L_M
                updated_cams.append(M_c_updated.reshape(B, H, W))
            
            # Stack and replace cam_refined (non-inplace)
            cam_refined = torch.stack(updated_cams, dim=1)  # (B, C, H, W)
        
        # Clip to valid CAM range
        cam_refined = torch.clamp(cam_refined, 0, 1)
        
        return cam_refined


class ACArLModule(nn.Module):
    """
    Main ACArL module integrating:
    - Learnable sub-class posterior
    - Class-conditional affinity learning
    - Anisotropic diffusion refinement
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        num_classes: int = 2,
        num_sub_classes: int = 8,
        diffusion_steps: int = 3,
        diffusion_step_size: float = 0.1,
        disable_affinity_learning: bool = False,
        disable_diffusion: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_sub_classes = num_sub_classes
        self.disable_affinity_learning = disable_affinity_learning
        self.disable_diffusion = disable_diffusion
        
        # Sub-class exploration
        self.sub_class_posterior = SubClassPosterior(encoder_dim, num_sub_classes)
        
        # Class-conditional affinity learning (Block A)
        self.affinity_module = ClassConditionalAffinity(encoder_dim, num_classes) if not disable_affinity_learning else None
        
        # Anisotropic diffusion (Block B)
        self.diffusion = AnisotropicDiffusion(diffusion_steps, diffusion_step_size) if not disable_diffusion else None
        
        # Learnable confidence parameters
        self.confidence_scale = nn.Parameter(torch.tensor(1.0))
        self.confidence_bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(
        self,
        features: torch.Tensor,
        cam_raw: torch.Tensor,
        y_primary: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ACArL module.
        
        Args:
            features: (B, D, H, W) encoder features
            cam_raw: (B, C, H, W) raw CAMs
            y_primary: (B, C) primary class labels
        
        Returns:
            dict containing refined CAMs, sub-class samples, affinity matrices, confidence
        """
        B, D, H, W = features.shape
        C = self.num_classes
        _, _, cam_H, cam_W = cam_raw.shape
        
        # Downsample features if too large for affinity computation
        # Keep spatial dims manageable (max 1024 = 32x32)
        max_hw = 1024  # 32x32
        if H * W > max_hw:
            downsample_factor = int((H * W / max_hw) ** 0.5)
            # Cache downsampled features
            features_ds = F.avg_pool2d(features, kernel_size=downsample_factor, stride=downsample_factor)
            H_ds, W_ds = features_ds.shape[2], features_ds.shape[3]
            # Downsample CAM to match feature dimensions (CAM might be higher resolution)
            cam_downsample_factor = max(1, cam_H // H_ds, cam_W // W_ds)
            cam_refined_ds = F.avg_pool2d(cam_raw, kernel_size=cam_downsample_factor, stride=cam_downsample_factor)
            # Ensure exact size match
            if cam_refined_ds.shape[2] != H_ds or cam_refined_ds.shape[3] != W_ds:
                cam_refined_ds = F.interpolate(cam_refined_ds, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
        else:
            features_ds = features
            H_ds, W_ds = H, W
            # Still need to downsample CAM if it's higher resolution than features
            if cam_H != H or cam_W != W:
                cam_refined_ds = F.interpolate(cam_raw, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
            else:
                cam_refined_ds = cam_raw
        
        # Global pooling for sub-class exploration (cached single operation)
        f_global = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)  # (B, D)
        
        # Step 1: Adaptive sub-class exploration
        z_mu, z_logstd, z_samples = self.sub_class_posterior(f_global)
        
        # Step 2: Learn class-conditional affinities (if Block A enabled)
        affinity_matrices = []
        if not self.disable_affinity_learning:
            for c in range(C):
                A_c = self.affinity_module(features_ds, c, H_ds, W_ds)
                affinity_matrices.append(A_c)
        
        # Step 3: Anisotropic diffusion refinement (if Block B enabled)
        if not self.disable_diffusion and self.diffusion is not None and len(affinity_matrices) > 0:
            cam_refined_ds = self.diffusion(cam_refined_ds, affinity_matrices, H_ds, W_ds)
        # If diffusion is disabled, cam_refined_ds stays as is (raw downsampled CAM)
        
        # Upsample refined CAM back to original CAM size (not feature size)
        if cam_refined_ds.shape[2] != cam_H or cam_refined_ds.shape[3] != cam_W:
            cam_refined = F.interpolate(cam_refined_ds, size=(cam_H, cam_W), mode='bilinear', align_corners=False)
        else:
            cam_refined = cam_refined_ds
        
        # Step 4: Compute uncertainty-aware confidence scores
        confidence = torch.sigmoid(
            self.confidence_scale * cam_refined + self.confidence_bias
        )
        
        return {
            'cam_refined': cam_refined,
            'z_mu': z_mu,
            'z_logstd': z_logstd,
            'z_samples': z_samples,
            'affinity_matrices': affinity_matrices,
            'confidence': confidence
        }
    
    def compute_loss(
        self,
        z_mu: torch.Tensor,
        z_logstd: torch.Tensor,
        affinity_matrices: List[torch.Tensor],
        y_primary: torch.Tensor,
        cam_refined: torch.Tensor,
        cam_raw: torch.Tensor,
        lambda_s: float = 1.0,
        lambda_a: float = 0.2,
        lambda_conf: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ACArL losses: L_s (sub-class exploration) + L_a (affinity) + L_conf
        
        Args:
            z_mu, z_logstd: sub-class posterior parameters
            affinity_matrices: learned affinity matrices
            y_primary: primary class labels
            cam_refined: refined CAMs
            cam_raw: raw CAMs
        
        Returns:
            dict with individual losses and total loss
        """
        # L_s: Sub-class exploration loss (KL divergence + entropy regularization)
        # KL(N(μ,σ²) || N(0,I))
        kl_loss = 0.5 * (
            z_mu.pow(2) + torch.exp(2 * z_logstd) - 2 * z_logstd - 1
        ).mean()
        
        # Entropy regularization: encourage diverse sub-class exploration
        z_std = torch.exp(z_logstd)
        entropy_reg = 0.5 * torch.log(2 * 3.14159 * z_std.pow(2) + 1e-8).sum(dim=1).mean()
        
        # L_s uses only KL divergence (always positive)
        # Entropy encourages diversity but KL dominates to keep distribution close to N(0,I)
        L_s = kl_loss + 0.05 * torch.clamp(entropy_reg, min=0)  # Clamp to ensure positivity
        
        # L_a: Affinity learning loss (contrastive)
        # Simplified version: encourage affinity matrix to be smooth and consistent
        L_a = torch.tensor(0.0, device=z_mu.device)
        if len(affinity_matrices) > 0:
            for A_c in affinity_matrices:
                # Encourage diagonal dominance (self-similarity)
                diag_vals = torch.diagonal(A_c, dim1=1, dim2=2)
                # Use MSE to encourage diagonal to be 1, off-diagonal to be 0
                B_size = A_c.shape[0]
                A_target = torch.eye(A_c.shape[1], device=A_c.device).unsqueeze(0).expand(B_size, -1, -1)
                L_a = L_a + F.mse_loss(torch.sigmoid(A_c), A_target)
            
            L_a = L_a / len(affinity_matrices)
        
        # L_conf: Confidence regularization (refined CAM should be close to raw but better)
        L_conf = F.mse_loss(cam_refined, cam_raw.detach()) * 0.1
        
        # Ensure all losses are positive and apply weights
        # Default weights: lambda_s=1.0 (implicit), lambda_a=0.2, lambda_conf=1.0 (implicit)
        # Note: Previous code had implicit linear combination: L_s + 0.2*L_a + L_conf
        
        L_total = (
            lambda_s * torch.clamp(L_s, min=0) + 
            lambda_a * torch.clamp(L_a, min=0) + 
            lambda_conf * torch.clamp(L_conf, min=0)
        )
        
        return {
            'L_s': L_s,
            'L_a': L_a,
            'L_conf': L_conf,
            'L_total': L_total
        }
