"""
Sparse SPECT Physics-Unrolled Reconstruction Network (SparseUnrollSPECTNet).
Specialized for multi-rate sparse projection reconstruction with co-registered CT attenuation prior (mu_CT).

Key Architectural Advances:
  1. Single Multi-Rate Model: A unified model handling 1/2 (30 views), 1/4 (15 views),
     1/8 (8 views), and 1/10 (6 views) sparse configurations simultaneously.
  2. Fixed mu_CT Single-Path Unrolling:
     - CT attenuation map mu_CT is anatomically known and fixed throughout all unrolled stages.
     - System matrix attenuation operator is strictly bound to A(mu_CT).
     - Activity x evolves through damped physical OSEM updates and learned multi-scale 3D ResUNet refinement.
  3. Multi-Rate Condition Injection (SparseConditionEncoder with FiLM):
     - Takes sparse sampling factor s = N_sparse / N_full, view count, and scanner geometry.
     - Emits multi-channel scale (gamma) and shift (beta) parameters to modulate 3D ResUNet features at every scale.
  4. Scale-Adaptive Additive Residual:
     - Residual update x_out = clamp(x_tilde + ResUNet(...) * scale, min=1e-6).
     - Conv-out is zero-initialized for exact physical OSEM identity mapping at step 0.
  5. Compound Dimensionless Loss (SparseSPECTLoss):
     - Scale-invariant relative L1 on body activity and bone structures + optional projection consistency.
"""

from __future__ import annotations
import math
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Ensure new1022 path is available for PyTomography operators
NEW1022_PATH = '/home/wengfenghua/sparseSPECT/0913ctless'
if NEW1022_PATH not in sys.path:
    sys.path.insert(0, NEW1022_PATH)

import pytomography
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from new1022.model import (
    DifferentiableAttenuationTransform,
    DifferentiablePSFTransform,
    DifferentiableRotationTransform,
    _numpy_vector
)
import pytomography.io.SPECT.dicom as pt_dicom


# =========================================================================
# 1. Conditioning Module (FiLM Modulation & Condition Encoder)
# =========================================================================

class SparseConditionEncoder(nn.Module):
    """
    Encodes sampling sparsity ratio, angle count, and scanner geometry into a latent conditioning vector.
    Condition inputs: [s, n_angles / 60.0, dr_x, dr_y, dr_z, mean_radius / 100.0] (dim=6)
    """
    def __init__(self, in_dim: int = 6, cond_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, cond_dim),
            nn.SiLU(inplace=True)
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, in_dim] -> [B, cond_dim]
        return self.net(cond.float())


class ResBlock3DWithFiLM(nn.Module):
    """
    3D Residual Block with Feature-wise Linear Modulation (FiLM).
    h_film = h * (1 + gamma) + beta
    Zero-initialized projection ensures identity behavior at initialization.
    """
    def __init__(self, channels: int, cond_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # FiLM projection: cond_dim -> 2 * channels (gamma and beta)
        self.film_proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))

        # FiLM modulation
        film = self.film_proj(cond_emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(film, 2, dim=1)
        res = res * (1.0 + gamma) + beta

        return self.act2(x + res)


# =========================================================================
# 2. Multi-Scale 3D ResUNet Activity Proximal Subnetwork
# =========================================================================

class AdaptiveScaleActivityResUNet3DWithFiLM(nn.Module):
    """
    Multi-scale 3D ResUNet for Activity Proximal Refinement with FiLM conditioning.
    Receptive field > 40x40x40 voxels (covers anatomical organ depths).
    Inputs: [x_tilde / scale, mu_CT / 0.154, res_bp_norm] (3 channels).
    Output: unbounded additive residual scaled by patient mean activity.
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 16, cond_dim: int = 64):
        super().__init__()
        c0 = base_ch
        c1 = base_ch * 2
        c2 = base_ch * 4

        # Level 0 (128^3)
        self.conv_in = nn.Conv3d(in_ch, c0, kernel_size=3, padding=1)
        self.norm_in = nn.InstanceNorm3d(c0, affine=True)
        self.act_in = nn.LeakyReLU(0.2, inplace=True)
        self.block0 = ResBlock3DWithFiLM(c0, cond_dim=cond_dim)

        # Down 1 (128^3 -> 64^3)
        self.down1 = nn.Conv3d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.norm_down1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_down1 = nn.LeakyReLU(0.2, inplace=True)
        self.block1 = ResBlock3DWithFiLM(c1, cond_dim=cond_dim)

        # Down 2 (64^3 -> 32^3, Bottleneck)
        self.down2 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.norm_down2 = nn.InstanceNorm3d(c2, affine=True)
        self.act_down2 = nn.LeakyReLU(0.2, inplace=True)
        self.bottleneck = ResBlock3DWithFiLM(c2, cond_dim=cond_dim)

        # Up 1 (32^3 -> 64^3)
        self.conv_up1 = nn.Conv3d(c2, c1, kernel_size=3, padding=1)
        self.norm_up1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_up1 = nn.LeakyReLU(0.2, inplace=True)
        self.fuse1 = nn.Conv3d(c1 * 2, c1, kernel_size=3, padding=1)
        self.norm_fuse1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_fuse1 = nn.LeakyReLU(0.2, inplace=True)
        self.block_up1 = ResBlock3DWithFiLM(c1, cond_dim=cond_dim)

        # Up 0 (64^3 -> 128^3)
        self.conv_up0 = nn.Conv3d(c1, c0, kernel_size=3, padding=1)
        self.norm_up0 = nn.InstanceNorm3d(c0, affine=True)
        self.act_up0 = nn.LeakyReLU(0.2, inplace=True)
        self.fuse0 = nn.Conv3d(c0 * 2, c0, kernel_size=3, padding=1)
        self.norm_fuse0 = nn.InstanceNorm3d(c0, affine=True)
        self.act_fuse0 = nn.LeakyReLU(0.2, inplace=True)
        self.block_up0 = ResBlock3DWithFiLM(c0, cond_dim=cond_dim)

        # Output head: Zero-initialized to guarantee identity mapping at step 0
        self.conv_out = nn.Conv3d(c0, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        x_tilde: torch.Tensor,
        mu_ct: torch.Tensor,
        res_bp: torch.Tensor,
        cond_emb: torch.Tensor
    ) -> torch.Tensor:
        b = x_tilde.shape[0]
        # Patient-specific mean activity scale
        scale = x_tilde.view(b, -1).mean(dim=-1).view(b, 1, 1, 1, 1).detach().clamp_min(1e-5)

        norm_x = x_tilde / scale
        norm_mu = mu_ct / 0.154  # normalized against linear attenuation of water at 140 keV
        norm_res = torch.clamp(res_bp, min=-3.0, max=3.0)

        feat = torch.cat([norm_x, norm_mu, norm_res], dim=1)  # [B, 3, 128, 128, 128]

        # Encoder with FiLM
        e0 = self.block0(self.act_in(self.norm_in(self.conv_in(feat))), cond_emb)
        e1 = self.block1(self.act_down1(self.norm_down1(self.down1(e0))), cond_emb)
        b_feat = self.bottleneck(self.act_down2(self.norm_down2(self.down2(e1))), cond_emb)

        # Decoder with trilinear interpolation + skip connections
        u1 = F.interpolate(b_feat, size=e1.shape[2:], mode='trilinear', align_corners=False)
        u1 = self.act_up1(self.norm_up1(self.conv_up1(u1)))
        d1 = self.block_up1(self.act_fuse1(self.norm_fuse1(self.fuse1(torch.cat([u1, e1], dim=1)))), cond_emb)

        u0 = F.interpolate(d1, size=e0.shape[2:], mode='trilinear', align_corners=False)
        u0 = self.act_up0(self.norm_up0(self.conv_up0(u0)))
        d0 = self.block_up0(self.act_fuse0(self.norm_fuse0(self.fuse0(torch.cat([u0, e0], dim=1)))), cond_emb)

        res = self.conv_out(d0)

        # Additive residual with scale adaptation + non-negative physical clamp
        x_out = torch.clamp(x_tilde + res * scale, min=1e-6)
        return x_out


# =========================================================================
# 3. Dynamic Sparse Projector Wrapper (No arbitrary 6-multiple limits)
# =========================================================================

class _SingleSubsetOp:
    """Internal single-subset forward and adjoint operator with cached sys_mat and sens_map."""
    def __init__(self, obj_meta: SPECTObjectMeta, p_meta: SPECTProjMeta, psf_meta: Any, device: str = "cuda:0"):
        self.obj_meta = obj_meta
        self.p_meta = p_meta
        self.psf_meta = psf_meta
        self.device = device
        self._cached_sys_mat: Optional[SPECTSystemMatrix] = None
        self._cached_sens_map: Optional[torch.Tensor] = None

    def _get_sys_mat(self, mu_3d: torch.Tensor) -> SPECTSystemMatrix:
        if self._cached_sys_mat is None:
            att_trans = DifferentiableAttenuationTransform(attenuation_map=mu_3d)
            psf_trans = DifferentiablePSFTransform(self.psf_meta)

            sys_mat = SPECTSystemMatrix(
                obj2obj_transforms=[att_trans, psf_trans],
                proj2proj_transforms=[],
                object_meta=self.obj_meta,
                proj_meta=self.p_meta
            )
            sys_mat.rotation_transform = DifferentiableRotationTransform()
            sys_mat.initialize_transforms()
            if hasattr(att_trans, "rotation_transform"):
                att_trans.rotation_transform = sys_mat.rotation_transform
            self._cached_sys_mat = sys_mat
        return self._cached_sys_mat

    def forward(self, lam_5d: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        b_size = lam_5d.shape[0]
        outs = []
        for b in range(b_size):
            lam_3d = lam_5d[b, 0]
            mu_3d = mu_5d[b, 0]
            with torch.autocast(device_type=lam_3d.device.type, enabled=False):
                H = self._get_sys_mat(mu_3d.float())
                fp = H.forward(lam_3d.float())
            outs.append(fp)
        return torch.stack(outs, dim=0)

    def backward(self, proj_4d: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        b_size = proj_4d.shape[0]
        outs = []
        for b in range(b_size):
            p_3d = proj_4d[b]
            mu_3d = mu_5d[b, 0]
            with torch.autocast(device_type=mu_3d.device.type, enabled=False):
                H = self._get_sys_mat(mu_3d.float())
                bp = H.backward(p_3d.float())
            outs.append(bp.unsqueeze(0))
        return torch.stack(outs, dim=0)

    def get_sensitivity_map(self, proj_sub: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        """Precomputes and caches sensitivity map A^T 1 for this subset and fixed CT attenuation."""
        if self._cached_sens_map is None:
            ones_proj = torch.ones_like(proj_sub)
            self._cached_sens_map = self.backward(ones_proj, mu_5d).clamp_min(1e-6)
        return self._cached_sens_map


class DynamicSparseProjectorWrapper:
    """
    Dynamic Subset Projector Wrapper designed for arbitrary sparse angle configurations.
    Partitions N_sparse angles into num_subsets orthogonal subsets.
    Supports N_sparse in [6, 8, 15, 30, 60] without failing any divisibility constraints.
    Caches subset operators and sensitivity maps for zero-overhead multi-stage unrolling.
    """
    def __init__(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        collimator_name: str = "GI-LEHR",
        num_subsets: int = 2,
        device: str = "cuda:0"
    ):
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self.collimator_name = collimator_name
        self.device = device

        n_angles = len(proj_meta.angles)
        self.num_subsets = max(1, min(num_subsets, n_angles))

        pytomography.set_device(device)
        pytomography.set_dtype(torch.float32)

        self.psf_meta = pt_dicom.get_psfmeta_from_scanner_params(collimator_name, energy_keV=140.5)

        # Partition angles into subsets
        all_indices = torch.arange(n_angles)
        self.subset_indices = [all_indices[i::self.num_subsets] for i in range(self.num_subsets)]

        # Pre-instantiate single subset operators to enable sys_mat and sens_map caching
        self.subset_ops: List[_SingleSubsetOp] = []
        for i in range(self.num_subsets):
            idx = self.subset_indices[i]
            idx_np = idx.cpu().numpy()
            sub_angles = _numpy_vector(self.proj_meta.angles)[idx_np]
            sub_radii = _numpy_vector(self.proj_meta.radii)[idx_np] if self.proj_meta.radii is not None else None
            sub_proj_meta = SPECTProjMeta(
                projection_shape=self.proj_meta.shape[1:],
                dr=(self.proj_meta.dr[0], self.proj_meta.dr[1]),
                angles=sub_angles,
                radii=sub_radii
            )
            self.subset_ops.append(
                _SingleSubsetOp(
                    obj_meta=self.object_meta,
                    p_meta=sub_proj_meta,
                    psf_meta=self.psf_meta,
                    device=self.device
                )
            )

    def get_subset(
        self,
        proj_primary: torch.Tensor,
        scatter_est: torch.Tensor,
        subset_idx: Optional[int] = None
    ) -> Tuple[_SingleSubsetOp, torch.Tensor, torch.Tensor]:
        if subset_idx is None or self.num_subsets <= 1:
            idx = torch.arange(len(self.proj_meta.angles), device=proj_primary.device)
            sub_op = self.subset_ops[0]
        else:
            s_idx = subset_idx % self.num_subsets
            idx = self.subset_indices[s_idx].to(proj_primary.device)
            sub_op = self.subset_ops[s_idx]

        proj_sub = proj_primary[:, idx, :, :]
        scatter_sub = scatter_est[:, idx, :, :]
        return sub_op, proj_sub, scatter_sub


# =========================================================================
# 4. Sparse SPECT Unrolled Stage
# =========================================================================

class SparseUnrollStage(nn.Module):
    """
    Single Stage k of Sparse SPECT Physics-Unrolled Network with Fixed CT Mu.
    Evolves activity:
      1. Damped OSEM step with learnable damping_alpha (gradients pass to x_curr & alpha).
      2. Projection residual backprojection feature res_bp_norm.
      3. 3D ResUNet refinement with FiLM condition injection.
    """
    def __init__(
        self,
        base_ch: int = 16,
        cond_dim: int = 64
    ):
        super().__init__()
        # Learnable damping parameter: sigmoid(-2.0) * 0.5 ≈ 0.059
        self.damping_alpha = nn.Parameter(torch.tensor(-2.0))
        self.act_prox_net = AdaptiveScaleActivityResUNet3DWithFiLM(in_ch=3, base_ch=base_ch, cond_dim=cond_dim)

    def forward(
        self,
        x_curr: torch.Tensor,           # [B, 1, 128, 128, 128]
        mu_ct: torch.Tensor,            # [B, 1, 128, 128, 128] Fixed CT attenuation
        proj_sub: torch.Tensor,         # [B, N_sub, 128, 128]
        scatter_sub: torch.Tensor,      # [B, N_sub, 128, 128]
        subset_projector: _SingleSubsetOp,
        cond_emb: torch.Tensor,         # [B, cond_dim]
        compute_fp: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # -------------------------------------------------------------
        # Step 1: Damped Physical OSEM Activity Step (no_grad for forward/backward memory)
        # -------------------------------------------------------------
        with torch.no_grad():
            fp_nogr = (subset_projector.forward(x_curr, mu_ct) + scatter_sub).clamp_min(1e-6)
            fp_eps = 1e-3 * proj_sub.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            ratio_nogr = proj_sub / (fp_nogr + fp_eps)
            bp_ratio = subset_projector.backward(ratio_nogr, mu_ct)
            sens_map = subset_projector.get_sensitivity_map(proj_sub, mu_ct)
            update_ratio = torch.clamp(bp_ratio / sens_map, min=0.1, max=10.0)

        # damp and x_tilde are outside no_grad so gradients flow to damping_alpha and x_curr
        damp = 0.5 * torch.sigmoid(self.damping_alpha)
        x_tilde = torch.clamp(x_curr * (1.0 + damp * (update_ratio - 1.0)), min=1e-6)

        # -------------------------------------------------------------
        # Step 2: Physical Residual Backprojection Feature
        # -------------------------------------------------------------
        with torch.no_grad():
            fp_ref = subset_projector.forward(x_tilde, mu_ct) + scatter_sub
            proj_residual = (proj_sub - fp_ref) / proj_sub.mean().clamp_min(1.0)
            res_bp = subset_projector.backward(proj_residual, mu_ct)
            res_bp_norm = torch.clamp(res_bp / (res_bp.abs().amax(dim=(2, 3, 4), keepdim=True) + 1e-6), -3.0, 3.0)

        # -------------------------------------------------------------
        # Step 3: Learned Activity Proximal Operator with FiLM Modulation
        # -------------------------------------------------------------
        x_next = self.act_prox_net(x_tilde, mu_ct, res_bp_norm, cond_emb)

        fp_next = None
        if compute_fp:
            fp_next = subset_projector.forward(x_next, mu_ct) + scatter_sub

        return x_next, fp_next


# =========================================================================
# 5. Complete SparseUnrollSPECTNet Architecture
# =========================================================================

class SparseUnrollSPECTNet(nn.Module):
    """
    Complete Multi-Rate Physics-Unrolled SPECT Network with Fixed CT Attenuation.
    Directly reconstructs the CT-AC activity volume from arbitrary sparse view projections!
    """
    def __init__(
        self,
        num_stages: int = 4,
        base_ch: int = 16,
        cond_in_dim: int = 6,
        cond_dim: int = 64,
        share_weights: bool = True
    ):
        super().__init__()
        self.num_stages = num_stages
        self.share_weights = share_weights

        # Condition Encoder: [s, n_angles_norm, dr_x, dr_y, dr_z, radius_norm] -> cond_dim
        self.cond_encoder = SparseConditionEncoder(in_dim=cond_in_dim, cond_dim=cond_dim)

        # Stages 1..K
        if share_weights and num_stages > 0:
            shared_stage = SparseUnrollStage(base_ch=base_ch, cond_dim=cond_dim)
            self.stages = nn.ModuleList([shared_stage for _ in range(num_stages)])
        else:
            self.stages = nn.ModuleList([
                SparseUnrollStage(base_ch=base_ch, cond_dim=cond_dim)
                for _ in range(num_stages)
            ])

    def forward(
        self,
        x_init: torch.Tensor,                         # [B, 1, 128, 128, 128] Initial activity estimate
        mu_ct: torch.Tensor,                          # [B, 1, 128, 128, 128] Fixed CT attenuation map
        proj_sparse: torch.Tensor,                    # [B, N_sparse, 128, 128] Sparse projections
        scatter_sparse: torch.Tensor,                 # [B, N_sparse, 128, 128] Sparse scatter
        projector_factory: DynamicSparseProjectorWrapper,
        cond: torch.Tensor,                           # [B, 6] Condition vector
        use_checkpoint: bool = True,
        compute_projection_loss: bool = False
    ) -> Dict[str, Any]:
        # Encode condition vector into latent embedding
        cond_emb = self.cond_encoder(cond)

        x_curr = x_init.float().clamp_min(1e-6)
        x_history = [x_curr]
        fp_history = []
        proj_sub_history = []

        for k, stage in enumerate(self.stages):
            is_last = (k == len(self.stages) - 1)
            stage_fp = compute_projection_loss and is_last
            subset_proj, proj_sub, scatter_sub = projector_factory.get_subset(
                proj_sparse, scatter_sparse, subset_idx=k
            )

            if use_checkpoint and self.training:
                x_curr, fp_sub = checkpoint(
                    stage,
                    x_curr,
                    mu_ct,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond_emb,
                    stage_fp,
                    use_reentrant=False
                )
            else:
                x_curr, fp_sub = stage(
                    x_curr,
                    mu_ct,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond_emb,
                    stage_fp
                )

            x_history.append(x_curr)
            if fp_sub is not None:
                fp_history.append(fp_sub)
                proj_sub_history.append(proj_sub)

        return {
            "activity_final": x_curr,
            "activity_history": x_history,
            "fp_history": fp_history,
            "proj_sub_history": proj_sub_history
        }


# =========================================================================
# 6. Compound Dimensionless Loss Function (SparseSPECTLoss)
# =========================================================================

class SparseSPECTLoss(nn.Module):
    """
    Scale-Invariant Relative L1 Activity Loss + High-Priority Bone L1 + Optional Projection Consistency.
    Dimensionless and invariant to absolute count scale.
    """
    def __init__(
        self,
        activity_weight: float = 1.0,
        bone_weight: float = 0.4,
        proj_weight: float = 0.05,
        eps: float = 1e-5
    ):
        super().__init__()
        self.activity_weight = activity_weight
        self.bone_weight = bone_weight
        self.proj_weight = proj_weight
        self.eps = eps

    def _rel_l1(self, pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = torch.abs(pred - gt)
        if mask is not None:
            diff = diff * mask
            denom = (gt.abs() * mask).sum().clamp_min(1.0) / mask.sum().clamp_min(1.0)
            return diff.sum() / mask.sum().clamp_min(1.0) / (denom + self.eps)
        return diff.mean() / (gt.abs().mean() + self.eps)

    def forward(
        self,
        outputs: Dict[str, Any],
        x_gt: torch.Tensor,
        mu_ct: torch.Tensor,
        support_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        x_pred = outputs["activity_final"]
        body_mask = (support_mask > 0.5).float()

        # 1. Body Activity Relative L1 (Dimensionless)
        l_act = self._rel_l1(x_pred, x_gt, mask=body_mask)

        # 2. High-Priority Bone Relative L1 (attenuation >= 0.175 cm^-1)
        bone_mask = ((mu_ct > 0.175) & (support_mask > 0.5)).float()
        if bone_mask.sum() > 50:
            l_bone = self._rel_l1(x_pred, x_gt, mask=bone_mask)
        else:
            l_bone = x_pred.new_zeros(())

        # 3. Optional Projection Data Consistency
        l_proj = x_pred.new_zeros(())
        if self.proj_weight > 0.0 and outputs.get("fp_history") and outputs.get("proj_sub_history"):
            fp = outputs["fp_history"][-1].clamp_min(1e-6)
            y = outputs["proj_sub_history"][-1]
            l_proj = torch.abs(fp - y).sum() / (y.abs().sum().clamp_min(1.0) + self.eps)

        total_loss = (
            self.activity_weight * l_act +
            self.bone_weight * l_bone +
            self.proj_weight * l_proj
        )

        return {
            "loss": total_loss,
            "activity_rel_l1": l_act.detach(),
            "bone_rel_l1": l_bone.detach(),
            "proj_rel_l1": l_proj.detach()
        }
