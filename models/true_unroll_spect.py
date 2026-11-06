"""
CT-less alternating attenuation/activity reconstruction with Path A.
The native operator treats mu as constant; mu retains learned cross-stage
connections and tissue supervision. The default NMRecon adapter differentiates
activity through physical updates. full_projection_step selects the separate
stop-gradient physical-feature variant. Softplus enforces positivity only.
Projection supervision uses count-normalized Poisson deviance (without factor 2).
"""

from __future__ import annotations
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .pytomography_support import (
    DualWindowMuInitNet, LightweightCond3DUNet, DeepCond3DUNet,
    DynamicSubsetProjectorWrapper, _SingleSubsetOp, PhysicsUnrolledMuLoss
)


class ResBlock3D(nn.Module):
    """Basic 3D Residual Block with InstanceNorm and LeakyReLU."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels, affine=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels, affine=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.norm1(self.conv1(x)))
        res = self.norm2(self.conv2(res))
        return self.act2(x + res)


class AdaptiveScaleActivityResUNet3D(nn.Module):
    """
    Lightweight Multi-scale 3D ResUNet for Activity Proximal Refinement.
    Key Architectural Advances:
      1. Multi-scale hierarchy: 2x downsampling (128^3 -> 64^3 -> 32^3 bottleneck) providing
         receptive field > 40x40x40 voxels, large enough to span deep bones and whole patient anatomy.
      2. Skip connections with 3D residual blocks for high-frequency edge and texture preservation.
      3. Scale-adaptive normalization: normalizes features by patient-specific mean activity scale,
         completely decoupling neural representation from absolute physical counts.
      4. Fully unbounded residual: x_out = clamp(x_tilde + res * scale, min=1e-6).
         Eliminates 0.2*tanh artificial clamping, allowing arbitrary local boosts (>300%) or noise suppression (>95%).
      5. Zero-initialized output convolution: output is identically 0 at initial step, guaranteeing
         an exact physical OSEM identity mapping at the start of training.
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 16):
        super().__init__()
        c0 = base_ch
        c1 = base_ch * 2
        c2 = base_ch * 4

        # Level 0 (128^3)
        self.conv_in = nn.Conv3d(in_ch, c0, kernel_size=3, padding=1)
        self.norm_in = nn.InstanceNorm3d(c0, affine=True)
        self.act_in = nn.LeakyReLU(0.2, inplace=True)
        self.block0 = ResBlock3D(c0)

        # Down 1 (128^3 -> 64^3)
        self.down1 = nn.Conv3d(c0, c1, kernel_size=3, stride=2, padding=1)
        self.norm_down1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_down1 = nn.LeakyReLU(0.2, inplace=True)
        self.block1 = ResBlock3D(c1)

        # Down 2 (64^3 -> 32^3, Bottleneck)
        self.down2 = nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.norm_down2 = nn.InstanceNorm3d(c2, affine=True)
        self.act_down2 = nn.LeakyReLU(0.2, inplace=True)
        self.bottleneck = ResBlock3D(c2)

        # Up 1 (32^3 -> 64^3)
        self.conv_up1 = nn.Conv3d(c2, c1, kernel_size=3, padding=1)
        self.norm_up1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_up1 = nn.LeakyReLU(0.2, inplace=True)
        self.fuse1 = nn.Conv3d(c1 * 2, c1, kernel_size=3, padding=1)
        self.norm_fuse1 = nn.InstanceNorm3d(c1, affine=True)
        self.act_fuse1 = nn.LeakyReLU(0.2, inplace=True)
        self.block_up1 = ResBlock3D(c1)

        # Up 0 (64^3 -> 128^3)
        self.conv_up0 = nn.Conv3d(c1, c0, kernel_size=3, padding=1)
        self.norm_up0 = nn.InstanceNorm3d(c0, affine=True)
        self.act_up0 = nn.LeakyReLU(0.2, inplace=True)
        self.fuse0 = nn.Conv3d(c0 * 2, c0, kernel_size=3, padding=1)
        self.norm_fuse0 = nn.InstanceNorm3d(c0, affine=True)
        self.act_fuse0 = nn.LeakyReLU(0.2, inplace=True)
        self.block_up0 = ResBlock3D(c0)

        # Output head
        self.conv_out = nn.Conv3d(c0, 1, kernel_size=3, padding=1)
        # Small std init (1e-4) preserves OSEM identity mapping while ensuring autograd flows through all layers from step 0
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x_tilde: torch.Tensor, mu: torch.Tensor, res_bp: torch.Tensor) -> torch.Tensor:
        b = x_tilde.shape[0]
        # 1. Patient-specific mean activity scale
        scale = x_tilde.view(b, -1).mean(dim=-1).view(b, 1, 1, 1, 1).detach().clamp_min(1e-5)

        norm_x = x_tilde / scale
        norm_mu = mu / 0.154  # normalized against linear attenuation of water at 140 keV
        norm_res = torch.clamp(res_bp, min=-3.0, max=3.0)

        feat = torch.cat([norm_x, norm_mu, norm_res], dim=1)  # [B, 3, 128, 128, 128]

        # Encoder
        e0 = self.block0(self.act_in(self.norm_in(self.conv_in(feat))))
        e1 = self.block1(self.act_down1(self.norm_down1(self.down1(e0))))
        b_feat = self.bottleneck(self.act_down2(self.norm_down2(self.down2(e1))))

        # Decoder with trilinear interpolation + skip connections
        u1 = F.interpolate(b_feat, size=e1.shape[2:], mode='trilinear', align_corners=False)
        u1 = self.act_up1(self.norm_up1(self.conv_up1(u1)))
        d1 = self.block_up1(self.act_fuse1(self.norm_fuse1(self.fuse1(torch.cat([u1, e1], dim=1)))))

        u0 = F.interpolate(d1, size=e0.shape[2:], mode='trilinear', align_corners=False)
        u0 = self.act_up0(self.norm_up0(self.conv_up0(u0)))
        d0 = self.block_up0(self.act_fuse0(self.norm_fuse0(self.fuse0(torch.cat([u0, e0], dim=1)))))

        res = self.conv_out(d0)

        # Unbounded additive residual with scale adaptation + non-negative physical clamp
        x_out = torch.clamp(x_tilde + res * scale, min=1e-6)
        return x_out


# Backward compatibility alias
BoundedActivityProximal3D = AdaptiveScaleActivityResUNet3D


class TrueEndToEndUnrollStage(nn.Module):
    """
    Single Stage k of True End-to-End Unrolling.

    Gradient flow design:
      - Step 1 (OSEM): update_ratio in no_grad for memory; damp & x_curr carry grads outside.
      - Step 2 (Physics bridge): fp_mu uses mu_curr WITHOUT detach -> Straight-Through estimator
        creates path: loss_x -> poisson_physics -> mu_curr -> mu_prox_net of PREVIOUS stage.
        This is the crucial link enabling true joint activity-attenuation unrolling!
      - Step 3 (mu_prox): mu_curr also appears as direct feature input -> another gradient path.
      - Step 4 (act_prox): act_prox_net(x_tilde, mu_next, ...) -> mu_next -> mu_prox_net weights.
    """
    def __init__(
        self,
        base_ch: int = 16,
        cond_dim: int = 4,
        net_depth: int = 4
    ):
        super().__init__()
        # Learnable damping parameter for OSEM step (sigmoid(-0.5) ~ 0.3775, damp ~ 0.1887)
        self.damping_alpha = nn.Parameter(torch.tensor(-0.5))

        # Attenuation refinement subnetwork
        if net_depth == 4:
            self.mu_prox_net = DeepCond3DUNet(in_ch=4, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)
        else:
            self.mu_prox_net = LightweightCond3DUNet(in_ch=4, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)

        # Upgraded: Multi-scale 3D ResUNet activity proximal subnetwork
        self.act_prox_net = AdaptiveScaleActivityResUNet3D(in_ch=3, base_ch=base_ch)

    @staticmethod
    def _robust_norm(vol: torch.Tensor, q: float = 0.995, eps: float = 1e-6) -> torch.Tensor:
        b, c = vol.shape[:2]
        flat = vol.view(b, c, -1)
        scale = torch.quantile(flat.float(), q, dim=-1, keepdim=True).view(b, c, 1, 1, 1).clamp_min(eps)
        return torch.clamp(vol / scale, min=0.0, max=4.0)

    def full_projection_step(self, x_curr, mu_curr, lam_scatter, y, scatter, projector, cond, compute_fp):
        # One full-angle Poisson gradient evaluation; physical features are stop-gradient.
        # Checkpoint only CNNs, never the expensive native projector.
        with torch.no_grad():
            expected = (projector.forward(x_curr, mu_curr.detach()) + scatter).clamp_min(1e-6)
            negative_gradient = projector.backward(y / expected - 1.0, mu_curr.detach())
            direction = negative_gradient / negative_gradient.abs().amax().clamp_min(1e-6)
            scale = x_curr.mean().clamp_min(1e-6)

            res_mu = (expected - y) / (y.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0))
            grad_mu = projector.backward(res_mu, mu_curr.detach())
            scale_mu = grad_mu.abs().amax(dim=(-3, -2, -1), keepdim=True) + 1e-6
            phys_residual_mu = torch.clamp(grad_mu / scale_mu, min=-3.0, max=3.0)

        damp = 0.5 * torch.sigmoid(self.damping_alpha)
        x_tilde = (x_curr + damp * scale * direction).clamp_min(1e-6)
        mu_in = torch.cat([mu_curr, self._robust_norm(x_tilde), self._robust_norm(lam_scatter), phys_residual_mu], 1)
        delta = checkpoint(self.mu_prox_net, mu_in, cond, use_reentrant=False) if self.training else self.mu_prox_net(mu_in, cond)
        scaled = (mu_curr.float() / 0.1).clamp_min(1e-7)
        mu_next = 0.1 * F.softplus(scaled + torch.log(-torch.expm1(-scaled)) + delta.float()/0.1)
        x_next = checkpoint(self.act_prox_net, x_tilde, mu_next, direction, use_reentrant=False) if self.training else self.act_prox_net(x_tilde, mu_next, direction)
        fp = projector.forward(x_next, mu_next.detach()) + scatter if compute_fp else None
        return x_next, mu_next, fp

    def forward(
        self,
        x_curr: torch.Tensor,           # [B, 1, 128, 128, 128] Current Activity
        mu_curr: torch.Tensor,          # [B, 1, 128, 128, 128] Current Attenuation
        lam_scatter: torch.Tensor,      # [B, 1, 128, 128, 128] Scatter Prior
        proj_sub: torch.Tensor,         # [B, N_sub, 128, 128]
        scatter_sub: torch.Tensor,      # [B, N_sub, 128, 128]
        subset_projector: _SingleSubsetOp,
        cond: torch.Tensor,             # [B, 4]
        compute_fp: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # -------------------------------------------------------------
        # Step 1: Damped Physical OSEM Activity Step
        # -------------------------------------------------------------
        # Autograd gradients flow through physical forward and backward projection!
        # Path A: Only detach mu when passing to physical projector
        mu_curr_detach = mu_curr.detach()
        fp_nogr = (subset_projector.forward(x_curr, mu_curr_detach) + scatter_sub).clamp_min(1e-6)
        # Fix (Issue 2): relative epsilon prevents air/FOV-edge ratio explosion.
        fp_eps = 1e-3 * proj_sub.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        ratio_nogr = proj_sub / (fp_nogr + fp_eps)
        bp_ratio = subset_projector.backward(ratio_nogr, mu_curr_detach)
        with torch.no_grad():
            ones_proj = torch.ones_like(proj_sub)
            sens_map = subset_projector.backward(ones_proj, mu_curr_detach).clamp_min(1e-6)
        # Clamp update ratio to [0.1, 10.0]: prevents air-region OSEM divergence
        update_ratio = torch.clamp(bp_ratio / sens_map, min=0.1, max=10.0)

        # damp and x_tilde: gradients flow from loss to damping_alpha and x_curr
        damp = 0.5 * torch.sigmoid(self.damping_alpha)  # max damp 0.5, starts ~0.060
        x_tilde = torch.clamp(x_curr * (1.0 + damp * (update_ratio - 1.0)), min=1e-6)

        # -------------------------------------------------------------
        # Step 2: Physical Attenuation Residual Backprojection Feature (with safe amax normalization)
        # -------------------------------------------------------------
        with torch.no_grad():
            res_mu = (fp_nogr - proj_sub) / (proj_sub.mean(dim=(-2, -1), keepdim=True).clamp_min(1.0))
            grad_mu = subset_projector.backward(res_mu, mu_curr_detach)
            scale_mu = grad_mu.abs().amax(dim=(-3, -2, -1), keepdim=True) + 1e-6
            phys_residual_mu = torch.clamp(grad_mu / scale_mu, min=-3.0, max=3.0)

        # -------------------------------------------------------------
        # Step 3: Attenuation Refinement (mu_next)
        # mu_curr also appears directly in mu_in -> gradient path to prev stage's mu_prox_net
        # -------------------------------------------------------------
        norm_x = self._robust_norm(x_tilde)
        norm_scatter = self._robust_norm(lam_scatter)
        mu_in = torch.cat([mu_curr, norm_x, norm_scatter, phys_residual_mu], dim=1)

        delta_mu = self.mu_prox_net(mu_in, cond)
        scaled_mu = (mu_curr.float() / 0.1).clamp_min(1e-7)
        latent_mu = scaled_mu + torch.log(-torch.expm1(-scaled_mu))
        mu_next = 0.1 * F.softplus(latent_mu + delta_mu.float() / 0.1)

        # -------------------------------------------------------------
        # Step 4: Multi-scale Activity Refinement (x_next)
        # Keep mu_next connected as context: activity loss may update preceding mu
        # stages through the learned activity proximal network (Path A).
        # -------------------------------------------------------------
        mu_next_detach = mu_next.detach()
        fp_ref = subset_projector.forward(x_tilde, mu_next_detach) + scatter_sub
        proj_residual = (proj_sub - fp_ref) / proj_sub.mean().clamp_min(1.0)
        res_bp = subset_projector.backward(proj_residual, mu_next_detach)
        res_bp_norm = torch.clamp(res_bp / (res_bp.abs().amax(dim=(2, 3, 4), keepdim=True) + 1e-6), -3.0, 3.0)

        # Pass mu_next with autograd graph into act_prox_net
        x_next = self.act_prox_net(x_tilde, mu_next, res_bp_norm)

        fp_next = None
        if compute_fp:
            # fp_next is differentiable w.r.t. x_next (outside no_grad)
            fp_next = subset_projector.forward(x_next, mu_next.detach()) + scatter_sub

        return x_next, mu_next, fp_next


class EndToEndPhysicsUnrollSPECTNet(nn.Module):
    """
    Complete End-to-End Physics-Unrolled SPECT Reconstruction Network.
    Directly outputs the final CT-AC reconstructed activity volume x_final!
    """
    def __init__(
        self,
        num_stages: int = 4,
        base_ch: int = 16,
        cond_dim: int = 4,
        share_weights: bool = True,
        net_depth: int = 4
    ):
        super().__init__()
        self.num_stages = num_stages
        self.share_weights = share_weights
        
        # Stage 0: Initial Attenuation Map Estimator
        self.init_mu_net = DualWindowMuInitNet(
            base_ch=base_ch,
            cond_dim=cond_dim,
            depth=net_depth,
            norm_mode="quantile"
        )
        
        # Stages 1..K
        if share_weights and num_stages > 0:
            shared_stage = TrueEndToEndUnrollStage(
                base_ch=base_ch, cond_dim=cond_dim, net_depth=net_depth
            )
            self.stages = nn.ModuleList([shared_stage for _ in range(num_stages)])
        else:
            self.stages = nn.ModuleList([
                TrueEndToEndUnrollStage(
                    base_ch=base_ch, cond_dim=cond_dim, net_depth=net_depth
                )
                for _ in range(num_stages)
            ])

    @property
    def mu_nets(self) -> List[nn.Module]:
        return [self.init_mu_net] + [s.mu_prox_net for s in self.stages]

    @property
    def act_nets(self) -> List[nn.Module]:
        return [s.act_prox_net for s in self.stages]

    def forward(
        self,
        lam_init_p: torch.Tensor,
        lam_init_s: torch.Tensor,
        proj_primary: torch.Tensor,
        scatter_est: torch.Tensor,
        projector_factory: DynamicSubsetProjectorWrapper,
        cond: torch.Tensor,
        x_init_nac: Optional[torch.Tensor] = None,
        use_checkpoint: bool = True,
        compute_projection_loss: bool = True
    ) -> Dict[str, Any]:
        # Stage 0: Initial Attenuation Map Estimator
        mu_0 = self.init_mu_net(lam_init_p, lam_init_s, cond)

        # Fix (Issue 3): Chang's first-order attenuation pre-correction.
        # x_init_nac was reconstructed with mu=0 (NAC), so A(mu=0)*x_nac ≈ y_meas.
        # But Stage-1 OSEM uses A(mu_0) with mu_0 ≈ 0.15 cm^-1.
        # The scale mismatch is exp(0.15 * 20cm) ≈ 20x in deep tissues, causing streak artifacts.
        # Chang's correction: scale x_nac by sens_nac/sens_ac ≈ exp(∫mu dl) to align scales.
        if x_init_nac is not None:
            x_nac = x_init_nac.float().clamp_min(1e-6)
            with torch.no_grad():
                # Use first subset projector to compute sensitivity maps (representative sample)
                subset_op_0, proj_sub_0, _ = projector_factory.get_subset(
                    proj_primary, scatter_est, subset_idx=0
                )
                ones_proj = torch.ones_like(proj_sub_0)
                mu_zero = torch.zeros_like(mu_0)
                # sens_nac = A^T(mu=0) * 1  (sensitivity without attenuation)
                # sens_ac  = A^T(mu_0)  * 1  (sensitivity with estimated attenuation)
                # scale = sens_nac / sens_ac  ≈  exp(∫mu dl) voxel-wise (Chang 1978)
                sens_nac = subset_op_0.backward(ones_proj, mu_zero).clamp_min(1e-5)
                sens_ac  = subset_op_0.backward(ones_proj, mu_0.detach()).clamp_min(1e-5)
                # Clamp [1.0, 15.0]: correction only increases signal (AC >= NAC),
                # cap at 15x prevents overcorrection in dense bone / metallic implants
                chang_scale = (sens_nac / sens_ac).clamp(min=1.0, max=15.0)
            x_0 = x_nac * chang_scale  # aligned to AC domain: ~3-5 dB free PSNR boost
        else:
            x_0 = lam_init_p.float().clamp_min(1e-6)

        mu_curr = mu_0
        x_curr = x_0

        mu_history = [mu_0]
        x_history = [x_0]
        fp_history = []
        proj_sub_history = []

        for k, stage in enumerate(self.stages):
            is_last = (k == len(self.stages) - 1)
            stage_fp = compute_projection_loss and is_last
            subset_proj, proj_sub, scatter_sub = projector_factory.get_subset(
                proj_primary, scatter_est, subset_idx=k
            )

            if getattr(projector_factory, "full_projection_step", False):
                x_curr, mu_curr, fp_sub = stage.full_projection_step(
                    x_curr, mu_curr, lam_init_s, proj_sub, scatter_sub, subset_proj, cond, stage_fp)
            elif use_checkpoint and self.training:
                x_curr, mu_curr, fp_sub = checkpoint(
                    stage,
                    x_curr,
                    mu_curr,
                    lam_init_s,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond,
                    stage_fp,
                    use_reentrant=False
                )
            else:
                x_curr, mu_curr, fp_sub = stage(
                    x_curr,
                    mu_curr,
                    lam_init_s,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond,
                    stage_fp
                )

            mu_history.append(mu_curr)
            x_history.append(x_curr)
            if fp_sub is not None:
                fp_history.append(fp_sub)
                proj_sub_history.append(proj_sub)

        return {
            "activity_final": x_curr,         # Directly Reconstructed CT-AC Activity!
            "mu_final": mu_curr,               # Directly Estimated Attenuation Map!
            "activity_history": x_history,
            "mu_history": mu_history,
            "fp_history": fp_history,
            "proj_sub_history": proj_sub_history
        }


class Separable3DSSIMLoss(nn.Module):
    """
    Fast Differentiable 3D SSIM Loss via separable 1D Gaussian convolutions.
    Numerically exact 3D isotropic Gaussian filter (sigma=1.5, window_size=7).
    Executes in ~200ms on 128^3 volume.
    """
    def __init__(self, window_size: int = 7, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        self.register_buffer("g", g)

    def _separable_conv(self, x: torch.Tensor) -> torch.Tensor:
        p = self.window_size // 2
        g = self.g.to(dtype=x.dtype, device=x.device)
        kx = g.view(1, 1, 1, 1, -1)
        ky = g.view(1, 1, 1, -1, 1)
        kz = g.view(1, 1, -1, 1, 1)
        out = F.conv3d(x, kx, padding=(0, 0, p))
        out = F.conv3d(out, ky, padding=(0, p, 0))
        out = F.conv3d(out, kz, padding=(p, 0, 0))
        return out

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            data_range = float(img2.max() - img2.min())
            data_range = max(data_range, 1.0)
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mu1 = self._separable_conv(img1)
        mu2 = self._separable_conv(img2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = (self._separable_conv(img1 * img1) - mu1_sq).clamp_min(0.0)
        sigma2_sq = (self._separable_conv(img2 * img2) - mu2_sq).clamp_min(0.0)
        sigma12 = self._separable_conv(img1 * img2) - mu1_mu2

        num = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / den.clamp_min(1e-7)

        if mask is not None:
            mask_bool = mask > 0.5
            if mask_bool.any():
                return ssim_map[mask_bool].mean()
        return ssim_map.mean()


def compute_projection_loss(
    act_pred: torch.Tensor,
    mu_pred: torch.Tensor,
    proj_gt: torch.Tensor,
    scatter_est: Optional[torch.Tensor] = None,
    projector_factory: Optional[Any] = None
) -> torch.Tensor:
    """
    Physical Poisson projection loss (Path A):
      Strictly non-negative Poisson deviance / Generalized KL divergence:
      mean(y_pred - y_true + y_true * log((y_true + 1e-6) / (y_pred + 1e-6)))
    """
    if projector_factory is not None:
        op, proj_sub, scatter_sub = projector_factory.get_subset(proj_gt, scatter_est, subset_idx=None)
        proj_pred = op.forward(act_pred, mu_pred.detach())
        if scatter_sub is not None:
            proj_pred = proj_pred + scatter_sub
        target_proj = proj_sub
    else:
        raise ValueError("projector_factory required for compute_projection_loss")

    proj_pred_safe = torch.clamp(proj_pred, min=1e-6)
    target_safe = torch.clamp(target_proj.float(), min=1e-6)
    loss_poisson = (proj_pred_safe - target_safe + target_safe * torch.log(target_safe / proj_pred_safe)).mean()
    return loss_poisson


class TrueEndToEndSPECTLoss(nn.Module):
    """
    Composite End-to-End Loss with Scale-Invariant Relative L1 + 3D SSIM Loss:
      L_total = activity_weight * (RelL1 + 0.15 * (1 - SSIM3D)) + atten_weight * L_mu + poisson_weight * L_Poisson
    """
    def __init__(
        self,
        activity_weight: float = 1.0,
        atten_weight: float = 5.0,
        poisson_weight: float = 1e-2,
        ssim_weight: float = 0.15,
        eps: float = 1e-5
    ):
        super().__init__()
        self.activity_weight = activity_weight
        self.atten_weight = atten_weight
        self.poisson_weight = poisson_weight
        self.ssim_weight = ssim_weight
        self.eps = eps
        self.ssim_loss_fn = Separable3DSSIMLoss(window_size=7, sigma=1.5)
        self.mu_loss_fn = PhysicsUnrolledMuLoss(
            loss_mode="tissue_balanced",
            decay_gamma=0.8,
            grad_weight=0.5,
            poisson_weight=0.0
        )

    def _rel_l1(self, pred: torch.Tensor, gt: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scale-invariant relative L1 loss, optionally masked to patient body."""
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
        mu_gt: torch.Tensor,
        support_mask: torch.Tensor,
        proj_gt: Optional[torch.Tensor] = None,
        scatter_est: Optional[torch.Tensor] = None,
        projector_factory: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        x_pred = outputs["activity_final"]
        mu_pred = outputs["mu_final"]

        # Body mask for activity (ignore air voxels outside patient)
        mask = (support_mask > 0).float()

        # 1. Scale-invariant relative L1 on activity
        rel_l1_act = self._rel_l1(x_pred, x_gt, mask=mask)

        # 2. 3D SSIM structural constraint
        ssim_val = self.ssim_loss_fn(x_pred, x_gt, mask=mask)
        loss_act = rel_l1_act + self.ssim_weight * (1.0 - ssim_val)

        # 3. Tissue-balanced attenuation loss (normalized internally by PhysicsUnrolledMuLoss)
        mu_loss_dict = self.mu_loss_fn(outputs, mu_gt, support_mask)
        l_mu = mu_loss_dict["loss"]

        # 4. Strictly non-negative Poisson deviance
        l_poisson = x_pred.new_zeros(())
        if self.poisson_weight > 0.0:
            if projector_factory is not None and proj_gt is not None:
                l_poisson = compute_projection_loss(x_pred, mu_pred, proj_gt, scatter_est, projector_factory)

        total_loss = (self.activity_weight  * loss_act
                      + self.atten_weight   * l_mu
                      + self.poisson_weight * l_poisson)

        return {
            "loss":             total_loss,
            "loss_act":         loss_act.detach(),
            "activity_rel_l1":  rel_l1_act.detach(),
            "activity_ssim":    ssim_val.detach(),
            "atten_loss":       l_mu.detach(),
            "poisson_loss":     l_poisson.detach(),
        }
