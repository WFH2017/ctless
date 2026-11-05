"""
Physics-Unrolled mu-map Alternating Estimation Network (PU-MAENet).
Follows 'CT-less SPECT深度学习重建方案.pdf'.

Architecture Components:
1. ResBlock3DWithFiLM: 3D residual block conditioned on scanner geometry (dx, dy, dz, ROR) via FiLM.
2. LightweightCond3DUNet: Memory-efficient 2-level 3D U-Net preserving high spatial resolution.
3. DualWindowMuInitNet: Stage 0 dual-energy-window initial attenuation estimator.
4. DynamicSubsetProjectorWrapper: Differentiable PyTomography wrapper for ordered-subset unrolling.
5. UnrolledMuRefinementStage: Alternating stage (Pure physical OSEM activity step + physical residual backprojection + mu-refinement).
6. PUMAENet: End-to-end unrolled network with deep supervision and gradient checkpointing.
7. PhysicsUnrolledMuLoss: Bone- & edge-weighted L1 loss + 3D spatial gradient loss + Poisson NLL loss.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import pytomography
from pytomography.io.SPECT import dicom as pt_dicom
from pytomography.metadata.SPECT import SPECTObjectMeta, SPECTProjMeta
from pytomography.projectors.SPECT import SPECTSystemMatrix
from pytomography.transforms.SPECT import SPECTAttenuationTransform, SPECTPSFTransform
from pytomography.transforms.shared import RotationTransform
try:
    from .contracts import geometry, scanner_config
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from new1022.contracts import geometry, scanner_config


# =========================================================================
# 1. Differentiable PyTomography Transforms
# =========================================================================
def _numpy_vector(value):
    """Accept metadata vectors or singleton-batched vectors, including CUDA tensors."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Geometry metadata must be a finite vector or [1, N] vector")
    return value


class DifferentiableRotationTransform(RotationTransform):
    """PyTomography 3.4 rotation transform unwrapped to preserve autograd computation."""
    def forward(self, object: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return RotationTransform.forward.__wrapped__(self, object, angles)

    def backward(self, object: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        return RotationTransform.backward.__wrapped__(self, object, angles)


class DifferentiableAttenuationTransform(SPECTAttenuationTransform):
    """PyTomography 3.4 attenuation transform unwrapped to retain gradients w.r.t. mu map."""
    def forward(self, object_i: torch.Tensor, ang_idx: torch.Tensor) -> torch.Tensor:
        return SPECTAttenuationTransform.forward.__wrapped__(self, object_i, ang_idx)

    def backward(self, object_i: torch.Tensor, ang_idx: torch.Tensor) -> torch.Tensor:
        return SPECTAttenuationTransform.backward.__wrapped__(self, object_i, ang_idx)


class DifferentiablePSFTransform(SPECTPSFTransform):
    """PyTomography 3.4 PSF transform unwrapped to retain autograd graph."""
    def forward(self, object: torch.Tensor, ang_idx: torch.Tensor) -> torch.Tensor:
        return SPECTPSFTransform.forward.__wrapped__(self, object, ang_idx)

    def backward(self, object: torch.Tensor, ang_idx: torch.Tensor) -> torch.Tensor:
        return SPECTPSFTransform.backward.__wrapped__(self, object, ang_idx)


# =========================================================================
# 2. Dynamic Subset Projector Wrapper
# =========================================================================
class DynamicSubsetProjectorWrapper:
    """
    Wraps PyTomography SPECTSystemMatrix into a dynamic, subset-based operator.
    Supports on-the-fly mu^{(k)} update per stage and ordered-subset angle partitioning.
    """

    def __init__(
        self,
        object_meta: SPECTObjectMeta,
        proj_meta: SPECTProjMeta,
        collimator_name: str = "GI-LEHR",
        num_subsets: int = 6,
        device: str = "cuda:0"
    ):
        self.object_meta = object_meta
        self.proj_meta = proj_meta
        self.collimator_name = collimator_name
        self.num_subsets = num_subsets
        self.device = device

        if getattr(pytomography, "__version__", "") != "3.4.0":
            raise RuntimeError("Differentiable transforms are validated for PyTomography 3.4.0 only")
        geometry(object_meta.shape, object_meta.dr, proj_meta.shape[1:], proj_meta.dr)
        if not 1 <= num_subsets <= len(proj_meta.angles):
            raise ValueError("num_subsets must be between 1 and the number of views")

        pytomography.set_device(device)
        pytomography.set_dtype(torch.float32)

        self.psf_meta = pt_dicom.get_psfmeta_from_scanner_params(collimator_name, energy_keV=140.5)

        # Partition angles into num_subsets orthogonal subsets
        n_angles = len(proj_meta.angles)
        all_indices = torch.arange(n_angles)
        self.subset_indices = [all_indices[i::num_subsets] for i in range(num_subsets)]

    def get_subset(
        self,
        proj_primary: torch.Tensor,
        scatter_est: torch.Tensor,
        subset_idx: int
    ) -> Tuple["_SingleSubsetOp", torch.Tensor, torch.Tensor]:
        """
        Extract subset data and return callable operator for stage k.
        proj_primary: [B, N_angles, 128, 128]
        scatter_est:  [B, N_angles, 128, 128]
        """
        # None selects all acquired views for final-state physical supervision.
        idx = (torch.arange(len(self.proj_meta.angles)) if subset_idx is None
               else self.subset_indices[subset_idx % self.num_subsets])

        proj_sub = proj_primary[:, idx, :, :]
        scatter_sub = scatter_est[:, idx, :, :]

        idx_np = idx.cpu().numpy()
        sub_angles = _numpy_vector(self.proj_meta.angles)[idx_np]
        sub_radii = _numpy_vector(self.proj_meta.radii)[idx_np] if self.proj_meta.radii is not None else None

        sub_proj_meta = SPECTProjMeta(
            projection_shape=self.proj_meta.shape[1:],
            dr=(self.proj_meta.dr[0], self.proj_meta.dr[1]),
            angles=sub_angles,
            radii=sub_radii
        )

        op = _SingleSubsetOp(
            obj_meta=self.object_meta,
            p_meta=sub_proj_meta,
            psf_meta=self.psf_meta,
            device=self.device
        )
        return op, proj_sub, scatter_sub


class _ExactAdjoint(torch.autograd.Function):
    """Transpose of the discretized forward operator, with explicit first VJPs.

    Inverse image rotation is not the transpose of bilinear interpolation.
    Computing the transpose by autograd fixes this; its mu derivative is
    d_mu <p, A(mu) g>, so no grid_sample second derivative is needed.
    """
    @staticmethod
    def forward(ctx, projection, mu, operator):
        ctx.operator = operator
        ctx.save_for_backward(projection, mu)
        with torch.enable_grad(), torch.autocast(device_type=mu.device.type, enabled=False):
            probe = torch.zeros_like(mu, requires_grad=True)
            matrix = operator._build_sys_mat(mu.detach())
            projected = matrix.forward(probe)
            return torch.autograd.grad(projected, probe, projection)[0]

    @staticmethod
    def backward(ctx, grad_output):
        projection, mu = ctx.saved_tensors
        with torch.enable_grad(), torch.autocast(device_type=mu.device.type, enabled=False):
            leaf_mu = mu.detach().requires_grad_(ctx.needs_input_grad[1])
            projected = ctx.operator._build_sys_mat(leaf_mu).forward(grad_output.detach())
            grad_mu = (torch.autograd.grad(projected, leaf_mu, projection)[0]
                       if ctx.needs_input_grad[1] else None)
        return projected.detach() if ctx.needs_input_grad[0] else None, grad_mu, None


class _SingleSubsetOp:
    """Internal single-subset forward and adjoint operator with dynamic mu."""

    def __init__(self, obj_meta: SPECTObjectMeta, p_meta: SPECTProjMeta, psf_meta: Any, device: str = "cuda:0"):
        self.obj_meta = obj_meta
        self.p_meta = p_meta
        self.psf_meta = psf_meta
        self.device = device

    def _build_sys_mat(self, mu_3d: torch.Tensor) -> SPECTSystemMatrix:
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
        # 3.4.0 rotates attenuation with rotate_detector_z directly. Only replace
        # a private rotation object if a compatible implementation actually has one.
        if hasattr(att_trans, "rotation_transform"):
            att_trans.rotation_transform = sys_mat.rotation_transform
        return sys_mat

    def forward(self, lam_5d: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        """
        lam_5d: [B, 1, 128, 128, 128]
        mu_5d:  [B, 1, 128, 128, 128]
        Returns: [B, N_sub_angles, 128, 128]
        """
        b_size = lam_5d.shape[0]
        outs = []
        for b in range(b_size):
            lam_3d = lam_5d[b, 0]
            mu_3d = mu_5d[b, 0]
            with torch.autocast(device_type=lam_3d.device.type, enabled=False):
                H = self._build_sys_mat(mu_3d.float())
                fp = H.forward(lam_3d.float())
            outs.append(fp)
        return torch.stack(outs, dim=0)

    def backward(self, proj_4d: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        """
        Native PyTomography backprojection for the stop-gradient OSEM step.
        The caller controls no_grad; this is not the exact discrete transpose.
        proj_4d: [B, N_sub_angles, 128, 128]
        mu_5d:   [B, 1, 128, 128, 128]
        Returns: [B, 1, 128, 128, 128]
        """
        b_size = proj_4d.shape[0]
        outs = []
        for b in range(b_size):
            p_3d = proj_4d[b]
            mu_3d = mu_5d[b, 0]
            with torch.autocast(device_type=mu_3d.device.type, enabled=False):
                H = self._build_sys_mat(mu_3d.float())
                bp = H.backward(p_3d.float())
            outs.append(bp.unsqueeze(0))
        return torch.stack(outs, dim=0)

    def exact_adjoint(self, proj_4d: torch.Tensor, mu_5d: torch.Tensor) -> torch.Tensor:
        """Differentiable discrete transpose for numerical tests, not OSEM updates."""
        outs = []
        for b in range(proj_4d.shape[0]):
            with torch.autocast(device_type=mu_5d.device.type, enabled=False):
                bp = _ExactAdjoint.apply(proj_4d[b].float(), mu_5d[b, 0].float(), self)
            outs.append(bp.unsqueeze(0))
        return torch.stack(outs, dim=0)


# =========================================================================
# 3. Neural Architecture: FiLM ResBlock and Lightweight 3D U-Net
# =========================================================================
class ResBlock3DWithFiLM(nn.Module):
    """
    3D Residual Block conditioned on scanner physics (dx, dy, dz, ROR) via FiLM.
    Features: InstanceNorm3d, LeakyReLU, zero-bias linear projection for affine scaling.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int = 4):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)

        self.film_proj = nn.Linear(cond_dim, out_ch * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        # FiLM condition: [B, cond_dim] -> [B, 2*out_ch, 1, 1, 1]
        if getattr(self, "use_film", True):
            film_params = self.film_proj(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            out = out * (1.0 + gamma) + beta

        return self.act(out + residual)


class LightweightCond3DUNet(nn.Module):
    """
    2-level downsampling 3D U-Net with FiLM conditioning.
    Maintains spatial resolution at 128x128x128 with low VRAM footprint.
    Final output conv is zero-initialized for stable residual learning.
    """

    def __init__(self, in_ch: int, out_ch: int = 1, base_ch: int = 24, cond_dim: int = 4):
        super().__init__()
        self.enc1 = ResBlock3DWithFiLM(in_ch, base_ch, cond_dim)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ResBlock3DWithFiLM(base_ch, base_ch * 2, cond_dim)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ResBlock3DWithFiLM(base_ch * 2, base_ch * 4, cond_dim)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock3DWithFiLM(base_ch * 4, base_ch * 2, cond_dim)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock3DWithFiLM(base_ch * 2, base_ch, cond_dim)

        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if any(size < 8 or size % 4 for size in x.shape[-3:]):
            raise ValueError("U-Net spatial dimensions must be multiples of 4 and at least 8")
        e1 = self.enc1(x, cond)                  # [B, base_ch, 128, 128, 128]
        e2 = self.enc2(self.pool1(e1), cond)     # [B, 2*base_ch, 64, 64, 64]
        b = self.bottleneck(self.pool2(e2), cond) # [B, 4*base_ch, 32, 32, 32]

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)
        return self.out_conv(d1)


class DeepCond3DUNet(nn.Module):
    """
    4-level downsampling 3D U-Net with FiLM conditioning.
    Receptive field encompasses the entire 128x128x128 volume (>60 cm).
    Bottleneck at 8x8x8. Final output conv is zero-initialized for stable residual learning.
    """

    def __init__(self, in_ch: int, out_ch: int = 1, base_ch: int = 16, cond_dim: int = 4):
        super().__init__()
        # Level 1: 128^3
        self.enc1 = ResBlock3DWithFiLM(in_ch, base_ch, cond_dim)
        self.pool1 = nn.MaxPool3d(2)
        # Level 2: 64^3
        self.enc2 = ResBlock3DWithFiLM(base_ch, base_ch * 2, cond_dim)
        self.pool2 = nn.MaxPool3d(2)
        # Level 3: 32^3
        self.enc3 = ResBlock3DWithFiLM(base_ch * 2, base_ch * 4, cond_dim)
        self.pool2_3 = nn.MaxPool3d(2)
        # Level 4: 16^3
        self.enc4 = ResBlock3DWithFiLM(base_ch * 4, base_ch * 6, cond_dim)
        self.pool3_4 = nn.MaxPool3d(2)
        # Bottleneck: 8^3
        self.bottleneck = ResBlock3DWithFiLM(base_ch * 6, base_ch * 8, cond_dim)

        # Up 4: 8^3 -> 16^3
        self.up4 = nn.ConvTranspose3d(base_ch * 8, base_ch * 6, kernel_size=2, stride=2)
        self.dec4 = ResBlock3DWithFiLM(base_ch * 12, base_ch * 6, cond_dim)
        # Up 3: 16^3 -> 32^3
        self.up3 = nn.ConvTranspose3d(base_ch * 6, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock3DWithFiLM(base_ch * 8, base_ch * 4, cond_dim)
        # Up 2: 32^3 -> 64^3
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock3DWithFiLM(base_ch * 4, base_ch * 2, cond_dim)
        # Up 1: 64^3 -> 128^3
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResBlock3DWithFiLM(base_ch * 2, base_ch, cond_dim)

        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if any(size < 16 or size % 16 for size in x.shape[-3:]):
            raise ValueError("Deep U-Net spatial dimensions must be multiples of 16 and at least 16")
        e1 = self.enc1(x, cond)                     # 128
        e2 = self.enc2(self.pool1(e1), cond)        # 64
        e3 = self.enc3(self.pool2(e2), cond)        # 32
        e4 = self.enc4(self.pool2_3(e3), cond)      # 16
        b = self.bottleneck(self.pool3_4(e4), cond) # 8

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1), cond)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), cond)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), cond)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), cond)
        return self.out_conv(d1)


# =========================================================================
# 4. Stage 0 Initial Attenuation Estimator (DualWindowMuInitNet)
# =========================================================================
class DualWindowMuInitNet(nn.Module):
    """
    Stage 0 network: maps normalized primary NAC reconstruction (bone hotspots)
    and scatter NAC reconstruction (body contour/lungs) to initial attenuation map mu^{(0)}.
    """

    def __init__(self, base_ch: int = 24, cond_dim: int = 4, depth: int = 2, norm_mode: str = "quantile", input_mode: str = "dual", use_film: bool = True):
        super().__init__()
        if input_mode not in ("dual", "primary"):
            raise ValueError("input_mode must be dual or primary")
        self.input_mode = input_mode
        self.depth = depth
        self.norm_mode = norm_mode
        if depth == 4:
            self.net = DeepCond3DUNet(in_ch=2, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)
        else:
            self.net = LightweightCond3DUNet(in_ch=2, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)
        for block in self.net.modules():
            if isinstance(block, ResBlock3DWithFiLM):
                block.use_film = use_film
                block.film_proj.requires_grad_(use_film)

    @staticmethod
    def _robust_norm(vol: torch.Tensor, q: float = 0.995, eps: float = 1e-6, clip_max: float = 4.0) -> torch.Tensor:
        b, c = vol.shape[:2]
        flat = vol.reshape(b, c, -1)
        scale = torch.quantile(flat.float(), q, dim=-1, keepdim=True).reshape(b, c, 1, 1, 1).clamp_min(eps)
        return torch.clamp(vol / scale, min=0.0, max=clip_max)

    def forward(self, lam_init_p: torch.Tensor, lam_init_s: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "quantile":
            p_norm = self._robust_norm(lam_init_p)
            s_norm = self._robust_norm(lam_init_s)
        else:
            p_max = lam_init_p.abs().amax(dim=(2, 3, 4), keepdim=True) + 1e-6
            s_max = lam_init_s.abs().amax(dim=(2, 3, 4), keepdim=True) + 1e-6
            p_norm = lam_init_p / p_max
            s_norm = lam_init_s / s_max

        # Keep architecture/initialization identical; remove only lower-window information.
        if self.input_mode == "primary":
            s_norm = torch.zeros_like(s_norm)
        x = torch.cat([p_norm, s_norm], dim=1)
        # Smooth nonnegative map, initialized at 0.10 cm^-1. No artificial bone ceiling.
        raw_delta = self.net(x, cond)
        mu_0 = 0.1 * F.softplus(raw_delta.float() / 0.1 + math.log(math.expm1(1.0)))
        return mu_0


# =========================================================================
# 5. Single Unroll Stage: Physical OSEM Step + Physical Residual + Mu Update
# =========================================================================
class UnrolledMuRefinementStage(nn.Module):
    """
    Single Stage k in PU-MAENet:
      Step A: Pure mathematical unparameterized OSEM update for activity lambda^{(k)}.
      Step B: Physical data-fidelity gradient sensitivity w.r.t attenuation map mu.
      Step C: 3D ResUNet refinement for mu^{(k)} conditioned on geometry.
    """

    def __init__(
        self,
        base_ch: int = 24,
        cond_dim: int = 4,
        residual_mode: str = "attenuation_grad",
        depth: int = 2,
        norm_mode: str = "quantile"
    ):
        super().__init__()
        self.residual_mode = residual_mode
        self.depth = depth
        self.norm_mode = norm_mode
        # Input channels = 4: [mu^{(k-1)}, norm(lam^{(k)}), norm(lam_scatter), phys_residual]
        if depth == 4:
            self.mu_prox_net = DeepCond3DUNet(in_ch=4, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)
        else:
            self.mu_prox_net = LightweightCond3DUNet(in_ch=4, out_ch=1, base_ch=base_ch, cond_dim=cond_dim)

    @staticmethod
    def _instance_max_norm(vol: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        max_val = vol.abs().amax(dim=(2, 3, 4), keepdim=True) + eps
        return vol / max_val

    @staticmethod
    def _robust_norm(vol: torch.Tensor, q: float = 0.995, eps: float = 1e-6, clip_max: float = 4.0) -> torch.Tensor:
        b, c = vol.shape[:2]
        flat = vol.reshape(b, c, -1)
        scale = torch.quantile(flat.float(), q, dim=-1, keepdim=True).reshape(b, c, 1, 1, 1).clamp_min(eps)
        return torch.clamp(vol / scale, min=0.0, max=clip_max)

    def forward(
        self,
        lam: torch.Tensor,           # [B, 1, 128, 128, 128]
        mu: torch.Tensor,            # [B, 1, 128, 128, 128]
        lam_scatter: torch.Tensor,    # [B, 1, 128, 128, 128]
        proj_sub: torch.Tensor,      # [B, N_sub, 128, 128]
        scatter_sub: torch.Tensor,   # [B, N_sub, 128, 128]
        subset_projector: _SingleSubsetOp,
        cond: torch.Tensor,          # [B, 4]
        compute_projection_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # -------------------------------------------------------------
        # Step A: 100% physically interpretable OSEM activity update
        # -------------------------------------------------------------
        with torch.no_grad():
            fp = (subset_projector.forward(lam, mu) + scatter_sub).clamp_min(1e-6)
            ratio = proj_sub / fp
            bp_ratio = subset_projector.backward(ratio, mu)
            ones_proj = torch.ones_like(proj_sub)
            sens_map = subset_projector.backward(ones_proj, mu).clamp_min(1e-6)

            lam_next = torch.clamp(lam * (bp_ratio / sens_map), min=0.0)

        # -------------------------------------------------------------
        # Step B: Physical data-fidelity sensitivity w.r.t attenuation mu
        # -------------------------------------------------------------
        if self.residual_mode == "attenuation_grad":
            # True attenuation gradient: d(Fidelity)/d(mu)
            with torch.enable_grad():
                mu_leaf = mu.detach().requires_grad_(True)
                fp_updated = (subset_projector.forward(lam_next.detach(), mu_leaf) + scatter_sub).clamp_min(1e-6)
                terms = fp_updated - proj_sub + torch.xlogy(proj_sub, proj_sub.clamp_min(1e-6) / fp_updated)
                axes = tuple(range(1, terms.ndim))
                subset_loss = (terms.sum(dim=axes) / proj_sub.sum(dim=axes).clamp_min(1.0)).mean()
                grad_mu = torch.autograd.grad(subset_loss, mu_leaf)[0]
            # Fixed robust scaling preserving physical scale (zero residual -> zero input)
            phys_residual = torch.clamp(grad_mu * 500.0, -2.0, 2.0)
        else:
            # Legacy activity backprojection mode
            with torch.no_grad():
                fp_updated = subset_projector.forward(lam_next, mu) + scatter_sub
                proj_residual = proj_sub - fp_updated
                res_bp = subset_projector.backward(proj_residual, mu)
            phys_residual = self._instance_max_norm(res_bp)

        # -------------------------------------------------------------
        # Step C: Deep residual refinement of attenuation map
        # -------------------------------------------------------------
        if self.norm_mode == "quantile":
            norm_lam = self._robust_norm(lam_next)
            norm_scatter = self._robust_norm(lam_scatter)
        else:
            norm_lam = self._instance_max_norm(lam_next)
            norm_scatter = self._instance_max_norm(lam_scatter)

        net_in = torch.cat([
            mu,
            norm_lam,
            norm_scatter,
            phys_residual
        ], dim=1)  # [B, 4, 128, 128, 128]

        delta_mu = self.mu_prox_net(net_in, cond)
        # A zero residual is exactly the identity. Stable inverse-softplus in float32.
        scaled_mu = (mu.float() / 0.1).clamp_min(1e-7)
        latent_mu = scaled_mu + torch.log(-torch.expm1(-scaled_mu))
        mu_next = 0.1 * F.softplus(latent_mu + delta_mu.float() / 0.1)
        # The loss must see the NEW mu, including the final refinement stage.
        fp_next = (subset_projector.forward(lam_next, mu_next) + scatter_sub
                   if compute_projection_loss else None)
        return lam_next, mu_next, fp_next


# =========================================================================
# 6. Full PU-MAENet Unrolled Network
# =========================================================================
class PUMAENet(nn.Module):
    """
    PU-MAENet: Dual-window initialization followed by K-stage alternating unrolling.
    """

    def __init__(
        self,
        num_stages: int = 8,
        base_ch: int = 24,
        cond_dim: int = 4,
        share_weights: bool = False,
        residual_mode: str = "attenuation_grad",
        net_depth: int = 2,
        norm_mode: str = "quantile",
        input_mode: str = "dual",
        use_film: bool = True,
    ):
        super().__init__()
        if num_stages and (input_mode != "dual" or not use_film):
            raise ValueError("Input/FiLM component ablations currently require num_stages=0")
        if num_stages < 0:
            raise ValueError("num_stages must be nonnegative")
        self.num_stages = num_stages
        self.share_weights = share_weights
        self.residual_mode = residual_mode
        self.net_depth = net_depth
        self.norm_mode = norm_mode

        # Stage 0
        self.init_mu_net = DualWindowMuInitNet(
            base_ch=base_ch, cond_dim=cond_dim, depth=net_depth, norm_mode=norm_mode,
            input_mode=input_mode, use_film=use_film
        )

        # Stages 1..K
        if share_weights and num_stages > 0:
            shared_stage = UnrolledMuRefinementStage(
                base_ch=base_ch, cond_dim=cond_dim, residual_mode=residual_mode,
                depth=net_depth, norm_mode=norm_mode
            )
            self.stages = nn.ModuleList([shared_stage for _ in range(num_stages)])
        else:
            self.stages = nn.ModuleList([
                UnrolledMuRefinementStage(
                    base_ch=base_ch, cond_dim=cond_dim, residual_mode=residual_mode,
                    depth=net_depth, norm_mode=norm_mode
                )
                for _ in range(num_stages)
            ])

    def forward(
        self,
        lam_init_p: torch.Tensor,
        lam_init_s: torch.Tensor,
        proj_primary: torch.Tensor,
        scatter_est: torch.Tensor,
        projector_factory: DynamicSubsetProjectorWrapper,
        cond: torch.Tensor,
        use_checkpoint: bool = True,
        compute_projection_loss: bool = True,
        poisson_stages: str = "last_full"
    ) -> Dict[str, Any]:
        """
        Forward pass with optional gradient checkpointing.
        """
        if poisson_stages not in ("all", "last", "last_full"):
            raise ValueError("poisson_stages must be all, last, or last_full")
        # Stage 0: Initial attenuation map
        mu_0 = self.init_mu_net(lam_init_p, lam_init_s, cond)

        mu_curr = mu_0
        # Multiplicative OSEM cannot revive exact zeros in a cached NAC image.
        lam_curr = lam_init_p.float().clamp_min(1e-6)

        mu_history = [mu_0]
        lam_history = []
        fp_sub_history = []
        proj_sub_history = []

        # Stage 1..K Alternating Unrolling
        for k, stage in enumerate(self.stages):
            stage_projection_loss = compute_projection_loss and (
                poisson_stages == "all" or
                (poisson_stages == "last" and k == len(self.stages) - 1))
            subset_proj, proj_sub, scatter_sub = projector_factory.get_subset(
                proj_primary, scatter_est, subset_idx=k
            )

            if use_checkpoint and self.training:
                lam_curr, mu_curr, fp_sub = checkpoint(
                    stage,
                    lam_curr,
                    mu_curr,
                    lam_init_s,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond,
                    stage_projection_loss,
                    use_reentrant=False
                )
            else:
                lam_curr, mu_curr, fp_sub = stage(
                    lam_curr,
                    mu_curr,
                    lam_init_s,
                    proj_sub,
                    scatter_sub,
                    subset_proj,
                    cond,
                    stage_projection_loss
                )

            mu_history.append(mu_curr)
            lam_history.append(lam_curr)
            fp_sub_history.append(fp_sub)
            proj_sub_history.append(proj_sub)

        if compute_projection_loss and poisson_stages == "last_full" and self.stages:
            full_op, full_y, full_scatter = projector_factory.get_subset(
                proj_primary, scatter_est, subset_idx=None)
            # Checkpoint the full projection to avoid retaining all angle graphs.
            if use_checkpoint and self.training:
                full_fp = checkpoint(full_op.forward, lam_curr, mu_curr, use_reentrant=False)
            else:
                full_fp = full_op.forward(lam_curr, mu_curr)
            fp_sub_history[-1] = full_fp + full_scatter
            proj_sub_history[-1] = full_y

        return {
            "mu_final": mu_curr,
            "lam_final": lam_curr,
            "mu_history": mu_history,          # [mu_0, mu_1, ..., mu_K]
            "lam_history": lam_history,
            "fp_sub_history": fp_sub_history,
            "proj_sub_history": proj_sub_history
        }


# =========================================================================
# 7. Physics-Informed Multi-Stage Composite Loss
# =========================================================================
class PhysicsUnrolledMuLoss(nn.Module):
    """
    Composite loss function:
    1. Bone-weighted or Tissue-balanced L1 loss:
       - Bone (mu_GT > 0.175 cm^-1)
       - Lungs (0.02 < mu_GT < 0.12 cm^-1)
       - Soft tissue (0.12 <= mu_GT <= 0.175 cm^-1)
       - Air inside body & empty space outside FOV
    2. 3D spatial gradient consistency loss (Sobel/finite difference)
    3. Projection Poisson negative log-likelihood (NLL)
    4. Deep supervision across stages with exponential decay gamma^(K-k)
    """

    def __init__(
        self,
        decay_gamma: float = 0.8,
        grad_weight: float = 0.5,
        poisson_weight: float = 1e-3,
        poisson_stages: str = "last_full",
        loss_mode: str = "legacy",
        bone_weight: float = 0.35,
        lung_weight: float = 0.30,
        soft_weight: float = 0.20,
        air_weight: float = 0.15,
        out_mask_weight: float = 0.25,
    ):
        super().__init__()
        self.decay_gamma = decay_gamma
        self.grad_weight = grad_weight
        self.poisson_weight = poisson_weight
        if poisson_stages not in ("all", "last", "last_full"):
            raise ValueError("poisson_stages must be all, last, or last_full")
        self.poisson_stages = poisson_stages
        self.loss_mode = loss_mode
        self.bone_weight = bone_weight
        self.lung_weight = lung_weight
        self.soft_weight = soft_weight
        self.air_weight = air_weight
        self.out_mask_weight = out_mask_weight

    @staticmethod
    def _spatial_gradient_3d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz

    def _bone_weighted_mu_loss(self, mu_pred: torch.Tensor, mu_gt: torch.Tensor, valid_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_map = torch.ones_like(mu_gt)
        weight_map = torch.where(mu_gt > 0.175, 3.0, weight_map)
        weight_map = torch.where((mu_gt > 0.02) & (mu_gt < 0.12), 2.0, weight_map)

        mask = torch.ones_like(mu_gt) if valid_mask is None else valid_mask.to(mu_gt)
        if mask.shape != mu_gt.shape:
            raise ValueError("Support mask shape must match mu labels")
        l1_weighted = (mask * weight_map * torch.abs(mu_pred - mu_gt)).sum() / mask.sum().clamp_min(1)

        # 3D Gradient Consistency
        gx_p, gy_p, gz_p = self._spatial_gradient_3d(mu_pred)
        gx_g, gy_g, gz_g = self._spatial_gradient_3d(mu_gt)
        masks = (mask[:, :, 1:] * mask[:, :, :-1],
                 mask[:, :, :, 1:] * mask[:, :, :, :-1],
                 mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1])
        grad_loss = sum((m * (p - g).abs()).sum() / m.sum().clamp_min(1)
                        for m, p, g in zip(masks, (gx_p, gy_p, gz_p), (gx_g, gy_g, gz_g)))

        return l1_weighted, grad_loss

    def _tissue_balanced_mu_loss(self, mu_pred: torch.Tensor, mu_gt: torch.Tensor, valid_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones_like(mu_gt) if valid_mask is None else valid_mask.to(mu_gt)
        if mask.shape != mu_gt.shape:
            raise ValueError("Support mask shape must match mu labels")

        inside = (mask > 0)
        diff = (mu_pred - mu_gt).abs()

        # 1. Bone: mu_gt > 0.175 cm^-1
        m_bone = inside & (mu_gt > 0.175)
        loss_bone = diff[m_bone].mean() if m_bone.any() else diff.new_zeros(())

        # 2. Soft tissue: 0.12 <= mu_gt <= 0.175 cm^-1
        m_soft = inside & (mu_gt >= 0.12) & (mu_gt <= 0.175)
        loss_soft = diff[m_soft].mean() if m_soft.any() else diff.new_zeros(())

        # 3. Lung: 0.02 < mu_gt < 0.12 cm^-1
        m_lung = inside & (mu_gt > 0.02) & (mu_gt < 0.12)
        loss_lung = diff[m_lung].mean() if m_lung.any() else diff.new_zeros(())

        # 4. Air inside support mask: mu_gt <= 0.02 cm^-1
        m_air = inside & (mu_gt <= 0.02)
        loss_air = diff[m_air].mean() if m_air.any() else diff.new_zeros(())

        # 5. Outside support mask: penalize non-zero mu in empty scanner bore/margins
        m_out = (~inside)
        loss_out = mu_pred[m_out].abs().mean() if m_out.any() else diff.new_zeros(())

        tissue_l1 = (
            self.bone_weight * loss_bone +
            self.soft_weight * loss_soft +
            self.lung_weight * loss_lung +
            self.air_weight * loss_air +
            self.out_mask_weight * loss_out
        )

        # 3D Gradient Consistency
        gx_p, gy_p, gz_p = self._spatial_gradient_3d(mu_pred)
        gx_g, gy_g, gz_g = self._spatial_gradient_3d(mu_gt)
        masks = (mask[:, :, 1:] * mask[:, :, :-1],
                 mask[:, :, :, 1:] * mask[:, :, :, :-1],
                 mask[:, :, :, :, 1:] * mask[:, :, :, :, :-1])
        grad_loss = sum((m * (p - g).abs()).sum() / m.sum().clamp_min(1)
                        for m, p, g in zip(masks, (gx_p, gy_p, gz_p), (gx_g, gy_g, gz_g)))

        return tissue_l1, grad_loss

    def _mu_loss(self, mu_pred: torch.Tensor, mu_gt: torch.Tensor, valid_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.loss_mode == "tissue_balanced":
            return self._tissue_balanced_mu_loss(mu_pred, mu_gt, valid_mask)
        if self.loss_mode == "uniform":
            # Change only in-support tissue averaging. Keep outside and gradient terms.
            mask = torch.ones_like(mu_gt) if valid_mask is None else valid_mask.to(mu_gt)
            if mask.shape != mu_gt.shape:
                raise ValueError("Support mask shape must match mu labels")
            inside = mask > 0
            diff = (mu_pred - mu_gt).abs()
            uniform = diff[inside].mean() if inside.any() else diff.new_zeros(())
            outside = mu_pred[~inside].abs().mean() if (~inside).any() else diff.new_zeros(())
            _, grad = self._bone_weighted_mu_loss(mu_pred, mu_gt, valid_mask)
            return uniform + self.out_mask_weight * outside, grad
        return self._bone_weighted_mu_loss(mu_pred, mu_gt, valid_mask)

    def forward(self, outputs: Dict[str, Any], mu_gt: torch.Tensor, valid_mask=None) -> Dict[str, torch.Tensor]:
        mu_history = outputs["mu_history"]
        fp_sub_history = outputs.get("fp_sub_history", [])
        proj_sub_history = outputs.get("proj_sub_history", [])

        K = len(mu_history) - 1
        total_mu_l1 = 0.0
        total_grad_loss = 0.0
        total_poisson_loss = mu_history[-1].new_zeros(())
        poisson_weight_sum = 0.0

        # Stage 0 initial prediction loss
        l1_0, grad_0 = self._mu_loss(mu_history[0], mu_gt, valid_mask)
        stage0_weight = self.decay_gamma ** K
        total_mu_l1 += stage0_weight * l1_0
        total_grad_loss += stage0_weight * grad_0

        # Stage 1..K deep supervision
        for k in range(K):
            stage_w = self.decay_gamma ** (K - 1 - k)
            mu_k = mu_history[k + 1]
            l1_k, grad_k = self._mu_loss(mu_k, mu_gt, valid_mask)
            total_mu_l1 += stage_w * l1_k
            total_grad_loss += stage_w * grad_k

            # Count-normalized half Poisson deviance (NLL minus data-only constant).
            if self.poisson_weight > 0 and (self.poisson_stages == "all" or k == K - 1):
                if len(fp_sub_history) != K or len(proj_sub_history) != K:
                    raise ValueError("Projection-loss outputs are required when poisson_weight > 0")
                fp_k = fp_sub_history[k]
                if fp_k is None:
                    raise ValueError("Selected Poisson stage has no forward projection")
                y_k = proj_sub_history[k]
                poisson_nll = self.poisson_deviance(fp_k, y_k)
                total_poisson_loss += stage_w * poisson_nll
                poisson_weight_sum += stage_w

        weight_sum = sum(self.decay_gamma ** i for i in range(K + 1))
        total_mu_l1 /= weight_sum
        total_grad_loss /= weight_sum
        if poisson_weight_sum:
            total_poisson_loss /= poisson_weight_sum
        if self.poisson_weight == 0:
            total_poisson_loss = mu_history[-1].new_zeros(())
        total_loss = (
            total_mu_l1
            + self.grad_weight * total_grad_loss
            + self.poisson_weight * total_poisson_loss
        )

        return {
            "loss": total_loss,
            "mu_l1_loss": total_mu_l1.detach(),
            "grad_loss": total_grad_loss.detach(),
            "weighted_grad_loss": (self.grad_weight * total_grad_loss).detach(),
            "weighted_poisson_loss": (self.poisson_weight * total_poisson_loss).detach(),
            "poisson_loss": (
                total_poisson_loss.detach()
                if isinstance(total_poisson_loss, torch.Tensor)
                else torch.tensor(0.0, device=total_loss.device)
            ),
            "stage0_loss": l1_0.detach()
        }

    @staticmethod
    def poisson_deviance(expected, observed):
        expected = expected.float().clamp_min(1e-6)
        observed = observed.float()
        terms = expected - observed + torch.xlogy(observed, observed.clamp_min(1e-6) / expected)
        axes = tuple(range(1, terms.ndim))
        return (terms.sum(dim=axes) / observed.sum(dim=axes).clamp_min(1.0)).mean()


def projector_from_batch(batch, device):
    """Create a per-patient operator; geometry is never shared between patients."""
    if batch["lam_init_p"].shape[0] != 1:
        raise ValueError("Physics unrolling requires batch_size=1; use gradient accumulation")
    angles = _numpy_vector(batch["angles"])
    radii = _numpy_vector(batch["radii"])
    if angles.shape != radii.shape or len(angles) != batch["proj_primary"].shape[1]:
        raise ValueError("Angles, radii and projection views must match")
    vendor = batch["vendor"]
    vendor = vendor[0] if isinstance(vendor, (list, tuple)) else vendor
    _, collimator, subsets = scanner_config(vendor, len(angles))
    shape, dr, pshape, pdr = geometry(
        batch["lam_init_p"].shape[-3:], _numpy_vector(batch["dr"]),
        batch["proj_primary"].shape[-2:], _numpy_vector(batch["projection_dr"]))
    pytomography.set_device(device)
    pytomography.set_dtype(torch.float32)
    return DynamicSubsetProjectorWrapper(
        SPECTObjectMeta(dr=dr, shape=shape),
        SPECTProjMeta(projection_shape=pshape, dr=pdr,
                      angles=angles, radii=radii),
        collimator_name=collimator, num_subsets=subsets, device=device)
