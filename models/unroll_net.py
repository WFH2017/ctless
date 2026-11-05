"""
Physics-Unrolled Multi-stage Attenuation Estimation Network (PU-MAENet) for CT-less SPECT.
Reference Implementation: /home/wengfenghua/sparseSPECT/0913ctless/new1022/model.py

Core Paradigm:
  1. Neural network NEVER directly predicts high-dynamic-range SPECT activity x (which causes
     black-box hallucination, denominator explosion, and negative-bias anatomical wipeout).
  2. Neural network STRICTLY predicts the 3D linear attenuation coefficient map mu(r) in [0.0, 0.35] cm^-1,
     guided by dual-window inputs (Primary NAC + Scatter NAC) and 4D scanner geometry conditioning (FiLM).
  3. SPECT activity lambda is updated SOLELY through non-trainable, guaranteed non-negative physical OSEM steps:
       lambda^{(k)} = lambda^{(k-1)} * [ A(mu^{(k-1)})^T ( y / (A(mu^{(k-1)}) lambda^{(k-1)} + s) ) ] / [ A(mu^{(k-1)})^T 1 + eps ]
  4. Physical Poisson data-fidelity gradient w.r.t attenuation map d(L_Poisson)/d(mu) drives the 3D ResUNet refinement:
       mu^{(k)} = mu^{(k-1)} + Delta mu^{(k)}
  5. Multi-tissue balanced loss (Bone 0.35, Lung 0.30, Soft 0.20, Air 0.15, Grad 0.50, Poisson 0.001).
  6. Downstream clinical reconstruction: True CT-AC SPECT is reconstructed using the predicted mu-map.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure new1022 is in sys.path
NEW1022_PATH = '/home/wengfenghua/sparseSPECT/0913ctless'
if NEW1022_PATH not in sys.path:
    sys.path.insert(0, NEW1022_PATH)

from new1022.model import (
    PUMAENet,
    DualWindowMuInitNet,
    UnrolledMuRefinementStage,
    LightweightCond3DUNet,
    DeepCond3DUNet,
    ResBlock3DWithFiLM,
    DynamicSubsetProjectorWrapper,
    PhysicsUnrolledMuLoss,
    projector_from_batch,
    _numpy_vector
)
from new1022.recon import reconstruct_ac_psf_osem, compute_tissue_mu_errors
from new1022.utils import calculate_psnr, calculate_ssim, calculate_nrmse


# =========================================================================
# Convenience Aliases and Helpers
# =========================================================================

# Backward-compatible model aliases
CTLessUnrollSPECTNet = PUMAENet
CTLessMultiplicativeStage = UnrolledMuRefinementStage
PseudoAttenuationEstimator = DualWindowMuInitNet


def load_pretrained_pumaenet(
    checkpoint_path: str = "/home/wengfenghua/sparseSPECT/0913ctless/new1022/checkpoints/ge_planB_v2_unrolled_balanced/checkpoint_best_unrolled.pt",
    device: str = "cuda:0"
) -> Tuple[PUMAENet, Dict[str, Any]]:
    """
    Loads the optimal pretrained PU-MAENet checkpoint.
    """
    if not os.path.exists(checkpoint_path):
        # Fallback to local symlink if present
        local_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "ge_planB_v2_unrolled_best.pt")
        if os.path.exists(local_path):
            checkpoint_path = local_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    model = PUMAENet(
        num_stages=args["num_stages"],
        base_ch=args["base_ch"],
        cond_dim=4,
        share_weights=args.get("share_weights", True),
        residual_mode=args.get("residual_mode", "attenuation_grad"),
        net_depth=args.get("net_depth", 4),
        norm_mode=args.get("norm_mode", "quantile"),
        input_mode=args.get("input_mode", "dual"),
        use_film=args.get("use_film", True),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def reconstruct_downstream_activity(
    proj_primary: np.ndarray,
    scatter_est: np.ndarray,
    mu_map: np.ndarray,
    angles_deg: np.ndarray,
    radii_cm: np.ndarray,
    dr_cm: float = 0.4414496,
    n_iters: int = 8,
    n_subsets: int = 4,
    device: str = "cuda:0",
    collimator_name: str = "GI-LEHR"
) -> np.ndarray:
    """
    Executes downstream clinical-grade AC-PSF-OSEM reconstruction using the predicted mu-map.
    """
    recon_tensor = reconstruct_ac_psf_osem(
        projections=proj_primary,
        scatter_est=scatter_est,
        mu_map=mu_map,
        angles=angles_deg,
        radii=radii_cm,
        dr_cm=dr_cm,
        n_iters=n_iters,
        n_subsets=n_subsets,
        collimator_name=collimator_name,
        device=device
    )
    return recon_tensor.detach().cpu().numpy()


__all__ = [
    "PUMAENet",
    "DualWindowMuInitNet",
    "UnrolledMuRefinementStage",
    "LightweightCond3DUNet",
    "DeepCond3DUNet",
    "ResBlock3DWithFiLM",
    "DynamicSubsetProjectorWrapper",
    "PhysicsUnrolledMuLoss",
    "projector_from_batch",
    "CTLessUnrollSPECTNet",
    "CTLessMultiplicativeStage",
    "PseudoAttenuationEstimator",
    "load_pretrained_pumaenet",
    "reconstruct_downstream_activity",
    "reconstruct_ac_psf_osem",
    "compute_tissue_mu_errors",
    "calculate_psnr",
    "calculate_ssim",
    "calculate_nrmse",
]
