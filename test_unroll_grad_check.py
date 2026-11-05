#!/usr/bin/env python3
"""
Verification Script for End-to-End Joint Physics-Unrolled SPECT Network
Tests:
1. 1 full training Step forward + backward on GPU 6.
2. Single-step total time (sec) and execution time.
3. Gradient norm of mu_net and act_net.
4. Gradient norm comparison of act_net with poisson_weight=0 vs poisson_weight=0.01.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

CTLESS_DIR = '/home/wengfenghua/workspace/SparseSPECT/ctless'
UNROLL_DIR = '/home/wengfenghua/workspace/SparseSPECT/ctless-unroll'
NEW1022_PATH = '/home/wengfenghua/sparseSPECT/0913ctless'
sys.path.insert(0, UNROLL_DIR)
sys.path.insert(0, CTLESS_DIR)
sys.path.insert(0, NEW1022_PATH)

from models.true_unroll_spect import EndToEndPhysicsUnrollSPECTNet, TrueEndToEndSPECTLoss
from train_true_unroll_siemens_nii import SiemensNiiFullUnrollDataset, collate_fn
from new1022.model import projector_from_batch

def test_step(poisson_weight: float, device: str = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\n=======================================================")
    print(f" Testing 1 Full Step with poisson_weight = {poisson_weight} on {device}")
    print(f"=======================================================")

    split_path = os.path.join(CTLESS_DIR, 'splits_siemens_nii/siemens_nii_train.json')
    cache_dir = os.path.join(CTLESS_DIR, 'data/cache_siemens_nii')

    dataset = SiemensNiiFullUnrollDataset(split_path, cache_dir)
    batch = collate_fn([dataset[0]])

    lam_p = batch['lam_init_p'].to(device)
    lam_s = batch['lam_init_s'].to(device)
    x_init_nac = batch['x_init_nac'].to(device)
    proj_p = batch['proj_primary'].to(device)
    scatter = batch['scatter_est'].to(device)
    cond = batch['cond'].to(device)
    x_gt = batch['x_gt'].to(device)
    mu_gt = batch['mu_gt'].to(device)
    support_mask = batch['support_mask'].to(device)

    # Initialize model with 4 stages (seed fixed for exact comparison)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model = EndToEndPhysicsUnrollSPECTNet(
        num_stages=4,
        base_ch=16,
        cond_dim=4,
        share_weights=True,
        net_depth=4
    ).to(device)

    pretrained = '/home/wengfenghua/sparseSPECT/0913ctless/new1022/checkpoints/ge_planB_v2_unrolled_balanced/checkpoint_best_unrolled.pt'
    if os.path.exists(pretrained):
        ckpt = torch.load(pretrained, map_location='cpu', weights_only=False)
        m_dict = model.state_dict()
        matched = {k: v for k, v in ckpt['model'].items() if k in m_dict and v.shape == m_dict[k].shape}
        m_dict.update(matched)
        model.load_state_dict(m_dict)
        print(f"Loaded {len(matched)} / {len(m_dict)} pretrained weights.")

    model.train()

    criterion = TrueEndToEndSPECTLoss(
        activity_weight=1.0,
        atten_weight=5.0,
        poisson_weight=poisson_weight
    ).to(device)

    # Build projector
    t_proj0 = time.perf_counter()
    projector = projector_from_batch(batch, device)
    proj_init_time = time.perf_counter() - t_proj0

    t_fwd0 = time.perf_counter()
    out = model(
        lam_init_p=lam_p,
        lam_init_s=lam_s,
        proj_primary=proj_p,
        scatter_est=scatter,
        projector_factory=projector,
        cond=cond,
        x_init_nac=x_init_nac,
        use_checkpoint=True,
        compute_projection_loss=True
    )
    fwd_time = time.perf_counter() - t_fwd0

    t_loss0 = time.perf_counter()
    loss_dict = criterion(
        outputs=out,
        x_gt=x_gt,
        mu_gt=mu_gt,
        support_mask=support_mask,
        proj_gt=proj_p,
        scatter_est=scatter,
        projector_factory=projector
    )
    loss = loss_dict['loss']
    loss_calc_time = time.perf_counter() - t_loss0

    t_bwd0 = time.perf_counter()
    loss.backward()
    bwd_time = time.perf_counter() - t_bwd0
    total_time = fwd_time + loss_calc_time + bwd_time

    mu_grad = next(model.mu_nets[0].parameters()).grad
    act_grad = next(model.act_nets[0].parameters()).grad

    mu_norm = mu_grad.abs().mean().item() if mu_grad is not None else None
    act_norm = act_grad.abs().mean().item() if act_grad is not None else None

    print(f"Results for poisson_weight = {poisson_weight}:")
    print(f"  Single-Step Total Time: {total_time:.3f} s (Forward: {fwd_time:.3f} s, Backward: {bwd_time:.3f} s)")
    print(f"  Projector Init Time: {proj_init_time*1000:.1f} ms")
    print(f"  Loss: {loss.item():.4f} (ActRelL1: {loss_dict['activity_rel_l1'].item():.4f}, MuLoss: {loss_dict['atten_loss'].item():.4f}, Poisson: {loss_dict['poisson_loss'].item():.4f})")
    print("Check Grad -> mu_net:", mu_norm)
    print("Check Grad -> act_net:", act_norm)

    return {
        "poisson_weight": poisson_weight,
        "total_time": total_time,
        "fwd_time": fwd_time,
        "bwd_time": bwd_time,
        "loss": loss.item(),
        "mu_norm": mu_norm,
        "act_norm": act_norm
    }

if __name__ == '__main__':
    r0 = test_step(poisson_weight=0.0)
    r1 = test_step(poisson_weight=0.01)

    print("\n" + "=" * 65)
    print("GRADIENT VERIFICATION & COMPARISON REPORT:")
    print(f"poisson_weight = 0.00 -> Check Grad -> mu_net: {r0['mu_norm']:.8f}")
    print(f"poisson_weight = 0.00 -> Check Grad -> act_net: {r0['act_norm']:.8e}")
    print(f"poisson_weight = 0.01 -> Check Grad -> mu_net: {r1['mu_norm']:.8f}")
    print(f"poisson_weight = 0.01 -> Check Grad -> act_net: {r1['act_norm']:.8e}")
    diff = abs(r1['act_norm'] - r0['act_norm'])
    rel_change = (diff / r0['act_norm']) * 100
    print(f"act_net Gradient Change: Delta={diff:.8e} ({rel_change:+.2f}%)")
    print("=" * 65)
