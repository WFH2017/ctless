#!/usr/bin/env python3
"""
Verification Script for nmrecon integrated training step on gn1 GPU 0.
Tests:
1. 1 full training Step forward + backward with nmrecon operator on GPU 0.
2. Single-step total time (sec) and execution time.
3. Gradient norm of mu_net and act_net.
4. Gradient norm comparison of act_net with poisson_weight=0 vs poisson_weight=0.001.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn

UNROLL_DIR = '/home/wengfenghua/workspace/SparseSPECT/ctless-unroll'
CTLESS_DIR = UNROLL_DIR
sys.path.insert(0, UNROLL_DIR)
sys.path.insert(0, CTLESS_DIR)

from models.true_unroll_spect import EndToEndPhysicsUnrollSPECTNet, TrueEndToEndSPECTLoss
from train_true_unroll_siemens_nii import SiemensNiiFullUnrollDataset, collate_fn
from nmrecon_training_projector import projector_from_batch

def test_step(poisson_weight: float, device: str = "cuda:0"):
    print(f"\n=======================================================")
    print(f" Testing NMRecon 1 Full Step with poisson_weight = {poisson_weight} on {device}")
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

    criterion = TrueEndToEndSPECTLoss(
        activity_weight=1.0,
        atten_weight=5.0,
        poisson_weight=poisson_weight
    ).to(device)

    model.train()
    model.zero_grad()

    # 1. Projector creation
    t_proj_0 = time.perf_counter()
    projector = projector_from_batch(batch, str(device))
    t_proj = time.perf_counter() - t_proj_0
    print(f"[*] NMRecon Projector initialized in {t_proj*1000:.2f} ms")

    # 2. Forward pass
    torch.cuda.synchronize()
    t_fwd_0 = time.perf_counter()
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
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t_fwd_0
    print(f"[*] Forward pass completed in {t_fwd:.3f} s")

    # 3. Loss computation
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
    act_ssim = loss_dict.get('activity_ssim', torch.tensor(0.0)).item()
    print(f"[*] Loss: {loss.item():.4f} | ActRelL1: {loss_dict['activity_rel_l1'].item():.4f} | ActSSIM: {act_ssim:.4f} | MuLoss: {loss_dict['atten_loss'].item():.4f} | Poisson: {loss_dict.get('poisson_loss', torch.tensor(0.0)).item():.4f}")

    # 4. Backward pass
    torch.cuda.synchronize()
    t_bwd_0 = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    t_bwd = time.perf_counter() - t_bwd_0
    print(f"[*] Backward pass completed in {t_bwd:.3f} s")

    # 5. Measure gradient norms
    mu_grad_norm = sum(p.grad.norm().item() for net in model.mu_nets for p in net.parameters() if p.grad is not None)
    act_grad_norm = sum(p.grad.norm().item() for net in model.act_nets for p in net.parameters() if p.grad is not None)
    damp_grad_norm = sum(s.damping_alpha.grad.norm().item() for s in model.stages if s.damping_alpha.grad is not None)
    print(f"[*] mu_net total grad norm:        {mu_grad_norm:.8f}")
    print(f"[*] act_net total grad norm:       {act_grad_norm:.8f}")
    print(f"[*] damping_alpha total grad norm: {damp_grad_norm:.8f}")

    return {
        'poisson_weight': poisson_weight,
        't_proj': t_proj,
        't_fwd': t_fwd,
        't_bwd': t_bwd,
        'loss': loss.item(),
        'mu_grad_norm': mu_grad_norm,
        'act_grad_norm': act_grad_norm,
        'damp_grad_norm': damp_grad_norm
    }

if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    print(f"Running test on device: {dev}")
    print(">>> RUNNING WITH POISSON WEIGHT = 0.01 (v2 target)")
    res = test_step(poisson_weight=0.01, device=dev)
    print("\n" + "=" * 65)
    print(" SUMMARY: NMRecon Zero-Copy Autograd v2 Test")
    print("=" * 65)
    print(f"Loss: {res['loss']:.4f}")
    print(f"mu_net grad norm:        {res['mu_grad_norm']:.8e}")
    print(f"act_net grad norm:       {res['act_grad_norm']:.8e}")
    print(f"damping_alpha grad norm: {res['damp_grad_norm']:.8e}")
    print(f"Timing: Forward {res['t_fwd']:.2f}s | Backward {res['t_bwd']:.2f}s | Projector {res['t_proj']*1000:.2f}ms")
    print("=" * 65)
