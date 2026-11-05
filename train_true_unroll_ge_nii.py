#!/usr/bin/env python3
"""
Full-Cohort Retraining of True End-to-End Physics-Unrolled SPECT Network
(EndToEndPhysicsUnrollSPECTNet) on 1,527 GE Clinical Cases (data/GE/nii).
Trains on 1,221 held-out cases, validates on 152 validation cases.
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from skimage.metrics import structural_similarity as ssim

# Workspace path setup
CTLESS_DIR = '/home/wengfenghua/workspace/SparseSPECT/ctless'
SPLITS_DIR = os.path.join(CTLESS_DIR, 'splits_ge_nii')
NEW1022_PATH = '/home/wengfenghua/sparseSPECT/0913ctless'
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
if LOCAL_DIR not in sys.path:
    sys.path.insert(0, LOCAL_DIR)
if CTLESS_DIR not in sys.path:
    sys.path.insert(0, CTLESS_DIR)
if NEW1022_PATH not in sys.path:
    sys.path.insert(0, NEW1022_PATH)

from models.true_unroll_spect import (
    EndToEndPhysicsUnrollSPECTNet,
    TrueEndToEndSPECTLoss
)
from new1022.model import projector_from_batch as pytomography_projector_from_batch
try:
    from nmrecon_training_projector import projector_from_batch as nmrecon_projector_from_batch
except ImportError:
    nmrecon_projector_from_batch = None


class GENiiFullUnrollDataset(Dataset):
    """Loads complete cached clinical cases for EndToEndPhysicsUnrollSPECTNet."""
    def __init__(self, split_json_path: str, cache_dir: str):
        with open(split_json_path, 'r') as f:
            self.cases = json.load(f)['cases']
        self.cache_dir = cache_dir

        # Filter available cached files
        self.valid_cases = []
        for c in self.cases:
            pid = c['patient_id']
            pt_path = os.path.join(cache_dir, f"{pid}.pt")
            if os.path.exists(pt_path):
                self.valid_cases.append(c)

        print(f"Loaded {len(self.valid_cases)} / {len(self.cases)} ready cached cases from {split_json_path}")

    def __len__(self):
        return len(self.valid_cases)

    def __getitem__(self, idx: int) -> dict:
        case = self.valid_cases[idx]
        pid = case['patient_id']
        pt_path = os.path.join(self.cache_dir, f"{pid}.pt")
        raw = torch.load(pt_path, map_location='cpu', weights_only=False)

        lam_init_p = raw['lambda_init_p'].float().unsqueeze(0)   # (1, 128, 128, 128)
        lam_init_s = raw['lambda_init_s'].float().unsqueeze(0)   # (1, 128, 128, 128)
        x_gt = raw['lambda_gt'].float().unsqueeze(0)             # (1, 128, 128, 128)
        mu_gt = raw['mu_gt'].float().unsqueeze(0)                # (1, 128, 128, 128)
        support_mask = raw['support_mask'].float().unsqueeze(0)  # (1, 128, 128, 128)

        proj_primary = raw['proj_primary'].float()                # (N, 128, 128)
        scatter_est = raw['scatter_est'].float()                  # (N, 128, 128)
        angles = raw['angles'].float()                            # (N,)
        radii = raw['radii'].float()                              # (N,)
        dr = raw['dr'].float()                                    # (3,)
        projection_dr = raw['projection_dr'].float()              # (2,)

        cond = torch.cat((dr, radii.mean().reshape(1) / 100.0))   # (4,)

        return {
            'lam_init_p': lam_init_p,
            'lam_init_s': lam_init_s,
            'x_init_nac': lam_init_p.clone(),
            'x_gt': x_gt,
            'mu_gt': mu_gt,
            'support_mask': support_mask,
            'proj_primary': proj_primary,
            'scatter_est': scatter_est,
            'angles': angles,
            'radii': radii,
            'dr': dr,
            'projection_dr': projection_dr,
            'cond': cond,
            'vendor': raw.get('vendor', 'GE MEDICAL SYSTEMS'),
            'case_id': pid,
            'patient_id': pid
        }


def collate_fn(batch):
    item = batch[0]
    return {
        'lam_init_p': item['lam_init_p'].unsqueeze(0),        # [1, 1, 128, 128, 128]
        'lam_init_s': item['lam_init_s'].unsqueeze(0),        # [1, 1, 128, 128, 128]
        'x_init_nac': item['x_init_nac'].unsqueeze(0),        # [1, 1, 128, 128, 128]
        'x_gt': item['x_gt'].unsqueeze(0),                    # [1, 1, 128, 128, 128]
        'mu_gt': item['mu_gt'].unsqueeze(0),                  # [1, 1, 128, 128, 128]
        'support_mask': item['support_mask'].unsqueeze(0),    # [1, 1, 128, 128, 128]
        'proj_primary': item['proj_primary'].unsqueeze(0),    # [1, N, 128, 128]
        'scatter_est': item['scatter_est'].unsqueeze(0),      # [1, N, 128, 128]
        'angles': item['angles'].unsqueeze(0),                # [1, N]
        'radii': item['radii'].unsqueeze(0),                  # [1, N]
        'dr': item['dr'].unsqueeze(0),                        # [1, 3]
        'projection_dr': item['projection_dr'].unsqueeze(0),  # [1, 2]
        'cond': item['cond'].unsqueeze(0),                    # [1, 4]
        'vendor': item['vendor'],
        'case_id': item['case_id'],
        'patient_id': item['patient_id']
    }


def compute_eval_metrics(
    x_pred: np.ndarray,
    x_gt: np.ndarray,
    mu_pred: np.ndarray,
    mu_gt: np.ndarray,
    support_mask: np.ndarray
) -> dict:
    """Computes comprehensive Activity & Attenuation validation metrics."""
    body_mask = support_mask > 0.5
    
    # 1. Activity PSNR & SSIM inside body mask
    diff_x = x_pred[body_mask] - x_gt[body_mask]
    mse_x = float(np.mean(diff_x ** 2)) if np.sum(body_mask) > 10 else 1.0
    max_x = float(np.max(x_gt[body_mask])) if np.sum(body_mask) > 10 else 100.0
    act_psnr = float(10 * np.log10((max_x ** 2) / max(mse_x, 1e-10)))

    # Activity SSIM sampled across axial slices
    valid_z = [z for z in range(x_gt.shape[0]) if np.sum(body_mask[z]) > 20]
    sample_z = valid_z[::max(1, len(valid_z) // 16)]
    x_ssims = []
    for z in sample_z:
        s = ssim(x_pred[z], x_gt[z], data_range=max_x)
        x_ssims.append(s)
    act_ssim = float(np.mean(x_ssims)) if x_ssims else 0.85

    # 2. Attenuation Metrics
    bone_mask = (mu_gt > 0.175) & body_mask
    lung_mask = (mu_gt > 0.02) & (mu_gt < 0.12) & body_mask
    soft_mask = (mu_gt >= 0.12) & (mu_gt <= 0.175) & body_mask

    diff_mu = mu_pred[body_mask] - mu_gt[body_mask]
    mse_mu = float(np.mean(diff_mu ** 2)) if np.sum(body_mask) > 10 else 1e-4
    max_mu = 0.35
    mu_psnr = float(10 * np.log10((max_mu ** 2) / max(mse_mu, 1e-12)))

    bone_mae = float(np.mean(np.abs(mu_pred[bone_mask] - mu_gt[bone_mask]))) if np.sum(bone_mask) > 10 else 0.0
    lung_mae = float(np.mean(np.abs(mu_pred[lung_mask] - mu_gt[lung_mask]))) if np.sum(lung_mask) > 10 else 0.0
    soft_mae = float(np.mean(np.abs(mu_pred[soft_mask] - mu_gt[soft_mask]))) if np.sum(soft_mask) > 10 else 0.0

    return {
        'act_psnr': round(act_psnr, 2),
        'act_ssim': round(act_ssim, 4),
        'mu_psnr': round(mu_psnr, 2),
        'bone_mae': round(bone_mae, 4),
        'lung_mae': round(lung_mae, 4),
        'soft_mae': round(soft_mae, 4)
    }


def parse_args():
    parser = argparse.ArgumentParser(description="EndToEndPhysicsUnrollSPECTNet Full-Cohort Training")
    parser.add_argument("--epochs", type=int, default=50, help="Total training epochs")
    parser.add_argument("--num-stages", type=int, default=4, help="Number of unrolled stages")
    parser.add_argument("--base-ch", type=int, default=16, help="Base channel width")
    parser.add_argument("--batch-size", type=int, default=1, help="Must be 1 for SPECT physics")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--poisson_weight", "--poisson-weight", type=float, default=1e-3, help="Physical Poisson projection loss weight")
    parser.add_argument("--projector_type", "--projector-type", type=str, default="nmrecon", choices=["nmrecon", "pytomography"], help="Physics projector backend")
    parser.add_argument("--output-dir", type=str, default=os.path.join(CTLESS_DIR, "experiments_true_unroll_ge_nii"))
    parser.add_argument("--cache-dir", type=str, default=os.path.join(CTLESS_DIR, "data", "cache_ge_nii"))
    parser.add_argument("--pretrained-mu-ckpt", type=str,
                        default="/home/wengfenghua/sparseSPECT/0913ctless/new1022/checkpoints/ge_planB_v2_unrolled_balanced/checkpoint_best_unrolled.pt")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    def log_print(msg: str):
        t_str = time.strftime("[%Y-%m-%d %H:%M:%S]")
        formatted = f"{t_str} {msg}"
        print(formatted, flush=True)
        with open(log_path, "a") as f:
            f.write(formatted + "\n")

    log_print("=" * 85)
    log_print("   EndToEndPhysicsUnrollSPECTNet 全量端到端联合展开训练工程 (GE 1,527例)")
    log_print("=" * 85)
    log_print(f"训练设备: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")
    log_print(f"输出目录: {out_dir}")
    log_print(f"网络配置: Stages={args.num_stages}, BaseCh={args.base_ch}, ShareWeights=True, NetDepth=4")
    log_print(f"超参数: Epochs={args.epochs}, LR={args.lr}, AccumSteps={args.grad_accum_steps}")

    # 1. Dataset & Loaders
    train_split_path = os.path.join(SPLITS_DIR, "ge_nii_train.json")
    val_split_path = os.path.join(SPLITS_DIR, "ge_nii_val.json")

    train_dataset = GENiiFullUnrollDataset(train_split_path, args.cache_dir)
    val_dataset = GENiiFullUnrollDataset(val_split_path, args.cache_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    log_print(f"训练集规模: {len(train_dataset)} 例 | 验证集规模: {len(val_dataset)} 例")

    # 2. Build Model
    model = EndToEndPhysicsUnrollSPECTNet(
        num_stages=args.num_stages,
        base_ch=args.base_ch,
        cond_dim=4,
        share_weights=True,
        net_depth=4
    ).to(device)

    # Load pretrained mu weights if available
    start_epoch = 1
    best_val_act_psnr = -float('inf')
    metrics_history = []

    best_ckpt_path = ckpt_dir / "checkpoint_best.pt"
    latest_ckpt_path = best_ckpt_path if best_ckpt_path.exists() else (ckpt_dir / "checkpoint_latest.pt")
    ckpt_resumed = None
    if args.resume and latest_ckpt_path.exists():
        log_print(f"Resuming training from checkpoint: {latest_ckpt_path}")
        ckpt_resumed = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resumed['model'])
        start_epoch = ckpt_resumed['epoch'] + 1
        best_val_act_psnr = ckpt_resumed.get('best_val_act_psnr', -float('inf'))
        metrics_history = ckpt_resumed.get('metrics_history', [])
        if metrics_history:
            best_val_act_psnr = max([m.get('val_act_psnr', -float('inf')) for m in metrics_history] + [best_val_act_psnr])
        log_print(f"Resumed at epoch {start_epoch}, previous best Act PSNR: {best_val_act_psnr:.2f} dB")
    elif os.path.exists(args.pretrained_mu_ckpt):
        log_print(f"Loading pretrained mu-network weights from: {args.pretrained_mu_ckpt}")
        ckpt = torch.load(args.pretrained_mu_ckpt, map_location='cpu', weights_only=False)
        model_dict = model.state_dict()
        matched = {k: v for k, v in ckpt['model'].items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        log_print(f"Successfully loaded {len(matched)} / {len(model_dict)} pretrained weights (DualWindowMuInitNet + MuProxNet).")
        log_print("AdaptiveScaleActivityResUNet3D initialized with exact zero-init on conv_out (Physical OSEM identity mapping).")

    # 3. Loss & Optimizer (aligned 1:1 scale between activity and attenuation)
    criterion = TrueEndToEndSPECTLoss(
        activity_weight=1.0,
        atten_weight=5.0,
        poisson_weight=args.poisson_weight
    ).to(device)
    if args.projector_type == "nmrecon":
        if nmrecon_projector_from_batch is None:
            raise RuntimeError("Requested nmrecon projector backend, but failed to import nmrecon_training_projector!")
        projector_fn = nmrecon_projector_from_batch
    else:
        projector_fn = pytomography_projector_from_batch
    log_print(f"物理投影算子后端: {args.projector_type} ({projector_fn.__module__ if hasattr(projector_fn, '__module__') else 'custom'})")
    log_print(f"泊松投影损失权重 (poisson_weight): {args.poisson_weight}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    if ckpt_resumed is not None and 'scheduler' in ckpt_resumed:
        try:
            scheduler.load_state_dict(ckpt_resumed['scheduler'])
            log_print(f"Restored scheduler state, current LR: {scheduler.get_last_lr()[0]:.7f}")
        except Exception as e:
            log_print(f"Scheduler load warning: {e}, using fresh scheduler.")

    log_print("正式开启全量数据端到端深度展开训练流水线...")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()

        train_loss_sum = 0.0
        train_act_rel_sum = 0.0
        train_mu_loss_sum = 0.0
        n_batches = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            lam_p = batch['lam_init_p'].to(device)
            lam_s = batch['lam_init_s'].to(device)
            x_init_nac = batch['x_init_nac'].to(device)
            proj_p = batch['proj_primary'].to(device)
            scatter = batch['scatter_est'].to(device)
            cond = batch['cond'].to(device)
            x_gt = batch['x_gt'].to(device)
            mu_gt = batch['mu_gt'].to(device)
            support_mask = batch['support_mask'].to(device)

            # Per-patient geometry projector
            projector = projector_fn(batch, str(device))

            # Forward pass with checkpointing
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

            loss_dict = criterion(
                outputs=out,
                x_gt=x_gt,
                mu_gt=mu_gt,
                support_mask=support_mask,
                proj_gt=proj_p,
                scatter_est=scatter,
                projector_factory=projector
            )
            loss = loss_dict['loss'] / args.grad_accum_steps
            loss.backward()

            # 防假死梯度有效性核验 (Step 1 打印一次)
            if step == 1 and epoch == start_epoch:
                mu_grads = [p.grad.abs().mean().item() for net in model.mu_nets for p in net.parameters() if p.grad is not None]
                act_grads = [p.grad.abs().mean().item() for net in model.act_nets for p in net.parameters() if p.grad is not None]
                mu_norm = (sum(mu_grads) / len(mu_grads)) if mu_grads else None
                act_norm = (sum(act_grads) / len(act_grads)) if act_grads else None
                log_print("=" * 60)
                log_print(f"Check Grad -> mu_net: {mu_norm} ({len(mu_grads)} params with grad)")
                log_print(f"Check Grad -> act_net: {act_norm} ({len(act_grads)} params with grad)")
                log_print("=" * 60)

            train_loss_sum += loss.item() * args.grad_accum_steps
            train_act_rel_sum += loss_dict['activity_rel_l1'].item()
            train_mu_loss_sum += loss_dict['atten_loss'].item()

            if step % args.grad_accum_steps == 0 or step == n_batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0 or step == n_batches:
                damp0 = 0.5 * torch.sigmoid(model.stages[0].damping_alpha).item()
                log_print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:04d}/{n_batches:04d}] | "
                    f"Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                    f"ActRelL1: {loss_dict['activity_rel_l1'].item():.4f} | "
                    f"MuLoss: {loss_dict['atten_loss'].item():.4f} | "
                    f"Damp0: {damp0:.4f}"
                )

        scheduler.step()
        epoch_time = time.time() - t0
        avg_train_loss = train_loss_sum / n_batches
        avg_act_rel = train_act_rel_sum / n_batches
        avg_mu_loss = train_mu_loss_sum / n_batches

        log_print(
            f"--- Epoch {epoch:02d} Complete [{epoch_time:.1f}s] | "
            f"Train Loss: {avg_train_loss:.4f} | ActRelL1: {avg_act_rel:.4f} | MuLoss: {avg_mu_loss:.4f} ---"
        )

        # 4. Validation Loop (every epoch)
        model.eval()
        val_act_psnr_list = []
        val_act_ssim_list = []
        val_mu_psnr_list = []
        val_bone_mae_list = []
        val_lung_mae_list = []
        val_soft_mae_list = []

        log_print(f"Running Validation on {len(val_loader)} cases...")
        t_val0 = time.time()
        with torch.no_grad():
            for v_batch in val_loader:
                v_lam_p = v_batch['lam_init_p'].to(device)
                v_lam_s = v_batch['lam_init_s'].to(device)
                v_x_init_nac = v_batch['x_init_nac'].to(device)
                v_proj_p = v_batch['proj_primary'].to(device)
                v_scatter = v_batch['scatter_est'].to(device)
                v_cond = v_batch['cond'].to(device)

                v_projector = projector_fn(v_batch, str(device))
                v_out = model(
                    lam_init_p=v_lam_p,
                    lam_init_s=v_lam_s,
                    proj_primary=v_proj_p,
                    scatter_est=v_scatter,
                    projector_factory=v_projector,
                    cond=v_cond,
                    x_init_nac=v_x_init_nac,
                    use_checkpoint=False,
                    compute_projection_loss=False
                )

                x_pred_np = v_out['activity_final'][0, 0].cpu().numpy()
                mu_pred_np = v_out['mu_final'][0, 0].cpu().numpy()
                x_gt_np = v_batch['x_gt'][0, 0].cpu().numpy()
                mu_gt_np = v_batch['mu_gt'][0, 0].cpu().numpy()
                mask_np = v_batch['support_mask'][0, 0].cpu().numpy()

                v_metrics = compute_eval_metrics(x_pred_np, x_gt_np, mu_pred_np, mu_gt_np, mask_np)
                val_act_psnr_list.append(v_metrics['act_psnr'])
                val_act_ssim_list.append(v_metrics['act_ssim'])
                val_mu_psnr_list.append(v_metrics['mu_psnr'])
                val_bone_mae_list.append(v_metrics['bone_mae'])
                val_lung_mae_list.append(v_metrics['lung_mae'])
                val_soft_mae_list.append(v_metrics['soft_mae'])

        val_time = time.time() - t_val0
        mean_act_psnr = float(np.mean(val_act_psnr_list))
        mean_act_ssim = float(np.mean(val_act_ssim_list))
        mean_mu_psnr = float(np.mean(val_mu_psnr_list))
        mean_bone_mae = float(np.mean(val_bone_mae_list))
        mean_lung_mae = float(np.mean(val_lung_mae_list))
        mean_soft_mae = float(np.mean(val_soft_mae_list))

        log_print(
            f"=== Validation Epoch {epoch:02d} [{val_time:.1f}s] ===\n"
            f"    Activity -> PSNR: {mean_act_psnr:.2f} dB | SSIM: {mean_act_ssim:.4f}\n"
            f"    Attenuation -> PSNR: {mean_mu_psnr:.2f} dB | Bone MAE: {mean_bone_mae:.4f} | Lung MAE: {mean_lung_mae:.4f} | Soft MAE: {mean_soft_mae:.4f}"
        )

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(avg_train_loss, 4),
            'train_act_rel': round(avg_act_rel, 4),
            'train_mu_loss': round(avg_mu_loss, 4),
            'val_act_psnr': round(mean_act_psnr, 2),
            'val_act_ssim': round(mean_act_ssim, 4),
            'val_mu_psnr': round(mean_mu_psnr, 2),
            'val_bone_mae': round(mean_bone_mae, 4),
            'val_lung_mae': round(mean_lung_mae, 4),
            'val_soft_mae': round(mean_soft_mae, 4),
            'damp0': round(0.5 * torch.sigmoid(model.stages[0].damping_alpha).item(), 4),
            'lr': round(scheduler.get_last_lr()[0], 7)
        }
        metrics_history.append(epoch_record)

        # Save metrics json
        with open(out_dir / "metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)

        # Save latest checkpoint
        ckpt_latest = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_act_psnr': best_val_act_psnr,
            'metrics_history': metrics_history
        }
        torch.save(ckpt_latest, ckpt_dir / "checkpoint_latest.pt")

        # Save best checkpoint
        if mean_act_psnr > best_val_act_psnr:
            best_val_act_psnr = mean_act_psnr
            ckpt_latest['best_val_act_psnr'] = best_val_act_psnr
            torch.save(ckpt_latest, ckpt_dir / "checkpoint_best.pt")
            log_print(f">>> New Best Activity PSNR: {best_val_act_psnr:.2f} dB! Saved to checkpoint_best.pt")

    log_print("================================================================================")
    log_print(f" EndToEndPhysicsUnrollSPECTNet 全量训练圆满完成! Best Val Act PSNR: {best_val_act_psnr:.2f} dB")
    log_print("================================================================================")


if __name__ == '__main__':
    main()
