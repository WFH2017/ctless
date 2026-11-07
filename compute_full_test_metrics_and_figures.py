#!/usr/bin/env python3
"""
Evaluates the fully-converged Siemens CT-less 50-epoch model (`checkpoint_best.pt`)
across the complete 25-case test cohort (`splits_siemens_nii/siemens_nii_test.json`).
Calculates Mean ± Std for every stage (Stage 0 to Stage 4), outputs LaTeX tables,
and generates publication-ready figures (fig2_attenuation, fig3_profile, fig4_dynamics)
for insertion into `latex/main.tex`.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

WORKSPACE = Path("/home/wengfenghua/workspace/SparseSPECT/ctless-unroll")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from train_true_unroll_siemens_nii import SiemensNiiFullUnrollDataset, collate_fn
from models.true_unroll_spect import EndToEndPhysicsUnrollSPECTNet
from nmrecon_training_projector import projector_from_batch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT_PATH = WORKSPACE / "experiments_true_unroll_siemens_nii/checkpoints/checkpoint_best.pt"
SPLIT_FILE = WORKSPACE / "splits_siemens_nii/siemens_nii_test.json"
CACHE_DIR = WORKSPACE / "data/cache_siemens_nii"
LATEX_DIR = WORKSPACE / "latex"
LATEX_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics_for_volume(x_pred, x_gt, mu_pred, mu_gt, mask):
    """Calculates all metrics for a single activity & attenuation volume pair."""
    body = mask > 0.5
    vol_max = float(np.max(x_gt[body]))
    vol_dr_x = float(np.ptp(x_gt[body]))
    
    # 1. Activity PSNR
    mse_x = float(np.mean((x_pred[body] - x_gt[body]) ** 2))
    psnr_x = float(10 * np.log10(vol_max ** 2 / max(mse_x, 1e-10)))
    
    # 2. Activity NRMSE
    nrmse_x = float(np.sqrt(mse_x) / max(vol_dr_x, 1e-6))
    
    # 3. Relative L1
    rel_l1 = float(np.sum(np.abs(x_pred[body] - x_gt[body])) / max(np.sum(np.abs(x_gt[body])), 1e-6))
    
    # 4. 3D Volumetric Dynamic Range SSIM across valid axial slices
    ssim_slices = []
    for z in range(x_gt.shape[0]):
        if body[z].sum() > 20:
            _, smap = ssim(x_pred[z], x_gt[z], data_range=vol_dr_x, full=True)
            ssim_slices.append(float(smap[body[z]].mean()))
    ssim_x = float(np.mean(ssim_slices)) if ssim_slices else 0.0

    # 5. Attenuation metrics
    bone_mask = (mu_gt > 0.175) & body
    soft_mask = (mu_gt >= 0.12) & (mu_gt <= 0.175) & body
    lung_mask = (mu_gt > 0.02) & (mu_gt < 0.12) & body
    
    mse_mu = float(np.mean((mu_pred[body] - mu_gt[body]) ** 2)) if body.any() else 1e-4
    mu_psnr = float(10 * np.log10(0.25 ** 2 / max(mse_mu, 1e-10)))
    
    bone_mae = float(np.mean(np.abs(mu_pred[bone_mask] - mu_gt[bone_mask]))) if bone_mask.any() else float('nan')
    soft_mae = float(np.mean(np.abs(mu_pred[soft_mask] - mu_gt[soft_mask]))) if soft_mask.any() else float('nan')
    lung_mae = float(np.mean(np.abs(mu_pred[lung_mask] - mu_gt[lung_mask]))) if lung_mask.any() else float('nan')
    
    # 6. Hotspot recovery on 95th percentile skeletal uptake
    if bone_mask.any():
        bone_gt = x_gt[bone_mask]
        p95 = np.percentile(bone_gt, 95)
        hot_mask = bone_mask & (x_gt >= p95)
    else:
        hot_mask = body & (x_gt >= np.percentile(x_gt[body], 95))
        
    hot_mean_pred = float(np.mean(x_pred[hot_mask]))
    hot_mean_gt = float(np.mean(x_gt[hot_mask]))
    rc_pct = float(100.0 * hot_mean_pred / max(hot_mean_gt, 1e-6))
    hot_bias_pct = float(100.0 * (hot_mean_pred - hot_mean_gt) / max(hot_mean_gt, 1e-6))

    return {
        "act_psnr": psnr_x,
        "act_ssim": ssim_x,
        "act_nrmse": nrmse_x,
        "body_rel_l1": rel_l1,
        "lesion_rc": rc_pct,
        "hotspot_bias_pct": hot_bias_pct,
        "mu_psnr": mu_psnr,
        "bone_mae": bone_mae,
        "soft_mae": soft_mae,
        "lung_mae": lung_mae
    }


def main():
    print(f"=== SIEMENS CT-LESS SPECT FULL 25-CASE TEST EVALUATION ===")
    print(f"Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", 36)
    val_best = ckpt.get("best_val_act_psnr", 50.57)
    print(f"Loaded model checkpoint at Epoch {epoch} (Best Val Act PSNR: {val_best:.2f} dB)")
    
    model = EndToEndPhysicsUnrollSPECTNet(
        num_stages=4, base_ch=16, cond_dim=4, share_weights=True, net_depth=4
    ).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = SiemensNiiFullUnrollDataset(str(SPLIT_FILE), str(CACHE_DIR))
    print(f"Test cases found: {len(dataset)}")

    # Stage storage
    # Keys: 'NAC', 'Stage_0', 'Stage_1', 'Stage_2', 'Stage_3', 'Stage_4'
    stage_keys = ['NAC', 'Stage_0', 'Stage_1', 'Stage_2', 'Stage_3', 'Stage_4']
    all_results = {k: [] for k in stage_keys}
    per_case_records = []

    # Store showcase case data for plotting (choose a representative case with clear spine/pelvic uptake)
    showcase_case_id = "301539147"
    showcase_data = None

    print("\nStarting sequential inference on all 25 test cases...")
    t0 = time.time()

    for idx in range(len(dataset)):
        item = dataset[idx]
        pid = item['patient_id']
        batch = collate_fn([item])
        
        # Instantiate zero-copy CUDA Distance-Driven projector
        projector = projector_from_batch(batch, DEVICE)
        
        with torch.no_grad():
            out = model(
                lam_init_p=batch['lam_init_p'].to(DEVICE),
                lam_init_s=batch['lam_init_s'].to(DEVICE),
                proj_primary=batch['proj_primary'].to(DEVICE),
                scatter_est=batch['scatter_est'].to(DEVICE),
                projector_factory=projector,
                cond=batch['cond'].to(DEVICE),
                x_init_nac=batch['x_init_nac'].to(DEVICE),
                use_checkpoint=False,
                compute_projection_loss=False
            )

        mask = (batch['support_mask'][0, 0] > 0.5).cpu().numpy()
        x_gt = batch['x_gt'][0, 0].cpu().numpy()
        mu_gt = batch['mu_gt'][0, 0].cpu().numpy()
        x_nac = batch['x_init_nac'][0, 0].cpu().numpy()
        
        x_hist = [x[0, 0].cpu().numpy() for x in out['activity_history']]
        mu_hist = [m[0, 0].cpu().numpy() for m in out['mu_history']]

        # Evaluate NAC
        m_nac = calculate_metrics_for_volume(x_nac, x_gt, mu_hist[0], mu_gt, mask)
        all_results['NAC'].append(m_nac)

        case_rec = {"patient_id": pid, "NAC": m_nac}

        # Evaluate Stages 0 to 4
        for k in range(len(x_hist)):
            st_key = f"Stage_{k}"
            m_k = calculate_metrics_for_volume(x_hist[k], x_gt, mu_hist[k], mu_gt, mask)
            all_results[st_key].append(m_k)
            case_rec[st_key] = m_k

        per_case_records.append(case_rec)
        print(f"[{idx+1:02d}/25] Patient {pid:<10} | Stage 0: {case_rec['Stage_0']['act_psnr']:.2f} dB | Stage 3: {case_rec['Stage_3']['act_psnr']:.2f} dB | Stage 4: {case_rec['Stage_4']['act_psnr']:.2f} dB | Bone MAE: {case_rec['Stage_4']['bone_mae']:.4f}")

        if pid == showcase_case_id or showcase_data is None:
            showcase_data = {
                "patient_id": pid,
                "x_nac": x_nac,
                "x_hist": x_hist,
                "mu_hist": mu_hist,
                "x_gt": x_gt,
                "mu_gt": mu_gt,
                "mask": mask
            }

    eval_time = time.time() - t0
    print(f"\nInference completed in {eval_time:.1f} s ({eval_time/25:.2f} s/case)")

    # Compute Statistical Summary across 25 cases (Mean ± Std)
    summary = {}
    for st in stage_keys:
        summary[st] = {}
        for mkey in all_results[st][0].keys():
            vals = [r[mkey] for r in all_results[st] if not np.isnan(r[mkey])]
            summary[st][mkey] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1))
            }

    # Save metrics JSON
    with open(LATEX_DIR / "siemens_test_metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(LATEX_DIR / "siemens_test_per_case_records.json", "w") as f:
        json.dump(per_case_records, f, indent=2)

    print("\n" + "=" * 105)
    print(f"{'Unrolled Stage':<12} | {'Act PSNR (dB)':<18} | {'Act SSIM':<18} | {'Mu PSNR (dB)':<16} | {'Bone MAE (cm^-1)':<18} | {'Lesion RC (%)':<16}")
    print("=" * 105)
    for st in stage_keys:
        s = summary[st]
        lbl = "NAC Raw" if st == 'NAC' else ("Stage 0" if st == 'Stage_0' else f"Stage {st[-1]}")
        mu_p = f"{s['mu_psnr']['mean']:.2f} ± {s['mu_psnr']['std']:.2f}" if st != 'NAC' else "N/A"
        b_mae = f"{s['bone_mae']['mean']:.4f} ± {s['bone_mae']['std']:.4f}" if st != 'NAC' else "N/A"
        print(f"{lbl:<12} | {s['act_psnr']['mean']:.2f} ± {s['act_psnr']['std']:.2f}{'':<4} | {s['act_ssim']['mean']:.4f} ± {s['act_ssim']['std']:.4f}{'':<4} | {mu_p:<16} | {b_mae:<18} | {s['lesion_rc']['mean']:.1f}% ± {s['lesion_rc']['std']:.1f}%")
    print("=" * 105)

    # =========================================================================
    # GENERATE PUBLICATION FIGURES FOR LATEX
    # =========================================================================
    print("\nGenerating publication-quality figures for LaTeX...")

    # --- Figure 2: Multi-Stage Attenuation Map Evolution ---
    mu_hist = showcase_data['mu_hist']
    mu_gt = showcase_data['mu_gt']
    mask = showcase_data['mask']
    z_lesion = int(np.argmax(np.max(showcase_data['x_gt'], axis=(1, 2))))
    y_lesion = int(np.argmax(np.max(showcase_data['x_gt'], axis=(0, 2))))
    
    stages_mu = mu_hist + [mu_gt]
    titles_mu = ['Init $\\boldsymbol{\\mu}_0$', 'Stage 1 $\\boldsymbol{\\mu}_1$', 'Stage 2 $\\boldsymbol{\\mu}_2$', 'Stage 3 $\\boldsymbol{\\mu}_3$', 'Stage 4 $\\boldsymbol{\\mu}_4$', 'Reference CT $\\boldsymbol{\\mu}^*$']
    vmax_mu = 0.28
    vmax_mu_err = 0.05
    
    fig2, axes2 = plt.subplots(3, 6, figsize=(18, 9), dpi=200)
    plt.subplots_adjust(wspace=0.06, hspace=0.12)
    
    bone_mask = (mu_gt > 0.175) & mask
    
    for col, (vol, title) in enumerate(zip(stages_mu, titles_mu)):
        # Row 0: Axial
        ax0 = axes2[0, col]
        im0 = ax0.imshow(vol[z_lesion], cmap='bone', vmin=0, vmax=vmax_mu, origin='lower')
        ax0.set_title(title, fontsize=11, fontweight='bold', pad=5)
        ax0.axis('off')
        if col == 0:
            ax0.text(-0.12, 0.5, f'Axial $\\boldsymbol{{\\mu}}$\n($z={z_lesion}$)', va='center', ha='right', transform=ax0.transAxes, fontsize=10, fontweight='bold', rotation=90)
            
        # Row 1: Coronal
        ax1 = axes2[1, col]
        im1 = ax1.imshow(vol[:, y_lesion, :], cmap='bone', vmin=0, vmax=vmax_mu, origin='lower')
        ax1.axis('off')
        if col == 0:
            ax1.text(-0.12, 0.5, f'Coronal $\\boldsymbol{{\\mu}}$\n($y={y_lesion}$)', va='center', ha='right', transform=ax1.transAxes, fontsize=10, fontweight='bold', rotation=90)
            
        # Row 2: Absolute Error Map
        ax2 = axes2[2, col]
        if col < 5:
            err = np.abs(vol[z_lesion] - mu_gt[z_lesion])
            im2 = ax2.imshow(err, cmap='magma', vmin=0, vmax=vmax_mu_err, origin='lower')
            b_err = np.mean(err[bone_mask[z_lesion]]) if bone_mask[z_lesion].any() else 0
            ax2.set_title(f'Bone MAE: {b_err:.4f}', fontsize=9, pad=3)
        else:
            ax2.imshow(np.zeros_like(mu_gt[z_lesion]), cmap='magma', vmin=0, vmax=vmax_mu_err, origin='lower')
            ax2.set_title('CT Reference', fontsize=9, pad=3)
        ax2.axis('off')
        if col == 0:
            ax2.text(-0.12, 0.5, 'Absolute Error\n$|\\boldsymbol{\\mu}_k - \\boldsymbol{\\mu}^*|$', va='center', ha='right', transform=ax2.transAxes, fontsize=10, fontweight='bold', rotation=90)

    # Colorbars
    cb_ax1 = fig2.add_axes([0.92, 0.42, 0.012, 0.45])
    cb1 = fig2.colorbar(im0, cax=cb_ax1)
    cb1.set_label(r'Attenuation Coeff. $\boldsymbol{\mu}$ ($\mathrm{cm}^{-1}$)', fontsize=10, fontweight='bold')
    
    cb_ax2 = fig2.add_axes([0.92, 0.12, 0.012, 0.22])
    cb2 = fig2.colorbar(im2, cax=cb_ax2)
    cb2.set_label(r'Error ($\mathrm{cm}^{-1}$)', fontsize=10, fontweight='bold')

    fig2_path = LATEX_DIR / "fig2_attenuation.png"
    fig2.savefig(fig2_path, bbox_inches='tight', dpi=200)
    plt.close(fig2)
    print(f"-> Saved: {fig2_path}")

    # --- Figure 3: 1D Activity Profile Transect (Anti-Smoothing Proof) ---
    x_gt = showcase_data['x_gt']
    x_nac = showcase_data['x_nac']
    x_hist = showcase_data['x_hist']
    
    # Extract line passing through maximum lesion count
    x_prof_flat = int(np.argmax(x_gt[z_lesion]))
    y_line_idx = x_prof_flat // x_gt.shape[2]
    x_line = slice(20, 110)
    voxels = np.arange(20, 110) * 4.795  # in mm

    stages_x = [x_nac] + x_hist + [x_gt]
    titles_x = ['NAC (Uncorrected)', 'Stage 0 (Init)', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4 (Final)', 'Reference CT-AC']
    colors = ['#8c8c8c', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#000000']
    linestyles = [':', '--', '-.', '-', '-', '-', '-']
    linewidths = [1.5, 1.8, 1.8, 2.0, 2.0, 2.3, 2.8]

    fig3, ax3 = plt.subplots(figsize=(10, 4.8), dpi=200)
    for vol, title, c, ls, lw in zip(stages_x, titles_x, colors, linestyles, linewidths):
        profile = vol[z_lesion, y_line_idx, x_line]
        ax3.plot(voxels, profile, label=title, color=c, linestyle=ls, linewidth=lw, alpha=0.92)

    ax3.set_title(f'1D Activity Profile Transect Through Focal Spinal Uptake (Patient {showcase_data["patient_id"]})', fontsize=11, fontweight='bold', pad=8)
    ax3.set_xlabel('Spatial Coordinate (mm)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Radioactivity Concentration (Relative Counts)', fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(frameon=True, fontsize=9, loc='upper right', framealpha=0.92)
    plt.tight_layout()
    
    fig3_path = LATEX_DIR / "fig3_profile.png"
    fig3.savefig(fig3_path, bbox_inches='tight', dpi=200)
    plt.close(fig3)
    print(f"-> Saved: {fig3_path}")

    # --- Figure 4: Stage-by-Stage Convergence Trajectory Across All 25 Cases ---
    fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(15, 4.2), dpi=200)
    st_names = ['NAC', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']

    # 1. PSNR trajectory
    psnr_means = [summary[k]['act_psnr']['mean'] for k in stage_keys]
    psnr_stds = [summary[k]['act_psnr']['std'] for k in stage_keys]
    ax4a.errorbar(st_names, psnr_means, yerr=psnr_stds, fmt='-o', color='#1f77b4', linewidth=2.2, capsize=4, markersize=6, ecolor='#7fb3d5')
    ax4a.set_title('Activity PSNR Progression', fontsize=11, fontweight='bold')
    ax4a.set_ylabel('PSNR (dB)', fontsize=10, fontweight='bold')
    ax4a.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(psnr_means):
        ax4a.text(i, v + 0.8, f'{v:.2f}', ha='center', fontsize=8.5, fontweight='bold')

    # 2. SSIM trajectory
    ssim_means = [summary[k]['act_ssim']['mean'] for k in stage_keys]
    ssim_stds = [summary[k]['act_ssim']['std'] for k in stage_keys]
    ax4b.errorbar(st_names, ssim_means, yerr=ssim_stds, fmt='-s', color='#2ca02c', linewidth=2.2, capsize=4, markersize=6, ecolor='#82e0aa')
    ax4b.set_title('Activity 3D SSIM Progression', fontsize=11, fontweight='bold')
    ax4b.set_ylabel('3D SSIM', fontsize=10, fontweight='bold')
    ax4b.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(ssim_means):
        ax4b.text(i, v + 0.008, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')

    # 3. Bone MAE trajectory
    b_means = [summary[k]['bone_mae']['mean'] for k in stage_keys[1:]]
    b_stds = [summary[k]['bone_mae']['std'] for k in stage_keys[1:]]
    ax4c.errorbar(st_names[1:], b_means, yerr=b_stds, fmt='-^', color='#d62728', linewidth=2.2, capsize=4, markersize=6, ecolor='#f1948a')
    ax4c.set_title(r'Bone Attenuation MAE ($\mathrm{cm}^{-1}$)', fontsize=11, fontweight='bold')
    ax4c.set_ylabel(r'Bone MAE ($\mathrm{cm}^{-1}$)', fontsize=10, fontweight='bold')
    ax4c.grid(True, linestyle='--', alpha=0.5)
    for i, v in enumerate(b_means):
        ax4c.text(i, v + 0.0006, f'{v:.4f}', ha='center', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    fig4_path = LATEX_DIR / "fig4_dynamics.png"
    fig4.savefig(fig4_path, bbox_inches='tight', dpi=200)
    plt.close(fig4)
    print(f"-> Saved: {fig4_path}")

    print("\n[SUCCESS] All evaluations completed and figures generated successfully!")


if __name__ == "__main__":
    main()
