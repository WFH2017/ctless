#!/usr/bin/env python3
"""
Creates a fully self-contained, end-to-end executable Jupyter Notebook:
`inspect_multistage_unroll_case.ipynb`
Contains all code from data loading to projector creation, model inference,
metrics calculation, multi-panel plotting, and interactive slider.
Pre-renders on test case 301539147 where Stage 4 strictly outperforms earlier stages.
"""

import os
import sys
import json
import base64
from pathlib import Path

WORKSPACE = Path("/home/wengfenghua/workspace/SparseSPECT/ctless-unroll")
NB_PATH = WORKSPACE / "inspect_multistage_unroll_case.ipynb"

# Create clean, well-documented cells
cells = []

# =========================================================================
# Cell 1: Header & Instructions
# =========================================================================
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# EndToEndPhysicsUnrollSPECTNet: Multi-Stage Unrolling Clinical Case Inspector\n",
        "\n",
        "This notebook provides an **end-to-end interactive workspace** to inspect the internal stage-by-stage evolution of the joint CT-less SPECT unrolling network.\n",
        "\n",
        "### How to Use:\n",
        "1. In **Cell 2**, set `PATIENT_ID` to any patient ID from the test/validation set (e.g., `'301539147'`, `'300210121'`, `'301415729'`, etc.).\n",
        "2. Click **Run All** (or run cell-by-cell).\n",
        "3. In seconds, the notebook loads the geometry, executes all 4 unrolled stages via the zero-copy CUDA `libnmrecon.so` physics engine, prints the quantitative evaluation table, and renders the high-resolution comparison figures!\n",
        "\n",
        "---"
    ]
})

# =========================================================================
# Cell 2: Imports & Environment Configuration
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%matplotlib inline\n",
        "import os\n",
        "import sys\n",
        "import json\n",
        "from pathlib import Path\n",
        "\n",
        "import torch\n",
        "import numpy as np\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "from IPython.display import display, Image\n",
        "matplotlib.rcParams['figure.dpi'] = 140\n",
        "from skimage.metrics import structural_similarity as ssim\n",
        "\n",
        "# Ensure ctless-unroll repository is in python path\n",
        "WORKSPACE = Path('/home/wengfenghua/workspace/SparseSPECT/ctless-unroll')\n",
        "if str(WORKSPACE) not in sys.path:\n",
        "    sys.path.insert(0, str(WORKSPACE))\n",
        "\n",
        "from models.true_unroll_spect import EndToEndPhysicsUnrollSPECTNet\n",
        "from train_true_unroll_siemens_nii import SiemensNiiFullUnrollDataset, collate_fn\n",
        "from nmrecon_training_projector import projector_from_batch\n",
        "\n",
        "print(f'PyTorch Version: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
    ]
})

# =========================================================================
# Cell 3: Patient Case Selection
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# =========================================================================\n",
        "# 🎯 CASE CONFIGURATION: CHANGE THIS TO ANY CASE YOU WANT TO INSPECT!\n",
        "# =========================================================================\n",
        "PATIENT_ID = '301539147'  # Default: Strictly monotonic Stage 0 -> 4 progression\n",
        "SPLIT_TYPE = 'test'       # 'test' (25 cases), 'val' (25 cases), or 'train' (185 cases)\n",
        "DEVICE = os.environ.get('CUDA_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')\n",
        "CKPT_PATH = str(WORKSPACE / 'checkpoints_v1_backup/checkpoint_best_48.61dB.pt')\n",
        "\n",
        "# Load test split to display available candidate IDs for copy-paste\n",
        "split_file = WORKSPACE / f'splits_siemens_nii/siemens_nii_{SPLIT_TYPE}.json'\n",
        "with open(split_file) as f:\n",
        "    available_cases = [c['patient_id'] for c in json.load(f)['cases']]\n",
        "\n",
        "print(f'Selected Target Patient ID: {PATIENT_ID} on {DEVICE}')\n",
        "print(f'Total {SPLIT_TYPE} cases available ({len(available_cases)}):')\n",
        "print(', '.join(available_cases[:12]) + ', ...')"
    ]
})

# =========================================================================
# Cell 4: Data Loading & Projector Setup
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "cache_dir = WORKSPACE / 'data/cache_siemens_nii'\n",
        "dataset = SiemensNiiFullUnrollDataset(str(split_file), str(cache_dir))\n",
        "\n",
        "# Find selected case\n",
        "matching_indices = [i for i, c in enumerate(dataset.valid_cases) if c['patient_id'] == PATIENT_ID]\n",
        "if not matching_indices:\n",
        "    raise ValueError(f'Patient ID {PATIENT_ID} not found in {SPLIT_TYPE} split! Available: {available_cases[:5]}...')\n",
        "\n",
        "case_idx = matching_indices[0]\n",
        "batch = collate_fn([dataset[case_idx]])\n",
        "\n",
        "# Build native zero-copy Distance-Driven projector via NMRecon\n",
        "projector = projector_from_batch(batch, DEVICE)\n",
        "\n",
        "print(f'Successfully loaded case {PATIENT_ID} (Index {case_idx})!')\n",
        "print(f'Primary projection views: {batch[\"proj_primary\"].shape[1]} | Matrix: {tuple(batch[\"x_gt\"].shape[2:])}')"
    ]
})

# =========================================================================
# Cell 5: Model Loading & Forward Unrolling
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(f'Loading model from checkpoint: {CKPT_PATH}...')\n",
        "ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)\n",
        "\n",
        "model = EndToEndPhysicsUnrollSPECTNet(\n",
        "    num_stages=4, base_ch=16, cond_dim=4, share_weights=True, net_depth=4\n",
        ").to(DEVICE)\n",
        "model.load_state_dict(ckpt['model'])\n",
        "model.eval()\n",
        "\n",
        "print('Executing 4-stage physics-unrolled forward pass...')\n",
        "with torch.no_grad():\n",
        "    out = model(\n",
        "        lam_init_p=batch['lam_init_p'].to(DEVICE),\n",
        "        lam_init_s=batch['lam_init_s'].to(DEVICE),\n",
        "        proj_primary=batch['proj_primary'].to(DEVICE),\n",
        "        scatter_est=batch['scatter_est'].to(DEVICE),\n",
        "        projector_factory=projector,\n",
        "        cond=batch['cond'].to(DEVICE),\n",
        "        x_init_nac=batch['x_init_nac'].to(DEVICE),\n",
        "        use_checkpoint=False,\n",
        "        compute_projection_loss=False\n",
        "    )\n",
        "\n",
        "print(f'Forward unrolling finished! Extracted {len(out[\"activity_history\"])} activity and {len(out[\"mu_history\"])} mu stages.')"
    ]
})

# =========================================================================
# Cell 6: Metrics Table Computation
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "mask = (batch['support_mask'][0, 0] > 0.5).cpu().numpy()\n",
        "x_nac = batch['x_init_nac'][0, 0].cpu().numpy()\n",
        "x_gt = batch['x_gt'][0, 0].cpu().numpy()\n",
        "mu_gt = batch['mu_gt'][0, 0].cpu().numpy()\n",
        "\n",
        "x_hist = [x[0, 0].cpu().numpy() for x in out['activity_history']]\n",
        "mu_hist = [m[0, 0].cpu().numpy() for m in out['mu_history']]\n",
        "\n",
        "max_x = float(np.max(x_gt[mask]))\n",
        "vol_dr_x = float(np.ptp(x_gt[mask]))\n",
        "\n",
        "bone_mask = (mu_gt > 0.175) & mask\n",
        "soft_mask = (mu_gt >= 0.12) & (mu_gt <= 0.175) & mask\n",
        "lung_mask = (mu_gt > 0.02) & (mu_gt < 0.12) & mask\n",
        "\n",
        "metrics = []\n",
        "# NAC Baseline\n",
        "mse_nac = float(np.mean((x_nac[mask] - x_gt[mask])**2))\n",
        "psnr_nac = float(10 * np.log10(max_x**2 / max(mse_nac, 1e-10)))\n",
        "ssim_slices_nac = []\n",
        "for z in range(x_nac.shape[0]):\n",
        "    if mask[z].sum() > 20:\n",
        "        _, smap = ssim(x_nac[z], x_gt[z], data_range=vol_dr_x, full=True)\n",
        "        ssim_slices_nac.append(float(smap[mask[z]].mean()))\n",
        "ssim_nac = float(np.mean(ssim_slices_nac)) if ssim_slices_nac else 0.0\n",
        "metrics.append({\n",
        "    'stage': 'NAC (Raw Unattenuated)',\n",
        "    'act_psnr': round(psnr_nac, 2),\n",
        "    'act_ssim': round(ssim_nac, 4),\n",
        "    'mu_psnr': float('nan'),\n",
        "    'bone_mae': float('nan'),\n",
        "    'soft_mae': float('nan')\n",
        "})\n",
        "\n",
        "for k in range(len(x_hist)):\n",
        "    xk = x_hist[k]\n",
        "    muk = mu_hist[k]\n",
        "    \n",
        "    mse_x = float(np.mean((xk[mask] - x_gt[mask])**2))\n",
        "    psnr_x = float(10 * np.log10(max_x**2 / max(mse_x, 1e-10)))\n",
        "    \n",
        "    # 3D global dynamic range SSIM across axial slices\n",
        "    ssim_slices = []\n",
        "    for z in range(xk.shape[0]):\n",
        "        if mask[z].sum() > 20:\n",
        "            _, smap = ssim(xk[z], x_gt[z], data_range=vol_dr_x, full=True)\n",
        "            ssim_slices.append(float(smap[mask[z]].mean()))\n",
        "    ssim_x = float(np.mean(ssim_slices)) if ssim_slices else 0.0\n",
        "    \n",
        "    mse_mu = float(np.mean((muk[mask] - mu_gt[mask])**2))\n",
        "    psnr_mu = float(10 * np.log10(0.25**2 / max(mse_mu, 1e-10)))\n",
        "    b_mae = float(np.mean(np.abs(muk[bone_mask] - mu_gt[bone_mask]))) if bone_mask.any() else float('nan')\n",
        "    s_mae = float(np.mean(np.abs(muk[soft_mask] - mu_gt[soft_mask]))) if soft_mask.any() else float('nan')\n",
        "    \n",
        "    lbl = 'Stage 0 (Chang Initial)' if k == 0 else f'Stage {k}'\n",
        "    metrics.append({\n",
        "        'stage': lbl,\n",
        "        'act_psnr': round(psnr_x, 2),\n",
        "        'act_ssim': round(ssim_x, 4),\n",
        "        'mu_psnr': round(psnr_mu, 2),\n",
        "        'bone_mae': round(b_mae, 4),\n",
        "        'soft_mae': round(s_mae, 4)\n",
        "    })\n",
        "\n",
        "# Print formatted table\n",
        "header = f\"{'Stage':<24} | {'Act PSNR (dB)':<14} | {'Act SSIM':<10} | {'Mu PSNR (dB)':<13} | {'Bone MAE':<10} | {'Soft MAE':<10}\"\n",
        "print('=' * len(header))\n",
        "print(f'   STAGE-BY-STAGE EVALUATION METRICS FOR PATIENT: {PATIENT_ID}')\n",
        "print('=' * len(header))\n",
        "print(header)\n",
        "print('-' * len(header))\n",
        "for m in metrics:\n",
        "    print(f\"{m['stage']:<24} | {str(m['act_psnr']):<14} | {str(m['act_ssim']):<10} | {str(m['mu_psnr']):<13} | {str(m['bone_mae']):<10} | {str(m['soft_mae']):<10}\")\n",
        "print('=' * len(header))"
    ]
})

# =========================================================================
# Cell 7: Figure 1 - Activity Evolution
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Identify slice with peak lesion/vertebral uptake\n",
        "z_lesion = int(np.argmax(np.max(x_gt, axis=(1, 2))))\n",
        "y_lesion = int(np.argmax(np.max(x_gt, axis=(0, 2))))\n",
        "\n",
        "stages_x = [x_nac] + x_hist + [x_gt]\n",
        "titles_x = ['NAC (Raw)', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4 (Final)', 'Ground Truth (CT-AC)']\n",
        "\n",
        "vmax_act = float(np.percentile(x_gt[mask], 99.8))\n",
        "vmax_err = vmax_act * 0.35\n",
        "\n",
        "fig, axes = plt.subplots(4, 7, figsize=(24, 14), dpi=140)\n",
        "plt.subplots_adjust(wspace=0.08, hspace=0.15)\n",
        "\n",
        "for col, (vol, title) in enumerate(zip(stages_x, titles_x)):\n",
        "    # Row 0: Axial\n",
        "    ax0 = axes[0, col]\n",
        "    im0 = ax0.imshow(vol[z_lesion], cmap='magma', vmin=0, vmax=vmax_act, origin='lower')\n",
        "    ax0.set_title(title, fontsize=12, fontweight='bold', pad=6)\n",
        "    ax0.axis('off')\n",
        "    if col == 0:\n",
        "        ax0.text(-0.15, 0.5, f'Axial Slice\\n(z={z_lesion})', va='center', ha='right', transform=ax0.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "        \n",
        "    # Row 1: Coronal\n",
        "    ax1 = axes[1, col]\n",
        "    im1 = ax1.imshow(vol[:, y_lesion, :], cmap='magma', vmin=0, vmax=vmax_act, origin='lower')\n",
        "    ax1.axis('off')\n",
        "    if col == 0:\n",
        "        ax1.text(-0.15, 0.5, f'Coronal Slice\\n(y={y_lesion})', va='center', ha='right', transform=ax1.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "        \n",
        "    # Row 2: MIP\n",
        "    ax2 = axes[2, col]\n",
        "    im2 = ax2.imshow(np.max(vol, axis=1), cmap='magma', vmin=0, vmax=vmax_act * 1.3, origin='lower')\n",
        "    ax2.axis('off')\n",
        "    if col == 0:\n",
        "        ax2.text(-0.15, 0.5, 'Maximum Intensity\\nProjection (MIP)', va='center', ha='right', transform=ax2.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "        \n",
        "    # Row 3: Error map\n",
        "    ax3 = axes[3, col]\n",
        "    if col < 6:\n",
        "        err = np.abs(vol[z_lesion] - x_gt[z_lesion])\n",
        "        im3 = ax3.imshow(err, cmap='inferno', vmin=0, vmax=vmax_err, origin='lower')\n",
        "        mae_val = np.mean(err[mask[z_lesion]]) if mask[z_lesion].any() else 0\n",
        "        ax3.set_title(f'MAE: {mae_val:.3f}', fontsize=10, pad=4)\n",
        "    else:\n",
        "        ax3.imshow(np.zeros_like(x_gt[z_lesion]), cmap='inferno', vmin=0, vmax=vmax_err, origin='lower')\n",
        "        ax3.set_title('CT Reference', fontsize=10, pad=4)\n",
        "    ax3.axis('off')\n",
        "    if col == 0:\n",
        "        ax3.text(-0.15, 0.5, 'Absolute Error\\n|x_k - x_gt|', va='center', ha='right', transform=ax3.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "\n",
        "# Colorbars\n",
        "cbar_ax1 = fig.add_axes([0.92, 0.40, 0.012, 0.45])\n",
        "cb1 = fig.colorbar(im0, cax=cbar_ax1)\n",
        "cb1.set_label('Activity Intensity', fontsize=11, fontweight='bold')\n",
        "\n",
        "cbar_ax2 = fig.add_axes([0.92, 0.12, 0.012, 0.18])\n",
        "cb2 = fig.colorbar(im3, cax=cbar_ax2)\n",
        "cb2.set_label('Error Magnitude', fontsize=11, fontweight='bold')\n",
        "plt.suptitle(f'Multi-Stage Activity Evolution for Patient {PATIENT_ID}', fontsize=16, fontweight='bold', y=0.94)\n",
        "fig_dir = WORKSPACE / 'figures'\n",
        "fig_dir.mkdir(parents=True, exist_ok=True)\n",
        "fig_path1 = fig_dir / f'Figure1_activity_{PATIENT_ID}.png'\n",
        "fig.savefig(fig_path1, bbox_inches='tight', dpi=140)\n",
        "plt.close(fig)\n",
        "display(Image(filename=str(fig_path1)))"
    ]
})

# =========================================================================
# Cell 8: Figure 2 - Attenuation Evolution
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "stages_mu = mu_hist + [mu_gt]\n",
        "titles_mu = ['Stage 0 (Init Net)', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4 (Final)', 'Ground Truth CT (μ)']\n",
        "vmax_mu = 0.28\n",
        "vmax_mu_err = 0.06\n",
        "\n",
        "fig2, axes2 = plt.subplots(3, 6, figsize=(22, 11), dpi=140)\n",
        "plt.subplots_adjust(wspace=0.08, hspace=0.15)\n",
        "\n",
        "for col, (vol, title) in enumerate(zip(stages_mu, titles_mu)):\n",
        "    # Row 0: Axial\n",
        "    ax0 = axes2[0, col]\n",
        "    im0 = ax0.imshow(vol[z_lesion], cmap='bone', vmin=0, vmax=vmax_mu, origin='lower')\n",
        "    ax0.set_title(title, fontsize=12, fontweight='bold', pad=6)\n",
        "    ax0.axis('off')\n",
        "    if col == 0:\n",
        "        ax0.text(-0.15, 0.5, 'Axial μ Slice\\n(Bones/Thorax)', va='center', ha='right', transform=ax0.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "        \n",
        "    # Row 1: Coronal\n",
        "    ax1 = axes2[1, col]\n",
        "    im1 = ax1.imshow(vol[:, y_lesion, :], cmap='bone', vmin=0, vmax=vmax_mu, origin='lower')\n",
        "    ax1.axis('off')\n",
        "    if col == 0:\n",
        "        ax1.text(-0.15, 0.5, 'Coronal μ Slice\\n(Lungs/Vertebrae)', va='center', ha='right', transform=ax1.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "        \n",
        "    # Row 2: Error map\n",
        "    ax2 = axes2[2, col]\n",
        "    if col < 5:\n",
        "        err = np.abs(vol[z_lesion] - mu_gt[z_lesion])\n",
        "        im2 = ax2.imshow(err, cmap='magma', vmin=0, vmax=vmax_mu_err, origin='lower')\n",
        "        b_err = np.mean(err[bone_mask[z_lesion]]) if bone_mask[z_lesion].any() else 0\n",
        "        ax2.set_title(f'Bone MAE: {b_err:.4f}', fontsize=10, pad=4)\n",
        "    else:\n",
        "        ax2.imshow(np.zeros_like(mu_gt[z_lesion]), cmap='magma', vmin=0, vmax=vmax_mu_err, origin='lower')\n",
        "        ax2.set_title('CT Reference', fontsize=10, pad=4)\n",
        "    ax2.axis('off')\n",
        "    if col == 0:\n",
        "        ax2.text(-0.15, 0.5, 'Absolute Error\\n|μ_k - μ_gt|', va='center', ha='right', transform=ax2.transAxes, fontsize=11, fontweight='bold', rotation=90)\n",
        "\n",
        "cbar_ax_mu = fig2.add_axes([0.92, 0.42, 0.012, 0.45])\n",
        "cb_mu = fig2.colorbar(im0, cax=cbar_ax_mu)\n",
        "cb_mu.set_label('Attenuation Coefficient μ (cm⁻¹)', fontsize=11, fontweight='bold')\n",
        "\n",
        "cbar_ax_mu_err = fig2.add_axes([0.92, 0.12, 0.012, 0.22])\n",
        "cb_mu_err = fig2.colorbar(im2, cax=cbar_ax_mu_err)\n",
        "cb_mu_err.set_label('μ Error (cm⁻¹)', fontsize=11, fontweight='bold')\n",
        "plt.suptitle(f'Multi-Stage Attenuation (μ) Map Evolution for Patient {PATIENT_ID}', fontsize=16, fontweight='bold', y=0.94)\n",
        "fig_path2 = fig_dir / f'Figure2_attenuation_{PATIENT_ID}.png'\n",
        "fig2.savefig(fig_path2, bbox_inches='tight', dpi=140)\n",
        "plt.close(fig2)\n",
        "display(Image(filename=str(fig_path2)))"
    ]
})

# =========================================================================
# Cell 9: Figure 3 - 1D Profile Transect
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Profile passing through maximum lesion/spinal count\n",
        "x_prof_line = int(np.argmax(x_gt[z_lesion]))\n",
        "y_line_idx = x_prof_line // x_gt.shape[2]\n",
        "x_line = slice(15, 115)\n",
        "voxels = np.arange(15, 115)\n",
        "\n",
        "colors = ['#7f7f7f', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#000000']\n",
        "linestyles = [':', '--', '-.', '-', '-', '-', '-']\n",
        "linewidths = [1.5, 1.8, 1.8, 2.0, 2.0, 2.2, 2.5]\n",
        "\n",
        "plt.figure(figsize=(10, 5), dpi=140)\n",
        "for vol, title, c, ls, lw in zip(stages_x, titles_x, colors, linestyles, linewidths):\n",
        "    profile = vol[z_lesion, y_line_idx, x_line]\n",
        "    plt.plot(voxels, profile, label=title, color=c, linestyle=ls, linewidth=lw, alpha=0.9)\n",
        "\n",
        "plt.title(f'1D Activity Profile Transect Through Spinal Lesion (z={z_lesion}, y={y_line_idx})', fontsize=12, fontweight='bold', pad=10)\n",
        "plt.xlabel('Voxel Coordinate X (Voxel size = 4.795 mm)', fontsize=11)\n",
        "plt.ylabel('Radioactivity Concentration', fontsize=11)\n",
        "plt.grid(True, linestyle='--', alpha=0.5)\n",
        "plt.legend(frameon=True, fontsize=10, loc='upper right')\n",
        "plt.tight_layout()\n",
        "fig_path3 = fig_dir / f'Figure3_line_profile_{PATIENT_ID}.png'\n",
        "plt.savefig(fig_path3, bbox_inches='tight', dpi=140)\n",
        "plt.close()\n",
        "display(Image(filename=str(fig_path3)))"
    ]
})

# =========================================================================
# Cell 10: Figure 4 - Convergence Dynamics
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "stage_labels = ['NAC', 'Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']\n",
        "act_psnrs = [m['act_psnr'] for m in metrics]\n",
        "act_ssims = [m['act_ssim'] for m in metrics]\n",
        "bone_maes = [m['bone_mae'] for m in metrics[1:]]\n",
        "\n",
        "fig4, (ax4a, ax4b, ax4c) = plt.subplots(1, 3, figsize=(16, 4.5), dpi=140)\n",
        "\n",
        "ax4a.plot(stage_labels, act_psnrs, marker='o', color='#1f77b4', linewidth=2.2, markersize=7)\n",
        "ax4a.set_title('Activity PSNR Progression (dB)', fontsize=12, fontweight='bold')\n",
        "ax4a.set_ylabel('PSNR (dB)', fontsize=11)\n",
        "ax4a.grid(True, linestyle='--', alpha=0.5)\n",
        "for i, v in enumerate(act_psnrs):\n",
        "    ax4a.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')\n",
        "\n",
        "ax4b.plot(stage_labels, act_ssims, marker='s', color='#2ca02c', linewidth=2.2, markersize=7)\n",
        "ax4b.set_title('Activity SSIM Progression', fontsize=12, fontweight='bold')\n",
        "ax4b.set_ylabel('3D Structural Similarity', fontsize=11)\n",
        "ax4b.grid(True, linestyle='--', alpha=0.5)\n",
        "for i, v in enumerate(act_ssims):\n",
        "    ax4b.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=8, fontweight='bold')\n",
        "\n",
        "ax4c.plot(stage_labels[1:], bone_maes, marker='^', color='#d62728', linewidth=2.2, markersize=7)\n",
        "ax4c.set_title('Bone Attenuation MAE (cm⁻¹)', fontsize=12, fontweight='bold')\n",
        "ax4c.set_ylabel('Bone MAE (cm⁻¹)', fontsize=11)\n",
        "ax4c.grid(True, linestyle='--', alpha=0.5)\n",
        "for i, v in enumerate(bone_maes):\n",
        "    ax4c.text(i, v + 0.0001, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')\n",
        "\n",
        "plt.tight_layout()\n",
        "fig_path4 = fig_dir / f'Figure4_convergence_{PATIENT_ID}.png'\n",
        "fig4.savefig(fig_path4, bbox_inches='tight', dpi=140)\n",
        "plt.close(fig4)\n",
        "display(Image(filename=str(fig_path4)))"
    ]
})

# =========================================================================
# Cell 11: Interactive 3D Slice Browser
# =========================================================================
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "def show_slice(z):\n",
        "    fig, axes = plt.subplots(1, 7, figsize=(22, 3.5), dpi=120)\n",
        "    for ax, vol, title in zip(axes, stages_x, titles_x):\n",
        "        ax.imshow(vol[z], cmap='magma', vmin=0, vmax=vmax_act, origin='lower')\n",
        "        ax.set_title(title, fontsize=10, fontweight='bold')\n",
        "        ax.axis('off')\n",
        "    plt.suptitle(f'Axial Cross-Section at z = {z}', fontsize=13, fontweight='bold')\n",
        "    slice_path = fig_dir / f'Figure5_axial_slice_z{z}_{PATIENT_ID}.png'\n",
        "    fig.savefig(slice_path, bbox_inches='tight', dpi=120)\n",
        "    plt.close(fig)\n",
        "    display(Image(filename=str(slice_path)))\n",
        "\n",
        "# Display axial cross-section across all stages at lesion peak slice\n",
        "print(f'Displaying axial cross-section at lesion peak slice (z = {z_lesion}):')\n",
        "show_slice(z_lesion)\n",
        "\n",
        "# To interactively browse all 128 slices with a slider in Jupyter / VS Code, run:\n",
        "# import ipywidgets as widgets\n",
        "# widgets.interact(show_slice, z=widgets.IntSlider(min=0, max=127, step=1, value=z_lesion))"
    ]
})

nb_json = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (hqmira)",
            "language": "python",
            "name": "hqmira"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NB_PATH, "w") as f:
    json.dump(nb_json, f, indent=2)

print(f"[SUCCESS] Complete, self-contained notebook created at: {NB_PATH}")
