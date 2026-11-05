"""
End-to-End PyTorch Autograd Certification for NMRecon
Runs on GPU 6
Validates:
1. Forward pass in 3D and 5D modes
2. Backward pass returning valid gradients for BOTH x and mu
3. Gradient propagation into trainable PyTorch Conv3d network parameters
4. Saves publication-grade verification plots into validation_mu/plots/
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append("/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd")
from nmrecon_torch import NMReconModule

GE_CONFIG = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/validation_mu/ge/config.json"
SIEMENS_CONFIG = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/validation_mu/siemens/config.json"
PLOTS_DIR = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/validation_mu/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


class MiniUnrollBlock(nn.Module):
    """Small neural network block to test gradient propagation to network weights"""
    def __init__(self, config_path: str, device_id: int):
        super().__init__()
        self.nmrecon = NMReconModule(config_path, device_id)
        # Small CNN updating mu
        self.mu_cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1)
        )
        # Small CNN updating activity x
        self.act_cnn = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=3, padding=1)
        )

    def forward(self, x_in, mu_in):
        # Neural update
        delta_mu = self.mu_cnn(mu_in)
        mu_refined = torch.clamp(mu_in + 0.01 * delta_mu, min=0.0)
        
        delta_x = self.act_cnn(x_in)
        x_refined = torch.clamp(x_in + 0.01 * delta_x, min=0.0)
        
        # Physical forward projection through NMRecon
        proj = self.nmrecon(x_refined, mu_refined)
        return proj, x_refined, mu_refined


def run_autograd_test(cohort_name: str, config_path: str, device_id: int = 6):
    print(f"\n==================================================================")
    print(f"Running Full PyTorch Autograd Certification: {cohort_name} on cuda:{device_id}")
    print(f"==================================================================")
    
    device = torch.device(f"cuda:{device_id}")
    case_dir = os.path.dirname(config_path)
    
    # Load patient data
    x_np = np.fromfile(os.path.join(case_dir, "x.bin"), dtype=np.float32).reshape((128, 128, 128))
    mu_np = np.fromfile(os.path.join(case_dir, "mu.bin"), dtype=np.float32).reshape((128, 128, 128))
    ref_pred_np = np.fromfile(os.path.join(case_dir, "pred.bin"), dtype=np.float32)
    n_views = 60 if cohort_name == "GE" else 64
    ref_pred_np = ref_pred_np.reshape((n_views, 128, 128))
    
    # Part 1: Direct 3D Tensor Autograd Function Test
    print("\n[PART 1]: Direct 3D Tensor Autograd Function Test")
    module = NMReconModule(config_path, device_id).to(device)
    
    x = torch.from_numpy(x_np).to(device).requires_grad_(True)
    mu = torch.from_numpy(mu_np).to(device).requires_grad_(True)
    target_proj = 1.05 * torch.from_numpy(ref_pred_np).to(device)
    
    pred_proj = module(x, mu)
    # Poisson Negative Log-Likelihood with small scatter epsilon
    eps = 1e-4
    poisson_loss = torch.mean(pred_proj - target_proj * torch.log(pred_proj + eps))
    
    print(f"  Forward projection range: [{pred_proj.min().item():.2f}, {pred_proj.max().item():.2f}]")
    print(f"  Poisson NLL Loss:         {poisson_loss.item():.6f}")
    
    poisson_loss.backward()
    
    assert x.grad is not None, "x.grad is None!"
    assert mu.grad is not None, "mu.grad is None!"
    assert torch.isfinite(x.grad).all(), "x.grad contains NaN or Inf!"
    assert torch.isfinite(mu.grad).all(), "mu.grad contains NaN or Inf!"
    
    x_grad_norm = torch.norm(x.grad).item()
    mu_grad_norm = torch.norm(mu.grad).item()
    print(f"  Activity gradient norm ||grad_x||:    {x_grad_norm:.6e}")
    print(f"  Attenuation gradient norm ||grad_mu||: {mu_grad_norm:.6e}")
    assert x_grad_norm > 0, "Activity gradient is all zeros!"
    assert mu_grad_norm > 0, "Attenuation gradient is all zeros!"
    print("  --> PASS: Direct 3D autograd successfully produced valid non-zero gradients for BOTH x and mu!")
    
    # Part 2: Batched 5D Unrolled Network Gradient Flow Test
    print("\n[PART 2]: Batched 5D Unrolled Network Parameter Update Test")
    unroll_net = MiniUnrollBlock(config_path, device_id).to(device)
    optimizer = torch.optim.Adam(unroll_net.parameters(), lr=1e-3)
    
    x_5d = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)
    mu_5d = torch.from_numpy(mu_np).unsqueeze(0).unsqueeze(0).to(device)
    target_5d = target_proj.unsqueeze(0).unsqueeze(0)
    
    proj_5d, x_ref, mu_ref = unroll_net(x_5d, mu_5d)
    loss_5d = torch.nn.functional.mse_loss(proj_5d, target_5d)
    
    optimizer.zero_grad()
    loss_5d.backward()
    
    # Check network parameter gradients
    mu_cnn_grad_norm = torch.norm(unroll_net.mu_cnn[0].weight.grad).item()
    act_cnn_grad_norm = torch.norm(unroll_net.act_cnn[0].weight.grad).item()
    print(f"  Mu CNN Conv3d weight grad norm:       {mu_cnn_grad_norm:.6e}")
    print(f"  Activity CNN Conv3d weight grad norm: {act_cnn_grad_norm:.6e}")
    assert mu_cnn_grad_norm > 0, "Mu CNN weights did not receive gradients!"
    assert act_cnn_grad_norm > 0, "Activity CNN weights did not receive gradients!"
    
    # Perform optimizer step to certify parameter update
    optimizer.step()
    print("  --> PASS: Network weights successfully updated via NMRecon physical autograd backpropagation!")
    
    # Part 3: Generate Publication-Grade Diagnostic Plot
    print("\n[PART 3]: Generating Diagnostic Verification Figure...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    slice_idx = 64
    view_idx = n_views // 2
    
    # Col 1: Forward inputs
    im0 = axes[0, 0].imshow(x_np[slice_idx], cmap='hot')
    axes[0, 0].set_title(f"{cohort_name} Activity x (Slice {slice_idx})")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    im1 = axes[1, 0].imshow(mu_np[slice_idx], cmap='bone')
    axes[1, 0].set_title(f"{cohort_name} Attenuation $\mu$ (cm$^{{-1}}$)")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # Col 2: Projection & Residual
    p_np = pred_proj.detach().cpu().numpy()
    im2 = axes[0, 1].imshow(p_np[view_idx], cmap='viridis')
    axes[0, 1].set_title(f"Forward Projection A(x, $\mu$) (View {view_idx})")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    diff_np = (p_np - target_proj.cpu().numpy())[view_idx]
    im3 = axes[1, 1].imshow(diff_np, cmap='coolwarm')
    axes[1, 1].set_title(f"Projection Residual $\partial\mathcal{{L}}/\partial y$ (View {view_idx})")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    # Col 3: Autograd Gradients
    gx_np = x.grad.cpu().numpy()[slice_idx]
    im4 = axes[0, 2].imshow(gx_np, cmap='bwr')
    axes[0, 2].set_title(f"Activity Grad $\partial\mathcal{{L}}/\partial x = A^T g$\n||g||={x_grad_norm:.2e}")
    plt.colorbar(im4, ax=axes[0, 2], fraction=0.046)
    
    gmu_np = mu.grad.cpu().numpy()[slice_idx]
    im5 = axes[1, 2].imshow(gmu_np, cmap='seismic')
    axes[1, 2].set_title(f"Attenuation Grad $\partial\mathcal{{L}}/\partial \mu$\n||g||={mu_grad_norm:.2e}")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.suptitle(f"Stage 4 Certification: Direct GPU NMRecon PyTorch Autograd ({cohort_name})", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"stage4_{cohort_name.lower()}_autograd_verification.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"  Saved verification plot to: {plot_path}")
    
    return {
        "cohort": cohort_name,
        "x_grad_norm": x_grad_norm,
        "mu_grad_norm": mu_grad_norm,
        "mu_cnn_grad_norm": mu_cnn_grad_norm,
        "act_cnn_grad_norm": act_cnn_grad_norm,
        "plot_path": plot_path
    }


if __name__ == "__main__":
    r_ge = run_autograd_test("GE", GE_CONFIG, device_id=6)
    r_siemens = run_autograd_test("Siemens", SIEMENS_CONFIG, device_id=6)
    print("\n==================================================================")
    print("ALL STAGE 4 AUTOGRAD VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print(f"GE:      ||grad_x||={r_ge['x_grad_norm']:.4e}, ||grad_mu||={r_ge['mu_grad_norm']:.4e}, CNN_mu_grad={r_ge['mu_cnn_grad_norm']:.4e}")
    print(f"Siemens: ||grad_x||={r_siemens['x_grad_norm']:.4e}, ||grad_mu||={r_siemens['mu_grad_norm']:.4e}, CNN_mu_grad={r_siemens['mu_cnn_grad_norm']:.4e}")
    print("==================================================================")
