"""
Test Mu-Gradient Formulations against Ground Truth Finite Difference Directional Derivatives
Runs on GPU 6
"""

import os
import sys
import ctypes
import numpy as np
import torch

sys.path.append("/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd")
from test_direct_capi import NMReconCAPI, GE_CONFIG

def main():
    device_id = 6
    device = torch.device(f"cuda:{device_id}")
    
    capi = NMReconCAPI()
    handle = capi.lib.nmrecon_create(GE_CONFIG.encode(), device_id)
    assert handle is not None, "Failed to create context"
    
    try:
        # Load patient activity and mu
        case_dir = os.path.dirname(GE_CONFIG)
        x_np = np.fromfile(os.path.join(case_dir, "x.bin"), dtype=np.float32).reshape((128, 128, 128))
        mu_np = np.fromfile(os.path.join(case_dir, "mu.bin"), dtype=np.float32).reshape((128, 128, 128))
        ref_pred_np = np.fromfile(os.path.join(case_dir, "pred.bin"), dtype=np.float32).reshape((60, 128, 128))
        
        x = torch.from_numpy(x_np).to(device).contiguous()
        mu = torch.from_numpy(mu_np).to(device).contiguous()
        
        # Target projection with slight mismatch: 1.1 * ref_pred
        y_target = 1.1 * torch.from_numpy(ref_pred_np).to(device).contiguous()
        
        def forward_nmrecon(x_tensor, mu_tensor):
            p = torch.zeros((60, 128, 128), dtype=torch.float32, device=device).contiguous()
            capi.lib.nmrecon_forward_device(
                handle,
                ctypes.c_void_p(x_tensor.data_ptr()),
                ctypes.c_void_p(mu_tensor.data_ptr()),
                ctypes.c_void_p(p.data_ptr())
            )
            torch.cuda.synchronize(device)
            return p
            
        def backproject_nmrecon(g_tensor, mu_tensor):
            bp = torch.zeros((128, 128, 128), dtype=torch.float32, device=device).contiguous()
            capi.lib.nmrecon_backproject_device(
                handle,
                ctypes.c_void_p(g_tensor.data_ptr()),
                ctypes.c_void_p(mu_tensor.data_ptr()),
                ctypes.c_void_p(bp.data_ptr())
            )
            torch.cuda.synchronize(device)
            return bp

        # Baseline forward
        p0 = forward_nmrecon(x, mu)
        # Loss: 0.5 * || p - y ||^2
        diff0 = p0 - y_target
        loss0 = 0.5 * torch.sum(diff0 ** 2).item()
        # Adjoint projection gradient g = dL/dp = (p - y)
        g = diff0
        
        print(f"Baseline Loss: {loss0:.6e}")
        print(f"Gradient norm ||g||: {torch.norm(g).item():.6e}")
        
        # Test 3 different perturbation directions
        torch.manual_seed(123)
        v1 = (mu / torch.norm(mu)).contiguous() # Along mu itself
        
        # v2: Smooth Gaussian-like perturbation
        v2_raw = torch.randn_like(mu)
        # Simple box blur for smoothness
        v2_smooth = torch.nn.functional.avg_pool3d(v2_raw.unsqueeze(0).unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze()
        v2 = (v2_smooth / torch.norm(v2_smooth)).contiguous()
        
        # v3: Bone region indicator
        v3_raw = (mu > 0.20).float()
        v3 = (v3_raw / torch.norm(v3_raw)).contiguous()
        
        directions = [("Along mu", v1), ("Smooth random", v2), ("Bone region", v3)]
        eps = 1e-2 # Proven optimal epsilon from Stage 3 sweep
        
        print("\n--- Finite Difference Directional Derivatives (FD) ---")
        fd_derivs = []
        for name, v in directions:
            mu_plus = (mu + eps * v).contiguous()
            mu_minus = (mu - eps * v).contiguous()
            p_plus = forward_nmrecon(x, mu_plus)
            p_minus = forward_nmrecon(x, mu_minus)
            
            loss_plus = 0.5 * torch.sum((p_plus - y_target)**2).item()
            loss_minus = 0.5 * torch.sum((p_minus - y_target)**2).item()
            
            fd = (loss_plus - loss_minus) / (2.0 * eps)
            fd_derivs.append(fd)
            print(f"  {name:15s}: FD dL/d(eps) = {fd:+.6e}")
            
        print("\n--- Candidate Gradient Formulations ---")
        # Candidate 1: - Backprojection of (g * p0) with mu
        # Formula: grad_mu ~ - A^T(g * p, mu)
        cand1 = - backproject_nmrecon(g * p0, mu)
        
        # Candidate 2: - Backprojection of (g * p0) without mu (A0^T)
        cand2 = - backproject_nmrecon(g * p0, torch.zeros_like(mu))
        
        # Candidate 3: - Backprojection of g * (forward of x) * effective chord length
        # Normalizing candidate 1 by total projection scale
        p_mean = p0.mean().item()
        cand3 = cand1 / max(p_mean, 1e-6)
        
        candidates = [
            ("Cand 1: -A^T(g * p, mu)", cand1),
            ("Cand 2: -A0^T(g * p)", cand2),
            ("Cand 3: -A^T(g * p)/p_mean", cand3),
        ]
        
        for cname, cgrad in candidates:
            print(f"\nEvaluating {cname}:")
            c_norm = torch.norm(cgrad).item()
            print(f"  ||grad|| = {c_norm:.6e}")
            for (dname, v), fd in zip(directions, fd_derivs):
                model_d = torch.sum(cgrad * v).item()
                ratio = model_d / fd if abs(fd) > 1e-12 else 0.0
                print(f"    {dname:15s}: Model = {model_d:+.6e} | FD = {fd:+.6e} | Ratio = {ratio:+.4f}")
                
    finally:
        capi.lib.nmrecon_destroy(handle)
        print("\nContext destroyed successfully.")

if __name__ == "__main__":
    main()
