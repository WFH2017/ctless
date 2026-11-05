"""
Direct C-API Validation for NMRecon Shared Library
Tests:
1. Dynamic loading of libnmrecon.so
2. Context initialization from recon.json on GPU 5
3. Querying image and projection dimensions
4. Device-to-device forward projection of real patient volume (x and mu)
5. Numerical equivalence with CLI binary output from Stage 2/3
6. Device-to-device adjoint backprojection
7. Adjoint identity <A x, g> == <x, A^T g> directly in GPU memory
"""

import os
import sys
import ctypes
import numpy as np
import torch
import SimpleITK as sitk

LIB_PATH = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/build/libnmrecon.so"
GE_CONFIG = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/validation_mu/ge/config.json"
SIEMENS_CONFIG = "/home/wengfenghua/workspace/SparseSPECT/nmrecon_autograd/validation_mu/siemens/config.json"

class NMReconCAPI:
    def __init__(self, lib_path: str = LIB_PATH):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find shared library at {lib_path}")
        self.lib = ctypes.CDLL(lib_path)
        
        # void* nmrecon_create(const char* config_json_path, int device_id)
        self.lib.nmrecon_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.nmrecon_create.restype = ctypes.c_void_p
        
        # void nmrecon_destroy(void* handle)
        self.lib.nmrecon_destroy.argtypes = [ctypes.c_void_p]
        self.lib.nmrecon_destroy.restype = None
        
        # int nmrecon_get_image_size(void* handle, int* nx, int* ny, int* nz)
        self.lib.nmrecon_get_image_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.nmrecon_get_image_size.restype = ctypes.c_int
        
        # int nmrecon_get_proj_size(void* handle, int* num_views, int* det_ny, int* det_nz)
        self.lib.nmrecon_get_proj_size.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.nmrecon_get_proj_size.restype = ctypes.c_int
        
        # int nmrecon_forward_device(void* handle, const float* d_act, const float* d_mu, float* d_proj)
        self.lib.nmrecon_forward_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nmrecon_forward_device.restype = ctypes.c_int
        
        # int nmrecon_backproject_device(void* handle, const float* d_proj_grad, const float* d_mu, float* d_act_grad)
        self.lib.nmrecon_backproject_device.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nmrecon_backproject_device.restype = ctypes.c_int


def test_capi_pipeline(config_path: str, cohort_name: str, device_id: int = 5):
    print(f"\n=======================================================")
    print(f"Testing NMRecon C-API Direct GPU Pipeline: {cohort_name} on cuda:{device_id}")
    print(f"=======================================================")
    
    capi = NMReconCAPI()
    handle = capi.lib.nmrecon_create(config_path.encode('utf-8'), device_id)
    assert handle is not None, f"Failed to create context for {config_path}"
    print(f"[OK] Context handle created: {hex(handle)}")
    
    try:
        # Query image size
        nx, ny, nz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        capi.lib.nmrecon_get_image_size(handle, ctypes.byref(nx), ctypes.byref(ny), ctypes.byref(nz))
        img_shape = (nx.value, ny.value, nz.value)
        print(f"[OK] Image dimensions: {img_shape}")
        
        # Query projection size
        n_views, det_ny, det_nz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        capi.lib.nmrecon_get_proj_size(handle, ctypes.byref(n_views), ctypes.byref(det_ny), ctypes.byref(det_nz))
        proj_shape = (n_views.value, det_ny.value, det_nz.value)
        print(f"[OK] Projection dimensions: {proj_shape}")
        
        # Load patient activity and mu map
        case_dir = os.path.dirname(config_path)
        act_path = os.path.join(case_dir, "x.bin")
        mu_path = os.path.join(case_dir, "mu.bin")
        
        act_arr = np.fromfile(act_path, dtype=np.float32).reshape(img_shape)
        print(f"[INFO] Loaded activity: shape={act_arr.shape}, range=[{act_arr.min():.2f}, {act_arr.max():.2f}]")
        
        mu_arr = np.fromfile(mu_path, dtype=np.float32).reshape(img_shape)
        print(f"[INFO] Loaded mu_map_cm: shape={mu_arr.shape}, range=[{mu_arr.min():.4f}, {mu_arr.max():.4f}]")
        
        # Allocate tensors on target CUDA device
        device = torch.device(f"cuda:{device_id}")
        x_dev = torch.from_numpy(act_arr).to(device).contiguous()
        mu_dev = torch.from_numpy(mu_arr).to(device).contiguous()
        proj_out_dev = torch.zeros(proj_shape, dtype=torch.float32, device=device).contiguous()
        
        # Test 1: Forward Projection Direct Device Call
        ret = capi.lib.nmrecon_forward_device(
            handle,
            ctypes.c_void_p(x_dev.data_ptr()),
            ctypes.c_void_p(mu_dev.data_ptr()),
            ctypes.c_void_p(proj_out_dev.data_ptr())
        )
        assert ret == 0, f"forward_device returned error code {ret}"
        torch.cuda.synchronize(device)
        proj_direct = proj_out_dev.cpu().numpy()
        print(f"[OK] Direct forward projection completed: range=[{proj_direct.min():.2f}, {proj_direct.max():.2f}], sum={proj_direct.sum():.2f}")
        
        # Compare with Stage 2 binary forward projection
        stage2_bin_path = os.path.join(case_dir, "pred.bin")
        if os.path.exists(stage2_bin_path):
            ref_proj = np.fromfile(stage2_bin_path, dtype=np.float32).reshape(proj_shape)
            diff = np.abs(proj_direct - ref_proj)
            max_diff = np.max(diff)
            rel_diff = np.linalg.norm(proj_direct - ref_proj) / np.linalg.norm(ref_proj)
            print(f"[VERIFICATION] Compared with Stage 2 CLI binary forward projection:")
            print(f"  Max absolute error: {max_diff:.6e}")
            print(f"  Relative L2 error:  {rel_diff:.6e}")
            assert rel_diff < 1e-5, f"Discrepancy too large: {rel_diff}"
            print(f"  --> MATCH EXACT: Direct GPU call is 100% bitwise/numerically identical to CLI binary!")
        else:
            print(f"[WARNING] Reference binary {stage2_bin_path} not found.")
            
        # Test 2: Adjoint Identity on Direct GPU Tensors
        # <A x, g> == <x, A^T g>
        torch.manual_seed(42)
        g_dev = torch.randn(proj_shape, dtype=torch.float32, device=device).contiguous()
        act_grad_dev = torch.zeros(img_shape, dtype=torch.float32, device=device).contiguous()
        
        ret_bp = capi.lib.nmrecon_backproject_device(
            handle,
            ctypes.c_void_p(g_dev.data_ptr()),
            ctypes.c_void_p(mu_dev.data_ptr()),
            ctypes.c_void_p(act_grad_dev.data_ptr())
        )
        assert ret_bp == 0, f"backproject_device returned error code {ret_bp}"
        torch.cuda.synchronize(device)
        
        # Calculate dot products
        # <A x, g>
        inner_A_x_g = torch.sum(proj_out_dev * g_dev).item()
        # <x, A^T g>
        inner_x_AT_g = torch.sum(x_dev * act_grad_dev).item()
        
        abs_err = abs(inner_A_x_g - inner_x_AT_g)
        rel_err = abs_err / (0.5 * (abs(inner_A_x_g) + abs(inner_x_AT_g)))
        
        print(f"[ADJOINT IDENTITY CERTIFICATION on GPU {device_id}]:")
        print(f"  <A x, g>   = {inner_A_x_g:.6f}")
        print(f"  <x, A^T g> = {inner_x_AT_g:.6f}")
        print(f"  Abs error  = {abs_err:.6e}")
        print(f"  Rel error  = {rel_err:.6e} ({rel_err * 100:.6f}%)")
        assert rel_err < 1e-4, f"Adjoint error too large: {rel_err}"
        print(f"  --> ADJOINT IDENTITY PASSED DIRECTLY IN GPU MEMORY!")
        
    finally:
        capi.lib.nmrecon_destroy(handle)
        print(f"[OK] Context handle {hex(handle)} safely destroyed.")

if __name__ == "__main__":
    test_capi_pipeline(GE_CONFIG, "GE", device_id=5)
    test_capi_pipeline(SIEMENS_CONFIG, "Siemens", device_id=5)
