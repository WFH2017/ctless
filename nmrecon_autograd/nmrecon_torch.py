"""
NMRecon PyTorch Autograd Interface (Zero-Copy GPU Direct)
Integrates compiled NMRecon C++/CUDA Distance-Driven physics into PyTorch Autograd.

Features:
- Direct GPU device-to-device execution (no disk I/O, no subprocess)
- Mathematically exact adjoint backprojection: <A x, g> == <x, A^T g> (error < 1e-5)
- Path A: attenuation is detached; only activity has a physical adjoint gradient
- Supports a single volume, optionally with leading [1, 1] dimensions
"""

import os
import ctypes
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

_CURR_DIR = Path(__file__).resolve().parent
_LOCAL_SO = _CURR_DIR / "build" / "libnmrecon.so"
DEFAULT_LIB_PATH = str(_LOCAL_SO)



class NMReconCAPI:
    """Low-level C-ABI binding to libnmrecon.so"""
    _instance = None
    
    def __init__(self, lib_path: str = DEFAULT_LIB_PATH):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Cannot find NMRecon shared library at {lib_path}")
        import hashlib, json
        expected = json.loads((_CURR_DIR / 'library_manifest.json').read_text())['sha256']
        actual = hashlib.sha256(Path(lib_path).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("NMRecon library SHA256 mismatch; validate and update manifest before use")
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


class NMReconDeviceOperator:
    """High-level wrapper managing context and GPU memory pointers"""
    def __init__(self, config_json_path: str, device_id: int, lib_path: str = DEFAULT_LIB_PATH):
        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"Config file not found: {config_json_path}")
        self.config_path = os.path.abspath(config_json_path)
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.capi = NMReconCAPI(lib_path)
        
        self.handle = self.capi.lib.nmrecon_create(self.config_path.encode('utf-8'), self.device_id)
        if not self.handle:
            raise RuntimeError(f"Failed to create NMRecon context from {self.config_path} on cuda:{self.device_id}")
            
        # Query dimensions
        nx, ny, nz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self.capi.lib.nmrecon_get_image_size(self.handle, ctypes.byref(nx), ctypes.byref(ny), ctypes.byref(nz))
        self.image_shape = (nx.value, ny.value, nz.value)
        
        n_views, det_ny, det_nz = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        self.capi.lib.nmrecon_get_proj_size(self.handle, ctypes.byref(n_views), ctypes.byref(det_ny), ctypes.byref(det_nz))
        self.proj_shape = (n_views.value, det_ny.value, det_nz.value)

    def __del__(self):
        if hasattr(self, 'handle') and self.handle:
            self.capi.lib.nmrecon_destroy(self.handle)
            self.handle = None

    def forward(self, activity: torch.Tensor, mu_cm: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward project activity with attenuation map mu_cm.
        Zero disk I/O, zero CPU copies.
        """
        act_c = activity.contiguous().to(self.device).float()
        # Keep the contiguous allocation alive until the native call completes.
        mu_c = mu_cm.contiguous().to(self.device).float() if mu_cm is not None else None
        mu_ptr = ctypes.c_void_p(mu_c.data_ptr()) if mu_c is not None else ctypes.c_void_p(0)
        proj_out = torch.zeros(self.proj_shape, dtype=torch.float32, device=self.device).contiguous()
        
        ret = self.capi.lib.nmrecon_forward_device(
            self.handle,
            ctypes.c_void_p(act_c.data_ptr()),
            mu_ptr,
            ctypes.c_void_p(proj_out.data_ptr())
        )
        if ret != 0:
            raise RuntimeError(f"nmrecon_forward_device failed with error code {ret}")
        return proj_out

    def backproject(self, proj_grad: torch.Tensor, mu_cm: torch.Tensor | None = None) -> torch.Tensor:
        """
        Adjoint backprojection of projection gradient with attenuation map mu_cm.
        """
        proj_c = proj_grad.contiguous().to(self.device).float()
        # Keep the contiguous allocation alive until the native call completes.
        mu_c = mu_cm.contiguous().to(self.device).float() if mu_cm is not None else None
        mu_ptr = ctypes.c_void_p(mu_c.data_ptr()) if mu_c is not None else ctypes.c_void_p(0)
        act_grad_out = torch.zeros(self.image_shape, dtype=torch.float32, device=self.device).contiguous()
        
        ret = self.capi.lib.nmrecon_backproject_device(
            self.handle,
            ctypes.c_void_p(proj_c.data_ptr()),
            mu_ptr,
            ctypes.c_void_p(act_grad_out.data_ptr())
        )
        if ret != 0:
            raise RuntimeError(f"nmrecon_backproject_device failed with error code {ret}")
        return act_grad_out


class NMReconFunction(torch.autograd.Function):
    """
    Exact PyTorch Autograd Function for NMRecon.
    Returns activity adjoint gradients only (Path A).
    """
    @staticmethod
    def forward(ctx, activity: torch.Tensor, mu_cm: torch.Tensor, operator: NMReconDeviceOperator):
        if mu_cm is not None and mu_cm.requires_grad:
            raise ValueError("Path A requires mu.detach(); physical mu derivatives are unsupported")
        if activity.ndim not in (3, 5) or (activity.ndim == 5 and activity.shape[:2] != (1, 1)):
            raise ValueError("NMRecon supports exactly one single-channel volume")
        if tuple(activity.shape[-3:]) != operator.image_shape:
            raise ValueError("Activity shape does not match native geometry")
        ctx.operator = operator
        is_batched = (activity.ndim == 5)
        ctx.is_batched = is_batched
        
        if is_batched:
            act_3d = activity[0, 0]
            mu_3d = mu_cm[0, 0] if mu_cm is not None else None
        else:
            act_3d = activity
            mu_3d = mu_cm
            
        with torch.no_grad():
            proj_3d = operator.forward(act_3d, mu_3d)
            
        ctx.save_for_backward(act_3d, mu_3d, proj_3d)
        
        if is_batched:
            return proj_3d.unsqueeze(0).unsqueeze(0)
        return proj_3d

    @staticmethod
    def backward(ctx, grad_proj: torch.Tensor):
        operator = ctx.operator
        act_3d, mu_3d, proj_3d = ctx.saved_tensors
        
        if ctx.is_batched:
            grad_p_3d = grad_proj[0, 0]
        else:
            grad_p_3d = grad_proj
            
        grad_act = None
        grad_mu = None
        
        # 1. Gradient w.r.t activity x: dL/dx = A(mu)^T (dL/dy)
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                grad_act = operator.backproject(grad_p_3d, mu_3d)
                if ctx.is_batched:
                    grad_act = grad_act.unsqueeze(0).unsqueeze(0)
                    
        # Path A: no physical attenuation derivative.
        return grad_act, grad_mu, None


class NMReconModule(nn.Module):
    """
    PyTorch Module wrapper for unrolled reconstruction networks.
    """
    def __init__(self, config_json_path: str, device_id: int):
        super().__init__()
        self.operator = NMReconDeviceOperator(config_json_path, device_id)
        
    def forward(self, activity: torch.Tensor, mu_cm: torch.Tensor) -> torch.Tensor:
        return NMReconFunction.apply(activity, mu_cm, self.operator)
        
    def backproject(self, proj_grad: torch.Tensor, mu_cm: torch.Tensor | None = None) -> torch.Tensor:
        """Direct backprojection for OSEM step"""
        if proj_grad.ndim == 5:
            p_3d = proj_grad[0, 0]
            mu_3d = mu_cm[0, 0] if mu_cm is not None else None
            bp = self.operator.backproject(p_3d, mu_3d)
            return bp.unsqueeze(0).unsqueeze(0)
        return self.operator.backproject(proj_grad, mu_cm)
