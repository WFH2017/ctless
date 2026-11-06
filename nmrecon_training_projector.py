"""In-process full-angle Path A adapter with a bounded cache keyed by complete geometry."""
import sys
import json
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
import torch

from nmrecon_autograd.nmrecon_torch import NMReconDeviceOperator, NMReconFunction

_CACHE = OrderedDict()


class NMReconBackprojectFunction(torch.autograd.Function):
    """
    Exact Adjoint Backprojection Autograd Function for NMRecon.
    Forward: backproject projection -> 3D activity.
    Backward: forward project activity grad -> projection grad (exact adjoint).
    """
    @staticmethod
    def forward(ctx, projection: torch.Tensor, mu_cm: torch.Tensor, operator: NMReconDeviceOperator):
        ctx.operator = operator
        is_batched = (projection.ndim == 4)
        ctx.is_batched = is_batched
        p_3d = projection[0] if is_batched else projection
        mu_3d = mu_cm[0, 0] if (mu_cm is not None and mu_cm.ndim == 5) else mu_cm

        with torch.no_grad():
            bp_3d = operator.backproject(p_3d, mu_3d)

        ctx.save_for_backward(mu_3d)
        if is_batched:
            return bp_3d.unsqueeze(0).unsqueeze(0)
        return bp_3d

    @staticmethod
    def backward(ctx, grad_act: torch.Tensor):
        operator = ctx.operator
        (mu_3d,) = ctx.saved_tensors
        grad_a_3d = grad_act[0, 0] if grad_act.ndim == 5 else grad_act

        grad_proj = None
        if ctx.needs_input_grad[0]:
            with torch.no_grad():
                grad_p_3d = operator.forward(grad_a_3d, mu_3d)
                grad_proj = grad_p_3d.unsqueeze(0) if ctx.is_batched else grad_p_3d
        return grad_proj, None, None


class Subset:
    def __init__(self, operator, indices):
        self.operator = operator
        self.indices = indices

    def forward(self, activity: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        activity: [B, 1, 128, 128, 128]
        mu: [B, 1, 128, 128, 128]
        returns: [B, N_views, 128, 128]
        """
        p_5d = NMReconFunction.apply(activity.float(), mu.detach().float(), self.operator)
        return p_5d[:, 0]

    def backward(self, projection: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        projection: [B, N_views, 128, 128]
        mu: [B, 1, 128, 128, 128]
        returns: [B, 1, 128, 128, 128]
        """
        return NMReconBackprojectFunction.apply(projection.float(), mu.detach().float(), self.operator)


class Projector:
    full_projection_step = False  # Run true unrolled stages with checkpointing

    def __init__(self, batch: dict, device: str):
        if batch['lam_init_p'].shape[0] != 1:
            raise ValueError('NMRecon requires batch size 1')

        pid = batch['patient_id']
        if isinstance(pid, (list, tuple)):
            pid = pid[0]

        candidates = [
            Path(__file__).resolve().parent / 'data/Siemens/nii' / pid / 'recon.json',
            Path(__file__).resolve().parent / 'data/GE/nii' / pid / 'recon.json',
            Path(f'/home/wengfenghua/workspace/SparseSPECT/data/Siemens/nii/{pid}/recon.json'),
            Path(f'/home/wengfenghua/workspace/SparseSPECT/data/GE/nii/{pid}/recon.json'),
            Path(__file__).resolve().parents[1] / 'data/Siemens/nii' / pid / 'recon.json',
            Path(__file__).resolve().parents[1] / 'data/GE/nii' / pid / 'recon.json',
        ]
        config = None
        for c in candidates:
            if c.exists():
                config = c
                break
        if config is None:
            raise FileNotFoundError(f"Cannot find recon.json for patient_id: {pid}")

        cfg = json.loads(config.read_text())
        geometry = [
            cfg['ScanInfo']['scanner']['setting'],
            cfg['ReconstructSetting']['output']['image_meta'],
            cfg['ReconstructSetting']['corrections']['psf']
        ]

        # Parse target device index
        if isinstance(device, torch.device):
            dev_idx = device.index
        elif isinstance(device, str):
            dev_idx = torch.device(device).index
        elif isinstance(device, int):
            dev_idx = device
        else:
            dev_idx = 0
        if dev_idx is None:
            dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0

        key = (f"cuda:{dev_idx}", hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest())
        self.create_seconds = 0.0
        if key not in _CACHE:
            start = time.perf_counter()
            _CACHE[key] = NMReconDeviceOperator(str(config), dev_idx)
            self.create_seconds = time.perf_counter() - start
        _CACHE.move_to_end(key)
        self.operator = _CACHE[key]
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)

        if self.operator.proj_shape != tuple(batch['proj_primary'].shape[1:]):
            raise ValueError(
                f"Projection geometry mismatch: operator={self.operator.proj_shape} vs batch={tuple(batch['proj_primary'].shape[1:])}"
            )
        self.num_subsets = 1

    def get_subset(self, proj_primary: torch.Tensor, scatter_est: torch.Tensor, subset_idx: int = None):
        idx = torch.arange(self.operator.proj_shape[0], device=proj_primary.device)
        if subset_idx is not None and self.num_subsets > 1:
            idx = idx[subset_idx % self.num_subsets::self.num_subsets]
        return Subset(self.operator, idx), proj_primary, scatter_est


def projector_from_batch(batch, device):
    return Projector(batch, device)
