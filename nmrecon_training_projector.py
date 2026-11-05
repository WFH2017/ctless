"""In-process full-angle Path A adapter with a bounded cache keyed by complete geometry."""
import sys
import json, hashlib, time
from collections import OrderedDict
_CACHE = OrderedDict()
from pathlib import Path
import torch
_local_nm = Path(__file__).resolve().parent / 'nmrecon_autograd'
_parent_nm = Path(__file__).resolve().parents[1] / 'nmrecon_autograd'
if _local_nm.exists():
    sys.path.insert(0, str(_local_nm))
elif _parent_nm.exists():
    sys.path.insert(0, str(_parent_nm))
from nmrecon_torch import NMReconDeviceOperator, NMReconFunction

class Subset:
    def __init__(self, operator, indices):
        self.operator, self.indices = operator, indices
    def forward(self, activity, mu):
        return NMReconFunction.apply(activity.float(), mu.detach().float(), self.operator)[:, 0]
    def backward(self, projection, mu):
        return self.operator.backproject(projection[0], mu.detach()[0, 0]).unsqueeze(0).unsqueeze(0)

class Projector:
    full_projection_step = True
    def __init__(self, batch, device):
        if batch['lam_init_p'].shape[0] != 1:
            raise ValueError('NMRecon requires batch size 1')
        config = Path(__file__).resolve().parents[1] / 'data/Siemens/nii' / batch['patient_id'] / 'recon.json'
        cfg = json.loads(config.read_text())
        # Native context uses scanner.setting, output.image_meta and corrections.psf.
        geometry = [cfg['ScanInfo']['scanner']['setting'], cfg['ReconstructSetting']['output']['image_meta'], cfg['ReconstructSetting']['corrections']['psf']]
        key = (str(device), hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest())
        self.create_seconds = 0.0
        if key not in _CACHE:
            start = time.perf_counter()
            _CACHE[key] = NMReconDeviceOperator(str(config), torch.device(device).index)
            self.create_seconds = time.perf_counter() - start
        _CACHE.move_to_end(key)
        self.operator = _CACHE[key]
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)
        if self.operator.proj_shape != tuple(batch['proj_primary'].shape[1:]):
            raise ValueError('Projection geometry mismatch')
        self.num_subsets = 1
    def get_subset(self, proj_primary, scatter_est, subset_idx):
        idx = torch.arange(self.operator.proj_shape[0], device=proj_primary.device)
        if subset_idx is not None:
            idx = idx[subset_idx % self.num_subsets::self.num_subsets]
        return Subset(self.operator, idx), proj_primary, scatter_est

def projector_from_batch(batch, device):
    return Projector(batch, device)
