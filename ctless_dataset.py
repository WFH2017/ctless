"""
Dataset and DataLoader pipeline for CT-less SPECT Reconstruction.
Pairs CT-less SPECT initial reconstructions (Input) with CT-AC high-fidelity reconstructions (Target),
along with physical projections, scatter estimates, and ground-truth CT attenuation maps.
"""

import os
import json
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import structural_similarity as ssim


def ct_hu_to_mu(ct_hu: np.ndarray | torch.Tensor, mu_water_cm: float = 0.154) -> np.ndarray | torch.Tensor:
    """
    Converts CT Hounsfield Units (HU) to 140 keV linear attenuation coefficient mu (in cm^-1)
    using NMRecon / GE standard bilinear mapping:
      HU < 0  => mu = mu_water * (HU / 1000 + 1)
      HU >= 0 => mu = mu_water * (0.665 * HU / 1000 + 1)
    """
    if isinstance(ct_hu, torch.Tensor):
        mu = torch.zeros_like(ct_hu)
        neg_mask = ct_hu < 0
        pos_mask = ~neg_mask
        mu[neg_mask] = mu_water_cm * (ct_hu[neg_mask] / 1000.0 + 1.0)
        mu[pos_mask] = mu_water_cm * (0.665 * ct_hu[pos_mask] / 1000.0 + 1.0)
        return torch.clamp(mu, min=0.0)
    else:
        mu = np.zeros_like(ct_hu, dtype=np.float32)
        neg_mask = ct_hu < 0
        pos_mask = ~neg_mask
        mu[neg_mask] = mu_water_cm * (ct_hu[neg_mask] / 1000.0 + 1.0)
        mu[pos_mask] = mu_water_cm * (0.665 * ct_hu[pos_mask] / 1000.0 + 1.0)
        return np.maximum(mu, 0.0).astype(np.float32)


class CTLessSPECTDataset(Dataset):
    """
    PyTorch Dataset for Unrolled Deep Learning CT-less SPECT Reconstruction.
    Each sample dictionary contains:
      - 'x_init': CT-less initial reconstruction volume (1, 128, 128, 128) [Input]
      - 'x_gt': CT-AC high-fidelity reconstruction volume (1, 128, 128, 128) [Target]
      - 'y_proj': Measured photopeak projections (60, 128, 128)
      - 'y_scatter': Estimated DEW scatter projections (60, 128, 128)
      - 'ct_mu': Ground-truth resampled CT 140 keV attenuation map (1, 128, 128, 128)
      - 'angles_deg': Detector projection angles (60,) in degrees
      - 'radii_cm': Dynamic rotation radii (60,) in cm
      - 'dr_cm': Voxel physical spacing (3,) in cm
      - 'case_id': Patient case identifier string
      - 'scale': Normalization intensity scale
    """
    def __init__(
        self,
        case_dirs: list[str],
        normalize_target: bool = True,
        cache_in_memory: bool = False
    ):
        self.normalize_target = normalize_target
        self.cache_in_memory = cache_in_memory
        self.cache = {}
        
        # Verify and retain only directories with complete prerequisites
        self.valid_dirs = []
        for d in case_dirs:
            req_files = [
                os.path.join(d, 'recon.json'),
                os.path.join(d, 'projection.nii.gz'),
                os.path.join(d, 'ct.nii.gz_resample.nii.gz'),
                os.path.join(d, 'nmrecon_out', 'recon.nii.gz'),
                os.path.join(d, 'nmrecon_out_no_ac', 'recon.nii.gz')
            ]
            if all(os.path.exists(f) for f in req_files):
                self.valid_dirs.append(os.path.abspath(d))
                
        print(f"Initialized CTLessSPECTDataset with {len(self.valid_dirs)} / {len(case_dirs)} verified complete cases.")

    def __len__(self):
        return len(self.valid_dirs)

    def _load_case(self, case_dir: str) -> dict:
        case_id = os.path.basename(case_dir)
        
        # 1. Parse scan geometry and metadata
        with open(os.path.join(case_dir, 'recon.json'), 'r') as f:
            cfg = json.load(f)
            
        scanner = cfg['ScanInfo']['scanner']['setting']
        start_deg = float(scanner['detector_start_degree'][0])
        end_deg = float(scanner['detector_end_degree'][0])
        num_views = int(scanner['num_views'])
        step_deg = (end_deg - start_deg) / num_views
        angles_deg = start_deg + np.arange(num_views, dtype=np.float32) * step_deg
        radii_cm = np.array(scanner['rotation_radii'], dtype=np.float32) / 10.0 # mm -> cm
        
        spacing_mm = float(cfg['ReconstructSetting']['output']['image_meta']['spacing'][0])
        dr_cm = np.array([spacing_mm / 10.0, spacing_mm / 10.0, spacing_mm / 10.0], dtype=np.float32)
        
        # 2. Load CT-less input volume (Input x_init)
        noac_path = os.path.join(case_dir, 'nmrecon_out_no_ac', 'recon.nii.gz')
        x_init = sitk.GetArrayFromImage(sitk.ReadImage(noac_path)).astype(np.float32)
        
        # 3. Load CT-AC target volume (Target x_gt)
        ac_path = os.path.join(case_dir, 'nmrecon_out', 'recon.nii.gz')
        x_gt = sitk.GetArrayFromImage(sitk.ReadImage(ac_path)).astype(np.float32)
        
        # 4. Load SPECT raw projections (Photopeak + Scatter)
        proj_path = os.path.join(case_dir, 'projection.nii.gz')
        proj_arr = sitk.GetArrayFromImage(sitk.ReadImage(proj_path)).astype(np.float32)
        y_proj = proj_arr[:num_views] # (60, 128, 128)
        y_scatter = proj_arr[num_views:2*num_views] * 0.55 # (60, 128, 128) with 0.55 DEW factor
        
        # 5. Load CT attenuation map (ct_mu)
        ct_path = os.path.join(case_dir, 'ct.nii.gz_resample.nii.gz')
        ct_hu = sitk.GetArrayFromImage(sitk.ReadImage(ct_path)).astype(np.float32)
        ct_mu = ct_hu_to_mu(ct_hu, mu_water_cm=0.154) # (128, 128, 128)
        
        # 6. Scatter Window 3D Reconstruction (x_scat) as Compton Attenuation Proxy
        scat_cache_path = os.path.join(case_dir, 'scatter_recon.npy')
        if os.path.exists(scat_cache_path):
            x_scat = np.load(scat_cache_path)
        else:
            try:
                from differentiable_projector import DifferentiableSPECTProjector
                dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                proj_helper = DifferentiableSPECTProjector(angles_deg=angles_deg, radii_cm=radii_cm, device=dev)
                y_scat_t = torch.from_numpy(y_scatter).to(dev)
                x_scat_t = proj_helper.backproject(y_scat_t)
                x_scat = x_scat_t.detach().cpu().numpy().astype(np.float32)
                x_scat = x_scat / (np.max(x_scat) + 1e-6)
                np.save(scat_cache_path, x_scat)
            except Exception:
                x_scat = np.zeros_like(x_init, dtype=np.float32)

        # 7. Intensity normalization scale (99.9th percentile of CT-AC target)
        val_ref = float(np.percentile(x_gt, 99.9))
        scale = max(val_ref, 1.0)
        
        if self.normalize_target:
            x_init = x_init / scale
            x_gt = x_gt / scale
            # Projections scaled accordingly
            y_proj = y_proj / (y_proj.mean() + 1e-6)
            y_scatter = y_scatter / (y_proj.mean() + 1e-6)
            
        return {
            'x_init': torch.from_numpy(x_init).unsqueeze(0).float(),      # (1, 128, 128, 128)
            'x_gt': torch.from_numpy(x_gt).unsqueeze(0).float(),          # (1, 128, 128, 128)
            'y_proj': torch.from_numpy(y_proj).float(),                   # (60, 128, 128)
            'y_scatter': torch.from_numpy(y_scatter).float(),             # (60, 128, 128)
            'x_scat': torch.from_numpy(x_scat).unsqueeze(0).float(),      # (1, 128, 128, 128)
            'ct_mu': torch.from_numpy(ct_mu).unsqueeze(0).float(),        # (1, 128, 128, 128)
            'angles_deg': torch.from_numpy(angles_deg).float(),           # (60,)
            'radii_cm': torch.from_numpy(radii_cm).float(),               # (60,)
            'dr_cm': torch.from_numpy(dr_cm).float(),                     # (3,)
            'case_id': case_id,
            'scale': scale
        }

    def __getitem__(self, idx: int) -> dict:
        if idx in self.cache:
            return self.cache[idx]
            
        case_dir = self.valid_dirs[idx]
        data = self._load_case(case_dir)
        
        if self.cache_in_memory:
            self.cache[idx] = data
            
        return data


def get_ctless_spect_dataloaders(
    data_root: str = '/home/wengfenghua/workspace/SparseSPECT/data/GE/nii',
    batch_size: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    max_cases: int | None = None
):
    """
    Builds reproducible, balanced Train, Validation, and Test DataLoaders
    for CT-less SPECT unrolled deep learning.
    """
    all_subdirs = sorted([
        os.path.join(data_root, d) for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith('recon_logs')
    ])
    
    # Pre-filter verified complete cases
    verified = []
    for d in all_subdirs:
        req_files = [
            os.path.join(d, 'recon.json'),
            os.path.join(d, 'projection.nii.gz'),
            os.path.join(d, 'ct.nii.gz_resample.nii.gz'),
            os.path.join(d, 'nmrecon_out', 'recon.nii.gz'),
            os.path.join(d, 'nmrecon_out_no_ac', 'recon.nii.gz')
        ]
        if all(os.path.exists(f) for f in req_files):
            verified.append(d)
            
    if max_cases is not None:
        verified = verified[:max_cases]
        
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(verified).tolist()
    
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    if n_total >= 3:
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_total:
            n_train = n_total - 2
            n_val = 1
            
    train_dirs = shuffled[:n_train]
    val_dirs = shuffled[n_train:n_train + n_val]
    test_dirs = shuffled[n_train + n_val:]
    if len(test_dirs) == 0:
        test_dirs = val_dirs
        
    train_dataset = CTLessSPECTDataset(train_dirs)
    val_dataset = CTLessSPECTDataset(val_dirs)
    test_dataset = CTLessSPECTDataset(test_dirs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_attenuation_gap_metrics(x_init: np.ndarray, x_gt: np.ndarray, data_range: float = 1.0) -> dict:
    """
    Computes quantitative gap between CT-less input and CT-AC target volume
    (SSIM, NMSE, Pearson r, PSNR, NRMSE, Central Cupping Ratio).
    """
    mse = float(np.mean((x_init - x_gt) ** 2))
    rmse = float(np.sqrt(mse))
    val_range = max(float(x_gt.max() - x_gt.min()), 1e-6)
    nrmse = (rmse / val_range) * 100.0
    psnr = 20.0 * np.log10(x_gt.max() / max(rmse, 1e-8))
    nmse = float(np.sum((x_init - x_gt) ** 2) / max(np.sum(x_gt ** 2), 1e-8))
    r_val = float(np.corrcoef(x_init.flatten(), x_gt.flatten())[0, 1])
    ssim_val = float(ssim(x_init, x_gt, data_range=data_range))
    
    # Central deep tissue cupping ratio (center 32x32 vs outer ring in transaxial plane)
    c_y, c_x = x_gt.shape[1] // 2, x_gt.shape[2] // 2
    r_center = 16
    center_gt = np.mean(x_gt[:, c_y - r_center:c_y + r_center, c_x - r_center:c_x + r_center])
    center_init = np.mean(x_init[:, c_y - r_center:c_y + r_center, c_x - r_center:c_x + r_center])
    cupping_deficit = max(0.0, float((center_gt - center_init) / max(center_gt, 1e-6) * 100.0))
    
    return {
        'pearson_r': r_val,
        'ssim_3d': ssim_val,
        'nmse': nmse,
        'psnr': psnr,
        'nrmse_pct': nrmse,
        'cupping_deficit_pct': cupping_deficit
    }
