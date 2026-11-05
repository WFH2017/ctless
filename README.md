# CT-less SPECT: Deep Physics-Unrolled Joint Activity and Attenuation Reconstruction

Official PyTorch implementation of **EndToEndPhysicsUnrollSPECTNet**: an end-to-end physics-unrolled neural network for joint SPECT activity reconstruction and attenuation map estimation without structural CT scans (CT-less SPECT).

---

## 🔬 Overview & Key Methodological Advances

Traditional CT-less SPECT reconstruction decouples attenuation estimation from activity reconstruction (predicting a static $\mu$-map followed by standard OSEM), which suffers from systematic error accumulation, mismatch artifacts, and count overestimation in high-uptake skeletal regions.

**EndToEndPhysicsUnrollSPECTNet** models the reconstruction as an unrolled, alternating optimization problem over $K$ physical stages:

1. **Stage 0 (Initial Estimation)**:
   - `DualWindowMuInitNet`: Maps dual-energy-window NAC reconstructions (photopeak bone uptake + scatter body contour) to an initial attenuation map $\mu^{(0)}$.
   - Chang's First-Order Attenuation Pre-correction: Resolves domain scale disparity before Stage 1 OSEM.
2. **Alternating Unrolled Stages ($k = 1 \dots K$)**:
   - **Damped Physical Data Consistency (DC)**:
     $$x_{k-1/2} = x_{k-1} \cdot \left[1 + \alpha_k \left(\frac{A^T(\mu_{k-1}) \cdot \frac{y}{A(\mu_{k-1}) x_{k-1} + s}}{A^T(\mu_{k-1}) \cdot 1} - 1\right)\right]$$
     Autograd gradients backpropagate through both forward and adjoint backprojection operators.
   - **Learned Attenuation Proximal Operator**:
     `DeepCond3DUNet` with FiLM conditioning updates $\mu_k$ based on activity context, scatter prior, and scanner orbit geometry.
   - **Learned Activity Proximal Operator**:
     `AdaptiveScaleActivityResUNet3D` refines $x_k$ across multi-scale 3D features with patient-specific scale adaptation and physical residual direction hints.
3. **Loss Function (Path A Composite Loss)**:
   - Scale-invariant relative $L_1$ on activity: $\frac{\|x - x_{\text{gt}}\|_1}{\|x_{\text{gt}}\|_1 + \epsilon}$
   - Multi-tissue balanced loss on attenuation (bone, soft tissue, lung, outer body) + 3D spatial gradient consistency.
   - Physical Poisson projection negative log-likelihood (NLL) directly connecting measured data consistency to neural proximal representations.

---

## 📁 Repository Structure

```text
ctless-unroll/
├── models/
│   ├── true_unroll_spect.py        # Core EndToEndPhysicsUnrollSPECTNet & TrueEndToEndSPECTLoss
│   ├── pytomography_support.py     # Differentiable PyTomography system matrix wrapper
│   ├── contracts.py                # Scanner collimator, angle, and orbit geometry contracts
│   ├── sparse_unroll_spect.py      # Sparse-view unrolled reconstruction modules
│   └── unroll_net.py               # Downstream evaluation utilities
├── nmrecon_autograd/               # High-Performance C++/CUDA Differentiable Physics Engine
│   ├── engine/                     # Distance-driven forward/adjoint projector & 6-sigma CDR kernels
│   ├── nmrecon_torch.py            # Zero-copy PyTorch Autograd Function (dL/dx and dL/dmu)
│   ├── test_full_autograd.py       # Exact adjointness & autograd gradient verification
│   └── STEP1_INTERFACE_AUDIT.md    # Physical interface specification & audit report
├── nmrecon_training_projector.py   # Bounded geometry LRU-cache adapter for true unroll training
├── train_true_unroll_siemens_nii.py# Full training pipeline on Siemens clinical cohort
├── train_true_unroll_ge_nii.py     # Full training pipeline on GE clinical cohort
├── launch_true_unroll_siemens_train.py # Detached daemon launcher (physical GPU 6 / nmrecon)
├── validate_nmrecon_fast.py        # Fast operator vs baseline L2 and adjointness validator
├── overfit_nmrecon_single_case.py  # 20-step single-case diagnostic convergence benchmark
├── test_unroll_grad_check.py       # Instant autograd gradient & timing verification
├── launch_gn1.py                   # Detached daemon cluster launcher (Node gn1)
├── ctless_dataset.py               # Dataset & DataLoader pipeline with HU-to-mu conversion
├── Figure1_architecture_pipeline.ipynb
├── Figure2_bias_and_correlations.ipynb
├── Figure3_multivendor_cohort_distributions.ipynb
├── Figure4_clinical_reconstructions.ipynb
├── Figure5_attenuation_map_fidelity.ipynb
├── Figure6_bland_altman_agreement.ipynb
├── Figure7_training_convergence_dynamics.ipynb
├── Figure8_multistage_unroll_evolution.ipynb
├── Figure9_GE_clinical_showcase.ipynb
├── Figure10_Siemens_test_showcase.ipynb
├── Table_benchmarks_summary.ipynb
├── figures/                        # High-resolution vector PDF & PNG evaluation figures
├── tables/                         # CSV benchmark quantitative summaries
└── README.md
```

---

## 🚀 Quick Start

### 1. Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 11.8+ (RTX 3090, A100, H100)
- Python 3.10+
- PyTorch >= 2.0
- PyTomography == 3.4.0
- SimpleITK, scikit-image, numpy, scipy

### 2. Physical Projector Verification (NMRecon C++/CUDA vs PyTomography)

To verify the accelerated in-process NMRecon distance-driven physics engine ($< 1\text{s}$ forward projection, adjoint error $< 10^{-6}$):

```bash
# Verify C++/CUDA NMRecon speed and mathematical adjointness
python validate_nmrecon_fast.py

# Verify autograd graph and non-zero gradient flow in unrolled network
python test_unroll_grad_check.py
```

### 3. Accelerated Training on Siemens Cohort (NMRecon Backend, ~8s / step)

Using the optimized, in-process C++/CUDA distance-driven physics backend:

```bash
python train_true_unroll_siemens_nii.py \
    --gpu-id 0 \
    --epochs 50 \
    --num-stages 4 \
    --batch-size 1 \
    --grad-accum-steps 4 \
    --lr 2e-4 \
    --poisson-weight 1e-3 \
    --projector-type nmrecon
```

Or launch as a background daemon process:
```bash
python launch_true_unroll_siemens_train.py --gpu 6 --epochs 50
```

### 4. Training on GE Cohort

```bash
python train_true_unroll_ge_nii.py \
    --gpu-id 0 \
    --epochs 50 \
    --num-stages 4 \
    --batch-size 1 \
    --grad-accum-steps 4 \
    --lr 2e-4 \
    --poisson-weight 1e-3
```

---

## 📊 Evaluation & Figure Reproduction

Jupyter notebooks are provided to reproduce the exact manuscript figures and benchmark tables:
- `Figure1_architecture_pipeline.ipynb`: High-level schematic of the alternating unroll architecture.
- `Figure4_clinical_reconstructions.ipynb`: Coronal, sagittal, and axial clinical reconstruction comparison.
- `Figure5_attenuation_map_fidelity.ipynb`: Attenuation map MAE across bone, soft tissue, and lung regions.
- `Figure6_bland_altman_agreement.ipynb`: Bland-Altman agreement analysis in skeletal hotspots.
- `Figure8_multistage_unroll_evolution.ipynb`: Intermediate unrolled stage progression ($k=1 \dots 4$).

---

## 📜 License

This project is licensed under the Apache 2.0 License.
