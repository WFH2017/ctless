#!/usr/bin/env python3
"""
Detached Daemon Process Launcher for EndToEndPhysicsUnrollSPECTNet v2 Training on gn1 GPU 1.
Hot-starts from 48.61 dB checkpoint with:
1. Damped physical updates (damping_alpha = -0.5, 10x lr = 2e-3).
2. grad_mu physical residual backprojection with amax normalization.
3. Separable 3D SSIM structural loss (ssim_weight = 0.15).
4. Strictly non-negative Poisson deviance (poisson_weight = 0.01).
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

EXP_DIR = "/home/wengfenghua/workspace/SparseSPECT/ctless-unroll/experiments_true_unroll_siemens_nii_v2"
CKPT_BACKUP = "/home/wengfenghua/workspace/SparseSPECT/ctless-unroll/checkpoints_v1_backup/checkpoint_best_48.61dB.pt"
PYTHON_BIN = "/home/wengfenghua/anaconda3/envs/hqmira/bin/python"
SCRIPT = "/home/wengfenghua/workspace/SparseSPECT/ctless-unroll/train_true_unroll_siemens_nii.py"
LOG_FILE = os.path.join(EXP_DIR, "train.log")
PID_FILE = os.path.join(EXP_DIR, "train.pid")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="1", help="Physical GPU ID to bind on gn1 (default: 1)")
parser.add_argument("--epochs", type=int, default=50, help="Total epochs for fine-tune (default: 50)")
parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate (default: 2e-4)")
parser.add_argument("--stages", type=int, default=4, help="Number of unrolled stages (default: 4)")
parser.add_argument("--poisson-weight", type=float, default=0.01, help="Poisson deviance loss weight (default: 0.01)")
parser.add_argument("--ssim-weight", type=float, default=0.15, help="3D SSIM structural loss weight (default: 0.15)")
args = parser.parse_args()

os.makedirs(EXP_DIR, exist_ok=True)
ckpt_dir = os.path.join(EXP_DIR, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# Copy baseline best checkpoint as initial anchor if not already present
initial_best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")
if not os.path.exists(initial_best_ckpt) and os.path.exists(CKPT_BACKUP):
    print(f"[*] Copying {CKPT_BACKUP} -> {initial_best_ckpt} as initial benchmark checkpoint.")
    shutil.copy2(CKPT_BACKUP, initial_best_ckpt)

worker_env = os.environ.copy()
worker_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
worker_env["OMP_NUM_THREADS"] = "4"
worker_env["MKL_NUM_THREADS"] = "4"

cmd = [
    PYTHON_BIN, "-u", SCRIPT,
    "--epochs", str(args.epochs),
    "--num-stages", str(args.stages),
    "--gpu-id", "0",
    "--lr", str(args.lr),
    "--weight-decay", "1e-4",
    "--grad-accum-steps", "4",
    "--num-workers", "4",
    "--projector-type", "nmrecon",
    "--poisson-weight", str(args.poisson_weight),
    "--ssim-weight", str(args.ssim_weight),
    "--hotstart-ckpt", CKPT_BACKUP,
    "--output-dir", EXP_DIR
]

log_fd = open(LOG_FILE, "a")

proc = subprocess.Popen(
    cmd,
    env=worker_env,
    stdout=log_fd,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True
)

with open(PID_FILE, "w") as f:
    f.write(str(proc.pid))

print(f"[SUCCESS] v2 Training launched on gn1 GPU {args.gpu} with PID: {proc.pid}")
print(f"Log: {LOG_FILE}")
print(f"PID file: {PID_FILE}")
