#!/usr/bin/env python3
"""
Detached Daemon Process Launcher for EndToEndPhysicsUnrollSPECTNet (Siemens Cohort).
Runs on physical GPU 6 (CUDA_VISIBLE_DEVICES=6) by default.
"""

import os
import sys
import argparse
import subprocess

EXP_DIR = "/home/wengfenghua/workspace/SparseSPECT/ctless/experiments_true_unroll_siemens_nii"
PYTHON_BIN = "/home/wengfenghua/anaconda3/envs/hqmira/bin/python"
SCRIPT = "/home/wengfenghua/workspace/SparseSPECT/ctless/train_true_unroll_siemens_nii.py"
LOG_FILE = os.path.join(EXP_DIR, "train.log")
PID_FILE = os.path.join(EXP_DIR, "train.pid")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="6", help="Physical GPU ID to bind (default: 6)")
parser.add_argument("--epochs", type=int, default=50, help="Total training epochs (default: 50)")
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
parser.add_argument("--stages", type=int, default=4, help="Number of unrolled stages (default: 4)")
parser.add_argument("--projector", type=str, default="nmrecon", choices=["nmrecon", "pytomography"], help="Physics projector (default: nmrecon)")
parser.add_argument("--resume", action="store_true", help="Resume from best/latest checkpoint")
args = parser.parse_args()

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)

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
    "--projector-type", str(args.projector),
    "--output-dir", EXP_DIR
]
if args.resume:
    cmd.append("--resume")

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

print("=" * 80)
print(f" EndToEndPhysicsUnrollSPECTNet (Siemens) 端到端展开训练守护进程已独立启动！")
print(f" 进程 PID: {proc.pid} | 绑定显卡: Physical GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")
print(f" 训练配置: 185 训练例 | 25 验证例 | 4 Stages | Epochs={args.epochs}")
print(f" 实时日志文件: {LOG_FILE}")
print(f" 模型存档路径: {os.path.join(EXP_DIR, 'checkpoints')}")
print("=" * 80)
