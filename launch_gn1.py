#!/usr/bin/env python3
"""
Detached Daemon Launcher for Siemens True Unroll Training on Node gn1.
Runs training in the background on gn1 GPU 0.
"""

import os
import sys
import subprocess

EXP_DIR = "/home/wengfenghua/workspace/SparseSPECT/ctless-unroll/experiments_true_unroll_siemens_nii"
PYTHON_BIN = "/home/wengfenghua/anaconda3/envs/hqmira/bin/python"
SCRIPT = "/home/wengfenghua/workspace/SparseSPECT/ctless-unroll/train_true_unroll_siemens_nii.py"
LOG_FILE = os.path.join(EXP_DIR, "train.log")
PID_FILE = os.path.join(EXP_DIR, "train.pid")

os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(os.path.join(EXP_DIR, "checkpoints"), exist_ok=True)

# Build command to execute on gn1
worker_cmd = (
    f"export CUDA_VISIBLE_DEVICES=0 && "
    f"export OMP_NUM_THREADS=4 && "
    f"export MKL_NUM_THREADS=4 && "
    f"nohup {PYTHON_BIN} -u {SCRIPT} "
    f"--epochs 50 "
    f"--num-stages 4 "
    f"--gpu-id 0 "
    f"--lr 2e-4 "
    f"--weight-decay 1e-4 "
    f"--grad-accum-steps 4 "
    f"--num-workers 4 "
    f"--projector-type pytomography "
    f"--poisson-weight 0.001 "
    f"--output-dir {EXP_DIR} "
    f">> {LOG_FILE} 2>&1 & echo $!"
)

# Launch via clean SSH to gn1
ssh_cmd = [
    "ssh", "-o", "StrictHostKeyChecking=no", "gn1",
    worker_cmd
]

clean_env = {
    "PATH": "/usr/bin:/bin",
    "HOME": os.environ["HOME"],
    "USER": os.environ["USER"]
}

result = subprocess.run(ssh_cmd, env=clean_env, capture_output=True, text=True, check=True)
remote_pid = result.stdout.strip().split()[-1]

with open(PID_FILE, "w") as f:
    f.write(remote_pid)

print("=" * 80)
print(f" EndToEndPhysicsUnrollSPECTNet (Siemens) 训练已成功在 gn1 节点后台启动！")
print(f" 目标节点: gn1.tek-pi.com")
print(f" 运行显卡: Physical GPU 0 (CUDA_VISIBLE_DEVICES=0)")
print(f" 远端进程 PID: {remote_pid}")
print(f" 实验输出目录: {EXP_DIR}")
print(f" 实时日志文件: {LOG_FILE}")
print("=" * 80)
