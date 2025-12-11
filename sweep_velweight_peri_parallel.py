# sweep_sigma_w_parallel.py
import subprocess
import math
import time
import os

# Sigmas you want to sweep
# sigma_ws = [0.25, 0.32, 0.44, 0.5, 0.55, 0.63]
# sigma_ws = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
sigma_ws = [0.38]

# Base command
base = ['python', 'train.py', 'config/train_small_model_exps.py', '--compile=False']

# How many experiments to run at once
MAX_PARALLEL = 4   # start with 2 or 3; you can adjust upward if VRAM is fine

procs = []

for sigma_w in sigma_ws:
    out = f'/content/drive/MyDrive/ml_projects/orthogonal_transformer/orthogonal-transformer-layers-16-d-256-lr-3e-4/sigma_w_{sigma_w:g}'
    os.makedirs(out, exist_ok=True)
    cmd = base + [
        f'--sigma_w={sigma_w}',
        f'--out_dir={out}',
    ]
    print('Starting:', ' '.join(cmd))
    p = subprocess.Popen(cmd)
    procs.append(p)

    # If we already have MAX_PARALLEL jobs running, wait until one finishes
    while len(procs) >= MAX_PARALLEL:
        alive = []
        for proc in procs:
            ret = proc.poll()
            if ret is None:
                alive.append(proc)
        procs = alive
        if len(procs) >= MAX_PARALLEL:
            time.sleep(10)  # avoid busy-waiting

# Wait for all remaining jobs to finish
for p in procs:
    p.wait()

print("All sigma_w runs finished.")
