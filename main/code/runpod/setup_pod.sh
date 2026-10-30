#!/bin/bash
# B-E0 (M2) pod setup: native Cosmos Policy install (their uv lock, cu128 wheels; replicates docker/Dockerfile),
# LIBERO with EGL (lerobot-libero checkout for assets/bddl/init files), CureWM engine, checkpoints and LIBERO-Goal demos.
# Usage on the pod: setsid nohup bash /workspace/code/setup_pod.sh > /workspace/logs/setup.log 2>&1 < /dev/null &
set -x
export DEBIAN_FRONTEND=noninteractive
source /workspace/code/env.sh
mkdir -p $W/logs $W/results $W/data $RB/src $HF_HOME $UV_CACHE_DIR $LIBERO_CONFIG_PATH
nvidia-smi
nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null; free -g; df -h / $W; cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null

# --- system packages (same list as cosmos-policy docker/Dockerfile, plus tools)
apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential cmake curl ffmpeg git git-lfs \
  libgl1 libegl1 libegl1-mesa-dev libgl1-mesa-dri libglib2.0-0 libglvnd0 rsync tmux jq bc unzip > /dev/null
mkdir -p /usr/share/glvnd/egl_vendor.d
[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || \
  printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
curl -LsSf https://astral.sh/uv/0.8.12/install.sh | sh   # same uv version as their Dockerfile
uv --version
echo "APT_UV_DONE"

# --- code
cd $RB/src
[ -d cosmos-policy ] || git clone -q https://github.com/nvlabs/cosmos-policy.git
[ -d CureWM ] || git clone -q https://github.com/jiuyixu25/CureWM.git
[ -d lerobot-libero ] || git clone -q https://github.com/huggingface/lerobot-libero.git
for d in cosmos-policy CureWM lerobot-libero; do (cd $RB/src/$d && echo "$d $(git log -1 --format='%H %cd')"); done | tee $W/results/repo_commits.txt
echo "CLONE_DONE"

# --- Cosmos Policy env: exactly the command in their LIBERO.md
cd $CP
uv sync --extra cu128 --group libero --python 3.10 2>&1 | tail -15
$VP -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.randn(512,512,device='cuda',dtype=torch.bfloat16); print('matmul ok', float((x@x).float().abs().mean()))"
$VP -c "import flash_attn, transformer_engine, natten; print('flash_attn', flash_attn.__version__, 'te ok, natten ok')"
echo "UV_SYNC_DONE"

# --- LIBERO paths -> lerobot-libero checkout (the PyPI sdist libero==0.1.1 that the lock installs has no assets)
L=$RB/src/lerobot-libero/libero/libero
cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $W/data/libero_datasets
assets: $L/assets
EOF
$VP -c "
import libero, os
from libero.libero import benchmark, get_libero_path
print('libero pkg', os.path.dirname(libero.__file__), 'assets', get_libero_path('assets'))
b = benchmark.get_benchmark_dict()['libero_goal']()
print([b.get_task(i).language for i in range(b.get_num_tasks())])
"
echo "LIBERO_CFG_DONE"

# --- public checkpoints only: the Cosmos Policy LIBERO checkpoint (public repo) and the public Wan2.1 VAE
# (Wan-AI/Wan2.1-T2V-1.3B, Apache-2.0), which is the video tokenizer this policy uses. How the cosmos-policy config is
# pointed at the local VAE file is decided after inspecting the checkout (see the grep below and patch_cosmos_cfg.py).
$VP - <<'PY'
import hashlib, os
from huggingface_hub import hf_hub_download, snapshot_download
p = snapshot_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B")
print("policy ckpt", p, os.listdir(p))
v = hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
h = hashlib.sha256()
with open(v, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
print("wan vae", v, "sha256", h.hexdigest())
PY
grep -rn "Cosmos-Predict2-2B-Video2World" $CP/cosmos_policy --include=*.py | grep -v "^.*#" | cut -c1-220
echo "CKPT_DONE"

# --- official LIBERO-Goal demonstrations (hdf5 with MuJoCo states), as read by CureWM's loader
$VP - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", allow_patterns=["libero_goal/*"],
                      local_dir="/workspace/data/libero_datasets")
print("datasets", p)
PY
ls -la $W/data/libero_datasets/libero_goal | head -20
du -sh $CP/.venv $UV_CACHE_DIR $HF_HOME $W/data 2>/dev/null; df -h / | tail -1
echo "SETUP_ALL_DONE"
