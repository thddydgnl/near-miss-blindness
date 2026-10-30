#!/bin/bash
# B M2 pod setup v2 (E0b and later M2 pods): everything setup_pod.sh did plus the fixes found in E0
# (python3.10-dev, LIBERO assets link, cosmos-policy config patch, checkpoint in the cache layout cosmos_utils uses).
# Usage on the pod: WITH_COSMOS=1 setsid nohup bash /workspace/code/setup_pod_v2.sh > /workspace/logs/setup.log 2>&1 < /dev/null &
set -x
export DEBIAN_FRONTEND=noninteractive
source /workspace/code/env.sh
WITH_COSMOS=${WITH_COSMOS:-1}
mkdir -p $W/logs $W/results $W/data $RB/src $HF_HOME $UV_CACHE_DIR $LIBERO_CONFIG_PATH
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null; df -h / $W | cat

apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential cmake curl ffmpeg git git-lfs \
  libgl1 libegl1 libegl1-mesa-dev libgl1-mesa-dri libglib2.0-0 libglvnd0 rsync tmux jq bc unzip python3.10-dev > /dev/null
mkdir -p /usr/share/glvnd/egl_vendor.d
[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] || \
  printf '{\n    "file_format_version" : "1.0.0",\n    "ICD" : {\n        "library_path" : "libEGL_nvidia.so.0"\n    }\n}\n' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
curl -LsSf https://astral.sh/uv/0.8.12/install.sh | sh
echo "APT_UV_DONE"

cd $RB/src
[ -d cosmos-policy ] || git clone -q https://github.com/nvlabs/cosmos-policy.git
[ -d CureWM ] || git clone -q https://github.com/jiuyixu25/CureWM.git
[ -d lerobot-libero ] || git clone -q https://github.com/huggingface/lerobot-libero.git
(cd cosmos-policy && git checkout -q 18a2accadf4e7a3531e56754102af5a24d2316da)
(cd CureWM && git checkout -q 8a7ee5dfd71eab09020db57754e72a4dc23c825b)
(cd lerobot-libero && git checkout -q 14fd6f022ff6647defc24e73563b95aa36670f43)
for d in cosmos-policy CureWM lerobot-libero; do (cd $RB/src/$d && echo "$d $(git log -1 --format='%H %cd')"); done | tee $W/results/repo_commits.txt

cd $CP
uv sync --extra cu128 --group libero --python 3.10 2>&1 | tail -5
$VP -c "import torch, flash_attn, transformer_engine; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
L=$RB/src/lerobot-libero/libero/libero
P=$CP/.venv/lib/python3.10/site-packages/libero/libero
[ -e $P/assets ] || ln -s $L/assets $P/assets
cat > $LIBERO_CONFIG_PATH/config.yaml <<EOF
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $W/data/libero_datasets
assets: $L/assets
EOF
echo "UV_SYNC_DONE"

# official LIBERO-Goal demonstrations (states + actions), needed by the DP data and the counterfactual generator
$VP - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("yifengzhu-hf/LIBERO-datasets", repo_type="dataset", allow_patterns=["libero_goal/*"],
                  local_dir="/workspace/data/libero_datasets")
PY
ls $W/data/libero_datasets/libero_goal | wc -l
echo "DATA_DONE"

if [ "$WITH_COSMOS" = "1" ]; then
  $VP - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
# cosmos_utils downloads with cache_dir=HF_HOME (not HF_HOME/hub): use the same layout so nothing is fetched twice
p = snapshot_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B", cache_dir=os.environ["HF_HOME"])
print("policy ckpt", p)
print("wan vae", hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"))
PY
  cd $CP && $VP /workspace/code/patch_cosmos_cfg.py | tail -3
  echo "COSMOS_DONE"
fi
du -sh $CP/.venv $UV_CACHE_DIR $HF_HOME $W/data 2>/dev/null; df -h / | tail -1
echo "SETUP_ALL_DONE"
