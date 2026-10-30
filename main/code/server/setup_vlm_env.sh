#!/bin/bash
# B E3 (server track): user-space uv environment for judge (d) = Qwen2.5-VL-7B-Instruct (same model and transformers
# version as M2: transformers 4.57.1). torch 2.7.0 (M2: 2.7.0+cu128) from PyPI, whose Linux wheel is built for CUDA 12.6
# (the server driver 560.35 supports CUDA <= 12.6; download.pytorch.org timed out from the server on 2026-10-21 23:52Z).
# No flash-attn wheel for this driver/torch combination: SDPA attention instead.
# Weights go to HF_HOME=~/pmx/hf (official repo Qwen/Qwen2.5-VL-7B-Instruct, about 16.6 GB).
# usage: bash setup_vlm_env.sh   (idempotent)
set -eu
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/b-python
export HF_HOME=$HOME/pmx/hf TORCH_HOME=$HOME/pmx/torch
PY=3.11.17
V=$HOME/pmx/envs/b-vlm
if [ ! -x $V/bin/python ]; then uv venv --python $PY $V; fi
uv pip install --python $V/bin/python "torch==2.7.0" "torchvision==0.22.0"
uv pip install --python $V/bin/python "transformers==4.57.1" "accelerate==1.10.1" "numpy==1.26.4" pillow "scipy==1.13.1"
$V/bin/python -c "import torch, torchvision, transformers, accelerate, huggingface_hub; print('b-vlm', torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, accelerate.__version__, huggingface_hub.__version__)"
$V/bin/python - <<'PY'
from huggingface_hub import snapshot_download, HfApi
rev = HfApi().model_info("Qwen/Qwen2.5-VL-7B-Instruct").sha
path = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct", revision=rev)
print("QWEN_REVISION", rev, path)
PY
echo SETUP_VLM_DONE
