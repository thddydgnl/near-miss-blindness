#!/bin/bash
# B E0 (server track): user-space uv environments for M1 (DINO-WM PushT), M3 (R2-9 controlled model) and the
# PushT Diffusion Policy. No sudo / conda / docker. Base Python lives in B's own dir (not shared with paper A).
#   b-dinowm : DINO-WM PushT imagination, judges, M3, analysis (+ DP deps so DP checkpoints can be loaded)
#   b-dp     : DP training and DP rollouts in the DINO-WM PushT env (same package set; kept separate so the
#              long training job is isolated from later package changes in b-dinowm)
# usage: bash setup_envs.sh   (idempotent; logs to stdout)
set -eu
export PATH=$HOME/.local/bin:$PATH
export UV_PYTHON_INSTALL_DIR=$HOME/pmx/envs/b-python
PY=3.11.17
uv python install $PY
TORCH_IDX=https://download.pytorch.org/whl/cu124
COMMON=(numpy==1.26.4 "hydra-core==1.3.2" "omegaconf==2.3.0" "gym==0.23.1" einops "pymunk==6.2.1" "shapely==2.0.6"
        "opencv-python-headless==4.10.0.84" "pygame==2.6.1" "scikit-image==0.24.0" "matplotlib==3.8.4" "scipy==1.13.1"
        pandas tqdm "accelerate==0.34.2" "decord==0.6.0" psutil pillow
        "diffusers==0.30.3" "huggingface_hub==0.25.2" safetensors "zarr==2.18.3" "numcodecs==0.13.1" "numba==0.60.0"
        dill termcolor)
for E in b-dinowm b-dp; do
  V=$HOME/pmx/envs/$E
  if [ ! -x $V/bin/python ]; then uv venv --python $PY $V; fi
  uv pip install --python $V/bin/python --index-url $TORCH_IDX "torch==2.4.1" "torchvision==0.19.1"
  uv pip install --python $V/bin/python "${COMMON[@]}"
  $V/bin/python -c "import torch, numpy, gym, pymunk, pygame, shapely, cv2, skimage, diffusers, zarr, numba, hydra; print('$E', torch.__version__, torch.version.cuda, numpy.__version__, gym.__version__, pymunk.version, pygame.version.ver, shapely.__version__, cv2.__version__, diffusers.__version__, zarr.__version__, numba.__version__)"
done
# DINOv2 hub (DINO-WM encoder and the M3 encoder) into B's TORCH_HOME
export TORCH_HOME=$HOME/pmx/torch HF_HOME=$HOME/pmx/hf
mkdir -p $TORCH_HOME $HF_HOME
$HOME/pmx/envs/b-dinowm/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'); print('DINOV2_OK')"
echo SETUP_ENVS_DONE
