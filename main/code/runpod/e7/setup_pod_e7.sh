#!/bin/bash
# B E7 pod setup (RunPod runpod-torch-v240 image: Ubuntu 22.04, Python 3.11, root). Idempotent. Prints SETUP_E7_DONE.
# Code is pushed to /workspace/pmx/B/code and the PointMaze .pth metadata to /workspace/pmx/data/dinowm/point_maze first.
set -u
W=/workspace; P=$W/pmx
mkdir -p $P/B/results $P/B/feat $P/B/ext $P/B/data_links $P/data/dinowm $P/envs $P/torch $P/hf $W/logs
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq > $W/logs/apt.log 2>&1 && apt-get install -y -qq rsync unzip git build-essential libosmesa6-dev libgl1-mesa-dev \
  libglew-dev libglfw3 patchelf curl >> $W/logs/apt.log 2>&1 && echo APT_OK || echo APT_FAIL
export PATH=$HOME/.local/bin:$PATH
command -v uv > /dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh > $W/logs/uv_install.log 2>&1)
command -v uv > /dev/null || pip install -q uv
V=$P/envs/b-pm
[ -x $V/bin/python ] || uv venv -q --python 3.11 $V
uv pip install -q --python $V/bin/python --index-url https://download.pytorch.org/whl/cu124 "torch==2.4.1" "torchvision==0.19.1" && echo TORCH_OK
uv pip install -q --python $V/bin/python numpy==1.26.4 "hydra-core==1.3.2" "omegaconf==2.3.0" "gym==0.23.1" einops "pymunk==6.2.1" \
  "shapely==2.0.6" "opencv-python-headless==4.10.0.84" "pygame==2.6.1" "scikit-image==0.24.0" "matplotlib==3.8.4" "scipy==1.13.1" \
  pandas tqdm "accelerate==0.34.2" "decord==0.6.0" psutil pillow statsmodels "setuptools==65.5.0" "wheel==0.38.0" "cython<3" patchelf \
  h5py termcolor click imageio lockfile glfw fasteners cffi && echo PKGS_OK
# MuJoCo 2.1.0 binaries + mujoco-py 2.1.2.14 (CPU/OSMesa builder) + d4rl (no deps), as paper A's server setup
MJ=$P/envs/mujoco210
if [ ! -f $MJ/bin/libmujoco210.so ]; then
  curl -sSL --retry 3 -o /tmp/mj.tgz https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xzf /tmp/mj.tgz -C $P/envs && rm -f /tmp/mj.tgz
fi
ls $MJ/bin/libmujoco210.so > /dev/null && echo MUJOCO210_OK || echo MUJOCO210_FAIL
export MUJOCO_PY_MUJOCO_PATH=$MJ MUJOCO_PY_FORCE_CPU=1 LD_LIBRARY_PATH=$MJ/bin:${LD_LIBRARY_PATH:-} PATH=$V/bin:$PATH
uv pip install -q --python $V/bin/python --no-build-isolation "mujoco-py==2.1.2.14" && echo MJPY_PIP_OK
uv pip install -q --python $V/bin/python --no-deps "d4rl @ git+https://github.com/Farama-Foundation/D4RL.git@89141a689b0353b0dac3da5cba60da4b1b16254d" && echo D4RL_PIP_OK
$V/bin/python -c "import mujoco_py; print('MUJOCO_PY_OK', mujoco_py.__file__)" > $W/logs/mujoco_build.log 2>&1 && echo MUJOCO_PY_OK || { echo MUJOCO_PY_FAIL; tail -20 $W/logs/mujoco_build.log; }
$V/bin/python -c "import d4rl.offline_env; print('D4RL_OK')" 2>/dev/null | grep -q D4RL_OK && echo D4RL_OK || echo D4RL_FAIL
# dino_wm at the pilot commit
[ -d $P/B/ext/dino_wm/.git ] || git clone -q https://github.com/gaoyuezhou/dino_wm.git $P/B/ext/dino_wm
(cd $P/B/ext/dino_wm && git checkout -q 0a9492fa && echo "DINO_WM $(git log --oneline -1)")
# checkpoint from OSF (same link as the server's download.sh); the pre-registered hash is checked by the caller
cd $P/data/dinowm
if [ ! -f outputs/point_maze/checkpoints/model_latest.pth ]; then
  curl -sSL --retry 3 -o outputs.zip "https://osf.io/download/xvzs4/?view_only=a56a296ce3b24cceaf408383a175ce28" \
    && unzip -q -o outputs.zip 'outputs/point_maze/*' && rm -f outputs.zip
fi
sha256sum outputs/point_maze/checkpoints/model_latest.pth outputs/point_maze/hydra.yaml
ls -la $P/data/dinowm/point_maze && sha256sum $P/data/dinowm/point_maze/*.pth
ln -sfn $P/data/dinowm/point_maze $P/B/data_links/point_maze
# DINOv2 ViT-S/14 hub code + weights into TORCH_HOME
TORCH_HOME=$P/torch $V/bin/python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'); print('DINOV2_OK')" 2>&1 | grep -E "DINOV2_OK|Error" 
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; free -g | head -2; df -h /workspace | tail -1
echo SETUP_E7_DONE
