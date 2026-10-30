# Common environment for B-E0 (M2: Cosmos Policy LIBERO) on the RunPod pod. Source before every remote command.
# /workspace is a network filesystem (MooseFS) on this pod type, so code checkouts, the uv venv/cache and the HF cache
# live on the local container disk (/root/b); bulk data (LIBERO demos, counterfactual episodes), results and logs on /workspace.
export W=/workspace
export RB=/root/b
export HF_HOME=$RB/cache/hf
export UV_CACHE_DIR=$RB/cache/uv
export HF_HUB_DISABLE_XET=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LIBERO_CONFIG_PATH=$RB/libero_cfg
export PATH=$HOME/.local/bin:$PATH
export CP=$RB/src/cosmos-policy
export VP=$RB/src/cosmos-policy/.venv/bin/python
export PYTHONPATH=$RB/src/CureWM/src:$W/code
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
# one thread per process: several simulator/render workers share a 7.65-vCPU cgroup quota (oversubscription stalls)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
