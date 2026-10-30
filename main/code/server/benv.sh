# Common environment for B server jobs (source this). GPU 1 only (paper A uses GPU 0).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export SDL_VIDEODRIVER=dummy
export TORCH_HOME=$HOME/pmx/torch
export HF_HOME=$HOME/pmx/hf
export B_NPROC=${B_NPROC:-16}
export PYTHONUNBUFFERED=1
export PATH=$HOME/.local/bin:$PATH
BPY_DINOWM=$HOME/pmx/envs/b-dinowm/bin/python
BPY_DP=$HOME/pmx/envs/b-dp/bin/python
BCODE=$HOME/pmx/B/code
# share GPU 1 politely with paper A: less reserved-but-unused memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# share the CPU politely (BLAS / OpenMP threads per process)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
