# B E7 environment on the RunPod pod (source this). One GPU, pinned explicitly; task PointMaze; paths under /workspace/pmx.
export B_PLACE=runpod B_TASK=pointmaze
export W=/workspace
export B_PMX=$W/pmx
export B_RES=$B_PMX/B/results/E7/pm B_FEAT=$B_PMX/B/feat/E7/pm
export B_NPROC=${B_NPROC:-12}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export TORCH_HOME=$B_PMX/torch HF_HOME=$B_PMX/hf
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_PY_MUJOCO_PATH=$B_PMX/envs/mujoco210
export MUJOCO_PY_FORCE_CPU=1                     # mujoco-py CPU (OSMesa) builder / renderer
export LD_LIBRARY_PATH=$MUJOCO_PY_MUJOCO_PATH/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export VP=$B_PMX/envs/b-pm/bin/python
export BCODE=$B_PMX/B/code
export PATH=$B_PMX/envs/b-pm/bin:$HOME/.local/bin:$PATH
