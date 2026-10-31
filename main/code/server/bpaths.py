"""Paths for paper B (E0 onwards). Server: everything under ~/pmx (MAIN_RULES section 1). RunPod (E7, DESIGN_main
section 12): B_PLACE=runpod, B_PMX=/workspace/pmx. Stage-specific result / feature dirs come from the environment
(B_RES, B_FEAT) and the task from B_TASK ('pusht' = M1 default, 'pointmaze' = E7), so one code copy serves all stages."""
import os

HOME = os.path.expanduser('~')
PLACE = os.environ.get('B_PLACE', 'server')                 # 'server' (university GPU server) or 'runpod' (E7)
TASK = os.environ.get('B_TASK', 'pusht')                    # 'pusht' (M1) or 'pointmaze' (E7)
PMX = os.environ.get('B_PMX', os.path.join(HOME, 'pmx'))
B = os.path.join(PMX, 'B')
DINO_WM = os.environ.get('B_DINO_WM', os.path.join(B, 'ext', 'dino_wm'))      # commit 0a9492fa (same as R2-12)
DP_REPO = os.path.join(B, 'ext', 'diffusion_policy')        # commit 5ba07ac6
CKPT_BASE = os.environ.get('B_CKPT_BASE', os.path.join(PMX, 'data', 'dinowm'))  # holds outputs/{pusht,point_maze} (read-only)
DS_DIR = os.environ.get('B_DS_DIR', os.path.join(B, 'data_links'))           # symlinks pusht_noise / point_maze
DP_DATA = os.path.join(PMX, 'data', 'dp_pusht', 'pusht_cchi_v7_replay.zarr')   # standard PushT demos
TORCH_HOME = os.environ.get('TORCH_HOME', os.path.join(PMX, 'torch'))
HUB_DINOV2 = os.path.join(TORCH_HOME, 'hub', 'facebookresearch_dinov2_main')
RES = os.environ.get('B_RES', os.path.join(B, 'results', 'E0', 'm1'))
FEAT = os.environ.get('B_FEAT', os.path.join(B, 'feat', 'E0', 'm1'))
NPROC = int(os.environ.get('B_NPROC', '16'))                # CPU workers (CPU is shared with paper A / other users)

for _d in (RES, FEAT):
    os.makedirs(_d, exist_ok=True)

# Guard (added 14:43Z after an incident): on the server paper B runs on GPU 1 only, and torch-hub / HF caches stay inside
# ~/pmx. tmux sessions started with an inline command do not inherit the caller's exports, so every B job must source
# benv.sh (or use bjob.sh); refuse to run otherwise. On a RunPod pod (one GPU) the device must still be pinned explicitly.
os.environ.setdefault('TORCH_HOME', os.path.join(PMX, 'torch'))
os.environ.setdefault('HF_HOME', os.path.join(PMX, 'hf'))
if PLACE == 'server':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('paper B must run on GPU 1 only: source ~/pmx/B/code/benv.sh (CUDA_VISIBLE_DEVICES=%r, '
                           'CUDA_DEVICE_ORDER=%r)' % (os.environ.get('CUDA_VISIBLE_DEVICES'), os.environ.get('CUDA_DEVICE_ORDER')))
elif os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    raise RuntimeError('B_PLACE=%s: set CUDA_VISIBLE_DEVICES explicitly (source e7_env.sh)' % PLACE)
