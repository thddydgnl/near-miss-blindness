#!/bin/bash
# Add the Cosmos Policy checkpoint + config patch to a pod set up with WITH_COSMOS=0 (setup_pod_v2.sh).
source /workspace/code/env.sh
$VP - <<'PY'
import os
from huggingface_hub import hf_hub_download, snapshot_download
print("policy ckpt", snapshot_download("nvidia/Cosmos-Policy-LIBERO-Predict2-2B", cache_dir=os.environ["HF_HOME"]))
print("wan vae", hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth"))
PY
cd $CP && $VP /workspace/code/patch_cosmos_cfg.py | tail -2
echo "COSMOS_DONE"
