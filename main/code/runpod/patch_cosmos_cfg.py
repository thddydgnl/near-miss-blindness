"""Point the cosmos-policy checkout at public files only (run once inside the cosmos-policy venv, cwd = checkout).

1) The LIBERO policy's tokenizer is the Wan2.1 VAE. The config downloads it from the gated repo
   nvidia/Cosmos-Predict2-2B-Video2World (tokenizer/tokenizer.pth). The same file is published openly as
   Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth (identical SHA256, checked in setup_pod.sh), so the config is pointed at
   that local file.
2) cosmos_policy_experiment_configs.py resolves the *training* base checkpoint (model-480p-16fps.pt in the same gated
   repo) eagerly at import time. Inference loads every weight from the policy checkpoint, so this training-only
   field is set to "" (no base checkpoint) instead of being downloaded.
The resulting git diff is written to /workspace/results/cosmos_cfg_patch.diff.
"""
import hashlib
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

SHA = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
vae = hf_hub_download("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1_VAE.pth")
h = hashlib.sha256()
with open(vae, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == SHA, h.hexdigest()

edits = [
    ("cosmos_policy/config/defaults/tokenizer.py",
     'vae_pth="hf://nvidia/Cosmos-Predict2-2B-Video2World/tokenizer/tokenizer.pth",',
     f'vae_pth="{vae}",  # B-E0: public Wan2.1 VAE, identical SHA256 to the gated tokenizer.pth'),
    ("cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py",
     'load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),',
     'load_path="",  # B-E0: training-only base checkpoint (gated); inference loads the policy checkpoint'),
]
for path, old, new in edits:
    p = Path(path)
    s = p.read_text()
    if new in s:
        print("already patched", path)
        continue
    assert s.count(old) == 1, (path, s.count(old))
    p.write_text(s.replace(old, new))
    print("patched", path)
diff = subprocess.run(["git", "diff"], capture_output=True, text=True).stdout
Path("/workspace/results/cosmos_cfg_patch.diff").write_text(diff)
print(diff)
