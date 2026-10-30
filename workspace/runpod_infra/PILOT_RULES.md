# Rules for pilot workers (read fully before starting)

Project root (has spaces, always quote it): `/Users/song-yonghwi/Documents/physical ai - Neural Networks`
Infra: `<root>/runpod_infra/`. Your pilot folder: `<root>/R1_<n>_<name>/` (given in your task).

## 1. Compute only on the pod
- The user's MacBook overheats. On the Mac you may only: write/edit files, run `python3 -m py_compile` syntax checks,
  run tiny pure-Python sanity checks that finish in seconds, and read results. **No local training, evaluation,
  simulation rollouts, rendering, model or dataset downloads.** Everything else runs on your RunPod pod.

## 2. Pod lifecycle (use only these scripts; tag = your pilot tag, e.g. `r1n1`)
- Create: `bash "<root>/runpod_infra/pod_up.sh" <tag> "NVIDIA RTX A5000|NVIDIA GeForce RTX 3090|NVIDIA A40|NVIDIA RTX A6000|NVIDIA GeForce RTX 4090|NVIDIA L4" [container_gb] [volume_gb]`
  - Default image is the runpod-torch-v240 template (torch 2.4, CUDA 12.4, Python 3.11, Ubuntu 22.04). Never pick Blackwell GPUs (RTX 5090, RTX PRO *, B200) with it.
  - Creation can take several minutes; it waits for ssh. The pod auto-terminates 4 h after creation.
  - It writes `runpod_infra/pods/<tag>.env` (POD_ID, POD_IP, POD_PORT) and a ledger entry. Run it in the foreground with a 15-minute Bash timeout.
- Shell: `bash "<root>/runpod_infra/rsh.sh" <tag> '<remote command>'`
- Copy: `bash "<root>/runpod_infra/sync.sh" <tag> push "<local path>" /workspace/<dir>/` and `... pull /workspace/<dir>/ "<local path>"`
  (needs `apt-get install -y rsync` on the pod once; local paths contain spaces, quote them).
- Delete: `bash "<root>/runpod_infra/pod_down.sh" <tag>` — only after pulling results. Then confirm with `python3 "<root>/runpod_infra/budget.py" status` that your pod shows `gone`.
- **Never touch any other pod.** The account is shared with other studies (`nnc-*`, `wmc-*`, other `pai-*` tags). Do not run `runpodctl pod delete/stop` on any id except through `pod_down.sh <your tag>`. Do not create pods outside `pod_up.sh`.
- If `pod_up.sh` fails on every GPU, wait a few minutes and retry once; if it still fails, stop and report.

## 3. Budget and time
- Your cap is in your task (also in `runpod_infra/caps.json`). A watchdog deletes your pod when your estimated spend reaches the cap, so keep a margin: plan to finish by ~80% of it.
- Check spend: `python3 "<root>/runpod_infra/budget.py" status`.
- Wall-clock target: finish within about 2.5 h of creating the pod. Prefer a smaller but decisive pilot over a large unfinished one.

## 4. Running long jobs
- Start long remote jobs detached: `rsh.sh <tag> 'cd /workspace && setsid nohup bash run.sh > /workspace/logs/run.log 2>&1 < /dev/null &'`.
- Poll with short commands, putting any waiting **on the remote side**, e.g. `rsh.sh <tag> 'sleep 240; tail -n 20 /workspace/logs/run.log; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'` (keep each Bash call under 10 minutes).
- Use the pod's CPUs for simulation (check `nproc`); parallelize independent runs with multiprocessing or several processes sharing the GPU.
- Write intermediate results to `/workspace/results/` as JSON/CSV as you go, so a crash or the auto-termination does not lose everything. Pull partial results periodically.

## 5. Pre-registration and judging
- Before the main runs, write `<pilot folder>/PLAN.md` (Korean; technical terms may stay English): hypotheses, exact metrics, and the pass criteria **copied from `IDEAS_ROUND1.md`**. You may fill in thresholds the criteria leave open, but do not change the given ones. Note any deviation (e.g. a fallback task) in PLAN.md with the reason, before seeing the main results.
- Smoke-test the pipeline on the pod with a tiny configuration first.
- Judge with the pre-registered rules. Label anything decided after seeing results as post-hoc (사후 분석).
- Mandatory controls from the earlier study: random-rotation basis and/or untrained-model null where neuron selectivity is claimed; never evaluate steering or decoding with a probe trained on the same layer and data; include an input-copy control when analyzing action/state inputs; prefer paired counterfactual swaps over mean ablation for mechanism claims.

## 6. Outputs
- `<pilot folder>/code/` (all code you ran), `<pilot folder>/results/` (JSON/CSV and a few small PNG figures; no checkpoints or datasets), `<pilot folder>/LOG.md` (short timeline: pod, GPU, errors and fixes).
- `<pilot folder>/REPORT.md` in Korean:
  1. 판정: 있음 / 조건부 있음 / 없음 / 있음(다른 방향), criterion by criterion (S1, S2, ...), with the numbers
  2. 핵심 결과 표
  3. 해석: what this means for the idea as a physical-AI paper
  4. 위험과 한계, confounds you could not rule out
  5. 본 실험으로 키울 때의 다음 단계(구체적으로)
  6. 비용(USD), GPU, 소요 시간
- If the Write tool fails, write with a Bash heredoc. If that fails too, put the full REPORT.md text in your final message.

## 7. Final message to the coordinator
Short: verdict per criterion with key numbers, cost, anything surprising, and explicit confirmation that your pod is deleted (`gone` in budget.py status).
