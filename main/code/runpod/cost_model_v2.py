"""M2 cost model v2 (E0b): E1/E2/E3 with the multi-pod runner, from E0 + E0b measurements on A40 pods.
Standard library only. Usage: python3 cost_model_v2.py > cost_model_v2.json
Unit times (A40, 7.65 vCPU, measured):
  Cosmos query (1 chunk, 1-step action pass)          0.90 s  (E0 throughput; batching gives no gain)
  imagine job (k = 1..4 chains + lift = 14 queries,
               real-state value, DINOv2 features,
               CNN judge (a), VLM 7B judge (d) x 5)    14.4 s  (E0b runner smokes: 13 s chains + 0.5 s + judges)
  cf job (physics rollout + settle/lift + 7 renders)   2.4 cf/s with 7 workers (E0b runner validation: 1,056 in 602 s, 5 workers)
  policy episode (DP, real rollout + renders)          ~7 s per worker on a dedicated pod (successes ~4 s, failures to
                                                       300 steps ~10 s; E0b dp_eval under CPU contention ~20 s)
  VLM 7B judgment (compare2, batch 1 in-job)           0.07 s
Pod price A40 $0.59/h; per-pod setup ~0.25 h (setup_pod_v2.sh 8 min + model load); OVERHEAD 15% for restarts and
idle polling; runner pods use TERM_H = 8 (each pod: setup + <= 7.5 h of work).
"""
import json
import math

PRICE, OVERHEAD, SETUP_H, POD_H = 0.59, 1.15, 0.25, 7.5
Q, IMAG_JOB, CF_RATE, EP_S, VLM_S = 0.90, 14.4, 2.4, 7.0, 0.07
TASKS = 6
out = {}


def pods_for(work_h):
    return math.ceil(work_h / POD_H)


def cost(work_h):
    n = pods_for(work_h)
    pod_h = (work_h + n * SETUP_H) * OVERHEAD
    return n, pod_h, pod_h * PRICE


# ---- E1: candidates (phase 1) then selected counterfactuals (phase 2)
cand = 11280                                      # make_manifest.py e1-candidates with e1_severity_settings_v2.json
sel = TASKS * (500 + 100 + 100)                   # 500 in r [0.8, 3] (11 bins) + 100 successes + 100 far (r >= 3)
phase1_h = cand / CF_RATE / 3600
phase2_h = sel * IMAG_JOB / 3600
n, ph, usd = cost(phase1_h + phase2_h)
out["E1"] = dict(candidates=cand, selected=sel, cosmos_queries=sel * 14, phase1_cpu_h=round(phase1_h, 2),
                 phase2_gpu_h=round(phase2_h, 2), pods=n, pod_hours=round(ph, 1), usd=round(usd, 2),
                 note="phase 1 is CPU-bound with the GPU idle; phase 2 GPU-bound with the CPU mostly idle")

# ---- E2: real episodes + imagination (mode O for all, mode C for DP checkpoints)
H_C = {1: 9, 2: 11, 4: 9, 5: 16, 6: 11, 8: 9}
avg_hc = sum(H_C.values()) / len(H_C)
pairs = dict(dp=4 * TASKS, ref=2 * TASKS, pert=4 * TASKS)
EP = 50
n_ep = sum(pairs.values()) * EP
avg_chunks_O = 12                                  # mean executed chunks per episode (successes ~6-9, failures 19)
real_h = n_ep * EP_S / 7 / 3600                    # 7 CPU workers
mode_O_h = n_ep * ((avg_chunks_O + 1) * Q + 0.5 + 0.2 + 3 * VLM_S) / 3600
mode_C_h = pairs["dp"] * EP * ((avg_hc + 1) * Q + avg_hc * 0.075) / 3600
ref_setup_h = 2.0                                  # openpi pi0.5 env + OpenVLA-OFT env and checkpoints, smoke tests
n, ph, usd = cost(real_h + mode_O_h + mode_C_h + ref_setup_h)
out["E2"] = dict(episodes=n_ep, pairs=pairs, avg_H_C=round(avg_hc, 1), real_rollout_h=round(real_h, 2),
                 mode_O_gpu_h=round(mode_O_h, 2), mode_C_gpu_h=round(mode_C_h, 2), ref_setup_h=ref_setup_h, pods=n,
                 pod_hours=round(ph, 1), usd=round(usd, 2),
                 note="reference-policy (pi0.5, OFT) real rollouts assumed ~0.15 s per query, not measured")

# ---- E3: judges inside the E1/E2 jobs ((c) values, (a) features + logistic/CNN); VLM only if (d) passes
vlm_judgments = sel * 5 + n_ep * 3
e3_h = vlm_judgments * VLM_S / 3600 + 0.5            # + oracle training / AUROC analyses on the pod
n, ph, usd = cost(e3_h)
out["E3"] = dict(vlm_judgments=vlm_judgments, gpu_h=round(e3_h, 2), pods=n, pod_hours=round(ph, 1), usd=round(usd, 2),
                 note="(c) and (a) are computed inside E1/E2 jobs (included there); this line is the VLM pass if (d) is kept")
out["total_usd"] = round(out["E1"]["usd"] + out["E2"]["usd"] + out["E3"]["usd"], 2)
print(json.dumps(out, indent=1))
