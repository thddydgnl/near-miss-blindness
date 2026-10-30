"""M2 (RunPod) cost model for B's E1-E3 and the LIBERO policy training, from the E0 throughput measurements.
Standard library only. Usage: python3 cost_model.py <results/E0_runpod dir>  -> prints JSON, writes cost_model.json
All workload sizes are the design's (EXPERIMENT_DESIGN.md 6.0-6.3, 6.8); the per-unit times are measured on the
E0 pod (A40 48 GB, 7.65 vCPU cgroup quota). Pod price: A40 secure $0.59/h (13:18Z quote); storage ignored (< 2%).
"""
import json
import os
import sys

D = sys.argv[1]
cos = json.load(open(os.path.join(D, "throughput", "cosmos.json")))
sim = json.load(open(os.path.join(D, "throughput", "sim.json")))
dp = json.load(open(os.path.join(D, "throughput", "dp_bench.json")))
vlm_p = os.path.join(D, "throughput", "vlm_bench.json")
vlm = json.load(open(vlm_p)) if os.path.exists(vlm_p) else None
PRICE = 0.59
OVERHEAD = 1.25  # setup, model loading, restarts, analysis on the pod, idle while polling

q_single = cos["cf_query_act1_s"]
best_b = min((k for k in cos if k.startswith("batch") and k.endswith("_s_per_query")), key=lambda k: cos[k])
q_batch = cos[best_b]
v_real = cos["value_real_future_s"]
sim_phys = sim["par7_res0_aggregate_steps_per_s"]          # 7 parallel physics-only processes
sim_r256 = sim["par7_res256_aggregate_steps_per_s"]        # 7 parallel, 2 cameras at 256 every step
dp_main = dp[0]
vlm_s = (vlm or {}).get("batch8_s_per_judgment", 0.3)


def h(sec):
    return sec / 3600.0


out = {"unit_times": dict(cf_query_single_s=q_single, cf_query_batched_s=q_batch, batched_key=best_b,
                          value_real_future_s=v_real, sim_phys_steps_per_s_7proc=sim_phys,
                          sim_render256_steps_per_s_7proc=sim_r256, dp_train_steps_per_s=dp_main["train_steps_per_s"],
                          dp_infer_s=dp_main["infer_ddim10_s"], vlm_s_per_judgment=vlm_s)}

# ---------------- E1: ~3,000 counterfactuals with r in [0.8, 3], >= 1,000 near-misses; horizons k = 1..4 chunks
N_CF = 3000
N_ROLL = int(N_CF * 1.3)            # + calibration sweeps and discards (r outside [0.8, 3])
STEPS = 165                          # ~150 action steps + 15 settle steps (Goal demos 80-350, selected tasks ~110-150)
RENDERS = 5                          # frames needed per counterfactual: 4 horizon start states + the final state
sim_s = N_ROLL * STEPS / sim_phys + N_CF * RENDERS / (sim_r256 / 1.0) * 1.0
q_e1 = N_CF * (1 + 2 + 3 + 4)        # chained queries for k = 1..4
gpu_e1_batched = q_e1 * q_batch + N_CF * v_real
gpu_e1_single = q_e1 * q_single + N_CF * v_real
vlm_e1 = N_CF * 5 * vlm_s            # real final + 4 imagined finals (counted under E3 in the design table)
out["E1"] = dict(counterfactual_rollouts=N_ROLL, cosmos_queries=q_e1, sim_hours=h(sim_s),
                 cosmos_hours_batched=h(gpu_e1_batched), cosmos_hours_single=h(gpu_e1_single),
                 pod_hours=OVERHEAD * (h(sim_s) + h(gpu_e1_batched)),
                 usd=OVERHEAD * (h(sim_s) + h(gpu_e1_batched)) * PRICE,
                 usd_if_unbatched=OVERHEAD * (h(sim_s) + h(gpu_e1_single)) * PRICE)

# ---------------- policy training (E0/E2 input): per-task DP, 6 tasks, 1 seed, 4 checkpoint stages
TASKS, DP_STEPS = 6, 50000
render_demo_s = TASKS * 50 * 130 / sim_r256          # re-render 50 demos/task from recorded states at 256 px
train_s = TASKS * DP_STEPS / dp_main["train_steps_per_s"]
out["policy_training"] = dict(tasks=TASKS, steps_per_task=DP_STEPS, checkpoints="5k,10k,20k,50k", render_hours=h(render_demo_s),
                              train_hours=h(train_s), pod_hours=1.15 * (h(render_demo_s) + h(train_s)),
                              usd=1.15 * (h(render_demo_s) + h(train_s)) * PRICE,
                              note="3 trainings can share the A40 (VRAM %.1f GiB each) if GPU-bound throughput allows" % dp_main["vram_peak_gib"])

# ---------------- E2: policies x tasks x 50 episodes, modes O and C
N_DP_PAIRS = 4 * TASKS                # DP checkpoints
N_REF_PAIRS = 2 * TASKS               # openpi pi0.5 LIBERO, OpenVLA-OFT
N_PERT_PAIRS = 4 * TASKS              # R2-12-style offset / iid perturbation families of a strong policy (auxiliary)
EP = 50
n_ep = (N_DP_PAIRS + N_REF_PAIRS + N_PERT_PAIRS) * EP
AVG_STEPS, AVG_CHUNKS = 200, 12
# real rollouts: physics every step, render + policy query every 16 steps (DP) or every 8 steps (reference VLAs)
sim_e2 = n_ep * AVG_STEPS / sim_phys + n_ep * (AVG_STEPS / 16) * 2 / sim_r256
dp_q = (N_DP_PAIRS + N_PERT_PAIRS) * EP * (AVG_STEPS / 16) * dp_main["infer_ddim10_s"]
REF_QUERY_S = 0.15                    # assumption for pi0.5 / OpenVLA-OFT chunk queries on an A40 (not measured here)
ref_q = N_REF_PAIRS * EP * (AVG_STEPS / 8) * REF_QUERY_S
q_O_full = n_ep * AVG_CHUNKS            # mode O (design 4.1): replay the executed action sequence from the initial obs
q_C = N_DP_PAIRS * EP * AVG_CHUNKS      # mode C: DP policies act on imagined observations
q_O_lastk = n_ep * 10                   # optional diagnostic: last k = 1..4 chunks from the real obs at T - 16k
gpu_e2 = (q_O_full + q_C) * q_batch + dp_q + ref_q + q_C * dp_main["infer_ddim10_s"]
gpu_e2_lastk = q_O_lastk * q_batch
out["E2"] = dict(episodes=n_ep, cosmos_queries_O_full=q_O_full, cosmos_queries_C=q_C,
                 sim_hours=h(sim_e2), gpu_hours=h(gpu_e2), pod_hours=OVERHEAD * (h(sim_e2) + h(gpu_e2)),
                 usd=OVERHEAD * (h(sim_e2) + h(gpu_e2)) * PRICE,
                 optional_O_lastk=dict(cosmos_queries=q_O_lastk, gpu_hours=h(gpu_e2_lastk), usd=OVERHEAD * h(gpu_e2_lastk) * PRICE),
                 reference_policy_setup_note="openpi (JAX) and OpenVLA-OFT each need their own env + checkpoint download (~1 pod-hour each, incl. smoke test)")
out["E2"]["usd_incl_ref_setup"] = out["E2"]["usd"] + 2 * 1.0 * PRICE

# ---------------- E3: judges on real vs imagined frames (value head + VLM); oracle judge training is negligible
n_vlm = N_CF * 5 + n_ep * (1 + 4 + 1) + N_DP_PAIRS * EP
gpu_e3 = n_vlm * vlm_s + n_ep * v_real
out["E3"] = dict(vlm_judgments=n_vlm, value_real_queries=n_ep, gpu_hours=h(gpu_e3),
                 pod_hours=OVERHEAD * h(gpu_e3), usd=OVERHEAD * h(gpu_e3) * PRICE)
out["total_usd_E1_E2_E3_training"] = out["E1"]["usd"] + out["E2"]["usd_incl_ref_setup"] + out["E3"]["usd"] + out["policy_training"]["usd"]
json.dump(out, open(os.path.join(D, "cost_model.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
