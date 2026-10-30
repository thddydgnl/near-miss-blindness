#!/bin/bash
# Sequential E0 jobs after the main probe: t9 probe pass, E1 smoke test, Cosmos throughput, DP benchmark,
# VLM judge benchmark, then the LIBERO simulation benchmark (last, on an otherwise idle CPU quota).
source /workspace/code/env.sh
L=$W/logs; mkdir -p $W/results/throughput
while pgrep -f "[p]robe_cf.py --max_fail" > /dev/null; do sleep 10; done
echo "chain start $(date -u +%H:%M:%S)"
cd $CP
$VP /workspace/code/probe_cf.py --only_tasks t9 --max_fail 400 --max_surv 40 > $L/probe_cf_t9.log 2>&1; echo "t9 probe rc=$? $(date -u +%H:%M:%S)"
$VP /workspace/code/e1_smoke.py > $L/e1_smoke.log 2>&1; echo "e1_smoke rc=$? $(date -u +%H:%M:%S)"
$VP /workspace/code/throughput.py --part cosmos > $L/throughput_cosmos.log 2>&1; echo "tp cosmos rc=$? $(date -u +%H:%M:%S)"
$VP /workspace/code/dp_bench.py > $L/dp_bench.log 2>&1; echo "dp_bench rc=$? $(date -u +%H:%M:%S)"
mkdir -p $W/cache/hf_vlm
HF_HOME=$W/cache/hf_vlm $VP /workspace/code/vlm_bench.py > $L/vlm_bench.log 2>&1; echo "vlm_bench rc=$? $(date -u +%H:%M:%S)"
while pgrep -f "[n]earmiss_sweep.py" > /dev/null; do sleep 10; done
cd /workspace/code && $VP /workspace/code/throughput.py --part sim > $L/throughput_sim.log 2>&1; echo "tp sim rc=$? $(date -u +%H:%M:%S)"
echo "CHAIN_DONE $(date -u +%H:%M:%S)"
