#!/bin/bash
# Rerun of the LIBERO simulation benchmark with spawned workers (the first run hung: EGL contexts inherited by forked
# children), then the VLM prompt/resolution follow-up.
source /workspace/code/env.sh
L=$W/logs
cd /workspace/code && $VP /workspace/code/throughput.py --part sim > $L/throughput_sim2.log 2>&1; echo "tp sim2 rc=$? $(date -u +%H:%M:%S)"
HF_HOME=$W/cache/hf_vlm $VP /workspace/code/vlm_bench2.py > $L/vlm_bench2.log 2>&1; echo "vlm_bench2 rc=$? $(date -u +%H:%M:%S)"
echo "CHAIN2_DONE $(date -u +%H:%M:%S)"
