#!/bin/bash
# Judge (d) step 2: Qwen2.5-VL-32B-Instruct (4-bit NF4), both prompt variants, all held rows + 400 random training rows.
source /workspace/code/env.sh
while pgrep -f "judge_a.py --stage" > /dev/null || pgrep -f "run_judge_a2.sh" > /dev/null; do sleep 15; done
cd /workspace/code
HF_HOME=/workspace/cache/hf_vlm $VP judge_d.py --model 32b --variants compare2 compare3 --max_train 400 > /workspace/logs/judge_d_32b.log 2>&1
echo "judge_d 32b rc=$? $(date -u +%H:%M:%S)"
