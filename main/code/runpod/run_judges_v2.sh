#!/bin/bash
# Judges on the corrected E0b data (cf_v2): (a) DINOv2+logistic and CNN with the pre-designated training split d00-d16,
# held d17-d19; (d) 7B compare2/compare3, then 32B only if no 7B variant reaches held AUROC 0.80.
source /workspace/code/env.sh
while pgrep -f "gen_e0b.py --workers" > /dev/null; do sleep 15; done
echo "gen v2 finished $(date -u +%H:%M:%S), $(ls /workspace/e0b/cf_v2/*.json | wc -l) cf"
cd /workspace/code
A="--cf /workspace/e0b/cf_v2 --out /workspace/results/judge_a_v2 --train_demos 0-16 --tag _v2"
$VP judge_a.py --stage features $A > /workspace/logs/judge_a_v2_features.log 2>&1; echo "features rc=$?"
$VP judge_a.py --stage logreg $A > /workspace/logs/judge_a_v2_logreg.log 2>&1; echo "logreg rc=$?"
$VP judge_a.py --stage cnn $A > /workspace/logs/judge_a_v2_cnn.log 2>&1; echo "cnn rc=$? $(date -u +%H:%M:%S)"
HF_HOME=/workspace/cache/hf_vlm $VP judge_d.py --model 7b --variants compare2 compare3 --cf /workspace/e0b/cf_v2 --out /workspace/results/judge_d_v2 > /workspace/logs/judge_d_v2_7b.log 2>&1
echo "judge_d 7b rc=$? $(date -u +%H:%M:%S)"
PASS=$($VP -c "import json; d=json.load(open('/workspace/results/judge_d_v2/judge_d_7b.json')); print(int(any(d[v]['passes_criterion'] for v in ('compare2','compare3'))))")
if [ "$PASS" = "0" ]; then
  HF_HOME=/workspace/cache/hf_vlm $VP judge_d.py --model 32b --variants compare2 compare3 --max_train 400 --cf /workspace/e0b/cf_v2 --out /workspace/results/judge_d_v2 > /workspace/logs/judge_d_v2_32b.log 2>&1
  echo "judge_d 32b rc=$? $(date -u +%H:%M:%S)"
fi
echo "JUDGES_V2_DONE $(date -u +%H:%M:%S)"
