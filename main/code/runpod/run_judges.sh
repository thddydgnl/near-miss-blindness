#!/bin/bash
# After the E0b counterfactual generation: judge (a) features -> logistic regression (+ CNN fine-tune), then judge (d)
# with the 7B VLM (two prompt variants); the 32B step is run separately only if no 7B variant passes.
source /workspace/code/env.sh
while pgrep -f "gen_e0b.py --workers" > /dev/null; do sleep 15; done
echo "gen finished $(date -u +%H:%M:%S), $(ls /workspace/e0b/cf/*.json | wc -l) cf"
cd /workspace/code
$VP judge_a.py --stage features > /workspace/logs/judge_a_features.log 2>&1; echo "features rc=$? $(date -u +%H:%M:%S)"
$VP judge_a.py --stage logreg > /workspace/logs/judge_a_logreg.log 2>&1; echo "logreg rc=$? $(date -u +%H:%M:%S)"
$VP judge_a.py --stage cnn > /workspace/logs/judge_a_cnn.log 2>&1; echo "cnn rc=$? $(date -u +%H:%M:%S)"
HF_HOME=/workspace/cache/hf_vlm $VP judge_d.py --model 7b --variants compare2 compare3 > /workspace/logs/judge_d_7b.log 2>&1; echo "judge_d 7b rc=$? $(date -u +%H:%M:%S)"
echo "JUDGES_DONE $(date -u +%H:%M:%S)"
