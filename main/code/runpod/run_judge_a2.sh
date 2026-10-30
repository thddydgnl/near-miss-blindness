#!/bin/bash
# Judge (a) second attempt: training demos d00-d16 (calibration + judge-train start states), held d17-d19 unchanged.
# Features of d00-d09 are computed first (the first attempt only featurized d10-d19).
source /workspace/code/env.sh
while pgrep -f "judge_d.py --model" > /dev/null; do sleep 15; done
cd /workspace/code
$VP judge_a.py --stage features --train_demos 0-16 --tag _d0-16 > /workspace/logs/judge_a_features2.log 2>&1; echo "features2 rc=$?"
$VP judge_a.py --stage logreg --train_demos 0-16 --tag _d0-16 > /workspace/logs/judge_a_logreg2.log 2>&1; echo "logreg2 rc=$?"
$VP judge_a.py --stage cnn --train_demos 0-16 --tag _d0-16 > /workspace/logs/judge_a_cnn2.log 2>&1; echo "cnn2 rc=$?"
echo "JUDGE_A2_DONE $(date -u +%H:%M:%S)"
