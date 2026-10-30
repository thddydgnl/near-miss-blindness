#!/bin/bash
# calibration of the training length T (p=0.3 std models only; PLAN.md 4장 rule)
set -e
cd /workspace/code
mkdir -p /workspace/results /workspace/logs
python train.py --name cal --ps 0.3 --modes std --seeds 0,1,2 --Ts 5000,10000,20000,40000 --N 100000
python select_T.py /workspace/results/cal/final_metrics.json /workspace/results/cal/T_choice.json
echo CAL_OK
