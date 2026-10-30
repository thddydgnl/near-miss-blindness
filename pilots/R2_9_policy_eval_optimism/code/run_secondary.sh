#!/bin/bash
# optional secondary conditions (PLAN 4장), launched only if time allows. usage: run_secondary.sh SEED
set -e
cd /workspace/code9
S=$1
T1=$(python -c "import json;print(json.load(open('/workspace/results9/T1_choice.json'))['T1'])")
T4=$((4*T1))
run() { if [ ! -f /workspace/ckpt9/$1.pt ]; then python wm.py train --data $2 --mode $3 --seed $S --T $4 --name $1; fi; python evaluate.py $1; }
run p0.03_cp_4x_s$S cp cp $T4
run p0.03_rw_4x_s$S std rw $T4
echo SECONDARY_OK_$S
