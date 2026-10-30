#!/bin/bash
# primary conditions for one seed: train then evaluate (PLAN 4장). usage: run_seed.sh SEED
set -e
cd /workspace/code9
S=$1
T1=$(python -c "import json;print(json.load(open('/workspace/results9/T1_choice.json'))['T1'])")
T4=$((4*T1))
run() { if [ ! -f /workspace/ckpt9/$1.pt ]; then python wm.py train --data $2 --mode $3 --seed $S --T $4 --name $1; fi; python evaluate.py $1; }
run p0.03_std_1x_s$S std std $T1
run p0.03_cp_1x_s$S cp cp $T1
run p0.03_rw_1x_s$S std rw $T1
run p0.03_std_4x_s$S std std $T4
run p0.3_std_1x_s$S p03 std $T1
run p0.03_nm_1x_s$S nm nm $T1
echo PRIMARY_OK_$S
