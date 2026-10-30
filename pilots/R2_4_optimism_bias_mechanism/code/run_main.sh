#!/bin/bash
# main run: 24 models (p x {std,rw} x 3 seeds) at N=100k, analyses, fine-tune; secondary N runs in parallel
set -e
cd /workspace/code
T=$(python -c "import json;print(json.load(open('/workspace/results/cal/T_choice.json'))['T'])")
echo "T=$T"
python train.py --name main --Ts $T --N 100000
( python train.py --name n20k --Ts $T --N 20000 > /workspace/logs/n20k.log 2>&1; \
  python train.py --name n300k --Ts $T --N 300000 > /workspace/logs/n300k.log 2>&1 ) &
SEC=$!
python analyze.py --name main
python finetune.py --name main
wait $SEC
python summarize.py /workspace/results > /workspace/results/summarize_console.txt
python figures.py /workspace/results main
echo MAIN_OK
