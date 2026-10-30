#!/bin/bash
# tiny end-to-end smoke test (separate output dirs; not used for any judgement)
set -e
cd /workspace/code
python train.py --name main --ps 0.3,0.1,0.03,0.01 --modes std,rw --seeds 0 --Ts 400 --N 5000 --eval_every 200 --out /workspace/smoke --ckpt /workspace/smoke_ckpt
python analyze.py --name main --out /workspace/smoke --ckpt /workspace/smoke_ckpt
python finetune.py --name main --out /workspace/smoke --ckpt /workspace/smoke_ckpt --steps 20
python summarize.py /workspace/smoke > /workspace/smoke/summarize_console.txt
python figures.py /workspace/smoke main
echo SMOKE_OK
