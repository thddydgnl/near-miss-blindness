#!/bin/bash
# tiny end-to-end smoke test in /workspace/smoke (not used for any judgement)
set -e
cd /workspace/code
O="--out /workspace/smoke --ckpt /workspace/smoke_ckpt"
python train.py --name s_n100k_T1 --version state --N 5000 --T 400 --eval_every 200 --ps 0.3,0.01 --seeds 0 $O
python train.py --name s_n100k_T4 --version state --N 5000 --T 800 --eval_every 200 --snap_every 200 --ps 0.3,0.01 --modes std --seeds 0 $O
python analyze.py --name s_n100k_T1 $O
python analyze.py --name s_n100k_T4 $O
python analyze.py --name s_n100k_T4 --timeseries --modes std $O
python visual.py 0 2000
python train.py --name v_T1_std --version visual --N 2000 --T 400 --eval_every 200 --ps 0.3,0.01 --modes std --seeds 0 $O
python train.py --name v_T4_std --version visual --N 2000 --T 800 --eval_every 200 --snap_every 200 --ps 0.3,0.01 --modes std --seeds 0 $O
python analyze.py --name v_T1_std $O
python analyze.py --name v_T4_std $O
python analyze.py --name v_T4_std --timeseries --modes std $O
python summarize.py /workspace/smoke > /workspace/smoke/summarize_console.txt
python figures.py /workspace/smoke
echo SMOKE_OK
