#!/bin/bash
set -e
cd /workspace/code
O="--out /workspace/smoke --ckpt /workspace/smoke_ckpt"
python analyze.py --name v_T1_std $O
python analyze.py --name v_T4_std $O
python analyze.py --name v_T4_std --timeseries --modes std $O
python summarize.py /workspace/smoke > /workspace/smoke/summarize_console.txt
python figures.py /workspace/smoke
echo SMOKE_OK
