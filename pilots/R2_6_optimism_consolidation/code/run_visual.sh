#!/bin/bash
# visual-version main runs (after features and the p=0.3 feasibility check) + analyses
set -e
cd /workspace/code
tr() { python train.py --version visual --N 100000 "$@"; }
tr --name v_T1_std --T 10000 --ps 0.3,0.03,0.01 --modes std
tr --name v_T1_rem --T 10000 --ps 0.01 --modes rw,hn
tr --name v_T4_std --T 40000 --ps 0.3,0.03,0.01 --modes std --snap_every 2500 --eval_every 2500
echo VIS_TRAIN_OK
python analyze.py --name v_T1_std
python analyze.py --name v_T1_rem
python analyze.py --name v_T4_std
python analyze.py --name v_T4_std --timeseries --modes std
echo VIS_OK
