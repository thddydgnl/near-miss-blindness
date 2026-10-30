#!/bin/bash
# state-version main runs (PLAN.md 4장) + analyses
set -e
cd /workspace/code
tr() { python train.py --version state "$@"; }
tr --name s_n100k_T1 --N 100000 --T 10000 --ps 0.3,0.03,0.01
tr --name s_n100k_T4 --N 100000 --T 40000 --ps 0.3,0.03,0.01 --snap_every 2500 --eval_every 2500
tr --name s_n20k_T1 --N 20000 --T 10000 --ps 0.01
tr --name s_n20k_T4 --N 20000 --T 40000 --ps 0.01 --snap_every 2500 --eval_every 2500
tr --name s_n300k_T1 --N 300000 --T 10000 --ps 0.01
tr --name s_n300k_T4 --N 300000 --T 40000 --ps 0.01 --snap_every 2500 --eval_every 2500
tr --name s_p03_n6k_T1 --N 6667 --T 10000 --ps 0.03
tr --name s_p03_n6k_T4 --N 6667 --T 40000 --ps 0.03 --snap_every 2500 --eval_every 2500
tr --name s_p03_n33k_T1 --N 33333 --T 10000 --ps 0.03
tr --name s_p03_n33k_T4 --N 33333 --T 40000 --ps 0.03 --snap_every 2500 --eval_every 2500
echo STATE_TRAIN_OK
for r in s_n100k_T1 s_n100k_T4 s_n20k_T1 s_n20k_T4 s_n300k_T1 s_n300k_T4 s_p03_n6k_T1 s_p03_n6k_T4 s_p03_n33k_T1 s_p03_n33k_T4; do
  python analyze.py --name $r
done
for r in s_n100k_T4 s_n20k_T4 s_n300k_T4 s_p03_n6k_T4 s_p03_n33k_T4; do
  python analyze.py --name $r --timeseries --modes std
done
echo STATE_OK
