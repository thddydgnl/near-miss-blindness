#!/bin/bash
# R2-11 main training run (pre-registered in PLAN.md): two processes (seed 0, seed 1), 7 crushing configs each, lockstep.
cd /workspace/code
for s in 0 1; do
  python train.py --data /workspace/data --out /workspace/results --ckpt /workspace/ckpt --seed $s \
    --steps 6000 --bs 256 --lr 6e-4 --warm 300 --lr_final 0.1 --n_eval 3000 \
    --evals 0,500,1000,2000,3000,4000,5000,6000 > /workspace/logs/train_s$s.log 2>&1 &
done
wait
echo TRAIN_DONE
