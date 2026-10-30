#!/bin/bash
# R2-11 post-training: analysis (probes, confusion, neurons, merged check, shares) and CEM in parallel, then judging.
cd /workspace/code
python analyze.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results --seeds 0,1 \
  --n_boot 200 --n_rot 10 > /workspace/logs/analyze.log 2>&1 &
python cem.py --data /workspace/data --ckpt /workspace/ckpt --out /workspace/results > /workspace/logs/cem.log 2>&1 &
wait
python judge.py --res /workspace/results > /workspace/logs/judge.log 2>&1
echo AFTER_DONE
