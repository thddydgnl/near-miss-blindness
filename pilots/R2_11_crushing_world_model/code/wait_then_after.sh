#!/bin/bash
while [ $(ls /workspace/ckpt/*_final.pt 2>/dev/null | wc -l) -lt 14 ]; do sleep 10; done
sleep 20
bash /workspace/code/run_after.sh > /workspace/logs/after.log 2>&1
