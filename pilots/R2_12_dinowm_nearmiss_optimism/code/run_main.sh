#!/bin/bash
# main evaluation set first (judging), then the two secondary sets, then the analysis.
cd /workspace/code
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor'
for S in main H10 trainsplit; do
  python main12.py $S 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/main_$S.log
done
python analyze12.py main H10 trainsplit > /workspace/logs/analyze.log 2>&1
echo RUN_MAIN_DONE >> /workspace/logs/analyze.log
