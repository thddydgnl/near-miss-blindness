#!/bin/bash
cd /workspace/code
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor'
python main12.py smoke 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/main_smoke.log
