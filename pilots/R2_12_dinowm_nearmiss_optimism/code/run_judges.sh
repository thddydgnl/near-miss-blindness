#!/bin/bash
cd /workspace/code
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded'
python judges12.py 2>&1 | grep --line-buffered -v -i "$F" > /workspace/logs/judges.log
