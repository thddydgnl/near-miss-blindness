#!/bin/bash
# frame banks (3 seeds x 30k episodes + variants) and evaluation bank, then the frozen classifier
set -e
cd /workspace/code9
python world.py 0,1,2 30000
python wm.py clf
echo FEAT_OK
