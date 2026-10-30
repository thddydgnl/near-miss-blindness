#!/bin/bash
# tiny end-to-end smoke test (separate ckpt/results dirs; not used for judgement)
set -e
cd /workspace/code9
export R29_CK=/workspace/smoke_ckpt9 R29_OUT=/workspace/smoke9
python world.py 0 600
python wm.py clf
python wm.py train --data std --mode std --seed 0 --T 200 --N 600 --name p0.03_std_1x_s0
python wm.py train --data std --mode rw --seed 0 --T 200 --N 600 --name p0.03_rw_1x_s0
python wm.py train --data cp --mode cp --seed 0 --T 200 --N 600 --name p0.03_cp_1x_s0
python wm.py train --data nm --mode nm --seed 0 --T 200 --N 600 --name p0.03_nm_1x_s0
python wm.py train --data p03 --mode std --seed 0 --T 200 --N 600 --name p0.3_std_1x_s0
python wm.py train --data std --mode std --seed 0 --T 400 --N 600 --name p0.03_std_4x_s0
python evaluate.py p0.03_std_1x_s0,p0.03_rw_1x_s0,p0.03_cp_1x_s0,p0.03_nm_1x_s0,p0.3_std_1x_s0,p0.03_std_4x_s0
python summarize.py /workspace/smoke9 > /workspace/smoke9/summarize_console.txt
echo SMOKE_OK
