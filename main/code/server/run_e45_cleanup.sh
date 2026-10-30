#!/bin/bash
# B E4/E5 disk cleanup (CPU only, inside ~/pmx/B): delete intermediate feature banks as soon as no later step needs them.
#   R1 / R2 sweep training banks  after the three R1 / R2 models are trained (evaluation needs only eval_sweep.pt, norm, classifier)
#   E1 M3 training banks          after the R3 / R4 sweep models are trained (last user); eval.pt / eval_e1.pt / norm.pt stay
#   E1-cell caches of E4          after lane C has re-imagined all variants
# All are deterministic to regenerate (scripts in code/server).
K=$HOME/pmx/B/ckpt/E4/m3_sweep; FS=$HOME/pmx/B/feat/E4/m3_sweep
done_r1=0; done_r2=0; done_e1=0; done_cells=0
while [ $((done_r1 + done_r2 + done_e1 + done_cells)) -lt 4 ]; do
  if [ $done_r1 = 0 ] && [ -f $K/sweep_R1_s0.pt ] && [ -f $K/sweep_R1_s1.pt ] && [ -f $K/sweep_R1_s2.pt ]; then
    rm -f $FS/R1/bank_s0.pt $FS/R1/bank_s1.pt $FS/R1/bank_s2.pt; done_r1=1; echo "removed R1 banks $(date -u +%FT%TZ)"; fi
  if [ $done_r2 = 0 ] && [ -f $K/sweep_R2_s0.pt ] && [ -f $K/sweep_R2_s1.pt ] && [ -f $K/sweep_R2_s2.pt ]; then
    rm -f $FS/R2/bank_s0.pt $FS/R2/bank_s1.pt $FS/R2/bank_s2.pt; done_r2=1; echo "removed R2 banks $(date -u +%FT%TZ)"; fi
  if [ $done_e1 = 0 ] && ls $K/sweep_R3_s{0,1,2}.pt $K/sweep_R4_s{0,1,2}.pt >/dev/null 2>&1; then
    rm -f $HOME/pmx/B/feat/E1/m3/bank_s0_N30000.pt $HOME/pmx/B/feat/E1/m3/bank_s1_N30000.pt $HOME/pmx/B/feat/E1/m3/bank_s2_N30000.pt
    done_e1=1; echo "removed E1 M3 training banks $(date -u +%FT%TZ)"; fi
  if [ $done_cells = 0 ] && grep -q "E45_LANEC_IMAG_DONE" $HOME/pmx/B/logs/B-E45-laneC.log 2>/dev/null; then
    rm -f $HOME/pmx/B/feat/E4/m1/cell_*.npz; done_cells=1; echo "removed E4 cell caches $(date -u +%FT%TZ)"; fi
  sleep 60
done
echo "E45_CLEANUP_DONE $(date -u +%FT%TZ)"
