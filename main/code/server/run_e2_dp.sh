#!/bin/bash
# B E2 M1: Diffusion-Policy rollouts on the evaluation seeds 100000-100099, split over two lanes (B keeps <= 2 GPU jobs
# on GPU 1, shared with paper A). Same env as the E1 a_std judge-training rollouts (b-dinowm).
# usage: run_e2_dp.sh A|B
source $HOME/pmx/B/code/benv.sh
export B_RES=$HOME/pmx/B/results/E2/m1 B_FEAT=$HOME/pmx/B/feat/E2/m1
cd $BCODE
F='warn\|gym\|migration\|torch.load\|pkg_resources\|xformers\|Loaded\|emb_dim\|repeat\|encoder\|Conv1d\|^)\|predictor\|distutils\|Gymnasium\|Please upgrade\|See the migration\|parse\|resetCache\|Using cache\|get_cmap'
case $1 in
  A) L=s42_ep0020,s42_ep0025,s42_ep0030,s42_ep0040,s42_ep0050,s42_ep0080,s42_ep0220,s42_ep1000,s43_ep0020,s43_ep0025,s43_ep0030,s43_ep0040 ;;
  B) L=s44_ep0020,s44_ep0025,s44_ep0030,s44_ep0040,s44_ep0050,s44_ep0080,s44_ep0220,s44_ep1000,s43_ep0050,s43_ep0080,s43_ep0220,s43_ep1000 ;;
esac
$BPY_DINOWM e2_dp_rollouts.py $L 2>&1 | grep -v -i "$F"
echo "DP_LANE_$1_DONE $(date -u +%FT%TZ)"
