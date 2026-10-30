#!/bin/bash
# B-E0-dp: PushT Diffusion Policy (low-dim CNN, standard demos), 3 seeds concurrently on GPU 1.
# Each seed has a retry loop; train_dp.py resumes from latest.pt. Main log: one status line every 10 min.
source $HOME/pmx/B/code/benv.sh
EPOCHS=1000
SNAPS=2,4,6,8,10,13,16,20,25,30,40,50,65,80,100,130,170,220,300,400,550,750,1000
OUT=$HOME/pmx/B/ckpt/dp
LOG=$HOME/pmx/B/logs
cd $BCODE
echo "$(date -u +%FT%TZ) B-E0-dp start: epochs $EPOCHS, snapshots $SNAPS, seeds 42 43 44, GPU $CUDA_VISIBLE_DEVICES"
run_seed() {
  local s=$1 d=$OUT/s$1
  mkdir -p $d
  for try in 1 2 3 4 5; do
    [ -f $d/DONE ] && break
    echo "$(date -u +%FT%TZ) seed $s start (try $try)"
    $BPY_DP train_dp.py --seed $s --out $d --epochs $EPOCHS --snaps $SNAPS --ckpt_every 25 >> $LOG/B-E0-dp_s$s.log 2>&1
    echo "$(date -u +%FT%TZ) seed $s exit $? (try $try)"
    sleep 20
  done
}
for s in 42 43 44; do run_seed $s & done
sleep 60
while true; do
  n=0; line="$(date -u +%FT%TZ) status"
  for s in 42 43 44; do
    e=$(tail -1 $OUT/s$s/train_log.jsonl 2>/dev/null | python3 -c "import sys,json; t=sys.stdin.read().strip(); d=json.loads(t) if t else {}; print('epoch', d.get('epoch','-'), 'loss', round(d.get('train_loss',0),4), 'sec/ep', round(d.get('sec',0),1))" 2>/dev/null)
    line="$line | s$s $e"
    [ -f $OUT/s$s/DONE ] && n=$((n+1))
  done
  echo "$line; disk free $(df -h ~ | tail -1 | awk '{print $4}')"
  [ $n -eq 3 ] && break
  sleep 600
done
wait
echo "$(date -u +%FT%TZ) B-E0-dp DONE"
