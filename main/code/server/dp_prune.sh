#!/bin/bash
# B E0: keep only the 8 chosen DP training stages (EMA-only files) per seed; delete every other EMA snapshot, including
# ones the running training writes later (300/400/550/750), until all seeds are DONE. latest.pt (one full resumable
# checkpoint per seed, overwritten in place) is kept. usage: dp_prune.sh "0013 0020 ..."   (4-digit epochs)
KEEP="$1"
[ -z "$KEEP" ] && { echo "usage: dp_prune.sh \"0013 0020 ...\""; exit 1; }
echo "$(date -u +%FT%TZ) keep: $KEEP"
while true; do
  for s in 42 43 44; do
    for f in $HOME/pmx/B/ckpt/dp/s$s/ema_ep*.pt; do
      [ -e "$f" ] || continue
      e=$(basename $f .pt | sed 's/ema_ep//')
      keep=0
      for k in $KEEP; do [ "$e" = "$k" ] && keep=1; done
      if [ $keep -eq 0 ]; then rm -f "$f"; echo "$(date -u +%FT%TZ) pruned $f"; fi
    done
  done
  all=1
  for s in 42 43 44; do [ -f $HOME/pmx/B/ckpt/dp/s$s/DONE ] || all=0; done
  [ $all -eq 1 ] && break
  sleep 120
done
echo "$(date -u +%FT%TZ) prune watcher done; $(du -sh $HOME/pmx/B/ckpt/dp | cut -f1) in ckpt/dp"
