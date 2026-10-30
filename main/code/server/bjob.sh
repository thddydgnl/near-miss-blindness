#!/bin/bash
# Run any B command with the B environment (GPU 1 only, caches in ~/pmx). Use for every tmux launch:
#   tmux new-session -d -s B-X "bash ~/pmx/B/code/bjob.sh <command ...> > ~/pmx/B/logs/B-X.log 2>&1"
source $HOME/pmx/B/code/benv.sh
cd $BCODE
exec "$@"
