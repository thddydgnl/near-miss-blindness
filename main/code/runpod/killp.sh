#!/bin/bash
# Kill processes whose command line matches the pattern $1, excluding this script and the shell that called it
# (a plain `pkill -f` from an ssh command also matches the ssh shell whose command line contains the pattern).
for pid in $(pgrep -f "$1"); do
  if [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ]; then continue; fi
  kill "$pid" 2>/dev/null && echo "killed $pid"
done
