#!/bin/bash
# Shared settings for the RunPod pod(s) of paper B stage E7 (PointMaze; DESIGN_main section 12; sourced by the other scripts).
# The RunPod account is shared with the user's other studies: only pods recorded in pods.ledger are ours.
INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERM_H=${TERM_H:-4}   # every pod auto-terminates this many hours after it is created (coordinator may allow up to 8)
DEADLINE=${DEADLINE:-$(date -u -v+"${TERM_H}"H +%Y-%m-%dT%H:%M:%SZ)}
PREFIX=bx7   # B E7 (PointMaze on RunPod); not "arx" (A RSS extension), "pmx" (main experiments), "pai", "nnc" or "wmc"
LEDGER="$INFRA/pods.ledger"
SSH_KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
mkdir -p "$INFRA/pods"
