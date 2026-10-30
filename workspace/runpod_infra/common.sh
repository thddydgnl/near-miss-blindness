#!/bin/bash
# Shared settings for this study's pods (sourced by the other scripts).
# The RunPod account is shared with the user's other studies: only pods recorded in pods.ledger are ours.
INFRA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERM_H=${TERM_H:-4}   # every pod auto-terminates this many hours after it is created
DEADLINE=${DEADLINE:-$(date -u -v+"${TERM_H}"H +%Y-%m-%dT%H:%M:%SZ)}
PREFIX=pai   # not "nnc" (the "size" study) and not "wmc" (the earlier world-model study)
LEDGER="$INFRA/pods.ledger"
SSH_KEY="$HOME/.runpod/ssh/runpodctl-ssh-key"
mkdir -p "$INFRA/pods"
