#!/bin/bash
# Create the pod for one pilot. Tries GPU types in order until one is available.
# Usage: [IMAGE=<docker image>] [TERM_H=4] pod_up.sh <tag> "<gpu1>|<gpu2>|..." [container_gb=40] [volume_gb=60]
# Default is the runpod-torch-v240 template (torch 2.4, CUDA 12.4). Do not use Blackwell GPUs (RTX 5090, RTX PRO *) with it.
# Writes pods/<tag>.env (POD_ID/POD_IP/POD_PORT/POD_GPU/POD_COST) and appends an "up" event to pods.ledger.
set -u
source "$(dirname "$0")/common.sh"
TAG=$1
IFS='|' read -ra GPUS <<< "$2"
CDISK=${3:-40}
VOL=${4:-60}
NAME="$PREFIX-$TAG"
if [ -n "${IMAGE:-}" ]; then SRC=(--image "$IMAGE" --ports "22/tcp"); else SRC=(--template-id runpod-torch-v240); fi  # a bare image exposes no ssh port unless asked
if [ -f "$INFRA/pods/$TAG.env" ]; then echo "pods/$TAG.env exists; run pod_down.sh $TAG first"; exit 1; fi
for G in "${GPUS[@]}"; do
  echo "trying: $G (terminate-after $DEADLINE)"
  if runpodctl pod create "${SRC[@]}" --gpu-id "$G" --cloud-type SECURE \
       --name "$NAME" --container-disk-in-gb "$CDISK" --volume-in-gb "$VOL" --volume-mount-path /workspace \
       --terminate-after "$DEADLINE" --wait --wait-timeout 12m > "$INFRA/pods/$TAG.json" 2> "$INFRA/pods/$TAG.err"; then
    ID=$(jq -r .id "$INFRA/pods/$TAG.json")
    COST=$(jq -r .costPerHr "$INFRA/pods/$TAG.json")
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    jq -nc --arg id "$ID" --arg tag "$TAG" --arg gpu "$G" --arg cost "$COST" --arg time "$NOW" \
      '{id:$id, tag:$tag, gpu:$gpu, cost:$cost, event:"up", time:$time}' >> "$LEDGER"
    runpodctl ssh info "$ID" > "$INFRA/pods/$TAG.ssh.json"
    {
      echo "POD_ID=$ID"
      echo "POD_IP=$(jq -r .ip "$INFRA/pods/$TAG.ssh.json")"
      echo "POD_PORT=$(jq -r .port "$INFRA/pods/$TAG.ssh.json")"
      echo "POD_GPU=\"$G\""
      echo "POD_COST=$COST"
      echo "POD_CREATED=$NOW"
      echo "POD_DEADLINE=$DEADLINE"
    } > "$INFRA/pods/$TAG.env"
    cat "$INFRA/pods/$TAG.env"
    exit 0
  fi
  echo "failed: $(head -c 300 "$INFRA/pods/$TAG.err")"
  # a pod may have been created even though the ssh wait timed out: record it, then delete it only if it carries our name
  ID=$(grep -oE '[a-z0-9]{14}' "$INFRA/pods/$TAG.err" | head -1)
  if [ -n "$ID" ] && [ "$(runpodctl pod get "$ID" 2>/dev/null | jq -r .name)" = "$NAME" ]; then
    COST=$(runpodctl pod get "$ID" 2>/dev/null | jq -r .costPerHr)
    T=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    jq -nc --arg id "$ID" --arg tag "$TAG-stray" --arg gpu "$G" --arg cost "${COST:-1}" --arg time "$T" \
      '{id:$id, tag:$tag, gpu:$gpu, cost:$cost, event:"up", time:$time}' >> "$LEDGER"
    runpodctl pod delete "$ID" > /dev/null 2>&1
    jq -nc --arg id "$ID" --arg tag "$TAG-stray" --arg time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      '{id:$id, tag:$tag, event:"deleted", time:$time}' >> "$LEDGER"
    echo "deleted stray $ID"
  fi
done
echo "no gpu available"
exit 1
