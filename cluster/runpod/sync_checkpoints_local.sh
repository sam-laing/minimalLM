#!/bin/bash
# Run this ON YOUR LOCAL MACHINE (not on the pod) — the pod can't push to your
# laptop (no public IP/listening SSH), so this pulls instead, on a loop.
#
# Use the pod's DIRECT SSH connection (Pod -> Connect -> "SSH over exposed TCP"),
# NOT the ssh.runpod.io proxy -- the proxy prints a banner on every connection
# that corrupts rsync's protocol handshake ("protocol version mismatch").
#
# Usage: bash cluster/runpod/sync_checkpoints_local.sh <user@pod-ip> [port]
# Example: bash cluster/runpod/sync_checkpoints_local.sh root@194.26.196.11 22134
set -euo pipefail

POD_HOST="${1:?Usage: $0 <user@pod-ip> [port]}"
POD_PORT="${2:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/minimalLM_checkpoints}"
INTERVAL="${INTERVAL:-300}"  # seconds between syncs, default 5 min

mkdir -p "$LOCAL_DIR"
echo "Syncing $POD_HOST:$POD_PORT:/workspace/checkpoints/ -> $LOCAL_DIR every ${INTERVAL}s. Ctrl+C to stop."

while true; do
  rsync -avz --partial -e "ssh -p $POD_PORT -i $SSH_KEY -o IdentitiesOnly=yes" \
    "${POD_HOST}:/workspace/checkpoints/" "$LOCAL_DIR/" \
    && echo "[$(date '+%H:%M:%S')] synced" \
    || echo "[$(date '+%H:%M:%S')] sync failed, will retry"
  sleep "$INTERVAL"
done
