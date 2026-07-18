#!/usr/bin/env bash
# Full-300 CORRECTED rollout-oracle capture (two-env, elapsed-reset fixes).
# Two GPUs, resumable, low-index-first. Each chunk writes a rich capture file
# outputs/active_min/oracle/capture/oracle_SSSS_EEEE.json AS IT COMPLETES, so
# the live dashboard (scripts/dashboard_watch.sh) fills in as the run proceeds.
#
# Resumable: cmd_capture skips a chunk whose output already exists, so re-running
# this after an interruption only does the missing chunks.
set -u
REPO=/home/yilong/actsemble
PY=/home/yilong/miniconda3/envs/actsemble/bin/python
OD="$REPO/outputs/active_min/oracle"
CHUNK=${CHUNK:-15}
TOTAL=${TOTAL:-300}

rm -f "$OD/CAPTURE_DONE"

worker () {                 # $1 = gpu id; runs chunks whose index %2 == gpu
  local gpu=$1 i=0 s
  for (( s=0; s<TOTAL; s+=CHUNK )); do
    if (( i % 2 == gpu )); then
      echo "[gpu$gpu $(date +%T)] capture start=$s count=$CHUNK"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" "$REPO/scripts/oracle_headroom.py" capture \
        --mode oracle --start "$s" --count "$CHUNK" --device cuda
    fi
    i=$((i+1))
  done
  echo "[gpu$gpu $(date +%T)] worker done"
}

worker 0 > "$OD/capture_gpu0.log" 2>&1 &
P0=$!
worker 1 > "$OD/capture_gpu1.log" 2>&1 &
P1=$!
wait "$P0" "$P1"
touch "$OD/CAPTURE_DONE"
echo "ALL CAPTURE CHUNKS DONE $(date)"
