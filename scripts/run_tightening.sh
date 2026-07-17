#!/usr/bin/env bash
# Tighten the selection-headroom ceiling upward, in the order requested:
#   1) M=5   (variance reduction, K=16)      -> honest depth-1 ceiling
#   2) K-sweep  K=32, K=64  (M=1)            -> proposal axis
#   3) both  M=5 x K=64                      -> combined tightest bound
# Each variant is sharded across both GPUs, then analyzed vs cz / verifier /
# baseline oracle. Smaller panels for the expensive (high K*M) runs.
set -u
cd /home/yilong/actsemble
PY=/home/yilong/miniconda3/envs/actsemble/bin/python
LOG=outputs/pushonly_min/logs
mkdir -p "$LOG"

# job: K M N   (N episodes from the compare panel, sharded N/2 per GPU)
JOBS=(
  "16 5 120"
  "32 1 120"
  "64 1 120"
  "64 5 40"
)

for job in "${JOBS[@]}"; do
  read -r K M N <<< "$job"
  HALF=$(( N / 2 ))
  echo "=== [tighten] K=$K M=$M N=$N (shards 0:$HALF | $HALF:$N) $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/oracle_headroom.py run --k "$K" --m "$M" --start 0     --count "$HALF"        --device cuda > "$LOG/tighten_K${K}_M${M}_g0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/oracle_headroom.py run --k "$K" --m "$M" --start "$HALF" --count $(( N-HALF )) --device cuda > "$LOG/tighten_K${K}_M${M}_g1.log" 2>&1 &
  P1=$!
  wait $P0 $P1
  echo "=== [tighten] K=$K M=$M done; analyzing $(date +%H:%M) ==="
  $PY scripts/oracle_headroom.py analyze --k "$K" --m "$M" 2>&1 | grep -vE "warn|Warn" | tail -14
done
echo "=== [tighten] ALL DONE $(date +%H:%M) ==="
