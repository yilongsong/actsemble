#!/usr/bin/env bash
# Re-run the rollout-oracle headroom with the DRIFT-FIXED two-env oracle.
#   1) K16 M1 N300  -> corrected headline headroom (candidate_zero/verifier unchanged)
#   2) K32 M1 N120  -> proposal axis
#   3) K64 M1 N120  -> proposal axis
#   4) K16 M5 N120  -> re-verify M (variance) with the fixed oracle
# then oracle capture (150) for the corrected failure viz + videos.
set -u
cd /home/yilong/actsemble
PY=/home/yilong/miniconda3/envs/actsemble/bin/python
LOG=outputs/pushonly_min/logs
mkdir -p "$LOG"

JOBS=("16 1 300" "32 1 120" "64 1 120" "16 5 120")
for job in "${JOBS[@]}"; do
  read -r K M N <<< "$job"; HALF=$(( N / 2 ))
  echo "=== [fixed] K=$K M=$M N=$N  $(date +%H:%M) ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/oracle_headroom.py run --k "$K" --m "$M" --start 0      --count "$HALF"        --force --device cuda > "$LOG/fixed_K${K}_M${M}_g0.log" 2>&1 &
  P0=$!
  CUDA_VISIBLE_DEVICES=1 $PY scripts/oracle_headroom.py run --k "$K" --m "$M" --start "$HALF" --count $(( N-HALF )) --force --device cuda > "$LOG/fixed_K${K}_M${M}_g1.log" 2>&1 &
  P1=$!
  wait $P0 $P1
  echo "=== [fixed] K=$K M=$M analyze  $(date +%H:%M) ==="
  $PY scripts/oracle_headroom.py analyze --k "$K" --m "$M" 2>&1 | grep -vE "warn|Warn" | tail -13
done

echo "=== [fixed] oracle capture (150) for viz  $(date +%H:%M) ==="
CUDA_VISIBLE_DEVICES=0 $PY scripts/oracle_headroom.py capture --mode oracle --start 0  --count 75 --force --device cuda > "$LOG/fixed_capA.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/oracle_headroom.py capture --mode oracle --start 75 --count 75 --force --device cuda > "$LOG/fixed_capB.log" 2>&1 &
wait
$PY scripts/visualize_oracle_failures.py 2>&1 | grep -vE "warn|Warn" | tail -4
CUDA_VISIBLE_DEVICES=0 $PY scripts/oracle_headroom.py video --device cuda > "$LOG/fixed_video.log" 2>&1
grep -E "video\]" "$LOG/fixed_video.log" | tail
echo "=== [fixed] ALL DONE  $(date +%H:%M) ==="
