#!/usr/bin/env bash
# Live dashboard: rebuild whenever the ingested inputs (oracle capture, paired
# eval, recorded captures, videos) change on disk. Reusable across experiments —
# it just re-runs build_dashboard.py, which ingests whatever exists right now.
#
# Terminates on the CAPTURE_DONE sentinel (after a final rebuild) or after MAXSEC.
# Usage: dashboard_watch.sh [interval_sec=90] [maxsec=36000]
set -u
REPO=/home/yilong/actsemble
PY=/home/yilong/miniconda3/envs/actsemble/bin/python
OD="$REPO/outputs/pushonly_min/oracle"
WATCH_DIRS=("$OD/capture" "$REPO/outputs/pushonly_min/compare" "$OD/videos" "$REPO/dashboard/captures")
INTERVAL=${1:-90}
MAXSEC=${2:-36000}
start=$(date +%s)

sig_of () { find "${WATCH_DIRS[@]}" -type f \( -name '*.json' -o -name '*.mp4' \) \
              -printf '%T@ %s %p\n' 2>/dev/null | sort | md5sum; }
build () { "$PY" "$REPO/scripts/build_dashboard.py" build; }

last=""
while true; do
  sig=$(sig_of)
  if [[ "$sig" != "$last" ]]; then
    if build; then
      echo "[watch $(date +%T)] rebuilt: $(ls "$OD"/capture/oracle_*.json 2>/dev/null | wc -l) oracle shard(s)"
    fi
    last="$sig"
  fi
  if [[ -f "$OD/CAPTURE_DONE" ]]; then
    build; echo "[watch $(date +%T)] CAPTURE_DONE -> final rebuild, exiting"; break
  fi
  now=$(date +%s)
  if (( now - start > MAXSEC )); then echo "[watch $(date +%T)] MAXSEC reached, exiting"; break; fi
  sleep "$INTERVAL"
done
