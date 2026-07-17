#!/usr/bin/env bash
# Populate the video-first dashboard: waits for the oracle re-run to finish, then
# renders an ALIGNED video set (same stratified seeds across every system, so the
# Compare tab shows true side-by-side rollouts) + full trajectories, and rebuilds.
set -u
cd /home/yilong/actsemble
PY=/home/yilong/miniconda3/envs/actsemble/bin/python
LOG=outputs/pushonly_min/logs
POL=outputs/pushonly_min/policy_seed_0/selected_policy.pt
VER=outputs/pushonly_min/verifier_seed_0/selected_verifier.pt

echo "[finalize] waiting for oracle re-run… $(date +%H:%M)"
while pgrep -f run_oracle_fixed.sh >/dev/null; do sleep 30; done
echo "[finalize] oracle re-run done $(date +%H:%M)"

# stratified video seeds (within first 150 panel seeds so every system incl. oracle has them)
SEEDS=$($PY - <<'PYEOF'
import sys, json, importlib.util
sys.path.insert(0, "src")
from actsemble.evaluation.panels import Panel, panel_episodes
spec = importlib.util.spec_from_file_location("bd", "scripts/build_dashboard.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)
first = {e.env_seed for e in panel_episodes(Panel(name="c", env_seed=20000, num_episodes=150))}
base = {e["env_seed"]: e for e in json.load(open("outputs/pushonly_min/oracle/capture/base_0000_0300.json"))["episodes"] if e["env_seed"] in first}
from collections import defaultdict
buck = defaultdict(list)
for es, e in base.items():
    k = "success" if e["success_once"] else bd.classify(False, e["obj_xy"], e["coverage"], e["goal_xy"])
    buck[k].append(es)
pick = []
for k, lim in [("success", 8), ("near_miss", 6), ("misaligned", 6), ("mispositioned", 8), ("never_engaged", 3)]:
    pick += sorted(buck.get(k, []))[:lim]
print(",".join(map(str, pick)))
PYEOF
)
echo "[finalize] video seeds: $SEEDS"

# selector systems: full 300-ep trajectories + videos for the aligned seeds (2 GPUs)
CUDA_VISIBLE_DEVICES=0 $PY scripts/build_dashboard.py record --policy $POL --count 300 --video-seeds "$SEEDS" \
  --system '{"name":"candidate_zero","label":"Candidate Zero (base)","kind":"base","color":"#6c757d","selection":{"type":"candidate_zero"},"num_candidates":16}' > $LOG/rec_cz.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/build_dashboard.py record --policy $POL --count 300 --video-seeds "$SEEDS" \
  --system '{"name":"verifier_argmax","label":"Verifier (learned)","kind":"selector","color":"#6a4c93","selection":{"type":"highest_component_score"},"num_candidates":16,"components":["'"$VER"'"]}' > $LOG/rec_ver.log 2>&1 &
wait
CUDA_VISIBLE_DEVICES=0 $PY scripts/build_dashboard.py record --policy $POL --count 300 --video-seeds "$SEEDS" \
  --system '{"name":"full_chunk_medoid","label":"Medoid (consensus)","kind":"selector","color":"#457b9d","selection":{"type":"full_chunk_medoid"},"num_candidates":16}' > $LOG/rec_med.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/oracle_headroom.py video --seeds "$SEEDS" > $LOG/rec_oracle_vid.log 2>&1 &
wait

$PY scripts/build_dashboard.py build 2>&1 | grep -vE "warn|Warn" | tail -6
echo "[finalize] dashboard rebuilt with aligned videos $(date +%H:%M)"
