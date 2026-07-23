#!/usr/bin/env bash
# Regenerate the teleop pool with the SOFT alignment gate (teleop_v2), cut the
# n=25 subset, and retrain the n=25 cell under the IDENTICAL protocol used for
# the v1 result (10500 steps, checkpoint every 500, horizon 160, 50-ep
# screening -> 200-ep confirmation -> 500-ep final test).
#
# The comparison this exists to make:
#     v1 (hard gate)  n=25 final = 47.0%  CI [42.7, 51.4]
#     v2 (soft gate)  n=25 final = ?
# Everything except the demonstrator's DESCEND rule is held fixed.
set -u
cd /home/yilong/actsemble
source ~/miniconda3/etc/profile.d/conda.sh
conda activate actsemble
export CUDA_VISIBLE_DEVICES=1

POOL=outputs/pickcube/teleop_v2/teleop_100.h5
SUB=outputs/pickcube/teleop_v2/teleop_100_n25.h5
OUT=outputs/pickcube/teleop_v2/sweep_n25

echo "########## GENERATE (100 episodes, soft gate) ##########"
python scripts/generate_pickcube_teleop.py --n 100 --out "$POOL" --seed 0 --device cuda \
  || { echo "GENERATE FAILED"; exit 1; }

echo "########## SUBSET n=25 ##########"
python scripts/subset_dataset.py --input "$POOL" --n 25 --output "$SUB" \
  || { echo "SUBSET FAILED"; exit 1; }

echo "########## TRAIN + SCREEN + CONFIRM + FINAL (n=25, v2) ##########"
python scripts/overnight_canonical.py --family diffusion \
  --dataset "$SUB" --out-root "$OUT" \
  --max-steps 160 --train-max-steps 10500 --checkpoint-every 500 --device cuda \
  || { echo "TRAIN FAILED"; exit 1; }

echo "########## V2 n=25 DONE ##########"
