#!/usr/bin/env bash
# SINGLE-VARIABLE test of the demonstrator's post-success HOLD length.
#
# Motivation (measured, on the v1 n=25 baseline over 120 episodes):
#   * 47% of failures end with the cube STILL GRASPED, off the table, not at
#     the goal -- a stall, not a drop.
#   * 31/32 of those DRIFT AWAY: closest approach median 0.041 m at median
#     step 57, final distance median 0.099 m. None ever entered the 0.025 win.
#   * The demos give only 4.2 frames per episode inside that success window
#     (~105 frames total at n=25) because --hold trims each episode 4 frames
#     after success. There is almost no data for "you have arrived: stay".
#
# Every argument below is COPIED FROM v1's recorded provenance so this differs
# from the 47.0% baseline in ONE intended variable:
#     hold 4 -> 30
# max_steps 110 -> 140 is forced by that change (v1's motion phase needed up
# to ~106 steps; the extra budget only lets the longer hold finish) and is the
# one unavoidable coupled change -- noted, not hidden.
#
# --hard-gate restores v1's exact DESCEND rule, because the generator's
# current default is the soft gate introduced for v2.
#
# NOTE on v2: it was run on generator DEFAULTS, which silently differed from
# v1 in align_tol (0.014->0.006), noise_sigma (0.15->0.05) and max_steps
# (110->90) on top of the soft gate. Its 44.2% result therefore does NOT
# isolate the alignment gate.
set -u
cd /home/yilong/actsemble
source ~/miniconda3/etc/profile.d/conda.sh
conda activate actsemble
export CUDA_VISIBLE_DEVICES=1

POOL=outputs/pickcube/teleop_v3/teleop_100.h5
SUB=outputs/pickcube/teleop_v3/teleop_100_n25.h5
OUT=outputs/pickcube/teleop_v3/sweep_n25

echo "########## GENERATE v3 (v1 args, hold 4->30) ##########"
python scripts/generate_pickcube_teleop.py \
  --n 100 --out "$POOL" --seed 0 --device cuda \
  --hard-gate --hold 30 --max-steps 140 \
  --align-tol 0.014 --noise-sigma 0.15 --noise-alpha 0.85 --noise-taper 0.06 \
  --approach-bias 0.012 --descent-cap 0.3 --carry-cap 0.35 --close-steps 4 \
  --grasp-gap 0.005 --hover-height 0.12 --hover-tol 0.035 --kp 6.0 \
  --kyaw 1.2 --yaw-cap 0.35 --yaw-noise 0.03 --lift-clear 0.06 \
  || { echo "GENERATE FAILED"; exit 1; }

echo "########## SUBSET n=25 ##########"
python scripts/subset_dataset.py --input "$POOL" --n 25 --output "$SUB" \
  || { echo "SUBSET FAILED"; exit 1; }

echo "########## TRAIN + SCREEN + CONFIRM + FINAL (n=25, v3) ##########"
python scripts/overnight_canonical.py --family diffusion \
  --dataset "$SUB" --out-root "$OUT" \
  --max-steps 160 --train-max-steps 10500 --checkpoint-every 500 --device cuda \
  || { echo "TRAIN FAILED"; exit 1; }

echo "########## V3 n=25 DONE ##########"
