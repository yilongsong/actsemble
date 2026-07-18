# The latency rule — "no-pause real-time contract"

**Status: ADOPTED 2026-07-17 (revisable).** Every reported closed-loop result must
be achievable under a real-time, no-pause deployment constraint. We do NOT credit a
replanning frequency the policy could not actually sustain on real hardware.

## The rule

For control frequency `f_c` (Hz) and a policy's single-query inference latency `τ` (s):

```
H_a ≥ ⌈τ · f_c⌉                         (minimum action horizon)
replan_freq ≤ f_c / ⌈τ · f_c⌉           (maximum sustainable replan frequency)
```

**Why:** while executing the current chunk you compute the next one in parallel. The
next chunk must be ready before the current buffer drains, i.e. `τ ≤ H_a / f_c`.
Equivalently, `τ·f_c` = the number of actions consumed during one inference, so the
buffer must hold at least `⌈τ·f_c⌉` (fast policy → 1; slow policy → more).

## Measurement discipline
- Measure `τ` as a **high percentile (p95/p99)**, not the mean — a single slow
  inference stalls the robot.
- Warm up the policy/component once before measurement and synchronize CUDA at
  both timing boundaries. Actsemble evaluation does both and stores synchronized
  mean, p95, and p99 policy/component/decision latencies in every result.
- Measure on a **named target platform** (state the GPU / edge device); `τ` is
  hardware-specific.
- Optional safety margin `α < 1`: `H_a ≥ ⌈τ·f_c / α⌉` for jitter headroom.

## The honesty clause (why it matters in sim)
In simulation the env is **frozen during inference**, so any policy can run H_a=1
(20 Hz) regardless of `τ` — a free lunch that does not exist on a real robot. We must
NOT report that as a deployable result. Every reported closed-loop number is tagged
with `(τ, platform, H_a_min, replan_freq)` and evaluated at `H_a ≥ H_a_min`.

## Example (`f_c = 20 Hz`, tick = 50 ms)
| policy | τ | `τ·f_c` (ticks) | `H_a_min` | max replan |
|---|---|---|---|---|
| ACT | 1.3 ms | 0.03 | 1 | 20 Hz |
| diffusion DDIM-10 | 17 ms | 0.34 | 1 | 20 Hz |
| diffusion DDPM-100 | 170 ms | 3.4 | 4 | 5 Hz |

Higher-dim policies (image / VLA) and higher real control rates (50–100 Hz) push
`τ·f_c` up → larger `H_a_min` → the latency wall bites, and fast policies (ACT,
distilled / consistency-model diffusion) win.

## Planned latency-constrained benchmark mode
The latency-constrained benchmark evaluates each policy at its `H_a_min` and reports
"best closed-loop success achievable under the no-pause contract" — making inference
speed a first-class performance lever (Actsemble's inference-time-compute thesis). A
mode on `scripts/sweep_replan_frequency.py`: measure p95 `τ` per policy, compute
`H_a_min`, evaluate there.
