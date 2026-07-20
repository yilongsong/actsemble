# Actsemble — agent instructions

## Explanation standard (user-mandated)

When explaining findings or mechanisms (in chat or in docs/), use the
interpretable ground-up format, not compressed expert summaries:

1. Define every quantity before using it, one line each, even if defined in
   an earlier session.
2. Numbered facts, each anchored to a named measurement and its number —
   never to an impression, a video, or an earlier conclusion.
3. Spell out mechanisms as explicit step-by-step loops; no metaphors doing
   load-bearing work.
4. Separate measured from hypothesized; do not use causal words ("OOD",
   "shortcut", "conservative") until a measurement has earned them — state
   what measurement would.
5. Show the control row (where the effect is absent) next to the effect row.
6. End with a one-paragraph plain-language version.

Repetition of known material is acceptable and preferred; the reader wants a
chain they can verify by reading once.

## Standing project rules

- Oracles are diagnostic-tier only: never used for training or checkpoint
  selection. Teleport results are labeled non-physical.
- Reference methods (ACT, Diffusion Policy, ...) are implemented faithful to
  the original recipe; deviations are labeled variants. Never claim to
  refute/reproduce a paper without running its exact method and comparison.
- Pin inference parameters in checkpoint + config + eval artifacts. Commit a
  clean tree before claim-grade runs. Commit only when asked.
- Development-tier results (single seed, diagnostic panels) are never
  presented as claims; claims need >= 5 seeds on the final-test protocol.
