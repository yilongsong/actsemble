# Failure Detection + Recovery: Formal Scheme and Ceiling Oracle

Status: DESIGN (2026-07-18). Part I formalizes the proposed detection+recovery
scheme (Tracks E2/E4 + F1/F3 instantiation). Part II specifies the simulation
oracle that measures the scheme-class ceiling before anything is trained.
Epistemic tier: the oracle is a **diagnostic** (same standing as C6 and the
selection oracle) — never used for training or checkpoint selection.

---

# Part I — The scheme, formalized

## 1. Setting and notation

**Process.** A discrete-time controlled Markov process with full simulator state
$s \in \mathcal{S}$, control action $a \in \mathcal{A} \subset \mathbb{R}^{d_a}$
(compact box), transition kernel $P(s' \mid s, a)$, control frequency $f_c$
(PushT: 20 Hz; ManiSkill3 physics is deterministic given $(s, a)$, so $P$ is a
point mass — we keep the kernel notation so the definitions survive stochastic
simulators and hardware). Episodes have a hard horizon $T_{\max}$ (100 steps).

**Observation.** The policy observes $x = \phi(s) \in \mathcal{X} \subset
\mathbb{R}^{d_x}$. In our state-mode PushT, $d_x = 31$ with the exact layout

| slice | dims | content | role |
|---|---|---|---|
| `agent.qpos` | 0–6 | 7 arm joint positions | robot |
| `agent.qvel` | 7–13 | 7 arm joint velocities | robot |
| `extra.tcp_pose` | 14–20 | stick-tip pose (3 pos + 4 quat); FK-redundant with qpos | robot |
| `extra.goal_pos` | 21–23 | goal position | constant context |
| `extra.obj_pose` | 24–30 | T-block pose (3 pos + 4 quat) | environment |

The **ambient** dimension is 31; the **intrinsic** dimension of demonstrated
states is ~5 (planar tip position, planar block SE(2): $x, y, \theta_z$; the
velocities live near a slow manifold). Nonparametric estimators below scale with
intrinsic, not ambient, dimension — this is why kNN is viable here.

**Success.** A goal set $G \subset \mathcal{S}$ (coverage $\ge \theta$). The
episode outcome is success-once:
$$Y(\tau) \;=\; \mathbb{1}\!\left[\exists\, t \le T_{\max}: s_t \in G\right].$$

**Data.** $\mathcal{D} = \{\tau^{(i)}\}_{i=1}^{N}$, $N = 400$ demonstrations,
each with $Y = 1$ (success-only; RL-generated, trailing-hold-trimmed; mean
length 35, range 17–78). Let $\mu_{\mathcal{D}}$ be the empirical state
distribution of $\mathcal{D}$ on $\mathcal{X}$ with density $p_{\mathcal{D}}$
(w.r.t. Lebesgue on the intrinsic chart). "The demonstrated manifold" is
shorthand for the **effective support**
$$\mathcal{M}_\varepsilon \;=\; \{x : p_{\mathcal{D}}(x) \ge \varepsilon\},$$
not a literal manifold.

**Nominal system.** The deployed system $\Sigma_{\text{nom}}$ (policy $\pi$ +
replanning at chunk boundaries) carries internal state $h_t$ = (observation
history of length $H_o$, cached chunk, position within chunk). The closed-loop
process is Markov on the extended state $(s_t, h_t)$. All values are defined at
**replan boundaries** (cached chunk exhausted; $h$ reduces to the observation
history), on the grid $\mathcal{T} = \{0, H_a, 2H_a, \dots\}$.

**Values.** With $k = T_{\max} - t$ remaining steps,
$$V_{\text{nom}}(s, h, k) \;=\; \Pr\!\left(\exists\, t' \le k :\ s_{t'} \in G
\;\middle|\; (s_0,h_0) = (s,h),\ \text{follow } \Sigma_{\text{nom}}\right).$$
The probability is over policy sampling noise (and simulator noise if any).
Values are **time-inhomogeneous** through $k$: a state near the goal with 5
steps left differs from the same state with 60. Any estimator must carry the
clock.

## 2. The factorization at the core of the proposal

**Assumption A-split.** There is a measurable splitting
$x = (x_R, x_E)$ into robot-referenced coordinates $x_R \in \mathcal{X}_R$
(proprioception; dims 0–20 above) and environment coordinates $x_E \in
\mathcal{X}_E$ (dims 24–30; the constant goal is context). In image
observations, $x_R$ is **still the low-dimensional proprioception** (always
available on a real robot) and $x_E$ is the image with the robot masked/cropped
out (requires a robot mask — the one nontrivial high-dim prerequisite).

The demonstrated density factors as
$$p_{\mathcal{D}}(x_R, x_E) \;=\; p_{\mathcal{D}}(x_E)\; \rho(x_R \mid x_E),
\qquad \rho(x_R \mid x_E) := p_{\mathcal{D}}(x_R \mid x_E),$$
and deployment-time covariate shift decomposes into two distinct events:

- **environment-marginal shift**: $p_{\mathcal{D}}(x_E)$ small — the scene
  itself is somewhere demos never were (block overshot into an undemonstrated
  configuration);
- **robot-conditional shift**: $p_{\mathcal{D}}(x_E)$ fine but
  $\rho(x_R \mid x_E)$ small — the scene is familiar, the robot is somewhere
  no demo ever had it *given this scene*.

The proposal targets the second event: it is (i) detectable from
$\rho$, and (ii) **actionable** — it can be repaired by moving only the robot,
which is the one thing we can command directly without touching the task.

## 3. Population objects

All components of the scheme are estimators of exactly three population
objects (this is what makes the low-dim and high-dim versions the *same
method*):

1. the conditional $\rho(x_R \mid x_E)$ — used twice: its **log-density at the
   current state** is the detection statistic, and its **samples** are the
   recovery goal generator;
2. the marginal $p_{\mathcal{D}}(x_E)$ — the applicability gate (recovery
   cannot repair environment-marginal shift);
3. the nominal value $V_{\text{nom}}$ — never estimated directly by the
   deployed scheme; the two densities act as computable proxies for its
   degradation (Hypotheses H1–H3 below say when the proxy is faithful).

## 4. The detector, formalized

**Information constraint.** An intervention rule is a stopping time
$\tau_{\text{stop}}$ adapted to the deployment filtration
$\mathcal{F}_t = \sigma(x_0, \dots, x_t,\ \text{policy internals})$ — no
privileged simulator access.

**Ideal trigger (definition of what detection is *for*).** With the recovery
operator $\mathcal{R}$ of §5 and its value $V_{\text{rec}}$, define the
advantage
$$\Delta(s, h, k) \;=\; V_{\text{rec}}(s, k) - V_{\text{nom}}(s, h, k).$$
The ideal detector fires iff $\Delta > 0$. $\Delta$ is not computable at
deployment; the scheme replaces it with support statistics.

**Detector statistics.** At each replan boundary compute
$$c_t \;=\; -\log \hat\rho(x_{R,t} \mid x_{E,t})
\qquad\text{(conditional surprisal — the aligned statistic)},$$
$$e_t \;=\; -\log \hat p_{\mathcal{D}}(x_{E,t})
\qquad\text{(environment familiarity — the applicability gate)},$$
optionally $u_t$ = policy-internal dispersion (Track E1; e.g. variance across
the $K$ candidate chunks already sampled).

**Rule class.** Fire at
$$\tau_{\text{stop}} \;=\; \min\{t \in \mathcal{T} :\ c_t > \kappa
\ \wedge\ e_t \le \kappa_E \ \wedge\ n_{\text{int}} < N_{\max}
\ \wedge\ t - t_{\text{last}} \ge \Delta_{\text{cool}}\},$$
with threshold $\kappa$, applicability gate $\kappa_E$, intervention budget
$N_{\max}$ (v0: 1), and cooldown $\Delta_{\text{cool}}$ (prevents intervention
loops when the nominal policy repeatedly re-drifts). Varying $\kappa$ traces
the operating curve (detection vs false alarms); §II.7 defines the
deadline-aware evaluation of this rule class against oracle labels.

**Why $c_t$ and not raw joint OOD-ness.** $c_t$ is aligned with
*actionability*: it is large exactly on the shift type the recovery operator
can repair, and the gate $e_t$ excludes the type it cannot. A joint-density
detector conflates the two and fires on unrepairable states.

## 5. The recovery operator, formalized

Recovery is a **semi-Markov option** $o_{\text{rec}} = (\mathcal{I}, \xi,
\beta)$: initiation set $\mathcal{I}$ = the detector's firing region, internal
policy $\xi$ = the return controller, termination $\beta$ = arrival or timeout.
Executing it from $(s, t)$:

1. **Generate.** Sample $M$ candidate robot configurations
   $x_R^{(1..M)} \sim \hat\rho(\cdot \mid x_{E,t})$.
2. **Select.** $x_R^\star = \sigma\big(x_t, \{x_R^{(j)}\}\big)$ for a selection
   rule $\sigma$ (v0: nearest-to-current reachable candidate; oracle version in
   Part II tries all).
3. **Transport.** The controller $\xi$ drives the robot from $x_{R,t}$ to
   $x_R^\star$, consuming $T_{\text{ret}}$ steps of the *same* episode clock,
   subject to the **no-disturbance requirement**: the environment must not
   change, formalized as the event
   $$B \;=\; \{\, d_E(x_{E,\,t+T_{\text{ret}}},\ x_{E,t}) > \epsilon_B \,\}$$
   ("bump") having small probability $p_B$. PushT instantiation: lift the
   stick above the block plane, translate, descend at the target, settle —
   `pd_ee_delta_pose` supports this, so $p_B \approx 0$ by construction and
   $T_{\text{ret}} \approx \lceil \text{path}/\delta_{\max}\rceil$ steps.
4. **Resume.** Clear the nominal system's internal state (obs history and
   chunk cache are invalid across the discontinuity), then follow
   $\Sigma_{\text{nom}}$ for the remaining $k - T_{\text{ret}}$ steps.

The recovery value is
$$V_{\text{rec}}(s, k) \;=\;
\mathbb{E}\left[\, V_{\text{nom}}\big(s',\ h_\emptyset,\ k - T_{\text{ret}}\big)\,\right],$$
the expectation over generation, selection, transport (including $B$), with
$h_\emptyset$ = fresh internal state. Note $V_{\text{rec}}$ does not depend on
$h$: recovery discards history. This is the object whose ceiling Part II
measures.

**The switched system.** $\Sigma_{\text{rec}}(\kappa, \kappa_E, N_{\max},
\Delta_{\text{cool}})$ = run $\Sigma_{\text{nom}}$; at replan boundaries apply
the rule of §4; on firing, execute $o_{\text{rec}}$ once, resume. Objective
$J(\Sigma) = \Pr_{s_0 \sim \mu_0}(Y = 1)$, reported as a **frontier** over
$\kappa$: success vs intervention rate vs false-alarm cost.

**Paired accounting.** All comparisons between $\Sigma_{\text{nom}}$ and
$\Sigma_{\text{rec}}$ use common random numbers (same env seeds; sim state
replay), so the outcome decomposes per-episode into
$\{+\text{saved}, -\text{broken}, =\}$ exactly as the selection oracle's
$+70/{-0}$ ledger.

## 6. Why it can work: the hypotheses, stated measurably

The scheme's soundness is equivalent to four hypotheses. Each is a measurable
quantity in Part II — none is assumed.

- **H1 (detectability with slack).** Failing nominal trajectories enter the
  high-$c$ region *before* recovery stops being possible:
  $\Pr(\exists\, t \le \tau^* : c_t > \kappa \mid Y{=}0)$ is high while
  $\Pr(\exists\, t : c_t > \kappa \mid Y{=}1)$ is low, for some common
  $\kappa$. (Measured: deadline-aware ROC, §II.7.)
- **H2 (conditional-shift dominance).** A substantial fraction of failures
  have $e_t \le \kappa_E$ (environment still supported) at some time with
  $\Delta > 0$ — i.e. the failure mass is of the repairable type. (Measured:
  failure decomposition, §II.7.)
- **H3 (support restoration ⇒ value restoration).** Define the *restored
  value*
  $$\nu(x_E, k) \;=\; \mathbb{E}_{x_R \sim \rho(\cdot \mid x_E)}
  \big[V_{\text{nom}}\big((x_R, x_E),\ h_\emptyset,\ k\big)\big].$$
  H3 says $\nu(x_E, k)$ is high when $p_{\mathcal{D}}(x_E)$ is high — the
  nominal policy is good *on its support*, so putting the robot back on the
  conditional support restores its success probability. (Measured: this is
  exactly what the teleport oracle $\mathcal{R}_{\text{tp}}$ estimates.)
- **H4 (cheap transport).** $T_{\text{ret}}$ is small relative to the
  remaining budget and $p_B \approx 0$. (Measured: the
  $\mathcal{R}_{\text{tp}} - \mathcal{R}_{\text{ex}}$ gap.)

**Advantage decomposition.** Combining §5 with H3–H4:
$$\Delta(s, h, k) \;\approx\;
\underbrace{\big[\nu(x_E,\, k - T_{\text{ret}}) - V_{\text{nom}}(s, h, k)\big]}_{\text{value of being back on support}}
\;-\; \underbrace{p_B \cdot L_B}_{\text{bump risk}}
\;-\; \underbrace{\big[\nu(x_E, k) - \nu(x_E, k - T_{\text{ret}})\big]}_{\text{clock cost}}.$$
On healthy states $V_{\text{nom}} \approx \nu$ and $\Delta < 0$ (the costs
dominate): recovery is a **specialist**, worse than nominal on-support. This is
what makes detection a real problem — cf. the degeneracy remark in §II.1. The
firing region $\{\Delta > 0\}$ opens at the crossover where degrading
$V_{\text{nom}}$ falls below $\nu$ minus costs, and closes at the viability
boundary $\tau^*$ (§II.6).

## 7. Estimators: v0 (low-dim) and the high-dim regimen

Both versions estimate the same three population objects of §3. **Only the
estimator changes; the scheme, statistics, thresholds, and oracle are
identical.**

**v0 (state mode, this repo, train-nothing).**
- $\hat\rho(x_R \mid x_E)$: retrieval — the $m$ demo states nearest in
  $d_E$ (below) define an empirical conditional; sampling = returning their
  (jittered) robot configurations; the surprisal $c_t$ = conditional kNN
  distance $\min_j \|x_{R,t} - x_R^{(j)}\|$ over those neighbors (a monotone
  surrogate of $-\log\rho$; exact enough at intrinsic dim ~5 with ~14k demo
  states).
- $\hat p_{\mathcal{D}}(x_E)$: kNN distance in $\mathcal{X}_E$ under
  $$d_E\big((p,q),(p',q')\big) = \|p_{xy} - p'_{xy}\|/L \;+\;
  \lambda\, d_{\angle}(\theta_z, \theta'_z),$$
  planar position scaled by workspace size $L$, yaw geodesic distance weighted
  by $\lambda$ (v0: $\lambda = L/\pi$ so a half-turn "costs" a workspace
  length; a sensitivity check over $\lambda$ is part of the oracle run).
  Thresholds $\kappa, \kappa_E$ are set as percentiles (e.g. 99th) of the same
  statistics computed on held-out demo states.
- A pleasant asymmetry: retrieval returns **full joint configurations** (qpos),
  so the recovery target needs no IK.

**High-dim regimen (the version that translates; not built now).**
- The **generated object never becomes high-dimensional**: $x_R$ is always
  low-dim proprioception. Only the *conditioning* changes:
  $\hat\rho_\psi(x_R \mid \text{enc}_\psi(x_E))$ with $x_E$ = robot-masked
  image and $\text{enc}_\psi$ an image encoder. Model class: any conditional
  generative model with tractable sampling and a usable log-density or proxy —
  conditional flow (exact log-density; preferred), conditional diffusion
  (ELBO / denoising-loss proxy), CVAE (ELBO). Training = masked conditional
  MLE on demo pairs $(m(x), x_R)$ — the literal "mask the robot, generate the
  robot" regimen proposed. No image generation, no image density estimation is
  ever required for the conditional path.
- The marginal gate $\hat p_{\mathcal{D}}(x_E)$ is the genuinely harder
  high-dim object: v1 = kNN in the encoder's latent space; upgrades = latent
  flow / one-class methods (Track E2 proper).
- Generated $x_R$ may be an EE pose rather than qpos → transport needs IK; the
  oracle's teleport stage (Part II) is unaffected (it retrieves qpos).
- Calibration transfers: thresholds remain percentile-based on held-out demos;
  validation against oracle labels (§II.7) is modality-independent.

**Refinements (recorded, not v0).**
- **Phase conditioning**: $\rho(x_R \mid x_E)$ marginalizes over demonstration
  phase $\zeta$ (approach vs push poses for the same block pose). Refinement:
  $\rho(x_R \mid x_E, \hat\zeta)$ with $\hat\zeta$ from the Track-B5
  temporal-order model. v0 accepts the aliasing (resumed nominal re-executes
  the phase).
- **Goal-pose selection** $\sigma$ beyond nearest: max-$\hat\rho$, reachability
  filtering, or (Track C) predicted-value ranking.
- **Multi-intervention** ($N_{\max} > 1$) with hysteresis.

---

# Part II — The ceiling oracle

## 1. What is being bounded, and why a *scheme-class* ceiling

**Degeneracy remark.** For an unconstrained (dominating) recovery operator —
one with $V_{\text{rec}} \ge V_{\text{nom}}$ everywhere, e.g. oracle-K
selection or sim-optimal control — the firing region is the whole episode,
timing carries no information, and the "recovery ceiling" reduces to the value
of the improved policy: it re-measures selection / Track C headroom (already
priced: +23.3pt), and a detector adds nothing. A meaningful E+F ceiling
therefore bounds a **scheme class**. We bound the class of Part I by replacing
each of its four stages (when / which pose / transport / resume) with its
idealization, independently. The result upper-bounds any implementation of the
class *under the stated scope restrictions* (single intervention,
grid-restricted timing), and the per-stage gaps later localize where a real
implementation loses headroom.

**Estimand.** For recovery operator $\mathcal{R}$ (below):
$$J_{\text{ceil}}(\mathcal{R}) \;=\;
\frac{1}{n}\Big[\, |S| + \sum_{i \in F} \max\big(0,\ \max_{t \in \mathcal{T}_i}
V_{\text{rec}}^{(i)}(t)\big) \Big],$$
where $S/F$ partition the reference episodes by realized outcome,
$\mathcal{T}_i$ is the grid of replan boundaries of episode $i$, and
$V_{\text{rec}}^{(i)}(t)$ is the true value of intervening at $t$ on episode
$i$'s realized prefix. Successes contribute 1 (success-once is already
realized; the oracle abstains). The max over $t$ is the single-intervention
optimal-stopping value on the grid.

**The re-roll control (mandatory).** Continuing nominal from a failing
episode's prefix with *fresh policy noise* already has positive value — a
failure is one unlucky draw. Define
$$J_{\text{reroll}} \;=\; \frac{1}{n}\Big[\, |S| + \sum_{i \in F}
\max_{t \in \mathcal{T}_i} V_{\text{nom}}^{(i)}(t) \Big],$$
the value of **oracle-timed doing-nothing** (pick the luckiest moment to simply
keep going). Every mechanism claim uses
$$\text{recovery headroom} \;=\; J_{\text{ceil}}(\mathcal{R}) - J_{\text{reroll}},$$
never $J_{\text{ceil}} - J_{\text{nom}}$; otherwise the ceiling is credited
with what fresh dice achieve. This is the recovery analog of the selection
oracle's random-candidate control.

## 2. Reference run and snapshot protocol

1. Fix a panel $E$ of $n$ episodes (dev: $n = 100$; env seeds from a dedicated
   diagnostic panel — disjoint from screening/confirmation/final_test).
2. Run $\Sigma_{\text{nom}}$ once per episode (the benchmark standalone
   config), recording outcomes $\to S, F$, and at every replan boundary
   $t \in \mathcal{T}_i$ saving the **extended snapshot**
   $(s_t, h_t)$ = full simulator state + the system's observation history.
   (A2: sim save/restore is exact — already validated by the selection
   oracle machinery.)
3. Failures run to $T_{\max}$ (success-once), so $|\mathcal{T}_i| \le
   \lceil T_{\max}/H_a \rceil = 13$ at $H_a = 8$.

## 3. Monte-Carlo value estimators

For a snapshot $(s_t, h_t)$ with $k = T_{\max} - t$:

- $\hat V_{\text{nom}}(t)$: restore $(s_t, h_t)$ — including the observation
  history, so the continuation is exact — and roll $\Sigma_{\text{nom}}$ to the
  horizon with $K_{\text{nom}}$ fresh policy seeds; average success.
- $\hat V_{\text{rec}}(t; j)$: apply recovery variant $\mathcal{R}$ with
  candidate pose $j$ (below), reset internal state to $h_\emptyset$ (recovery
  invalidates history — deliberate and matching deployment), roll
  $\Sigma_{\text{nom}}$ with $K$ fresh seeds; average success.

Estimators are unbiased per $(t, j)$ with standard error
$\le 1/(2\sqrt{K})$. All randomness enumerated: env is deterministic given
state; policy generators are seeded per rollout.

## 4. Stage oracles

**(when) Oracle timing** — exhaustive sweep of $t$ over $\mathcal{T}_i$
(≤13 points; no coarsening needed).

**(which pose) Oracle generation + selection** — retrieval instantiation of
$\hat\rho$: the $M$ demo states nearest in $d_E$ to $x_{E,t}$ contribute their
robot configurations (qpos; velocities zeroed) as candidates; oracle selection
tries **all** $M$ and keeps the best-by-value.

**(transport) Two operators** — the ladder:

- $\mathcal{R}_{\text{tp}}$ (**teleport**; non-physical, labeled): overwrite
  the arm's qpos with the candidate's, zero qvel, leave `obj_pose` untouched;
  $T_{\text{ret}} = 0$, $p_B = 0$. Requires restore-of-modified-state
  (**feasibility check F1 before anything else**: verify the ManiSkill state
  vector can be edited at the articulation slice and re-set exactly; fallback
  is $\mathcal{R}_{\text{ex}}$-only). $\mathcal{R}_{\text{tp}}$ estimates the
  pure H3 quantity $\nu(x_E, k)$: the value of *being* back on the conditional
  support.
- $\mathcal{R}_{\text{ex}}$ (**executed**; physical): scripted
  lift–translate–descend return under `pd_ee_delta_pose`, real physics, real
  clock; measures $T_{\text{ret}}$ and bump events $B$. Timeout
  $T_{\text{ret}} \le T_{\text{ret}}^{\max}$ (e.g. 16 steps), after which
  resume regardless.

Expected ordering $J_{\text{ceil}}(\mathcal{R}_{\text{ex}}) \le
J_{\text{ceil}}(\mathcal{R}_{\text{tp}})$ — not a theorem (an accidental
beneficial bump can invert it locally); a violation is itself a diagnostic
(beneficial contact), so both are reported.

**(resume) Nominal** — always $\Sigma_{\text{nom}}$ with fresh internal state.
(A stacked variant resuming with oracle-selection is a *later* cell; it
conflates tracks and is excluded from the primary ceiling.)

## 5. Selection bias control (two-phase, winner's curse)

$\max_{t, j}$ over noisy MC estimates is upward-biased. Protocol, per failing
episode:

- **Phase 1 (search):** estimate $\hat V_{\text{rec}}(t; j)$ for all
  $t \in \mathcal{T}_i$, $j \le M$ with small $K_{\text{sel}}$; pick
  $(t^\star, j^\star) = \arg\max$. Same for $t^\star_{\text{nom}} = \arg\max_t
  \hat V_{\text{nom}}(t)$.
- **Phase 2 (report):** re-estimate only the winners with fresh seeds and
  larger $K_{\text{eval}}$. **All reported estimands ($J_{\text{ceil}}$,
  $J_{\text{reroll}}$) use phase-2 numbers only.** Phase-1 numbers appear only
  inside curves labeled as search-phase.

This mirrors the screening→confirmation discipline of the checkpoint-selection
protocol, for the same reason. Aggregate CIs: nonparametric bootstrap over
episodes of the per-episode phase-2 values (per-episode estimates are unbiased,
so the episode-mean is unbiased and the bootstrap CI is valid).

## 6. Derived per-episode objects

From the same sweep, for each failing episode (search-phase curves):

- the **advantage curve** $\hat\Delta^{(i)}(t) = \hat V_{\text{rec}}^{(i)}(t) -
  \hat V_{\text{nom}}^{(i)}(t)$ on $\mathcal{T}_i$;
- **crossover** $\hat t_x = \min\{t : \hat\Delta(t) \ge \delta\}$ with noise
  margin $\delta = \max(0.05,\ 2\,\text{SE})$;
- **viability boundary** $\hat\tau^* = \max\{t : \hat V_{\text{rec}}(t) \ge
  p_{\min}\}$, reported at $p_{\min} \in \{0.25, 0.5\}$;
- **detection slack** $= \hat\tau^* - \hat t_x$ (steps, and seconds at
  $f_c = 20$ Hz) — the deadline a deployable detector must meet.

**False-alarm arm.** On a sample of successes ($\sim$20 episodes × 3 grid
times), apply $\mathcal{R}_{\text{tp/ex}}$ anyway and estimate
$\hat\Delta \le 0$: the specialist property (§I.6) verified, and the cost side
of the $\kappa$-frontier calibrated.

## 7. Deliverables

1. **Ceiling table**: $J_{\text{nom}}$, $J_{\text{reroll}}$,
   $J_{\text{ceil}}(\mathcal{R}_{\text{tp}})$,
   $J_{\text{ceil}}(\mathcal{R}_{\text{ex}})$, with bootstrap CIs; recovery
   headroom = ceiling − re-roll; transport tax = tp − ex.
2. **Slack distribution**: histograms of $\hat t_x$, $\hat\tau^*$, slack — the
   detector's spec sheet.
3. **Additivity 2×2** vs the selection oracle *run on the same panel and same
   nominal policy* (one extra run of the existing rollout-oracle machinery):
   per-episode {saved by oracle selection} × {saved by oracle recovery}. The
   recovery-only cell is the genuinely new headroom; the overlap cell is
   repackaged Track-C mass. This is the strategic deliverable.
4. **Failure decomposition** of $F$ with the v0 statistics on the *reference*
   trajectories (no training): (a) environment-marginal OOD
   ($e_t > \kappa_E$ throughout the pre-$\tau^*$ window — unrepairable by this
   class), (b) robot-conditional OOD ($c_t > \kappa$, $e_t \le \kappa_E$
   somewhere in-window — the addressable mass, H2), (c) in-support doom
   (never OOD, fails anyway — invisible to support statistics; bounded by the
   re-roll and selection ceilings instead). Thresholds at the 99th percentile
   of held-out demo statistics.
5. **Detector calibration set**: labeled pairs
   $\big(x_t^{(i)},\ \mathbb{1}[\hat\Delta^{(i)}(t) > \delta]\big)$ for every
   snapshot. Cheap statistics ($c_t$, $e_t$, joint kNN, policy dispersion
   $u_t$) evaluated two ways: pointwise AUROC, and **episode-level
   deadline-aware ROC** — a threshold $\kappa$ scores a hit on a recoverable
   failure iff it fires within $[\hat t_x, \hat\tau^*]$, and a false alarm on a
   success iff it fires at all; sweep $\kappa$. This is the validation
   harness all future E-track detectors are judged by, decoupled from any
   deployment run.
6. **Sensitivity**: $\lambda$ (yaw weight in $d_E$), $M$, $p_{\min}$,
   grid density.

## 8. Cost model and budget (dev scale)

Per failing episode: $|\mathcal{T}_i| \le 13$ grid points ×
[$M{=}3$ poses × $K_{\text{sel}}{=}3$ + $K_{\text{nom}}{=}4$] + phase-2
$K_{\text{eval}}{=}10$ ≈ **180 partial rollouts** of mean remaining length
~50 steps ≈ 9k env-steps ≈ 1.1k policy queries at $H_a = 8$. At $n = 100$
(≈48 failures at the current ~52% base): ≈ **430k env-steps ≈ 54k policy
queries** + false-alarm arm (noise) + one selection-oracle run on the same
panel. Order **1–2 GPU-hours** with the DP nominal (40 ms/query), minutes-level
policy cost with ACT (2 ms). Knobs: $K$'s, $M$, $n$; all scale linearly.

## 9. Assumptions, scope, and validity discipline

- **A1** exact sim state save/restore (validated); **A2** measurable robot/env
  split (given by the state schema; in image mode requires a robot mask);
  **A3** success-only demos suffice for the *scheme* (it never needs failure
  data — §I.7); the *oracle* additionally needs the simulator; **A4**
  deterministic simulator given (state, action) — randomness is policy-side
  and enumerated; **A5** interventions at replan boundaries only.
- **Scope restrictions on the "ceiling" claim**: single intervention,
  grid-restricted timing, retrieval-restricted pose set, this recovery class.
  A multi-intervention or finer-grid method could exceed it; the unconstrained
  ceiling is the viability/selection quantity, measured elsewhere.
- **PushT optimism caveat**: PushT is physically near-reversible (nothing
  breaks, everything re-approachable), so $\tau^*$ is dominated by the episode
  clock, and windows will be wide relative to contact-rich/irreversible tasks.
  The result establishes the mechanism and the measurement methodology, not
  the hard case.
- **Tier**: diagnostic, single policy seed, dev panels. Never used for
  training or checkpoint selection. $\mathcal{R}_{\text{tp}}$ labeled
  non-physical wherever reported. Claim-grade versions follow the standard
  protocol (≥5 seeds, paired, frozen tree).

## 10. Open decisions before implementation

1. **F1 feasibility check**: articulation-slice state editing for
   $\mathcal{R}_{\text{tp}}$ (build $\mathcal{R}_{\text{ex}}$ first
   regardless — it needs no state surgery).
2. Nominal family for the first run (post-rerun benchmark choice).
3. Panel identity and size ($n = 100$ dev proposed).
4. Constants: $M, K_{\text{sel}}, K_{\text{nom}}, K_{\text{eval}}, \lambda,
   T_{\text{ret}}^{\max}$, thresholds' percentile.
5. Whether the selection-oracle same-panel run happens in the same batch
   (recommended: yes — the 2×2 is the point).
