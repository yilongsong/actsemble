# Failure Detection + Recovery: the Scheme, and the Oracle that Measures its Ceiling

Status: DESIGN (2026-07-18).
Part I formalizes the proposed detection+recovery scheme (an instantiation of
Tracks E2/E4 + F1/F3). Part II specifies the simulation oracle that measures
the scheme's ceiling **before anything is trained**.
Epistemic tier: the oracle is a **diagnostic** — same standing as C6 and the
selection oracle. It is never used for training or checkpoint selection.

**How to read this.** Every section follows the same pattern: first *why the
object needs to exist* (what goes wrong without it), then the plain-language
version, then the precise definition, then the concrete PushT instantiation.
The formal content is identical to a compressed spec; the words around it are
the reasoning that produced it.

---

# Part 0 — The problem in plain language

## 0.1 What recovery is supposed to add

Our deployed policy fails on roughly half of PushT episodes. We already know
two ways to convert some failures into successes: pick better among the
policy's own proposals (selection — oracle headroom +23.3pt), and replan more
often (+12.7pt). Detection+recovery proposes a third, different mechanism:

> Watch the rollout as it unfolds. Notice when it has gone off the rails.
> Execute a *corrective maneuver* that ordinary policy execution would never
> produce. Then hand control back.

The bet is that some failures are not "the policy chose slightly wrong at some
decision point" (selection's territory) but "the robot drifted somewhere the
policy has never seen, and once there, it stays lost." For those, no amount of
choosing-among-proposals helps — every proposal is conditioned on an
observation the policy can't handle. What helps is *going back to somewhere it
can handle*.

## 0.2 A running example

We will reuse one stylized failure throughout (illustrative, not a claim about
the dominant real failure mode — the oracle will measure that):

> **The overshoot.** Demos push the T-block toward the goal by pressing on its
> right face, tip always *behind* the push face. At deployment the policy
> pushes a bit too hard; the block rotates past the demonstrated orientation
> band and the stick tip slides off the edge, ending up *between the block and
> the goal* — a tip-relative-to-block configuration that appears in no
> demonstration. Conditioned on this unfamiliar relative pose, the policy
> outputs averaged, jittery motions that never re-establish contact with the
> push face. The clock burns; the episode fails.
>
> A human watching says: *"just lift the stick, go back around behind the
> block, and push again."* That sentence, made precise, is the entire scheme:
> **detect** (the tip's position is surprising *given* the block's position),
> **recover** (find where demos put the tip when the block was posed like
> this; lift, travel there, descend), **resume** (let the policy continue from
> a configuration it recognizes).

## 0.3 Why this needs care: recovery is not obviously its own thing

There is a trap in formalizing "recovery," discovered in our earlier
discussion, and it shapes everything below. If your "recovery" is simply *a
better way to act* — e.g. re-plan with oracle-selected proposals — then it is
better than the nominal policy **everywhere**, you should run it at every
step, and there is nothing to *detect*. "Detection+recovery" collapses into
"act well," which is Track C, already measured. Detection is only a real
problem when recovery is a **specialist**: a maneuver that is *worse* than
nominal on healthy states (it interrupts the task, burns clock, risks
disturbing the scene) and better only after things have gone wrong. The value
of the whole scheme then lives in the *timing* — and the ceiling we build must
respect that structure, or it silently measures something else.

## 0.4 Why measure a ceiling first

The selection oracle earned its keep by replacing a vague hope ("maybe
selection helps") with three numbers (+23.3pt headroom, ~9% captured, ~35%
proposal wall) that redirected the research plan. We want the same for
recovery *before* training anything: how much failure mass is recoverable *by
this kind of maneuver*, how much warning does a detector actually get, and —
the strategic question — is that mass **additive** to selection's, or the same
episodes wearing a different costume?

---

# Part I — The scheme, formalized

## 1. The setting, carefully

### 1.1 The process

A discrete-time controlled Markov process. Full simulator state
$s \in \mathcal{S}$; control action $a \in \mathcal{A} \subset \mathbb{R}^6$
(compact box; normalized end-effector delta pose); control frequency
$f_c = 20$ Hz; episode horizon $T_{\max} = 100$ steps. The transition kernel
$P(s' \mid s, a)$ is genuinely stochastic in practice: GPU PhysX
(`physx_cuda`) contact resolution is nondeterministic run-to-run — measured
2026-07-18 by re-running identical deterministic-policy evals: ~96%
per-episode outcome agreement for sparse-replan systems, ~81% for
dense-replan systems (closed-loop chaos amplifies contact noise). So
randomness has two sources — policy sampling *and* sim noise — and every
"value" below is a probability over both.

**Success is "success-once."** There is a goal set $G \subset \mathcal{S}$
(block coverage $\ge \theta$), and an episode counts as a success if it
*touches* $G$ at any time:

$$Y(\tau) \;=\; \mathbb{1}\!\left[\exists\, t \le T_{\max}: s_t \in G\right].$$

This matters later: once an episode has succeeded, nothing that happens
afterward can un-succeed it — which is why the oracle never needs to intervene
on realized successes.

### 1.2 The observation, dissected

The policy sees $x = \phi(s) \in \mathbb{R}^{31}$. The layout (from the
dataset schema) — with the role each block plays in this document:

| slice | dims | content | role |
|---|---|---|---|
| `agent.qpos` | 0–6 | 7 arm joint positions | **robot** ($x_R$) |
| `agent.qvel` | 7–13 | 7 joint velocities | **robot** ($x_R$) |
| `extra.tcp_pose` | 14–20 | stick-tip pose (3 pos + 4 quat) | **robot** ($x_R$) — redundant: a deterministic function of qpos (forward kinematics) |
| `extra.goal_pos` | 21–23 | goal position | constant context |
| `extra.obj_pose` | 24–30 | T-block pose (3 pos + 4 quat) | **environment** ($x_E$) |

So the robot/environment split that the whole scheme rests on is not an
assumption we hope for — it is literally given by the state schema:
$x = (x_R, x_E, \text{goal})$ with $x_R$ = dims 0–20, $x_E$ = dims 24–30, and
the goal constant across the dataset.

### 1.3 The data — and what "success-only" implies

$\mathcal{D} = \{\tau^{(i)}\}_{i=1}^{400}$: 400 demonstrations, every one
successful ($Y = 1$), RL-generated, trailing-hold-trimmed, mean length 35
steps (range 17–78) — about **14,000 states** total. Let $\mu_{\mathcal{D}}$
denote the empirical distribution of demonstrated states, with density
$p_{\mathcal{D}}$.

The crucial negative fact: **the dataset contains no failures, no corrections,
no examples of "what to do when lost."** Any recovery mechanism that needs
corrective supervision (state-off-manifold → corrective action) is starved at
birth — this is the classic Track-F obstruction (F2/F5). A central virtue of
the scheme below is that it sidesteps this entirely: it never learns
corrections. It learns *where demonstrated robots are*, and uses a planner to
go there. Success-only data is exactly enough for that.

### 1.4 "The demonstrated manifold": ambient vs intrinsic dimension

We will constantly refer to states being "on" or "off" the demonstrated
manifold, and the v0 implementation estimates this with nearest-neighbor
distances. Whether that is statistically sound hinges on a distinction worth
spelling out completely, because it decides whether "train nothing" is
legitimate here and *why* images will later require an encoder.

**Ambient dimension** is a property of the *representation*: our states are
vectors of 31 numbers, so the data lives in $\mathbb{R}^{31}$. Full stop.

**Intrinsic dimension** is a property of the *distribution*: how many
independent directions the data actually varies along — the dimension of the
subset of $\mathbb{R}^{31}$ that demonstrated states occupy. The data does not
fill the ambient space; constraints tie coordinates together, and every
constraint removes a direction of variation. Counting for our state, block by
block:

| block | ambient dims | intrinsic contribution | why |
|---|---|---|---|
| `goal_pos` | 3 | 0 | constant over the whole dataset |
| `tcp_pose` | 7 | 0 | forward-kinematics function of qpos; given qpos it adds nothing |
| `obj_pose` quat | 4 | ~1 | unit-norm (≤3 DOF); the block lies flat → only yaw varies |
| `obj_pose` xyz | 3 | ~2 | z pinned to the table; x, y free |
| `agent.qpos` | 7 | ~2 | see below — the subtle row |
| `agent.qvel` | 7 | ~1–2 (weak) | velocities ride along the path; small, correlated |

Total: about **5 strong directions** (tip x, y; block x, y, yaw) plus a few
weak ones.

The qpos row is where the concept earns its keep. The arm mechanically *has*
7 degrees of freedom — but intrinsic dimension is about the **data**, not the
mechanism. The demonstrations were produced by a controller that keeps the
stick tip at fixed height and orientation and resolves the arm's redundancy
consistently, so the 7-dim joint vectors *visited by the data* trace out a
~2-dimensional sheet parameterized by tip (x, y), winding through joint space.
Configurations off that sheet are mechanically reachable but never
demonstrated — and "never demonstrated" is precisely what our density
$p_{\mathcal{D}}$ is about.

**Where is the line, formally?** Two equivalent-in-spirit definitions:

- *Geometric*: the data concentrates near a $d$-dimensional submanifold of
  $\mathbb{R}^{31}$. Operationally: take a state, collect its neighbors, run
  local PCA — about $d$ eigenvalues sit above the noise floor and $31 - d$
  near zero. The large-eigenvalue directions are *tangent* (move that way and
  you stay in the data); the rest are *normal* (any motion immediately leaves
  the support). Detection-by-support-distance is literally "measure how far
  you have moved in the normal directions."
- *Statistical* (the one our claims rest on): $d$ is the exponent in how
  neighbor distances shrink with sample size — nearest-neighbor distance
  $\sim n^{-1/d}$.

One honesty note: the line is **scale-dependent**, not sharp. At task scale
the demo states look 5-dimensional; zoom into contact jitter and the weak
directions (velocities, z-vibration) light up. "Intrinsic ≈ 5" means: *at the
resolution relevant to detection and retrieval, local neighborhoods of the
demo set are ~5-dimensional*. It is a measurable number (TwoNN / local-PCA
spectrum on the 14k states) and measuring it is a listed sanity check.

**Why it decides everything — the arithmetic.** kNN's resolution is its
nearest-neighbor distance, $\sim n^{-1/d}$ at $n \approx 14{,}000$:

- if $d = 31$: $14000^{-1/31} \approx 0.73$ — the "nearest" neighbor is ~73%
  of the data diameter away, barely closer than a random point. kNN useless.
- if $d = 5$: $14000^{-1/5} \approx 0.15$ — neighbors are genuinely local.
  kNN works.

And the resolution of the apparent paradox ("but we run kNN in 31
dimensions!"): **the curse of dimensionality follows the intrinsic dimension,
and kNN adapts to it automatically.** Nobody identifies which directions are
"the real 5." Distances between demo states are dominated by displacement
*along* the manifold: constant dims contribute exactly zero, and the
FK-redundant dims move in lockstep with qpos (a fixed factor, not a new
dimension). The line is drawn by the geometry of the data, not by us. The one
place we *do* intervene by hand is the **metric**: adaptivity fixes the rate,
but badly scaled nuisance coordinates inflate the constants — e.g.
standardization gives velocity dims unit variance, letting task-irrelevant
velocity jitter blur neighborhoods. Hence the hand-built environment metric
$d_E$ in §6, and "which dims, what weights" as a sensitivity check rather
than an afterthought.

**The image case, in the same language.** Images have ambient $\sim 10^5$,
but the *scene distribution's* intrinsic dimension is still tiny — the scene
still only has ~5 degrees of freedom. kNN on raw pixels fails anyway, and not
because of intrinsic dimension: the **pixel metric does not respect the
manifold's geometry**. Two images of nearly identical scenes can be far apart
in L2 (lighting, texture, tiny shifts), so nuisance directions dominate
distances and the *effective* $d$ explodes. This is the precise sense in
which the learned encoder in the high-dim regimen (§6) is **metric repair,
not dimension reduction**: its job is to produce coordinates in which distance
again tracks displacement along the ~5-dim scene manifold — at which point
the same kNN/conditional-density machinery from v0 applies unchanged.

### 1.5 The nominal system, and values as probabilities with a clock

The deployed system $\Sigma_{\text{nom}}$ (policy + replanning every $H_a = 8$
steps) carries **internal state** $h_t$: the observation history (length
$H_o$), the cached action chunk, and the position within it. The closed-loop
process is Markov on the extended state $(s_t, h_t)$. We define everything at
**replan boundaries** — the moments the cached chunk is exhausted — on the
grid $\mathcal{T} = \{0, H_a, 2H_a, \dots\}$; there $h$ reduces to the
observation history.

**Values.** With $k = T_{\max} - t$ steps remaining:

$$V_{\text{nom}}(s, h, k) \;=\; \Pr\!\left(\exists\, t' \le k :\ s_{t'} \in G
\;\middle|\; \text{start at } (s,h),\ \text{follow } \Sigma_{\text{nom}}\right),$$

the probability over the policy's sampling noise. Two features deserve
emphasis because they drive the oracle's design:

1. **Values are probabilities, not verdicts.** A state does not "lead to
   failure"; it leads to failure *with some probability under future
   stochasticity*. A realized failure is **one unlucky draw** from
   $V_{\text{nom}}$. This single observation kills the naive
   "point-of-doom" framing (there is no step where success probability
   discontinuously hits zero), forces Monte-Carlo estimation everywhere in
   Part II, and creates the need for the *re-roll control* (§9.2) — without
   which a recovery oracle takes credit for plain luck.
2. **Values carry the clock.** The same physical state with 60 steps left and
   with 5 steps left have very different values. Every estimator must be
   truncated at the true remaining horizon; "value of a state" without $k$ is
   not well defined here.

### 1.6 Notation summary

| symbol | meaning |
|---|---|
| $s, x = \phi(s)$ | simulator state; observation (31-dim) |
| $x_R, x_E$ | robot part (dims 0–20), environment part (dims 24–30) |
| $Y$ | success-once episode outcome |
| $\mathcal{D}, p_{\mathcal{D}}$ | 400 success-only demos; their state density |
| $\rho(x_R \mid x_E)$ | demonstrated conditional: where robots are, given the scene |
| $\Sigma_{\text{nom}}, h$ | deployed nominal system; its internal state |
| $V_{\text{nom}}(s,h,k)$ | success probability, continuing nominally, $k$ steps left |
| $V_{\text{rec}}(s,k)$ | success probability if we recover now, then resume |
| $\Delta = V_{\text{rec}} - V_{\text{nom}}$ | the advantage of intervening now |
| $\mathcal{T}$, $H_a$ | replan-boundary grid; chunk execution length (8) |
| $T_{\max}, k, f_c$ | horizon (100); remaining steps; control rate (20 Hz) |

## 2. The idea, and the two kinds of being lost

In one sentence: **learn from the demos where the robot belongs given the
scene; notice when it is not there; put it back; resume.**

The reason the robot/environment split is the load-bearing choice: when a
rollout drifts off the demonstrated manifold, the drift can live in two
different places, and they differ in what can be done about them.

- **Environment-marginal shift**: the *scene itself* is somewhere demos never
  were — $p_{\mathcal{D}}(x_E)$ is small. (The block got shoved into a
  configuration no demo contains.) Repairing this requires *manipulating the
  world* — which is the hard task itself. Not what this scheme fixes.
- **Robot-conditional shift**: the scene is familiar, but the robot is
  somewhere no demo ever had it *given this scene* — $p_{\mathcal{D}}(x_E)$
  fine, $\rho(x_R \mid x_E)$ small. (The overshoot: familiar block pose, tip
  on the wrong side.) This is repairable by moving **only the robot** — the
  one thing we command directly, without touching the task.

The scheme targets the second kind, detects it with the conditional, gates on
the marginal, and repairs it by transport. Formally, the demonstrated density
factors as

$$p_{\mathcal{D}}(x_R, x_E) \;=\; p_{\mathcal{D}}(x_E)\;\rho(x_R \mid x_E),
\qquad \rho(x_R \mid x_E) := p_{\mathcal{D}}(x_R \mid x_E),$$

and the two shift types are exactly "which factor got small."

**The three population objects.** Everything in the scheme — low-dim v0 and
high-dim version alike — is an estimator of exactly three things:

1. the conditional $\rho(x_R \mid x_E)$, used twice: its **log-density at the
   current state** is the detection statistic, and its **samples** are the
   recovery targets;
2. the marginal $p_{\mathcal{D}}(x_E)$ — the applicability gate;
3. $V_{\text{nom}}$ — never estimated by the deployed scheme; the two
   densities act as computable proxies for its degradation, and the
   hypotheses of §5 state *measurably* when the proxy is faithful.

Pinning the scheme to population objects is what makes "translates to
high-dim" a precise statement rather than a hope: swapping kNN for a learned
conditional model changes the *estimator*, not the method (§6).

## 3. The detector

### 3.1 What we would ideally trigger on

Suppose recovery (§4) exists, with value $V_{\text{rec}}(s, k)$ = success
probability if we recover *now* and then resume. Intervening is worth it
exactly when recovery beats continuing:

$$\Delta(s, h, k) \;=\; V_{\text{rec}}(s, k) - V_{\text{nom}}(s, h, k)
\;>\; 0.$$

The ideal detector fires the moment $\Delta$ crosses zero. Three consequences
of this definition, all established in our earlier analysis:

- **Do not wait for certainty.** $V_{\text{nom}}$ degrades *gradually* (it is
  a probability, §1.5); the crossover generically happens well before
  $V_{\text{nom}}$ reaches zero. A detector that waits until failure is
  certain fires after the last moment recovery could still act.
- **Do not fire from the start either.** On healthy states recovery is worse
  than nominal ($\Delta < 0$: it burns clock and interrupts a working
  rollout — the specialist property, §4.5). The firing region opens at a
  *crossover*, after degradation begins, before recovery stops working.
- **The firing region is a window in time**: it opens at the crossover
  $t_x$ and closes at the viability boundary $\tau^*$ (the last time recovery
  still has meaningful value). Its width is the **detection slack** — how much
  warning a real detector gets. The oracle measures both endpoints (§10.5).

### 3.2 From the ideal trigger to computable statistics

$\Delta$ requires counterfactual futures — not computable at deployment. The
deployment-facing constraint, stated once precisely: an intervention rule is a
**stopping time** adapted to the filtration
$\mathcal{F}_t = \sigma(x_0, \dots, x_t, \text{policy internals})$ — in plain
words, *the decision to intervene at $t$ may use only what has been observed
up to $t$*, never privileged simulator access. The scheme's statistics, at
each replan boundary:

$$c_t \;=\; -\log \hat\rho(x_{R,t} \mid x_{E,t})
\qquad\text{“how surprising is the robot's configuration, given the scene?”}$$

$$e_t \;=\; -\log \hat p_{\mathcal{D}}(x_{E,t})
\qquad\text{“how familiar is the scene itself?”}$$

optionally $u_t$ = policy-internal dispersion (Track E1: e.g. variance across
the $K$ candidate chunks the system already samples — free), read as *"how
much does the policy disagree with itself here?"*

**Why the conditional, not joint OOD-ness.** $c_t$ is aligned with
*actionability*: it is large exactly on the shift type recovery can repair,
and the gate $e_t$ excludes the type it cannot. A joint-density detector
conflates the two and fires on unrepairable states. There is a pleasing
economy here: **the recovery model doubles as its own best-aligned detector**
— the same $\hat\rho$ that generates targets scores the current state.

Two known blind spots, recorded so nobody is surprised later: policy-internal
$u_t$ can be *confidently wrong* off-manifold (low variance $\ne$ safe), and
all support statistics are blind to **in-support doom** — states that look
demonstrated and fail anyway (bad luck, dynamics). The failure decomposition
(§11.4) measures how much mass each blind spot hides.

### 3.3 The rule

$$\tau_{\text{stop}} \;=\; \min\bigl\{\, t \in \mathcal{T} :\
c_t > \kappa
\ \wedge\ e_t \le \kappa_E
\ \wedge\ n_{\text{int}} < N_{\max}
\ \wedge\ t - t_{\text{last}} \ge \Delta_{\text{cool}} \,\bigr\}$$

with each knob doing one job: $\kappa$ — the sensitivity (sweeping it traces
the frontier of §4.6); $\kappa_E$ — the applicability gate (don't fire where
recovery can't help); $N_{\max}$ — intervention budget (v0: 1);
$\Delta_{\text{cool}}$ — cooldown, preventing intervention *loops* when the
nominal policy re-drifts from the same region. Thresholds are set as
percentiles (e.g. 99th) of the same statistics on held-out demo states — a
convention that transfers unchanged to high-dim.

## 4. The recovery operator

### 4.1 The maneuver, in the running example

The overshoot, replayed with the scheme: at some replan boundary, $c_t$
spikes (tip-given-block surprising) while $e_t$ stays low (block pose
familiar). Fire. Retrieve demo states whose block pose is near the current one
— they all have the tip *behind the push face*. Pick one; lift the stick,
travel over the block, descend behind the push face; hand control back. The
policy now sees a configuration it was trained on and simply continues the
task.

### 4.2 The four stages, precisely

Executing recovery from $(s, t)$:

1. **Generate.** Sample $M$ candidate robot configurations
   $x_R^{(1..M)} \sim \hat\rho(\cdot \mid x_{E,t})$.
2. **Select.** Pick $x_R^\star = \sigma\big(x_t, \{x_R^{(j)}\}\big)$ — v0
   rule: nearest reachable candidate. (Which rule is best is an open
   sub-problem; the oracle sidesteps it by trying all candidates.)
3. **Transport.** A scripted controller $\xi$ drives the robot to
   $x_R^\star$, consuming $T_{\text{ret}}$ steps of the *same* episode clock,
   under the **no-disturbance requirement**: the environment must not change.
   Formally, the "bump" event
   $B = \{ d_E(x_{E,\,t+T_{\text{ret}}},\ x_{E,t}) > \epsilon_B \}$ must have
   small probability $p_B$. PushT instantiation: **lift–translate–descend**.
   The `pd_ee_delta_pose` controller can lift the stick clear of the block, so
   $p_B \approx 0$ *by construction*, and
   $T_{\text{ret}} \approx \lceil \text{path length}/\delta_{\max} \rceil$
   steps. (The lifted path is itself off-manifold — irrelevant: the policy is
   not consulted during transport.)
4. **Resume.** Clear the nominal system's internal state — the observation
   history and chunk cache describe a trajectory that no longer exists across
   the discontinuity; feeding them forward would condition the policy on a
   lie — then follow $\Sigma_{\text{nom}}$ for the remaining
   $k - T_{\text{ret}}$ steps.

### 4.3 Formal wrapper

Recovery is a **semi-Markov option** $o_{\text{rec}} = (\mathcal{I}, \xi,
\beta)$ — an "option" being the standard RL formalization of a temporally
extended macro-action: an initiation set $\mathcal{I}$ (here: the detector's
firing region), an internal policy $\xi$ (the return controller), and a
termination condition $\beta$ (arrival or timeout). Its value:

$$V_{\text{rec}}(s, k) \;=\;
\mathbb{E}\Big[\, V_{\text{nom}}\big(s',\, h_\emptyset,\, k - T_{\text{ret}}\big)\Big],$$

expectation over generation, selection, and transport (including bumps), with
$h_\emptyset$ = fresh internal state. Note $V_{\text{rec}}$ does not depend on
$h$ — recovery discards history. This is the object whose ceiling Part II
measures.

### 4.4 Why success-only data is enough — the central elegance

The classic obstruction to learned recovery (Track F): correcting requires
examples of *being lost and getting un-lost*, which success-only demos do not
contain. This scheme never needs them, because it decomposes the maneuver
into:

- a **target distribution** — $\rho(x_R \mid x_E)$, which the demos supply
  directly and densely (every demo state is a training pair), and
- **transport** — outsourced to a planner, which needs geometry, not data.

Nothing in the pipeline ever consumes a failure example. The scheme is,
precisely, a *covariate-shift resetter*: it cannot fix "the policy is bad at
this demonstrated situation" (that is selection/training territory); it fixes
"the policy was never shown this situation, but a nearby shown one is
reachable by moving the robot."

### 4.5 The specialist property — why worse-on-healthy is a feature

On a healthy state, recovery interrupts a working rollout, burns
$T_{\text{ret}}$ steps, resets useful history, and adds a small bump risk:
$\Delta < 0$. This is not a defect to engineer away — it is what makes
detection a real problem (§0.3). If recovery dominated nominal everywhere,
the right system would run it always, no detector needed, and the whole
construct would collapse into policy improvement. The falsifiable signature
of the specialist structure — negative advantage on successes — is directly
measured by the oracle's false-alarm arm (§10.6).

### 4.6 The switched system and its accounting

$\Sigma_{\text{rec}}(\kappa, \kappa_E, N_{\max}, \Delta_{\text{cool}})$: run
$\Sigma_{\text{nom}}$; evaluate the rule at replan boundaries; on firing,
execute $o_{\text{rec}}$ once; resume. Objective:
$J(\Sigma) = \Pr_{s_0 \sim \mu_0}(Y = 1)$, reported as a **frontier** over
$\kappa$ — success vs intervention rate vs false-alarm cost — because a
single number hides the trade that thresholds actually navigate.

All nominal-vs-switched comparisons use common random numbers (same env
seeds, sim state replay), so outcomes decompose per-episode into
$\{+\text{saved}, -\text{broken}, =\}$ — the same ledger that made the
selection oracle's $+70/-0$ legible.

### 4.7 The advantage, decomposed

Putting §4.2–4.5 together (informally but term-by-term):

$$\Delta(s, h, k) \;\approx\;
\underbrace{\big[\nu(x_E,\, k - T_{\text{ret}}) - V_{\text{nom}}(s, h, k)\big]}_{\text{value of being back on support}}
\;-\;\underbrace{p_B \, L_B}_{\text{bump risk}}
\;-\;\underbrace{\big[\nu(x_E, k) - \nu(x_E, k - T_{\text{ret}})\big]}_{\text{clock cost}}$$

where the **restored value**

$$\nu(x_E, k) \;=\; \mathbb{E}_{x_R \sim \rho(\cdot \mid x_E)}
\Big[V_{\text{nom}}\big((x_R, x_E),\, h_\emptyset,\, k\big)\Big]$$

is "the success probability of a robot placed where demos would have it, in
this scene, with $k$ steps left." Reading the decomposition: intervene when
the current state's value has fallen well below the restored value
(off-conditional-support), and the transport costs are small. On healthy
states $V_{\text{nom}} \approx \nu$ and the costs make $\Delta < 0$; the
window opens at the crossover, closes at $\tau^*$.

## 5. The hypotheses — the scheme's load-bearing beliefs, stated measurably

The scheme is sound iff four things are true. None is assumed; each maps to a
specific oracle deliverable.

- **H1 (detectability with slack).** Failing rollouts enter the
  high-$c$ region *before* recovery stops being possible — there exists a
  common threshold $\kappa$ with
  $\Pr(\exists\, t \le \tau^* : c_t > \kappa \mid Y{=}0)$ high and
  $\Pr(\exists\, t : c_t > \kappa \mid Y{=}1)$ low.
  *If false:* failures are invisible to support statistics, or visible only
  too late. → measured by the deadline-aware ROC (§11.5).
- **H2 (conditional-shift dominance).** A substantial fraction of failure
  mass is of the *repairable* kind — scene familiar, robot lost.
  *If false:* the scheme addresses a sliver. → measured by the failure
  decomposition (§11.4).
- **H3 (support restoration ⇒ value restoration).** $\nu(x_E, k)$ is high
  where $p_{\mathcal{D}}(x_E)$ is high — the nominal policy is good *on its
  support*, so putting the robot back restores success probability.
  *If false:* the policy fails even from demonstrated configurations, and no
  repositioning helps. → this is exactly what the teleport oracle estimates
  (§10.3).
- **H4 (cheap transport).** $T_{\text{ret}}$ is small relative to remaining
  budget and $p_B \approx 0$.
  *If false:* the clock/bump costs eat the restored value. → measured by the
  teleport-vs-executed gap (§10.3).

## 6. Estimators: v0 (train nothing) and the high-dim regimen

Both versions estimate the three population objects of §2. **Only the
estimator changes** — scheme, statistics, thresholds, rule, and oracle are
identical. This section is the "translates to high-dim" contract.

### 6.1 v0 — state mode, nonparametric, this repo

Justified by §1.4 (intrinsic dim ≈ 5, n ≈ 14k):

- $\hat\rho(x_R \mid x_E)$ by **retrieval**: the $m$ demo states nearest in
  the environment metric $d_E$ define an empirical conditional. *Sampling* =
  returning their robot configurations (optionally jittered); *surprisal*
  $c_t$ = distance from the current $x_{R}$ to those neighbors' robot
  configurations — a monotone surrogate of $-\log\rho$.
- $\hat p_{\mathcal{D}}(x_E)$ by kNN distance in $\mathcal{X}_E$ under

  $$d_E\big((p,\theta),(p',\theta')\big) \;=\; \|p_{xy} - p'_{xy}\|/L
  \;+\; \lambda\, d_\angle(\theta_z, \theta_z'),$$

  planar position scaled by workspace size $L$; yaw geodesic distance
  weighted by $\lambda$ (v0 default $\lambda = L/\pi$: a half-turn "costs"
  one workspace length; a sensitivity sweep over $\lambda$ is part of the
  oracle run). This hand-built metric is the §1.4 point about metrics in
  action: we spend our design effort where adaptivity does not reach.
- Thresholds $\kappa, \kappa_E$: percentiles of the same statistics on
  held-out demo states.
- A pleasant bonus: retrieval returns **full joint configurations** (qpos),
  so the recovery target needs no inverse kinematics.

**Known v0 limitation — no compositionality or sparse-region interpolation.**
kNN's implicit manifold model is a union of balls around the 14k samples
(0th-order): it cannot credit a valid state *between* distant samples, and —
more fundamentally — cannot credit a **novel combination of familiar
factors** (block pose from one demo, relative approach from another). The
policy, being a DNN, generalizes along exactly those directions, so the
detector's true target (the policy's *competence region*, §3.1) is a
thickened completion of the sample that kNN under-covers → systematic
over-flagging of healthy states (false-alarm cost; a frontier shift, not a
breakage). The converse caveat: learned density models fail in the opposite
direction — flows/likelihood models notoriously assign high likelihood
off-manifold — so kNN is *paranoid* where DNN estimators are *credulous*;
neither dominates a priori. Mitigation ladder (cheap first):
1. **Relative-frame encoding**: express $x_R$ in the block's frame before
   any estimation. Captures PushT's dominant compositional structure
   analytically, and upgrades the *generator*: transfer relative approach
   configurations from ALL demo states onto the current block pose
   (reachability/limit-checked) — compositional generation without learning.
2. **Local-tangent kNN**: distance to the local PCA plane through the k
   neighbors (1st-order manifold completion — principled interpolation).
3. **Learned conditional model** (§6.2) — justified when the calibration
   harness (§11.5), whose labels are ground-truth *competence* ($\Delta$),
   shows kNN false-firing on states MC rollouts prove healthy. The
   kNN-vs-learned choice is thereby an empirical outcome, not a design
   argument.

### 6.2 The high-dim regimen

The observation that makes this cheaper than it sounds: **the generated
object never becomes high-dimensional.** $x_R$ is proprioception — always
low-dim, always available on a real robot. Only the *conditioning* side goes
to images:

$$\hat\rho_\psi\big(x_R \,\big|\, \text{enc}_\psi(x_E)\big),
\qquad x_E = \text{robot-masked image}.$$

- **Training** = masked conditional MLE on demo pairs $(m(x), x_R)$ — the
  literal "mask the robot, generate the robot" regimen. Model class: any
  conditional generative model with tractable sampling and a usable
  log-density or proxy — conditional flow (exact log-density; preferred),
  conditional diffusion (ELBO / denoising-loss proxy), CVAE (ELBO). **No
  image generation and no image density estimation ever occurs on the
  conditional path.**
- The encoder is the §1.4 *metric repair*: it restores coordinates in which
  distance tracks the ~5-dim scene manifold; the rest of the machinery is v0
  unchanged.
- The genuinely hard high-dim object is the **marginal gate**
  $\hat p_{\mathcal{D}}(x_E)$: v1 = kNN in the encoder latent; upgrades =
  latent flow / one-class methods (Track E2 proper).
- Two honest deltas from v0: the robot mask itself must come from somewhere
  (segmentation — the one nontrivial prerequisite), and a generated EE pose
  may need IK where retrieval's qpos did not.
- Calibration transfers as-is: percentile thresholds on held-out demos;
  validation against oracle labels (§11.5) is modality-independent.

### 6.3 Refinements (recorded, deliberately not v0)

- **Phase conditioning.** $\rho(x_R \mid x_E)$ marginalizes over
  demonstration *phase*: for one block pose, demos contain both
  approach-phase and push-phase tip positions, and v0 may return the robot to
  an early-phase pose late in the task. Not fatal (the resumed policy re-does
  the phase; it costs clock, not correctness). Refinement:
  $\rho(x_R \mid x_E, \hat\zeta)$ with progress $\hat\zeta$ from the
  Track-B5 temporal-order model — a clean cross-track reuse.
- **Selection rule** $\sigma$ beyond nearest: max-$\hat\rho$, reachability
  filtering, or (Track C) predicted-value ranking.
- **Multi-intervention** ($N_{\max} > 1$) with hysteresis.

---

# Part II — The ceiling oracle

## 7. What is being bounded, and why it must be a *scheme-class* ceiling

**The degeneracy problem, one more time, because it dictates the design.**
Suppose we "idealized" recovery as *oracle-K selection from the current
state* or *sim-optimal control*. Both dominate nominal everywhere
($V_{\text{rec}} \ge V_{\text{nom}}$ at every state), so the "when to
intervene" question evaporates (answer: always), and the measured number is
the value of a better policy — i.e. it *re-measures selection / Track C
headroom* (+23.3pt, already priced), telling us nothing about recovery as a
distinct lever. A meaningful E+F ceiling must bound a **class of recovery
maneuvers with the specialist structure** — and §4 defines exactly such a
class.

**The method: oracle-ize each stage independently.** The scheme is a
four-stage pipeline — *when / which pose / transport / resume*. Replace each
stage by its idealization while keeping the class structure:

| stage | deployed (later) | oracle (now) |
|---|---|---|
| when | threshold detector on $c_t, e_t$ | exhaustive sweep over intervention times |
| which pose | learned $\hat\rho$ + rule $\sigma$ | retrieval pose set; try **all**, keep best |
| transport | scripted controller | teleport (perfect) *and* executed (real) |
| resume | nominal, fresh state | same |

The product upper-bounds any implementation of the class (under the scope
restrictions of §13), and — the practical payoff — the **per-stage gaps**
later localize where a real implementation loses headroom: if the learned
system undershoots the ceiling, comparing against the stage oracles says
whether detection timing, pose generation, or transport ate the difference.

**Epistemic status.** Diagnostic, like C6 and the selection oracle: sim-only,
never used for training or checkpoint selection, teleport explicitly labeled
non-physical.

## 8. The estimand

For a recovery operator $\mathcal{R}$ (two versions in §10.3):

$$J_{\text{ceil}}(\mathcal{R}) \;=\; \frac{1}{n}\Big[\,|S| \;+\;
\sum_{i \in F} \max\Big(0,\ \max_{t \in \mathcal{T}_i}
V^{(i)}_{\text{rec}}(t)\Big)\Big]$$

read: successes count 1 (success-once is already realized — the oracle
abstains; nothing after can un-succeed them); each failure contributes the
value of intervening at its **best** grid time on its realized prefix — the
single-intervention optimal-stopping value. $S/F$ partition the reference
episodes by realized outcome; $\mathcal{T}_i$ is episode $i$'s grid of replan
boundaries; $V^{(i)}_{\text{rec}}(t)$ is the true value of intervening at $t$
on episode $i$'s prefix.

## 9. The two controls that keep the number honest

### 9.1 The re-roll control (mandatory)

Recall §1.5: a realized failure is one unlucky draw. Continuing nominally
from a failing episode's prefix *with fresh policy noise* already has positive
value — sometimes a lot. So define the value of **oracle-timed doing
nothing**:

$$J_{\text{reroll}} \;=\; \frac{1}{n}\Big[\,|S| \;+\; \sum_{i \in F}
\max_{t \in \mathcal{T}_i} V^{(i)}_{\text{nom}}(t)\Big]$$

— pick the luckiest moment to simply keep going, with re-rolled dice. Every
mechanism claim uses

$$\textbf{recovery headroom} \;=\; J_{\text{ceil}}(\mathcal{R}) -
J_{\text{reroll}},$$

never $J_{\text{ceil}} - J_{\text{nom}}$. Otherwise the ceiling is credited
with what fresh randomness achieves on its own. This is the recovery analog
of the selection oracle's random-candidate control, and skipping it is the
single easiest way to fake a positive result with this experiment.

### 9.2 The winner's-curse control (two-phase estimation)

The estimand takes $\max_{t,\,\text{pose}}$ over Monte-Carlo *estimates*, and
the max of noisy estimates is biased upward. Toy version: twenty coins, each
with true heads-rate 30%, each estimated from 5 flips — the *maximum*
estimate will typically read 60–80%. Same mechanism here, at ~13 grid times ×
3 poses per episode.

Protocol, per failing episode:

- **Phase 1 (search):** estimate $\hat V_{\text{rec}}(t; j)$ for all grid
  times $t$ and poses $j$ with small $K_{\text{sel}}$; pick the winner
  $(t^\star, j^\star)$. Likewise $t^\star$ for the re-roll control.
- **Phase 2 (report):** re-estimate *only the winners* with fresh seeds and
  larger $K_{\text{eval}}$. **All reported estimands use phase-2 numbers
  only**; phase-1 numbers appear only inside curves labeled search-phase.

This is the same discipline — and for the same reason — as
screening→confirmation in the checkpoint-selection protocol.

Aggregate uncertainty: nonparametric bootstrap over episodes of the
per-episode phase-2 values (each is unbiased, so the episode-mean is
unbiased and the bootstrap CI is valid).

## 10. The protocol

### 10.1 Reference run and snapshots

1. Fix a panel $E$ of $n$ episodes (dev: $n = 100$), env seeds from a
   dedicated diagnostic panel — disjoint from screening / confirmation /
   final_test.
2. Run $\Sigma_{\text{nom}}$ once per episode; record outcomes → $S, F$; at
   every replan boundary save the **extended snapshot** $(s_t, h_t)$ — full
   simulator state *plus the system's observation history*. Saving $h_t$ is
   what makes nominal continuations exact rather than approximately restarted
   (the sim save/restore machinery is already validated by the selection
   oracle).
3. Failures run to $T_{\max}$ (success-once), so
   $|\mathcal{T}_i| \le \lceil 100 / 8 \rceil = 13$ snapshots each.

### 10.2 Monte-Carlo value estimators

From a snapshot $(s_t, h_t)$, $k = T_{\max} - t$:

- $\hat V_{\text{nom}}(t)$: restore $(s_t, h_t)$ — including the history —
  and roll $\Sigma_{\text{nom}}$ to the horizon with $K_{\text{nom}}$ fresh
  policy seeds; average success.
- $\hat V_{\text{rec}}(t; j)$: apply the recovery variant with pose $j$,
  reset internal state to $h_\emptyset$ (recovery invalidates history — same
  as deployment, §4.2), roll with $K$ fresh seeds; average.

Both are unbiased per $(t, j)$ with standard error $\le 1/(2\sqrt{K})$. All
randomness is enumerated: the env is deterministic given state; policy
generators are seeded per rollout. (Why MC at all: §1.5 — values are
probabilities; single deterministic branches, which sufficed for the
selection oracle's one-shot comparisons, cannot estimate them.)

### 10.3 Stage oracles

**(when)** Exhaustive sweep of $t$ over $\mathcal{T}_i$ — at ≤13 points,
no coarsening needed.

**(which pose)** Retrieval instantiation of $\hat\rho$: the $M$ demo states
nearest in $d_E$ to $x_{E,t}$ contribute their robot configurations (qpos;
velocities zeroed). Oracle selection = try **all** $M$, keep best-by-value
(under the §9.2 two-phase discipline).

**(transport)** The two-rung ladder:

- $\mathcal{R}_{\text{tp}}$ — **teleport** (non-physical, labeled as such):
  overwrite the arm's qpos with the candidate's, zero qvel, leave the block
  untouched; $T_{\text{ret}} = 0$, $p_B = 0$. This is the pure **H3 probe**:
  it estimates the restored value $\nu(x_E, k)$ — the value of *being* back
  on the conditional support — with every transport cost deleted.
  *Feasibility check F1 (do first):* verify the ManiSkill state vector can be
  edited at the articulation slice and re-set exactly; fallback is
  executed-only.
- $\mathcal{R}_{\text{ex}}$ — **executed** (physical): the scripted
  lift–translate–descend return under `pd_ee_delta_pose`, real physics, real
  clock; measures $T_{\text{ret}}$ and bump events; timeout
  $T_{\text{ret}} \le T^{\max}_{\text{ret}}$ (e.g. 16 steps), then resume
  regardless.

Expected: $J_{\text{ceil}}(\mathcal{R}_{\text{ex}}) \le
J_{\text{ceil}}(\mathcal{R}_{\text{tp}})$ — the gap is the **transport tax**
(H4). Not a theorem: an accidental *beneficial* bump during the return can
locally invert it; a violation is itself a finding (beneficial contact), so
both rungs are always reported.

**(resume)** Always $\Sigma_{\text{nom}}$ with fresh internal state. A
stacked variant that resumes with oracle-selection exists as a *later* cell —
it deliberately conflates tracks and is excluded from the primary ceiling.

### 10.4 Derived per-episode objects

From the same sweep (search-phase curves), per failing episode:

- the **advantage curve**
  $\hat\Delta^{(i)}(t) = \hat V^{(i)}_{\text{rec}}(t) -
  \hat V^{(i)}_{\text{nom}}(t)$ on $\mathcal{T}_i$;
- **crossover** $\hat t_x = \min\{t : \hat\Delta(t) \ge \delta\}$, with noise
  margin $\delta = \max(0.05,\ 2\,\mathrm{SE})$ so MC jitter cannot
  manufacture a crossing;
- **viability boundary**
  $\hat\tau^* = \max\{t : \hat V_{\text{rec}}(t) \ge p_{\min}\}$, reported at
  $p_{\min} \in \{0.25, 0.5\}$;
- **detection slack** $= \hat\tau^* - \hat t_x$, in steps and in seconds
  ($f_c = 20$ Hz) — the deadline a deployable detector must meet.

### 10.5 The false-alarm arm

On ~20 sampled successes × 3 grid times, apply recovery *anyway* and estimate
$\hat\Delta$ (expected $\le 0$). Two purposes: verify the specialist property
(§4.5) empirically, and calibrate the cost side of the $\kappa$-frontier —
what one false alarm costs in success probability.

## 11. Deliverables — each with the question it answers

1. **Ceiling table** — *how much is recovery worth, at most?*
   $J_{\text{nom}}$, $J_{\text{reroll}}$,
   $J_{\text{ceil}}(\mathcal{R}_{\text{tp}})$,
   $J_{\text{ceil}}(\mathcal{R}_{\text{ex}})$ with bootstrap CIs;
   **recovery headroom** = ceiling − re-roll; **transport tax** = tp − ex.
2. **Slack distributions** — *how much warning does a detector get?*
   Histograms of $\hat t_x$, $\hat\tau^*$, slack. This is the detector's spec
   sheet: if slack is typically one replan period, only fast statistics
   qualify; if it is seconds, almost anything does.
3. **Additivity 2×2** — *new lever, or selection repackaged?* Run the
   existing selection-oracle machinery **on the same panel and same nominal
   policy**; cross-tabulate per-episode {saved by oracle selection} × {saved
   by oracle recovery}. The recovery-only cell is genuinely new headroom; the
   overlap cell is Track-C mass in a costume. This is the strategic
   deliverable — it decides whether E+F earns a lane of its own.
4. **Failure decomposition** — *what kinds of lost are we?* Classify each
   failure with the v0 statistics on its reference trajectory (no training):
   (a) **scene lost** — $e_t > \kappa_E$ throughout the pre-$\tau^*$ window:
   unrepairable by this class; (b) **robot lost** — $c_t > \kappa$,
   $e_t \le \kappa_E$ somewhere in-window: the addressable mass (H2);
   (c) **doomed while looking fine** — never OOD, fails anyway: invisible to
   support statistics, bounded instead by the re-roll and selection ceilings.
   Thresholds at the 99th percentile of held-out demo statistics.
5. **Detector calibration set** — *which cheap statistic tracks the truth?*
   Labeled pairs $\big(x_t^{(i)},\ \mathbb{1}[\hat\Delta^{(i)}(t) >
   \delta]\big)$ for every snapshot. Candidate statistics ($c_t$, $e_t$,
   joint kNN, policy dispersion $u_t$) evaluated two ways: pointwise AUROC,
   and the **episode-level deadline-aware ROC** — a threshold scores a hit on
   a recoverable failure only if it fires *inside* $[\hat t_x, \hat\tau^*]$
   (firing after the deadline is a miss, not a partial credit), and a false
   alarm on a success if it fires at all; sweep $\kappa$. This harness — not
   deployment runs — is what all future Track-E detectors are judged by.
6. **Sensitivity** — $\lambda$ (yaw weight), $M$, $p_{\min}$, grid density,
   plus the §1.4 intrinsic-dimension measurement.

## 12. Budget arithmetic (dev scale)

Per failing episode: ≤13 grid points × [$M{=}3$ poses × $K_{\text{sel}}{=}3$
+ $K_{\text{nom}}{=}4$] + phase-2 $K_{\text{eval}}{=}10$ ≈ **180 partial
rollouts**, mean remaining length ~50 steps → ~9k env-steps ≈ 1.1k policy
queries at $H_a = 8$. At $n = 100$ (≈48 failures at the current ~52% base):
≈ 430k env-steps ≈ 54k policy queries, plus the false-alarm arm (noise) and
one same-panel selection-oracle run. Order **1–2 GPU-hours** with the DP
nominal (40 ms/query); minutes-level policy cost with ACT (2 ms). All knobs
($K$'s, $M$, $n$) scale linearly.

## 13. Assumptions, scope, and caveats

- **A1** exact sim state save/restore (validated by the selection oracle).
  **A2** a measurable robot/environment split (given by the state schema; in
  image mode requires a robot mask). **A3** success-only demos suffice for
  the *scheme* (§4.4); the *oracle* additionally needs the simulator.
  **A4 (revised 2026-07-18)** the simulator is NOT deterministic given
  (state, action): `physx_cuda` contact resolution varies run-to-run (~96%
  per-episode agreement on repeated sparse-replan evals, ~81% dense-replan).
  The MC value estimators absorb this automatically — each rollout draws
  fresh sim noise along with its policy seed, so $\hat V$ estimates the value
  over *both* noise sources, which is the deployment-relevant quantity. Two
  residual consequences: the reference-run prefix is itself one draw of sim
  noise (fine — the estimand conditions on the realized prefix), and paired
  comparisons across separately executed runs carry sim-noise variance that
  within-run pairing does not. **A5** interventions at replan boundaries
  only.
- **Scope restrictions on the word "ceiling."** Single intervention,
  grid-restricted timing, retrieval-restricted pose set, this recovery
  class. A multi-intervention or finer-grained method could exceed it; the
  *unconstrained* ceiling is the viability/selection quantity, measured
  elsewhere.
- **PushT is a generous testbed for recovery — read the number accordingly.**
  The task is physically near-reversible: nothing breaks, everything can be
  re-approached, so $\tau^*$ is dominated by the episode clock and windows
  will be wide. The result establishes the mechanism and the measurement
  methodology; it does *not* forecast contact-rich or irreversible tasks,
  where the same machinery would report narrower windows.
- **Tier.** Diagnostic; single policy seed; dev panels. Never used for
  training or checkpoint selection. Teleport labeled non-physical wherever it
  appears. Claim-grade versions follow the standard protocol (≥5 seeds,
  paired, frozen tree).

## 13.5 First results (2026-07-19, flow + DP nominals, diagnostic n=100 each)

`scripts/recovery_oracle.py`; artifacts `outputs/active_min/recovery_oracle/{flow,dp}/`.
Single seed, dev tier. Bug-audit-relevant checks: teleport state-edit exact
(reused validated machinery); two-phase estimation and re-roll control active.

| estimand | flow | DP |
|---|---|---|
| $J_{nom}$ | 53.0% | 51.0% |
| $J_{reroll}$ (luck control) | 59.1% | 57.4% |
| $J_{ceil}(\mathcal{R}_{ex})$ | 61.7% (**+2.6** over luck) | 62.2% (**+4.8**) |
| $J_{ceil}(\mathcal{R}_{tp})$ [non-phys] | 65.2% (+6.1) | 65.1% (+7.7) |
| $J_{proto}$ (v0) | 52.0% (1 saved/2 broken) | 51.0% (8/8) |

The re-roll control earned its place: it carries **most** of the naive
headroom (+6.1/+6.4 of the +8.7/+11.2 tp-vs-nominal gap). The
mechanism-specific physical ceiling is **+2.6/+4.8pt** — an order smaller
than the selection headroom (+23.3) on this task family.

**The taxonomy inversion (replicated on both families).** Classifying each
failure by the v0 detector's own statistics:

| class | flow: n / mean tp / tp-specific | DP: n / mean tp / tp-specific |
|---|---|---|
| robot_lost (detector fires here) | 18 / **0.13** / 3 | 19 / **0.01** / 0 |
| scene_lost (gated out) | 17 / 0.05 / 0 | 15 / 0.20 / 5 |
| looks_fine (detector-blind) | 12 / **0.75** / 5 | 15 / **0.73** / 7 |

Hypothesis verdicts: **H4 holds** (transport tax 2.9–3.5pt — the executed
return is nearly as good as teleport). **H1 partially holds** (windows are
wide — median slack 56 steps ≈ 2.8 s — and the detector is precise: 2–4%
FPR). **H3 is refuted on the detector's own firing class**: a robot flagged
off-conditional-support is almost never saved by repositioning (tp
0.13/0.01) — going off-manifold is a late *symptom* of failure, not a
repairable cause. The recoverable mass (tp ≈ 0.75) sits in **in-support
doom** — episodes that never leave support and fail anyway — where
repositioning acts as a beneficial *re-randomization* of a
committed-but-doomed rollout. That mechanism is selection-adjacent
(inject diversity into a nominal-looking rollout), not covariate-shift
repair.

**Consequences.** (1) On PushT, recovery-as-repositioning is nearly
valueless *as designed*; the doc's reversibility caveat cut the other way —
PushT is so forgiving that failures are rarely caused by being physically
lost. A real Track-F test needs a task where lost-robot is the failure
cause. (2) The measurement infrastructure worked exactly as intended: the
oracle localized the value (looks_fine class), the controls prevented a
+8.7pt fake claim, and the detector concept is implementable (precision
fine) but aims at the wrong class here. (3) The looks_fine finding should
be cross-tabbed against the same-panel selection oracle (§11.3, still
queued) — expectation: substantial overlap with selection headroom.

## 13.6 Debug post-mortem, visualized (2026-07-19)

`scripts/visualize_recovery.py` replays panel episodes through the exact
stage-4 prototype loop with per-step (c, e), decision points, the takeover
span, and handback annotated on video + a metrics timeline. Artifacts in
`outputs/active_min/recovery_oracle/flow/debug_viz/` (both replays
reproduced the recorded fire steps exactly). Two episodes, three layers of
failure:

**Layer 1 — the executed return does not return (mechanical defect, own
fix).** `ReturnController` targets a 3-DOF *tip* position, but c is
measured on 7-DOF *qpos* — the 4-dim null space is uncontrolled. Episode
1229399060: the return **converged perfectly in tip space** (tip error
0.008 < tol, 11 steps, block bump 0.000) and left the firing statistic
**unchanged** — c 0.89 → 0.90, qpos still 0.90 away from the chosen
candidate, while the teleport counterfactual at the same instant is c =
0.00. The prototype restores a 3-dim shadow of the 7-dim state it promised.
Any real fix must land the *configuration*, not the tip: track a demo
arm-path in tip space (waypoint sequence recruits the null space
demo-like), or plan in joint space, or filter candidates by null-space
compatibility.

**Layer 2 — even perfect returns would not save the fired class (the
§13.5 inversion, per-episode).** The same episode's oracle grid: V_nom =
V_tp = V_ex = 0 at *every* boundary including t=0 — doomed before the
policy moved; the c-drift the detector caught (0.08 → 1.34, monotone, e
flat-familiar) is a **symptom** of doom, not its cause. 14/17 fired
failures have tp_final = 0. The exception that proves Layer 1 matters at
all: 1120782160 fires at t=16 where **V_tp = 1.0** (fully recoverable,
non-physically) yet V_ex = 0 — the return burned its 16-step budget
without converging (tip err 0.042), handed back at c = 0.82 with qpos 0.92
from the candidate, and the fix window closed. Fixing Layer 1 converts at
most ~2–3 episodes on this panel — the +2.6pt ceiling, rediscovered
episode-by-episode.

**Layer 3 — detector timing pathologies.** (a) 3/19 fires at t=0: c is
phase-blind — at reset the arm is at home but the m_cond neighborhood at
that block pose is mid-task states, so a perfectly nominal start reads as
OOD (one such fire broke a would-be success, 2105815467). Conditioning on
task phase / elapsed time would remove these. (b) The e ≤ κ_e gate opens
only briefly in scene-drift episodes (1120782160: e crosses κ_e at t≈24,
right after the fire) — correct behavior by design (scene lost ⇒ don't
reposition), but it means the fire window on drifting scenes is one or two
boundaries wide, and N_max = 1 spends the sole intervention inside it.

## 14. Open decisions before implementation

1. **F1 feasibility**: articulation-slice state editing for teleport (build
   $\mathcal{R}_{\text{ex}}$ first regardless — no state surgery).
2. Nominal family for the first run (post-rerun benchmark: flow 52.6 / DP
   51.2 / ACT 45.6 — ACT is ~20× cheaper to sweep, flow/DP are the stronger
   nominals).
3. Panel identity and size ($n = 100$ dev proposed).
4. Constants: $M$, $K_{\text{sel}}$, $K_{\text{nom}}$, $K_{\text{eval}}$,
   $\lambda$, $T^{\max}_{\text{ret}}$, threshold percentile.
5. Whether the same-panel selection-oracle run happens in the same batch
   (recommended: yes — the 2×2 of §11.3 is the point of the exercise).
