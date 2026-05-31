# Tier IV — End-to-end coupled system

**Forward model:** transport + RTM + multi-instrument fusion, from source parameters all the way to simulated radiances across multiple satellites simultaneously. This is the full operational pipeline.

```text
Source params (Q_{1:K}(t), x₀_{1:K}, t₀_{1:K},  ū, θ_wind, c_bg, α_BC, …)
       ↓  [Tier I/II/III transport]
Concentration field  c(x,t)
       ↓  [RTM / AK operator,  per instrument]
Simulated observations  {y_inst}_{inst ∈ {TROPOMI, EMIT, Tanager, GHGSat, …}}
       ↑
Cross-instrument bias correction  bias_inst
```

Tier IV is **assembly + multi-instrument fusion**, not new modelling: it composes transport (any of Tiers I–III) with the [RTM stack](04_rtm_stack.md) and joins observations from multiple satellites into a single coherent posterior. The contribution at this tier is the joint multi-instrument inference, the operational predictor, and the cross-instrument calibration.

---

## (1) Simple model — composed forward over multiple instruments {#tier4-simple-model}

### Per-instrument forward {#tier4-per-instrument-forward}

For a single instrument $\text{inst}$:

$$
\mathbf{y}_\text{inst} \;=\; \mathbf{A}_\text{inst}\, \mathrm{col}_z\!\bigl(\text{transport}(Q(t), x_0, \text{met}) + \mathbf{c}_\text{bg}\bigr) \;+\; \text{bias}_\text{inst} \;+\; \boldsymbol{\varepsilon}_\text{inst}
$$

$$
\boldsymbol{\varepsilon}_\text{inst} \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_{\text{retr},\text{inst}} + \mathbf{R}_{\text{repr},\text{inst}} + \mathbf{R}_{\text{align},\text{inst}})
$$

- $\text{transport}(Q(t), x_0, \text{met})$ — Tier I, II, or III. **$Q(t)$ is time-resolved**, not a static rate (see §3 below).
- $\mathbf{c}_\text{bg}$ — regional background, with prior from the [emission inventory loader](00_prerequisites.md#prereqs-emission-inventory).
- $\mathbf{A}_\text{inst}$, $\mathbf{R}_{\text{retr},\text{inst}}$ — per-instrument from the [RTM stack](04_rtm_stack.md) or directly from the L2 product.
- $\text{bias}_\text{inst} \sim \mathcal{N}(0, \sigma^{2}_\text{inst})$ — per-instrument additive bias, **first-class state element**. Documented inter-instrument biases are $O(\pm 10$ ppb); ignoring them double-counts agreement.
- $\mathbf{R}_{\text{repr},\text{inst}}$ — representation error (model-vs-pixel-footprint mismatch); rises with terrain complexity.
- $\mathbf{R}_{\text{align},\text{inst}}$ — temporal misalignment error (overpass at $t_\text{inst}$ vs. modelled state at $t$).

### State vector — full enumeration {#tier4-state-vector}

Single-overpass coupled inference works in a state space far larger than just $(Q, x_0, t_0)$:

$$
\mathbf{x} \;=\; \bigl(\,
   Q_{1:K}(t),\;
   x_{0,1:K},\;
   \bar{u},\, \theta_\text{wind},\;
   \mathbf{c}_\text{bg},\;
   \alpha_\text{BC},\;
   \text{bias}_\text{inst},\;
   A_\text{surf},\, \text{AOD},\,
   \dots
\bigr)
$$

with trans-dimensional $K = n_\text{sources}$ (basin case). Single-source single-instrument is the sanity-check special case, not the operational target.

### Multi-instrument fusion {#tier4-multi-instrument-fusion}

The joint observation operator is a **list-of-forwards** keyed on `instrument_id`, not a single forward:

$$
\mathbf{y} \;=\; [\mathbf{y}_\text{TROPOMI},\, \mathbf{y}_\text{EMIT},\, \mathbf{y}_\text{Tanager},\, \mathbf{y}_\text{GHGSat},\, \dots], \qquad
\mathbf{H}(\mathbf{x}) \;=\; \bigl[\, \mathbf{H}_\text{inst}(\mathbf{x}) \;:\; \text{inst} \in \text{instruments} \,\bigr]
$$

Each $\mathbf{H}_\text{inst}$ carries its own AK, footprint, native resolution, observation time, and quality-flag schema (see [s5p_tropomi,emit,carbon_mapper,ghgsat]).

!!! important "Don't pre-regrid to a common resolution"
    Pre-regridding loses information; do the AK + footprint averaging at native resolution per instrument, fuse at the likelihood level.

### Spatiotemporal alignment — $Q(t)$ as a stochastic process {#tier4-q-of-t}

Different satellites overpass at different times. With a static $Q$ the coupled forward implies the same source state at every overpass — wrong for intermittent/leak emissions and wrong over multi-day windows.

**Default:** $Q(t) \sim \text{Ornstein–Uhlenbeck}$ with a basin-typical correlation timescale (hours to days), or **Gaussian process** prior with Matérn-3/2 covariance:

$$
\mathrm{d} Q(t) \;=\; -\theta\, \bigl(Q(t) - \mu_Q\bigr)\, \mathrm{d}t \;+\; \sigma_Q\, \mathrm{d}W(t)
$$

Captures intermittent / burst emissions naturally.

### Build order

Start with the cheapest combination that is still physically coherent, with multi-instrument fusion enabled from day 1:

1. **Tier I + AK + L2 fusion across {TROPOMI, GHGSat, EMIT}** for static $Q$. This is the v1 target — Methane Alert and Response System (MARS, UNEP-IMEO) style attribution with multi-satellite cross-validation.
2. Lagrangian (Tier II) + AK + L2 fusion → handles wind-driven plumes; same fusion harness.
3. FV (Tier III) + neural RTM + L1 fusion → full L1-radiance inversion with end-to-end gradients.
4. $Q(t)$ stochastic-process upgrade once multi-day events appear in the catalog.

The point: **don't try to ship the most complex tier first.** Each upgrade replaces a single block in the diagram; the multi-instrument fusion harness, likelihood structure, and observational comparison stay the same.

---

## (2) Model-based inference {#tier4-inference}

### End-to-end gradient — honest cost {#tier4-gradient}

$$
\nabla_{\mathbf{x}} J \;=\; \nabla_{\mathbf{x}}\!\left[\, \sum_\text{inst} \tfrac{1}{2}\lVert \mathbf{H}_\text{inst}(\mathbf{x}) - \mathbf{y}_\text{inst} \rVert^{2}_{\mathbf{R}_\text{inst}} \;+\; \tfrac{1}{2}\lVert \mathbf{x} - \mathbf{x}_b \rVert^{2}_{\mathbf{B}} \right]
$$

JAX autodiff propagates through transport + RTM jointly — no chain rule by hand. **Cost is non-trivial:** each gradient call runs transport + RTM for every instrument's $\mathbf{H}_\text{inst}$. For Tier III + HAPI that's seconds-to-minutes per call; emulator-based inference (Step 4) is the operational path.

!!! caution "'JAX gives you gradients for free' is a half-truth"
    The gradient is automatic, but the *forward cost per gradient step* dominates wall time. Budget accordingly.

### Cost function {#tier4-cost}

Three terms:

$$
J(\mathbf{x}) \;=\;
   \underbrace{\sum_\text{inst} \tfrac{1}{2}\lVert \mathbf{H}_\text{inst}(\mathbf{x}) - \mathbf{y}_\text{inst} \rVert^{2}_{\mathbf{R}_\text{inst}}}_{\text{observations, per-instrument}}
   \;+\; \underbrace{\tfrac{1}{2}\lVert \mathbf{x} - \mathbf{x}_b \rVert^{2}_{\mathbf{B}}}_{\text{prior on full state}}
   \;+\; \underbrace{\tfrac{1}{2}\lVert Q(t) - \mu_Q(t) \rVert^{2}_{\mathbf{K}_Q}}_{Q(t)\text{ stochastic-process prior}}
$$

$\mathbf{B}$ carries the structured priors from §1 (lognormal $Q$, met-tight $\bar{u}$, $\theta_\text{wind}$, GP/OU on $Q(t)$, etc.); $\mathbf{K}_Q$ is the OU/GP kernel. $\mathbf{R}_\text{inst}$ includes representation, retrieval, and temporal-alignment terms.

### Quality-flag handling

Per-instrument quality flags from the RTM stack flow into the coupled forward. **Default policy:** flagged pixels contribute zero log-likelihood (mask multiplier in $\mathbf{R}^{-1}$).

!!! caution "Don't drop flagged pixels silently"
    Keep masks visible in diagnostics so the effective per-instrument observation count is auditable.

### Posterior covariance

Three paths, mirroring Tier III:

- **Laplace around MAP** — cheapest, default for Tier I + L2 fusion.
- **Gauss–Newton Hessian** via Krylov / [`gaussx`](https://github.com/jejjohnson/gaussx) — used when posterior is approximately Gaussian and tractable.
- **Ensemble (En-EKI / En-4D-Var)** via [`filterax`](https://github.com/jejjohnson/filterax) — required when the posterior is non-Gaussian (multi-modal across $n_\text{sources}$, heavy-tailed $Q$).

Posterior export to Tier V.A is via the same adapter pattern as Tiers I/II/III.

### Trans-dimensional $n_\text{sources}$ {#tier4-trans-dim}

$K = n_\text{sources}$ is itself unknown. Three options:

- **Reversible-jump MCMC (RJMCMC).** Birth/death proposals on source count. Exact but slow.
- **Max-K with masking.** Fix $K_\text{max}$, infer activity probabilities $p_k$ per slot. Tractable, biased toward $K_\text{max}$.
- **Hierarchical Dirichlet-process prior.** Variable $K$ with a non-parametric prior. Middle ground.

v1: max-K with masking ($K_\text{max} = 10$ per basin tile). Promote to RJMCMC when basin events exceed $K_\text{max}$ regularly.

---

## (3) Model emulator — coupled vs. stacked {#tier4-emulator}

Two architectural choices:

### Stacked emulators (tier-modular) {#tier4-stacked-emulator}

Compose Tier-N transport emulator + RTM emulator at runtime.

- **Pros:** any emulator can be swapped independently; intermediate $c(\mathbf{x},t)$ is materialised for diagnostics; modular validation chains directly into Steps 3/4 of each parent tier.
- **Cons:** two emulator calls per forward; no joint training signal.

### Coupled emulator (single network) {#tier4-coupled-emulator}

$$
g_\phi : (\text{met fields},\, \text{source params},\, \text{instrument metadata}) \;\longmapsto\; \text{simulated multi-instrument overpass tensor}
$$

- **Pros:** one network call replaces transport + RTM; massive speedup; potentially learns coupling biases the stack misses.
- **Cons:** intermediate $c(\mathbf{x},t)$ is no longer materialised; cross-tier diagnostics break; retraining required when any block changes.

### Decision rule

- **Development / interpretability / cross-tier diagnostics:** stacked.
- **Operational latency-bound deployment (e.g. real-time alerting):** coupled.

Both should exist; the coupled emulator is validated against the stacked composition before deployment.

### Training-data budget {#tier4-active-learning}

"Millions of pairs" naively needs $O(10^{6})$ transport+RTM simulations. For Tier III + HAPI that's CPU-years on a single machine.

!!! important "Active learning is mandatory"
    Sample sequentially, prioritise loss-residual hot spots and operationally important tails (sun-glint, high AOD, low PBL, multi-source basins). Reaches operational accuracy with $O(10^{5})$ or fewer simulations.

### Domain randomization

Sample the joint $(\text{met regime}, \text{source configuration}, \text{scene class}, \text{viewing geometry}, \text{instrument}, n_\text{sources})$ distribution with **stratified sampling**, not uniform. Naive uniform under-represents the tail regimes that actually drive operational failures.

---

## (4) Emulator-based inference {#tier4-emu-inference}

Use the coupled (or stacked) emulator in EKI ([`filterax`](https://github.com/jejjohnson/filterax)) or gradient-based inversion. Real-time capable.

- **Adjoint validation:** emulator-autodiff gradient ≈ physics-stack gradient on a held-out set. Same hard test as Tiers III and RTM — failure means the inversion is biased even when forward predictions look fine.
- **Posterior validation:** posterior from coupled-emulator inversion ≈ posterior from end-to-end physics inversion (Step 2).

---

## (5) Amortized inference (predictor) {#tier4-amortized}

$$
f_\theta : \bigl(\, \{(\text{instrument\_id}, \mathbf{y}_\text{inst}, \mathbf{A}_\text{inst}, \text{mask}_\text{inst}, \text{footprint}_\text{inst})\},\;
\text{met}_\text{reanalysis},\, \text{transport\_tier\_id} \,\bigr)
\;\longmapsto\;
p\!\bigl(\, Q_{1:K}(t),\, x_{0,1:K},\, K \,\big|\, \text{observations}, \text{met} \,\bigr)
$$

This is **the operational product**: a multi-instrument satellite-overpass list goes in, source-parameter posterior comes out.

### Multi-instrument list input

Input is a *list* of per-instrument observation tuples — same pattern as Tier II/III, generalised to a heterogeneous list. Each element keeps native resolution, AK, mask, and footprint. **No pre-regridding.**

### Per-instrument heads, tier-conditioned

- **Per-instrument summary networks** (TROPOMI 5 km vs. EMIT 60 m vs. Tanager 30 m vs. GHGSat hyperspectral need different encoders).
- **Transport tier as categorical context.** $\text{transport\_tier\_id} \in \{\text{I}, \text{II}, \text{III}\}$ conditions the posterior head — the predictor is one model that handles all tiers, not three separate models.
- **Met + tier conditioning** wired in via FiLM / hypernet primitives in [`pyrox.nn`](https://github.com/jejjohnson/pyrox) — same pattern as Tiers I/II/III.

### Trans-dimensional output

$K = n_\text{sources}$ varies. Default architecture: **max-K masked output**, predicting $(K, \{Q_k(t), x_{0,k}\}_{k=1}^{K_\text{max}}, \text{activity\_mask})$ jointly. Activity mask is a Bernoulli per slot. Promote to RJMCMC predictor head only if max-K masking shows systematic basin saturation.

### Posterior representation

- $Q_k(t)$ posterior is a 1D function — conditional flow over time-axis (`gauss_flows` 1D handles natively).
- $x_{0,k}$ posterior is 2D Gaussian per source.
- $K$ posterior is categorical.
- Joint via factorised flow + categorical head.

### Training data

Simulate millions of $(\text{source config}, \text{multi-instrument overpass})$ pairs spanning the realistic met regime distribution + scene-class distribution + instrument coverage distribution. Active learning over the training schedule (§3) is mandatory at this scale.

---

## (6) Improve {#tier4-improve}

- **Active learning loop.** Flag high-uncertainty scenes for targeted follow-up (e.g. trigger a GHGSat tasking based on a TROPOMI alert; [ghgsat,s5p_tropomi]). Posterior entropy from the predictor is the natural trigger metric.
- **Joint met + source posterior.** Currently we condition on met with tight priors; instead infer $p(Q, \bar{u}, \theta_\text{wind}, \dots \mid \mathbf{y})$ jointly with a looser met prior. WRF becomes informative prior, not hard constraint. Critical for remote regions where reanalysis is poor.
- **Hierarchical bias correction.** $\text{bias}_\text{inst}$ per instrument is currently a flat Gaussian. Promote to hierarchical: per-(instrument, basin, season) — captures known seasonal/regional structure in inter-instrument biases.
- **Multi-species coupling.** When CO + CH₄ are observed jointly (e.g. TROPOMI), use the source-ratio prior to constrain attribution (fossil vs. agricultural).
- **$Q(t)$ hierarchical prior.** Promote OU correlation timescale and amplitude to hyperparameters with their own posterior.

---

## Module layout (proposed) {#tier4-modules}

*Tier IV proposed module layout — step, concern, target module, status.*

| Step | Concern | Module | Status |
| --- | --- | --- | --- |
| 1 | Coupled forward (Tier I + AK + multi-inst) | [`coupled/forward.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/forward.py) (`CoupledForward`, `PlumeSource`) | ✓ |
| 1 | Per-instrument observation spec (receptors, AK, `R = R_retr+R_repr+R_align`) | [`coupled/instrument.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/instrument.py) (`Instrument`) | ✓ |
| 1 | Multi-instrument fusion harness (closed-form `(Q, bias)` posterior) | [`coupled/fusion.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/fusion.py) (`fuse_observations`) | ✓ |
| 1 | Cross-instrument bias model | first-class state in `coupled.fusion` (per-instrument additive bias) | ✓ |
| 1 | Coupled forward (Tier II + AK + multi-inst) | `plumax.coupled` (Tier II transport) | ☐ |
| 1 | Coupled forward (Tier III + RTM + multi-inst) | `plumax.coupled` (Tier III + RTM) | ☐ |
| 1 | Quality-flag aggregator | `plumax.coupled.quality` | ☐ |
| 1 | $Q(t)$ stochastic-process model (OU / GP) | `plume_simulation.coupled.q_dynamics` | ☐ |
| 1 | Trans-dimensional source-count handling | `plume_simulation.coupled.k_sources` | ☐ |
| 2 | End-to-end inversion | reuse [`assimilation/`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/assimilation/) with composed `forward` | ☐ |
| 2 | Posterior covariance (Laplace / Hessian / EnKF) | reuse Tier III's posterior modules | ☐ |
| 2 | Posterior export → Tier V | `plume_simulation.coupled.posterior_export` | ☐ |
| 3 | Stacked emulator runtime | `plume_simulation.coupled.stacked_emulator` | ☐ |
| 3 | Coupled emulator (end-to-end) | `plume_simulation.coupled.emulator` | ☐ |
| 3 | Active-learning training scheduler | `plume_simulation.coupled.active_learning` | ☐ |
| 5 | Operational predictor (per-instrument, tier-conditioned) | `plume_simulation.coupled.predictor` | ☐ |
| 6 | Joint met + source inversion | `plume_simulation.coupled.joint_met` | ☐ |

The `coupled` subpackage doesn't exist yet; this is the proposed shape. It's the only tier where new top-level modules are still needed once Tiers I–III and RTM are done.

---

## Validation strategy {#tier4-validation}

- **Composition correctness.** Apply identity AK, identity RTM, single-instrument list → coupled forward should equal the bare transport forward. Cheap, catches plumbing bugs.
- **Linear-conditional-Gaussian limit.** Tier I + linear AK + Gaussian noise → conditional posterior $p(Q \mid x_0, \bar{u}, \theta_\text{wind}, \mathbf{c}_\text{bg})$ is closed-form via [`gaussx`](https://github.com/jejjohnson/gaussx). Compare end-to-end JAX inversion to the closed-form result. (Note: only $Q$ is linear; the joint over $(x_0, \bar{u}, \dots)$ is nonlinear — phrase the test as conditional, not joint.)
- **Synthetic-truth recovery.** Simulate a known multi-instrument overpass through the full pipeline → run inversion → recover within reported posterior uncertainty. Stratify by instrument count (1, 2, 3+) — fusion benefit should be quantifiable.
- **Cross-instrument hold-out.** Invert with $N - 1$ instruments, predict the held-out instrument's observations from the source posterior + coupled forward, compare predicted to actual. Catches multi-instrument fusion bugs and exposes the value of fusion vs. single-instrument inversion.
- **Cross-tier consistency.** Run inversion with Tier I, II, III transports on the same observations under stationary met conditions. Posteriors should overlap within stated uncertainty. Catches systematic biases between transport tiers and confirms the fusion harness is tier-agnostic.
- **Real-data multi-pass benchmark.** **The validation that proves Tier IV works.** Invert a documented event using $(\text{TROPOMI} + \text{GHGSat} + \text{EMIT})$ simultaneously (e.g. published Permian super-emitter from Sherwin et al. 2024 cross-comparison campaigns); compare posterior to the reported emission. Without this the tier is unvalidated for its intended use.
- **Bias-correction calibration.** With known cross-instrument biases (Sherwin et al.'s controlled-release flights), the inferred $\text{bias}_\text{inst}$ posterior should recover the published values within 95% CI.
- **Predictor calibration.** SBC stratified by instrument count, met regime, scene class. Standard-deviation calibration vs. empirical RMSE on a held-out set.

---

## Open questions {#tier4-open-questions}

!!! attention "Tiering at inference time"
    Should the user choose the transport tier, or should the predictor figure it out (e.g. choose Tier I for stationary winds, Tier II for turbulent regimes)? Probably the former for v1 — explicit is safer; predictor-side dispatch as v2.

!!! attention "Trans-dimensional posterior — RJMCMC vs. masked-K"
    v1 commitment is masked-K ($K_\text{max} = 10$). When does basin saturation force the upgrade to RJMCMC? Operationally: when >5% of overpasses saturate $K_\text{max}$.

!!! attention "$Q(t)$ parameterisation"
    OU process is the simplest non-trivial choice; GP with Matérn-3/2 is more flexible but slower. v1: OU; v2: hierarchical kernel choice.

!!! attention "Cross-instrument bias structure"
    Flat per-instrument Gaussian (v1), per-(instrument, basin) (v1.5), per-(instrument, basin, season) (v2). Promotion driven by residual diagnostics on real-data benchmark.

!!! attention "Training-data budget"
    Coupled-emulator training cost is operationally prohibitive without active learning. Open: target $O(10^{5})$ simulations with sequential design vs. $O(10^{6})$ uniform — confirm the active-learning factor in pilot.

!!! attention "Coupled vs. stacked emulator default"
    Stacked for development. Coupled for deployment. Open: at what predictor latency budget does coupled become required? Likely when overall pipeline must respond <1 s to an alert trigger.

!!! attention "What goes into 'operational'?"
    This is the line between research artefact and product. Need to decide: SLA on inference latency, supported scene types, failure modes, monitoring. Probably outside scope of `plumax` itself — that's `plumax-deploy` or similar.

!!! attention "Posterior summaries"
    A full distribution over all state elements is unwieldy for downstream consumers. Canonical 1-page summary: per-source mode + 68/95% credible region + entropy budget + per-instrument bias posterior + activity flag. Pin the schema before Tier V.A starts consuming.

!!! attention "Quality-flag aggregation"
    Mask flagged pixels (default), or down-weight via inflated $\mathbf{R}$? Open: empirical comparison on real data.
