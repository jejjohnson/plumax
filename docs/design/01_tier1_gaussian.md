# Tier I — Gaussian family

**Forward model:** closed-form analytical solution for steady/transient point-source dispersion.

This is the **first tier to build to completion** (all six steps), because:

1. The forward model is microseconds, so MCMC is feasible end-to-end.
2. Every downstream tier validates against Tier I in the limit of weak turbulence and stationary winds — Tier I is the analytical reference.
3. The amortized predictor at Step 5 doubles as the working prototype for the inference UX (input/output shapes, posterior visualisation, uncertainty budget).

---

## (1) Simple model {#tier1-simple-model}

### Gaussian plume (steady-state, continuous source) {#tier1-gaussian-plume}

$$
c(x',y',z) \;=\; \frac{Q}{2\pi \, \sigma_y \, \sigma_z \, \bar{u}}
   \exp\!\left(-\frac{y'^{2}}{2\sigma_y^{2}}\right)
   \sum_{n} \!\left[
     \exp\!\left(-\frac{(z - H_\text{eff} - 2nL)^{2}}{2\sigma_z^{2}}\right) \;+\;
     \exp\!\left(-\frac{(z + H_\text{eff} - 2nL)^{2}}{2\sigma_z^{2}}\right)
   \right]
$$

The primed coordinates $(x', y')$ are the source-aligned frame: $x'$ is downwind along $\theta_\text{wind}$, $y'$ is crosswind. The image-source sum runs over $n \in \{\dots,-1, 0, 1, \dots\}$ to enforce no-flux at the ground ($z=0$) **and** at the capping inversion ($z=L = \text{PBL height}$). For $L \to \infty$ only the $n=0$ pair survives — that's the unbounded-domain case in most textbooks. For typical PBL $L \approx 1\,\text{km}$ and $\sigma_z$ growing past $\sim L/3$, the upper image is non-negligible.

**Parameters:**

- $Q$ — source strength (kg/s)
- $x_0 = (\text{lat}_0, \text{lon}_0)$ — source location
- $H_\text{stack}$ — physical stack height
- $\Delta h(F_b, F_m, \bar{u}, \text{stability})$ — Briggs plume rise from buoyancy/momentum fluxes ([briggs1973]); $H_\text{eff} = H_\text{stack} + \Delta h$
- $\bar{u}$, $\theta_\text{wind}$ — wind speed and direction (from met, with uncertainty — see [Inference](#tier1-inference))
- $\sigma_y(x'), \sigma_z(x')$ — crosswind/vertical spread; PG-class lookup ([pasquill1961,turner1970]) for v1, MO-similarity ([monin1954,stull1988]) for v2 (see [prereqs](00_prerequisites.md#prereqs-mo-similarity))
- $L$ — PBL capping height (from met)

For methane super-emitter inversions where stack height is poorly known, $H_\text{eff}$ is part of what you infer; for known facilities $H_\text{stack}$ is fixed and only $\Delta h$ is computed.

### Gaussian puff (time-varying, episodic source) {#tier1-gaussian-puff}

$$
c(\mathbf{x}, t) \;=\; \sum_{k} \frac{Q_k}{(2\pi)^{3/2} \sigma^{3}}
  \exp\!\left(-\frac{\lVert \mathbf{x} - \mathbf{x}_k(t) \rVert^{2}}{2\sigma^{2}}\right)
$$

Each puff $k$ advects with the wind and diffuses independently — handles intermittent / burst-mode releases that violate the steady-state assumption.

### From $c(x,y,z)$ to a satellite-comparable observation {#tier1-column-ak}

The forward used by inference is **not** $c(x,y,z)$ directly. It's:

$$
y_\text{model}(x,y) \;=\; \mathbf{A}\,\bigl(\textstyle\int c(x,y,z)\,\mathrm{d}z \;+\; c_\text{bg}(x,y)\bigr)
$$

- **Column integrate:** the satellite sees a column-integrated enhancement, not a single altitude.
- **Background $c_\text{bg}(x,y)$:** the regional background (typically ~1900 ppb for CH₄). **Not optional** — without it the model predicts the *enhancement above background*, which is what $c$ actually is, but the satellite delivers absolute column densities. Either subtract $c_\text{bg}$ from $y_\text{obs}$ upstream, or jointly model it. Either way, it's day-1 infrastructure, not a Step-6 upgrade.
- **Averaging kernel $\mathbf{A}$:** the satellite-product AK from the [prereqs](00_prerequisites.md#prereqs-ak-operator); see (eq-ak-operator). Required when comparing to L2 XCH₄ products. Skip it only when working with a flat-AK assumption (rare and worth flagging).

### Extended Gaussian (AERMOD-style)

Adds terrain corrections (plume rise over hills, beyond Briggs), building downwash (Huber–Snyder), and receptor-grid output. See [cimorelli2005aermod]. Useful as a sanity benchmark against operational tools, **not** a primary research target.

---

## (2) Model-based inference {#tier1-inference}

The plume is analytical, so the full forward $H : (Q, x_0, H_\text{eff}, \bar{u}, \theta_\text{wind}, \dots) \to y_\text{model}(x,y)$ is differentiable via JAX end-to-end (column integration + AK included).

### Likelihood model {#tier1-likelihood}

$$
y_\text{obs}(x,y) \;=\; y_\text{model}(x,y) + \varepsilon(x,y), \qquad
\varepsilon(x,y) \sim \mathcal{N}\!\bigl(0,\, \sigma_\text{retr}(x,y)^{2}\bigr)
$$

- **Default:** heteroscedastic Gaussian on column-XCH₄ enhancement, with per-pixel $\sigma$ from the L2 retrieval-error map.
- **Heavy-tail variant:** Student-$t$ with $\nu \approx 5$ for retrievals near the detection floor where outliers dominate (TROPOMI single-pass [s5p_tropomi], EMIT off-axis [emit]).
- **Mask:** quality-flag mask from the L2 product is passed through as an indicator weight on the likelihood — flagged pixels contribute zero log-likelihood.

!!! caution "Likelihood is load-bearing"
    The likelihood is load-bearing for MCMC convergence. Don't leave it to "default Gaussian" silently — diagnose the residual distribution per instrument before defaulting.

### Priors {#tier1-priors}

*Tier I prior specifications by parameter.*

| Parameter | Prior | Rationale |
| --- | --- | --- |
| $Q$ | $\operatorname{LogNormal}(\mu_Q, \sigma_Q)$ with $\mu_Q$ from the [emission inventory](00_prerequisites.md#prereqs-emission-inventory) prior, $\sigma_Q \approx 1.0$ | positive, heavy-tail; matches inventory uncertainty |
| $x_0$ | $\mathcal{N}(\text{facility}_\text{lat,lon}, \sigma_{x_0})$ with $\sigma_{x_0} \approx$ sub-pixel | facility location is known, sub-pixel uncertainty for cluster vs. point |
| $H_\text{eff}$ | $\operatorname{Uniform}(H_\text{stack},\, H_\text{stack} + \Delta h_\text{max})$ or $\mathcal{N}(H_\text{stack} + \Delta h_\text{briggs},\, \sigma_H)$ | depends on whether stack height is known |
| $\bar{u}$ | $\mathcal{N}(\bar{u}_\text{met}, \sigma_{\bar{u},\text{met}})$ — **tight prior from met** | $\bar{u}$ is data, not a free parameter |
| $\theta_\text{wind}$ | $\mathcal{N}(\theta_\text{met}, \sigma_{\theta,\text{met}})$ — tight prior from met | same |
| $c_\text{bg}$ | GP or Gaussian per-tile around climatology | regional background varies smoothly |

### MAP / MCMC

- **MAP estimation:** `jax.grad(log_posterior)` → L-BFGS or Adam. Convergence in <1 s for a single overpass.
- **MCMC:** NumPyro NUTS over $(Q, x_0, H_\text{eff}, \bar{u}, \theta_\text{wind}, c_\text{bg})$ jointly. Forward pass is ~µs, so 10k samples is seconds. Keep $\bar{u}$, $\theta_\text{wind}$ as inference variables (with tight met priors) so their posterior contracts can be diagnosed downstream.
- **Linear-Gaussian special case:** if observations are linear in $c$ (column-integrated XCH₄ with a fixed AK) and only $Q$ is free with all geometry fixed, the posterior over $Q$ is analytically tractable — exact Bayesian inversion via [`gaussx`](https://github.com/jejjohnson/gaussx). Useful as a sanity check on the NUTS implementation.

### $Q / \bar{u}$ identifiability — corrected {#tier1-q-ubar-identifiability}

The classic statement is "$Q$ and $\bar{u}$ enter only as $Q/\bar{u}$, so they're degenerate in a single transect" [varon2018quantifying]. This is true only if $\bar{u}$ has a flat prior. In production, **the tight met-derived prior on $\bar{u}$ resolves the degeneracy**: the posterior $p(Q \mid y, \bar{u}_\text{met}, \sigma_{\bar{u},\text{met}})$ contracts as long as $\sigma_{\bar{u},\text{met}} / \bar{u}_\text{met} \ll \sigma_Q / Q_\text{prior}$. The "need two crosswind transects" claim is a special case for fully-free $\bar{u}$ and should not be the default reading.

When the wind prior is *not* tight (rare — typically remote regions without good reanalysis coverage), then yes, multiple overpasses with different wind directions break the ratio. But the first-line fix is the prior, not the geometry.

This is the **first working end-to-end inverse pipeline.** Build it here, validate against synthetic releases with known truth, then move to real controlled-release data (see [Validation](#tier1-validation)).

---

## (3) Model emulator {#tier1-emulator}

!!! tip "Emulator is optional at Tier I"
    The Gaussian plume is cheap enough that an emulator is **optional** at this tier. Skip in production, but build it as a training exercise for the emulator infrastructure that Tier III will need.

- **Input:** $(Q, x_0, H_\text{eff}, \bar{u}, \theta_\text{wind}, \text{stability\_class}, L)$ — the *raw* inputs, not the evaluated $\sigma$ profiles. The emulator should learn the $\sigma$ functional too; otherwise you're benchmarking interpolation, not modelling.
- **Output:** $y_\text{model}(x,y)$ on the satellite-pixel grid (post column-integration, post AK).
- **Architecture:** MLP for low-dim parameters → small transposed-conv decoder for spatial output, or lightweight DeepONet.
- **Value:** validates emulator training pipeline (data generation, training loop, residual checks) before applying to expensive PDE models.

---

## (4) Emulator-based inference {#tier1-emu-inference}

Same inference loop as Step 2, but the forward pass is the neural network. Required validation:

- Posterior mean from emulator-MCMC ≈ posterior mean from analytical-MCMC, within $1\sigma$.
- Posterior covariance from emulator-MCMC ≈ analytical posterior covariance, in operator norm.

If those pass for Tier I, the same diagnostics apply unchanged at Tier III.

---

## (5) Amortized inference (predictor) {#tier1-amortized}

Train a **summary network + posterior network** mapping observation patches directly to the source posterior:

$$
f_\theta : (y_\text{patch}, \text{context}) \;\longmapsto\; p(Q, x_0, H_\text{eff} \mid y_\text{patch}, \text{context})
$$

### Input shape — commit to a 2D patch {#tier1-patch}

$y_\text{patch}$ is a fixed-size 2D image patch of column XCH₄ enhancement (background-subtracted) centred on the facility candidate, plus per-pixel uncertainty and quality-mask channels. Concretely:

- Shape: $(C, H_\text{patch}, W_\text{patch})$ with $C = 3$ channels (`enhancement`, $\sigma_\text{retr}$, `mask`).
- Resolution: instrument-native (e.g. 30 m for Tanager [carbon_mapper], 60 m for EMIT [emit], ~5 km for TROPOMI [s5p_tropomi]). Train one predictor per instrument family — don't try to share across resolutions.
- Footprint: large enough to capture the plume's downwind extent at expected wind speeds (typical: 2–10 km square for point-source instruments).

This commitment matters because it pins the architecture (CNN-style backbones over set-transformers or graph nets).

### Conditioning on context

The predictor is **not** observation-only. At inference time we know $\text{context} = (\text{facility}_\text{lat,lon}, \bar{u}_\text{met}, \theta_\text{wind,met}, \text{stability\_class}, L_\text{met})$. The clean way to wire this in is feature-wise modulation of the summary network — the FiLM / hypernet conditioning primitives being built in [`pyrox.nn`](https://github.com/jejjohnson/pyrox).

```python
summary   = CNN(y_patch)              # observation features
modulated = FiLM(summary, context)    # per-feature γ(context)·summary + β(context)
posterior = NPE_head(modulated)       # conditional flow over (Q, x₀, H_eff)
```

!!! important "Always condition on context"
    Without context conditioning the predictor has to relearn met-dependent behaviour from scratch, which is wasteful and well-documented to fail near regime boundaries (stable vs. unstable PBL, low vs. high wind speed).

### Architecture options

- **Conditional normalizing flow** (`gauss_flows`) for the posterior head — preferred for low-dim posteriors over $(Q, x_0, H_\text{eff})$.
- **BNN posterior head** (`pyrox`) when posterior multimodality is unlikely.
- **NPE / SBI-style** — drop-in for either, gives a clean SBC validation interface.

Training dataset is **free**: simulate millions of plume configurations in seconds, sampling met context from the WRF/ERA5 climatology. Validation: posterior calibration via simulation-based calibration (SBC) — uniform rank statistics across 1k held-out simulations, stratified by met regime.

---

## (6) Improve {#tier1-improve}

- **PG → MO $\sigma$ swap.** Replace PG lookup tables for $\sigma_y, \sigma_z$ with MO-similarity-derived functions parameterised by $(u_*, L_\text{Obukhov}, z_0)$ from the [prereqs](00_prerequisites.md#prereqs-mo-similarity). Closer to physics, continuous in stability, and the right pre-step for tier-II validation.
- **Multi-source.** Real basins host 5–50 simultaneous emitters [frankenberg2016airborne,jacob2022quantifying]. This is **not** an array-shape change — the inference becomes a *mixture model with unknown component count*, requiring reversible-jump MCMC (RJMCMC) or a Dirichlet-process prior over source count. Plan for this from the start: the posterior interface that Tier V.A consumes must handle variable-$K$ per overpass.
- **Learned $\sigma$ from LES.** Train a small NN against LES output to learn stability-dependent $\sigma$ functions that go beyond MO. Slot in as a swap-in for either PG or MO $\sigma$. Useful for super-emitter regimes that LES has resolved better than empirical fits.
- **Distributed-source field $Q(\mathbf{x})$.** Replace point source with a spatial source field — opens the door to Tier II/III for spatially extended emissions. At Tier I this is just a sum of point sources sharing met context; the multi-source mixture (above) is the same code path with finer support.

---

## Module layout {#tier1-modules}

*Tier I module layout — step, concern, target module, status.*

| Step | Concern | Module | Status |
| --- | --- | --- | --- |
| 1 | Plume forward | [`gauss_plume/plume.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_plume/plume.py) | ✓ |
| 1 | Puff forward | [`gauss_puff/puff.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_puff/puff.py) | ✓ |
| 1 | Plume rise (Briggs) | `gauss_plume.plume_rise` | ☐ |
| 1 | Stability + dispersion | [`gauss_plume/dispersion.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_plume/dispersion.py) | 🚧 partial |
| 1 | Puff turbulence | [`gauss_puff/turbulence.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_puff/turbulence.py) | ✓ |
| 1 | Column + AK pipeline | `gauss_plume.observation` (links to [`assimilation/obs_operator.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/assimilation/obs_operator.py)) | ☐ |
| 1 | Background $c_\text{bg}$ loader | `plume_simulation.priors.background` | ☐ |
| 2 | Likelihoods + priors | `gauss_plume.likelihoods` | ☐ |
| 2 | Plume MAP/MCMC | [`gauss_plume/inference.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_plume/inference.py) | ✓ |
| 2 | Puff inference | [`gauss_puff/inference.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/gauss_puff/inference.py) | ✓ |
| 2 | Posterior export → Tier V | `gauss_plume.posterior_export` (mark-likelihood adapter for [V.A](06a_instantaneous.md)) | ☐ |
| 3 | Plume emulator | `gauss_plume.emulator` | ☐ |
| 4 | Emulator-based MCMC | wired in `inference.py` once emulator exists | ☐ |
| 5 | NPE / flow predictor | `gauss_plume.predictor` | ☐ |
| 5 | Context-conditioning layer | uses `pyrox.nn` FiLM/hypernet primitives | external dep |
| 6 | Multi-source (RJMCMC) | extend `inference.py` with reversible-jump kernel | ☐ |
| 6 | MO $\sigma$ swap | `gauss_plume.dispersion_mo` | ☐ |

---

## Validation strategy {#tier1-validation}

- **Forward model — mass flux conservation.** Integrate **mass flux** $\int\!\!\int c \, u_\perp \, \mathrm{d}A$ through a transverse plane downwind of the source, compare to $Q$. Should be exact in the no-deposition, unbounded-domain limit. (Common bug: integrating $\int\!\!\int c \, \mathrm{d}A$ instead — that's mass per unit wind speed, dimensionally wrong.)
- **Ground reflection consistency.** Set $H = 0$; the two image-source terms must collapse to a single Gaussian with doubled prefactor. Catches sign/index bugs in the image sum.
- **PBL capping.** With $L \to \infty$ the upper-image series must vanish; with finite $L$ and $\sigma_z$ growing past $L/2$, concentrations must converge to a vertically well-mixed limit:

$$
c \;\longrightarrow\; \frac{Q}{\bar{u}\, L\, \sqrt{2\pi}\, \sigma_y}
   \exp\!\left(-\frac{y^{2}}{2\sigma_y^{2}}\right)
$$

  Tests both branches of the $n$-sum.
- **MAP recovery.** Synthetic release with known $(Q^*, x_0^*, H^*)$. Recovery target: $|Q_\text{MAP} - Q^*| < \sigma_\text{post}(Q)$ — i.e. recovery within the posterior's own claimed uncertainty. (Not a fixed % — that conflates SNR with method quality.)
- **MCMC calibration.** SBC on 1000 simulated releases — rank histograms uniform across all parameters, stratified by met regime.
- **$Q/\bar{u}$ identifiability — empirical.** Run MAP over a sweep of $\sigma_{\bar{u},\text{met}} / \bar{u}_\text{met} \in [0.05, 0.5]$. Posterior $\operatorname{CV}(Q)$ should grow monotonically with met-wind uncertainty; flat CV signals a wiring bug.
- **Real-data benchmark.** Invert a published controlled-release flight (e.g. Stanford / Sherwin et al. 2024 controlled releases over Tanager / GHGSat / EMIT). Posterior median for $Q$ must contain the metered release rate within 95% credible interval. **This is the only test that proves the inference works on real radiances** — synthetic validation is necessary but not sufficient.
- **Emulator agreement.** Step 4 posterior matches Step 2 posterior in mean and covariance to within Monte Carlo error.

---

## Aggregating across overpasses {#tier1-aggregation}

The MAP / MCMC posterior $p(Q \mid \text{overpass})$ produced here is the **per-event evidence** consumed by the population layer. See [Tier V.A — Instantaneous emission estimation](06a_instantaneous.md) for the formal interface that turns this posterior into a mark likelihood for the TMTPP fit, and [Tier V.D — Total emission estimation](06d_total_emission.md) for why per-overpass averages systematically misrepresent regional totals.

The multi-source extension (Step 6) makes this interface variable-$K$ per overpass — the V.A adapter must handle that.

## Open questions {#tier1-open-questions}

!!! attention "Coordinate frame for the puff cloud"
    Carry puff centroids in lat/lon (general but slower) or in a local frame anchored to the source (fast but breaks for puffs that travel far)?

!!! attention "Time-varying wind in the plume model"
    Strictly the steady-state plume assumes constant $(\bar{u}, \theta_\text{wind})$. For ~30-min satellite overpasses winds rotate; do we time-slice (multiple steady-state plumes), advect with hourly met (piecewise-stationary), or jump straight to puff?

!!! attention "Non-Gaussian noise"
    Detection-floor effects mean observation noise is heavy-tailed at low concentration. Default Student-$t$ with $\nu \approx 5$ for predictor training (Step 5); does heteroscedastic Gaussian suffice for MCMC at moderate-to-high SNR? Open until SBC results across SNR regimes are in.

!!! attention "Identifiability — quantitative target"
    Sherwin et al. (2024) report that ~3 overpasses with varied wind direction give $\operatorname{CV}(Q) < 50\%$ on isolated super-emitters. Adopt this as the operational target: the inference is "good enough" when 3-overpass $\operatorname{CV}(Q) \leq 50\%$, single-overpass $\operatorname{CV}(Q) \leq 100\%$ (with tight met prior). Document on the inference docstring.

!!! attention "PG sunset timing"
    When do we deprecate PG $\sigma_y, \sigma_z$? Either (a) immediately swap to MO once the prereq lands, or (b) keep PG as a v1 default and add MO as opt-in. **Leaning (b)** — PG matches the historic literature and gives a stable baseline.

!!! attention "Multi-source prior on $K$"
    RJMCMC needs a prior on source count. $\operatorname{Poisson}(\lambda_K)$ with $\lambda_K \sim 5$ (typical basin)? Geometric? Open — depends on basin and inventory coverage.

!!! attention "Background $c_\text{bg}$ model"
    GP residual on top of a regional climatology, or a simple per-tile constant? GP is "right" but slows MCMC; per-tile is fast but has bias near the source. v1: per-tile; v2: GP.
