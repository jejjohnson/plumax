---
title: "End-to-end: methane retrieval → persistency"
short_title: "E2E — retrieval → persistency"
subject: "plumax — end-to-end pipeline"
authors:
  - name: J. Emmanuel Johnson
    affiliations: [UNEP, IMEO, MARS]
    orcid: 0000-0002-6739-0053
    email: jemanjohnson34@gmail.com
license: CC-BY-4.0
keywords: [plumax, methane, retrieval, emission rate, point process, persistency, MARS, end-to-end, TROPOMI, EMIT, GHGSat]
---

# End-to-end: methane retrieval → persistency

> One satellite radiance in; a leak-recurrence forecast out. This page is the **connective walkthrough** that threads the per-tier notebooks into a single operational story.

The [roadmap](../design/roadmap.md) lays out the two axes of the architecture: a fixed six-step **inference cycle** and a **complexity axis** (Tier I → V). This page reads the architecture the *other* way — not "how does one tier work," but "how do the tiers **compose** into the pipeline an operational methane-monitoring system (MARS-style) actually runs?"

The pipeline has five stages. Each stage is a hand-off: it consumes the previous stage's output and produces the next stage's input. Every stage is independently swappable for a more complex model (the complexity axis) without changing the contract between stages.

```text
  L1/L2 radiance                                                          dispatch / tasking
        │                                                                        ▲
        ▼                                                                        │
 ┌──────────────┐   ΔXCH₄    ┌──────────────┐   p(Q)    ┌──────────────┐  f(Q),λ(t) ┌──────────────┐
 │ 1. RETRIEVAL │ ────────▶ │ 2. INVERSION │ ───────▶ │ 4. POPULATION │ ─────────▶ │ 5. PERSISTENCY│
 │  radiance →  │  column   │  column → Q  │  per-     │  events →     │  intensity │  λ(t) → wait  │
 │  ΔXCH₄ field │  enhance.  │  (per event) │  event    │  f(Q), λ(t)   │  + marks   │  time, P(occ.)│
 └──────────────┘           └──────┬───────┘  posterior└──────────────┘            └──────────────┘
   RTM stack                       │ 3. FUSION (optional)      Tier V.A / V.B            Tier V.C
   hapi_lut / radtran /            ▼ multi-instrument joint
   matched_filter            Tier IV coupled  (Q, bias_inst)
        ▲                    TROPOMI + EMIT + GHGSat
        └──── Tier I/II/III transport supplies the forward model used in stages 2–3
```

:::{tip} The whole pipeline runs at any complexity level
You can run all five stages with the **cheapest** models (Tier I Gaussian plume + L2 retrieval + single instrument) and get an end-to-end answer in seconds — that is the MARS v1 target. Then climb the complexity axis *one stage at a time*: a 4D-Var inversion (Tier III) here, multi-instrument fusion (Tier IV) there. The hand-off contracts below never change.
:::

---

(e2e-stage1-retrieval)=
## Stage 1 — Retrieval: radiance → column enhancement

**Question:** given the at-sensor radiance, how much *extra* methane is in this pixel's column relative to background?

The [RTM stack](../design/04_rtm_stack.md) maps a band-integrated radiance to a column enhancement $\Delta\mathrm{XCH_4}$ (mol m⁻² or ppb·m). For a single absorbing layer the Beer–Lambert forward model gives the normalised radiance $L = \exp(-\Delta\tau)$ with optical depth $\Delta\tau$ built from a HITRAN line-by-line Voigt LUT; the hyperspectral **matched filter** turns a per-pixel spectral residual into a column estimate against a robustly-estimated scene background.

- **Forward model:** [`hapi_lut`](xref:api#plumax.hapi_lut) Beer–Lambert + [`radtran`](xref:api#plumax.radtran) SRF band integration ([theory](hapi_lut/00_hapi_lut_derivation.md)).
- **Estimator:** [`matched_filter`](xref:api#plumax.matched_filter) (trimmed-mean background + low-rank covariance, Woodbury-dispatched via `gaussx`; [theory](matched_filter/00_mf_derivation.md)).
- **Notebooks:** [Beer–Lambert with a CH₄ LUT](../notebooks/hapi_lut/03_beers_law_with_lut.ipynb) · [SRF band integration](../notebooks/radtran/04_srf_band_integration.ipynb) · [matched-filter retrieval on a turbulent plume](../notebooks/radtran/05_matched_filter_retrieval.ipynb).
- **Produces:** a 2-D column-enhancement field $\Delta\mathrm{XCH_4}(x, y)$ with a per-pixel retrieval covariance $\mathbf{R}_\text{retr}$.

---

(e2e-stage2-inversion)=
## Stage 2 — Inversion: column → emission rate

**Question:** what source emission rate $Q$ produced this plume?

The transport forward model relates an emission rate to a concentration / column field. Inverting it gives the per-event **posterior** $p(Q \mid \Delta\mathrm{XCH_4})$ — the load-bearing object the rest of the pipeline consumes. The forward model is the complexity axis in action: a closed-form **Tier I** Gaussian plume/puff, a wind-realistic **Tier II** Lagrangian footprint, or a high-fidelity **Tier III** Eulerian field inverted by 4D-Var.

$$
\Delta\mathrm{XCH_4} \;=\; \mathbf{A}\,\mathrm{col}_z\!\bigl(\text{transport}(Q, x_0, \text{met})\bigr) \;+\; \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon}\sim\mathcal N(\mathbf 0,\mathbf R)
$$

- **Tier I (default):** [Gaussian plume / puff](../design/01_tier1_gaussian.md) emission-rate inference — [plume MAP/MCMC](../notebooks/gauss_plume/02_plume_emission_rate_estimation.ipynb), [puff emission-rate estimation](../notebooks/gauss_puff/02_puff_emission_rate_estimation.ipynb).
- **Tier III (high fidelity):** [Eulerian FV](../design/03_tier3_eulerian.md) strong-constraint **4D-Var** ([`les_fvm`](xref:api#plumax.les_fvm)) with a Laplace / Gauss–Newton posterior covariance — [variational methane retrieval](../notebooks/assimilation/06_variational_methane_retrieval.ipynb), [matched filter vs 3D-Var](../notebooks/matched_filter/01_mf_vs_3dvar.ipynb).
- **Produces:** a per-event posterior $p(Q)$ — full samples *or* a lognormal summary $(\mu_{\log Q}, \sigma_{\log Q})$ — **plus its per-event prior** $\pi_\text{per-event}(Q)$ (required downstream; see Stage 4).

---

(e2e-stage3-fusion)=
## Stage 3 — Fusion (optional): joint multi-instrument posterior

**Question:** when more than one satellite sees the same event, how do we combine them without double-counting agreement?

[Tier IV](../design/05_tier4_coupled.md) keeps each instrument at its **native resolution** (its own AK, footprint, overpass time, quality flags) and fuses at the *likelihood* level — never by pre-regridding. Because the plume is linear in $Q$, the joint posterior over $(Q, \text{bias}_\text{inst})$ across a list of satellites is closed-form, with the per-instrument additive bias a **first-class state element** ($O(\pm10\,\text{ppb})$ documented inter-instrument biases otherwise leak into $Q$).

$$
\mathbf y \;=\; [\mathbf y_\text{TROPOMI},\,\mathbf y_\text{EMIT},\,\mathbf y_\text{GHGSat},\dots], \qquad
\mathbf H(\mathbf x) \;=\; \bigl[\mathbf H_\text{inst}(\mathbf x) : \text{inst}\in\text{instruments}\bigr]
$$

- **Module:** [`plumax.coupled`](xref:api#plumax.coupled) — [`CoupledForward`](xref:api#plumax.coupled.CoupledForward), [`Instrument`](xref:api#plumax.coupled.Instrument), [`fuse_observations`](xref:api#plumax.coupled.fuse_observations), [`RadianceObservationOperator`](xref:api#plumax.coupled.RadianceObservationOperator).
- **Produces:** a *fused* per-event posterior $p(Q)$ — same payload contract as Stage 2, so Stage 4 doesn't care whether one instrument or five produced it.

---

(e2e-stage4-population)=
## Stage 4 — Population: events → mark distribution + intensity

**Question:** across *many* detected events (and the ones we missed), what is the emission-size distribution $f(Q)$ and the event rate $\lambda(t)$?

[Tier V](../design/06_tier5_population.md) sits *above* the per-event physics. It treats each per-event posterior as a **soft observation** of an unknown true mark $Q_i$ and fits a thinned marked temporal point process (TMTPP): intensity $\lambda(t)$, mark distribution $f(Q)$, and per-satellite probability-of-detection $P_d(\cdot)$. The three-term log-likelihood makes $\lambda$ and $P_d$ jointly identifiable:

$$
\log L \;=\; \sum_{i\in\mathcal D}\log\!\int P_d(Q)\,L_i(Q)\,f(Q)\,\mathrm dQ
\;+\; \sum_{i\in\mathcal D}\log\lambda(t_i)
\;-\; \int_0^T \lambda(t)\!\left[\int P_d(Q)\,f(Q)\,\mathrm dQ\right]\mathrm dt
$$

:::{important} Importance correction is mandatory
The per-event posterior from Stage 2/3 already absorbs $\pi_\text{per-event}$. Feeding its samples directly into the population fit double-counts that prior and biases both $f(Q)$ and the total-mass estimate. Every per-event posterior consumed here **must carry its prior log-density** so the importance weight $f/\pi_\text{per-event}$ can re-point it at the population mark distribution.
:::

- **Library:** [`plumax.population`](xref:api#plumax.population) — the intensity zoo ([`plumax.population.intensity`](xref:api#plumax.population.intensity)), the POD zoo ([`plumax.population.pod`](xref:api#plumax.population.pod)), the missing-mass paradox simulator ([`plumax.population.paradox`](xref:api#plumax.population.paradox)) and the NUTS fitter ([`plumax.population.fitting`](xref:api#plumax.population.fitting)); plus the cross-tier catalog ([`event_from_posterior`](xref:api#plumax.population.event_from_posterior)), the V.A lognormal [size-distribution fit](../design/06_tier5a_instantaneous.md) ([`fit_lognormal_size_distribution`](xref:api#plumax.population.fit_lognormal_size_distribution)), and the V.B [point-process core](../design/06_tier5b_point_process.md) ([`fit_poisson_rate`](xref:api#plumax.population.fit_poisson_rate), [`fit_inhomogeneous_intensity`](xref:api#plumax.population.fit_inhomogeneous_intensity)).
- **Theory & notebooks:** [TMTPP theory](population/01_mttpp_theory.md) · [missing-mass paradox](../notebooks/population/01_missing_mass_paradox.ipynb) · [stationary NUTS fit](../notebooks/population/04_stationary_numpyro_mcmc.ipynb).
- **Produces:** posteriors over $\lambda(t)$, $f(Q)$, $P_d$ — and, via the missing-mass correction, a POD-corrected **total emitted mass** ([Tier V.D](../design/06_tier5d_total_emission.md)).

---

(e2e-stage5-persistency)=
## Stage 5 — Persistency: intensity → operational forecast

**Question:** given the inverted $\lambda(t)$, when will this source next emit, and should we send a crew (or re-task a high-resolution satellite) now?

[Tier V.C](../design/06_tier5c_persistency.md) is the **operational layer** — what an LDAR crew or a satellite-tasking dispatcher consumes. It turns the intensity posterior into four metrics, each propagating full posterior uncertainty (no point estimates):

```{list-table} The four persistency metrics.
:label: tbl-e2e-persistency-metrics
:header-rows: 1

* - Metric
  - Formula (inhomogeneous)
  - Operational use
* - Expected wait time $\mathbb E[\Delta t\mid t_0]$
  - $\int_{t_0}^{\infty}\exp\!\big(-\!\int_{t_0}^{t}\lambda(u)\,\mathrm du\big)\,\mathrm dt$
  - Dispatch timing — arrive in a high-$\lambda$ window
* - Occurrence prob. $\mathbb P(N(t_1,t_2)\ge1)$
  - $1-\exp\!\big(-\!\int_{t_1}^{t_2}\lambda(t)\,\mathrm dt\big)$
  - "Wrench-turning" probability for a maintenance window
* - Conditional intensity $\lambda(t\mid t_\text{prev})$
  - $\mu+\alpha\,e^{-\beta(t-t_\text{prev})}$ (Hawkes)
  - Re-task GHGSat after a TROPOMI alert (clustering)
* - Cumulative count $\mathbb E[N(0,T)]$
  - $\Lambda(T)=\int_0^T\lambda(t)\,\mathrm dt$
  - Annual reporting / regulatory compliance
```

- **API (planned, [#112](https://github.com/jejjohnson/plumax/issues/112)):** a `plumax.population.persistency` module — `expected_wait_time`, `occurrence_probability`, `cumulative_count`, `next_event_quantile`, each taking a posterior sample of the intensity parameters and returning posterior samples of the metric. It does not exist yet.
- **Theory:** [persistency](population/03_persistency.md).
- **Produces:** the dispatch / tasking decision — closing the loop back to Stage 1 (a re-task triggers a new overpass).

---

(e2e-handoff-contract)=
## The hand-off contract

The pipeline is robust because each stage's output is a *fixed payload* regardless of which model produced it. This is the contract that lets you climb the complexity axis stage-by-stage:

```{list-table} Stage hand-offs — what each stage consumes and produces.
:label: tbl-e2e-handoff
:header-rows: 1

* - Stage
  - Consumes
  - Produces
  - Complexity-axis options
* - 1 Retrieval
  - L1/L2 radiance
  - $\Delta\mathrm{XCH_4}(x,y)$ + $\mathbf R_\text{retr}$
  - matched filter → optimal estimation → neural RTM
* - 2 Inversion
  - $\Delta\mathrm{XCH_4}$ field
  - per-event $p(Q)$ + $\pi_\text{per-event}$
  - Tier I plume → Tier II footprint → Tier III 4D-Var
* - 3 Fusion *(opt.)*
  - per-instrument $\{p(Q)\}$
  - fused $p(Q,\text{bias})$
  - single-instrument → multi-instrument list
* - 4 Population
  - $\{p(Q_i), \pi_i, t_i\}$
  - $\lambda(t),\,f(Q),\,P_d$, total mass
  - point estimate → importance-corrected full-sample
* - 5 Persistency
  - $\lambda(t)$ posterior
  - wait time, $P(\text{occur})$, dispatch
  - Poisson → Hawkes / Cox clustering
```

:::{important} The required field that is easy to forget
The single most common cross-tier bug is dropping $\pi_\text{per-event}$ on the floor between Stage 2 and Stage 4. Without the per-event prior log-density, the population fit (Stage 4) is biased — see the [Tier V importance-correction note](../design/06_tier5_population.md#tier5-mark-contribution).
:::

---

(e2e-reproduce)=
## How to reproduce a thin end-to-end slice

A minimal Tier I slice you can run today from the existing notebooks:

1. **Retrieve** a synthetic plume scene → column enhancement: [matched-filter retrieval](../notebooks/radtran/05_matched_filter_retrieval.ipynb).
2. **Invert** the column to an emission-rate posterior: [Gaussian plume emission-rate inference](../notebooks/gauss_plume/02_plume_emission_rate_estimation.ipynb).
3. **Aggregate** many such posteriors into a point-process fit: [stationary NUTS fit](../notebooks/population/04_stationary_numpyro_mcmc.ipynb).
4. **Forecast** wait times / occurrence from the fitted intensity: the closed-form metrics in the [persistency theory page](population/03_persistency.md) (the `persistency` module is planned, [#112](https://github.com/jejjohnson/plumax/issues/112)).

Stage 3 (multi-instrument fusion) already ships in [`plumax.coupled`](xref:api#plumax.coupled) but has no notebook yet; the importance-corrected Stage 4 path is still open ([#108](https://github.com/jejjohnson/plumax/issues/108)). Both are tracked in [Tier IV](../design/05_tier4_coupled.md) and [Tier V](../design/06_tier5_population.md); this page will gain a fully-executed companion notebook once they land.

---

(e2e-see-also)=
## See also

- [Roadmap — the two-axis modeling cycle](../design/roadmap.md#cycle-overview)
- [Tier I — Gaussian family](../design/01_tier1_gaussian.md) · [Tier III — Eulerian FV](../design/03_tier3_eulerian.md) · [RTM stack](../design/04_rtm_stack.md) · [Tier IV — Coupled E2E](../design/05_tier4_coupled.md)
- [Tier V — Population](../design/06_tier5_population.md) · [V.A instantaneous](../design/06_tier5a_instantaneous.md) · [V.B point process](../design/06_tier5b_point_process.md) · [V.C persistency](../design/06_tier5c_persistency.md) · [V.D total emission](../design/06_tier5d_total_emission.md)
