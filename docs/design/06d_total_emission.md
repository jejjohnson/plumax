# Tier V.D — Total emission estimation

**Question:** Given a population of detected plumes (and the ones we missed), what is the true total emitted mass over a region and time window?

This is the **inventory-grade output** of `plumax` — the number that gets reported into national greenhouse-gas inventories, climate models, and policy dashboards. It also requires the most care, because the satellite catalog you start from is **systematically biased**: detection thinning means the very things you can't see (small, frequent leaks) are exactly the things that matter for total mass.

---

## The missing-mass paradox {#vd-paradox}

The full Monte Carlo proof is in [`methane_pod/notebooks/03_missing_mass_paradox`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/notebooks/03_missing_mass_paradox.ipynb). The result, in one sentence:

!!! important "The paradox"
    A POD-thinned plume catalog simultaneously **overestimates the average emission rate** (because it oversamples big leaks) and **underestimates the total emitted mass** (because it misses many small leaks).

These two biases pull in opposite directions, but they don't cancel — averaging the wrong thing over the wrong sample size gives you the wrong total. The corrected estimator has to model the thinning explicitly.

---

## The corrected total-mass estimator {#vd-corrected-estimator}

Given a TMTPP fit (Tier V.B) with posterior $(\lambda, f, P_d)$:

$$
M_\text{total}(T) \;=\; \mathbb{E}[N_\text{true}(T)] \cdot \mathbb{E}[Q]
\;=\; \left(\int_{0}^{T} \lambda(t)\, \mathrm{d}t\right) \cdot \left(\int Q\, f(Q)\, \mathrm{d}Q\right)
$$

This is the **un-thinned** total — what would be emitted regardless of detection. Compare to the naive estimator:

$$
M_\text{naive}(T) \;=\; \sum_{i \in \mathcal{D}} Q_i \;\approx\; \mathbb{E}[N_\text{detected}(T)] \cdot \mathbb{E}[Q \mid \text{detected}]
$$

with

$$
\mathbb{E}[N_\text{detected}(T)] \;=\; \int_{0}^{T} \lambda(t) \int P_d(Q)\, f(Q)\, \mathrm{d}Q\, \mathrm{d}t.
$$

$M_\text{naive}$ is biased low because:

- $\mathbb{E}[N_\text{detected}] < \mathbb{E}[N_\text{true}]$ (some events missed).
- $\mathbb{E}[Q \mid \text{detected}] > \mathbb{E}[Q]$ (detected events are systematically bigger — heavy tail of $f$).

The two errors *compound* rather than cancel: the regional total is undercounted, and the per-event mean is inflated. Inverting the POD model is the **only** way to recover an unbiased total.

---

## Posterior over total mass {#vd-posterior}

With NUTS samples $(\lambda^{(s)}, f^{(s)}, P_d^{(s)})$, the posterior over $M_\text{total}(T)$ is:

$$
M_\text{total}^{(s)}(T) \;=\; \left(\int_{0}^{T} \lambda^{(s)}(t)\, \mathrm{d}t\right) \cdot \left(\int Q\, f^{(s)}(Q)\, \mathrm{d}Q\right)
$$

Reported as posterior median + 95% credible interval. Both integrals are tractable for the standard intensity / mark choices (closed-form for constant $\lambda$ + lognormal $f$; quadrature otherwise).

---

## Validation strategy {#vd-validation}

!!! important "Most important validation in the tier"
    Without these, the estimator is just a number.

- **MC ground truth (bias direction).** Reproduce the qualitative result of the paradox notebook: simulate a known $(\lambda^{*}, f^{*}, P_d^{*})$, compute $M_\text{true}$ exactly, and check that the corrected estimator recovers $M_\text{true}$ while $M_\text{naive}$ is biased low.
- **MC ground truth (calibration).** Across 1000 replicates of the previous test, the 95% credible interval on $M_\text{total}$ should contain $M_\text{true}$ ~95% of the time.
- **Per-satellite sensitivity.** Same population, two different $P_d$ (e.g. GHGSat-floor [ghgsat] vs. TROPOMI-floor [s5p_tropomi]) → corrected estimator should give the same $M_\text{total}$ posterior. The naive estimator gives wildly different $M_\text{naive}$. This is the test that *proves* the correction is doing its job.
- **Real-data benchmark.** Once [`07_pod_fitting_mcmc`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/notebooks/07_pod_fitting_mcmc.md) lands with IMEO + Tanager data, compare the corrected total for a well-studied basin (Permian) to published bottom-up inventories ([epa_ghgi,scarpelli2020sectoral], GHGRP) and to top-down inverse-modelling estimates ([maasakkers2023ghgi,jacob2022quantifying], Sherwin et al.). They will disagree; the question is whether the corrected estimator is *closer* to the top-down number than the naive one.

---

## Module layout {#vd-modules}

*Tier V.D module layout — concern, target module, status.*

| Concern | Module | Status |
| --- | --- | --- |
| Missing-mass MC simulator | [`methane_pod.paradox`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/src/methane_pod/paradox.py) | ✓ (NumPy) |
| Posterior fit | [`methane_pod.fitting`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/src/methane_pod/fitting.py) | ✓ (synthetic); 🚧 (real data) |
| $M_\text{total}$ estimator + uncertainty | `plume_simulation.population.totals` | ☐ |
| Per-satellite calibration loader | `plume_simulation.population.satellite_pod` | ☐ |
| Multi-satellite fusion | `plume_simulation.population.fusion` | ☐ |

---

## Multi-satellite fusion (Tier V.D extension) {#vd-multi-satellite-fusion}

For a region observed by $K$ satellites, each with its own POD, the unified detection probability is:

$$
P_d^{\cup}(Q) \;=\; 1 - \prod_{k=1}^{K} \bigl(1 - P_d^{k}(Q)\bigr)
$$

This is the "any satellite saw it" probability. Folds into the TMTPP likelihood as a single replacement of $P_d$ with $P_d^{\cup}$. Adds one strong assumption: detections by different satellites are conditionally independent given the leak size — defensible at the population level, possibly violated for clustered super-emitters.

!!! attention "Open: union vs. categorical mark"
    Whether to model **which** satellite detected each event (categorical mark) or just the union. The first gives more information per event but doubles the number of POD parameters.

---

## Open questions {#vd-open-questions}

!!! attention "Mass vs. mass-rate"
    $M_\text{total}(T)$ is mass. Most published inventories report mass-rate (Tg / yr). The conversion is $M_\text{total} / T$, but $T$ for a satellite catalog is fuzzy — what's the effective observing time when satellites overpass intermittently? Document the convention.

!!! attention "Spatial aggregation"
    Currently temporal-only. Aggregating $M_\text{total}$ over a basin requires either a spatial point process (cleaner) or stratifying the sources by facility class and combining (operational shortcut). v1: stratification; v2: spatial CGS / Cox process.

!!! attention "POD parameter sources"
    Per-satellite POD parameters can come from (a) fits in [`methane_pod`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/) on a held-out catalog, (b) published values from [varon2018quantifying] / Cusworth et al., or (c) joint inference with the population. Each has trade-offs around identifiability.

!!! attention "Reporting cadence"
    Inventories are annual; satellites are daily-ish. How do we smooth the $M_\text{total}$ time series? Rolling 30-day window? Bayesian time-series prior on $\lambda(t)$?
