# Tier V.C: Persistency

**Question:** Given the inverted intensity $\lambda(t)$ from Tier V.B, when will the next emission event happen, and what's the probability of an event during a specified window?

This is the operational layer, which is what an LDAR (Leak Detection and Repair) crew or a satellite-tasking dispatcher actually consumes. The full derivations of each metric live in the [persistency theory page](../theory/population/03_persistency.md); this page summarises the metrics and how they slot into the `plumax` API.

---

(vc-four-metrics)=
## The four operational metrics

(vc-wait-time)=
### 1. Expected wait time $\mathbb{E}[\Delta t \mid t_0]$

How long after time $t_0$ until the next event?

$$
\mathbb{E}[\Delta t] \;=\; \frac{1}{\lambda_0} \qquad \text{(homogeneous Poisson; memoryless)}
$$

$$
\mathbb{E}[\Delta t \mid t_0] \;=\; \int_{t_0}^{\infty} \exp\!\left(-\int_{t_0}^{t} \lambda(u)\, \mathrm{d}u\right) \mathrm{d}t \qquad \text{(inhomogeneous; depends on starting clock)}
$$

For a diurnal source, vastly different at noon vs. midnight.

**Operational use.** Dispatch decisions: arrive during a high-$\lambda$ window and the next event is imminent (worth waiting); arrive during a low-$\lambda$ window and you'd waste hours. Drives MARS-style dispatch suppression during dormant cycles.

(vc-occurrence)=
### 2. Probability of occurrence $\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr)$

What's the chance of at least one event in $[t_1, t_2]$?

$$
\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr) \;=\; 1 - \exp\!\bigl(-\lambda_0\, (t_2 - t_1)\bigr) \qquad \text{(homogeneous)}
$$

$$
\mathbb{P}\!\bigl(N(t_1, t_2) \geq 1\bigr) \;=\; 1 - \exp\!\left(-\int_{t_1}^{t_2} \lambda(t)\, \mathrm{d}t\right) \qquad \text{(inhomogeneous)}
$$

**Operational use.** This is the "wrench-turning" probability: if a maintenance window is 4 hours, what's the chance the leak shows itself during that window? It drives whether to schedule the visit.

(vc-conditional-intensity)=
### 3. Conditional intensity given prior detection $\lambda(t \mid t_\text{prev})$

For a source with a known recent detection at $t_\text{prev}$, what's the posterior intensity going forward?

For Poisson processes (no memory) the intensity is unchanged. For Hawkes / self-exciting processes it is bumped:

$$
\lambda(t \mid t_\text{prev}) \;=\; \mu + \alpha\, \exp\!\bigl(-\beta(t - t_\text{prev})\bigr)
$$

This captures the empirical observation that super-emitters "cluster".

**Operational use.** Prioritisation: a source with a recent detection is *more* likely to repeat-emit in the next 24 h. Re-task a high-resolution satellite (GHGSat [ghgsat], Carbon Mapper [carbon_mapper]) on top of a TROPOMI alert [s5p_tropomi].

(vc-cumulative-count)=
### 4. Cumulative event count $\mathbb{E}[N(0, T)]$ and credible bounds

Expected number of events in $[0, T]$, with credible interval from the posterior on $\lambda$.

$$
\mathbb{E}[N(0, T)] \;=\; \Lambda(T) \;=\; \int_{0}^{T} \lambda(t)\, \mathrm{d}t
$$

(Homogeneous: $\lambda_0 \cdot T$.)

**Operational use.** Annual reporting, regulatory compliance. "How many emission events should we expect this year at this facility class, with 95% credible interval?"

---

(vc-api)=
## API shape

A thin wrapper around `plumax.population.intensity`:

```python
import jax
from plumax.population.persistency import (
    expected_count,
    expected_wait_time,
    occurrence_probability,
    posterior_metric,
)
from plumax.population.point_process import PoissonRatePosterior

# Shapes: N_E = 1,000 posterior draws of the event rate λ

# today (plumax)
# λ | n ~ Gamma(α₀ + n, β₀ + A T), with 200 events over 100 days
posterior = PoissonRatePosterior(alpha=201.0, beta=100.0, n_events=200, exposure=100.0)
key = jax.random.key(0)
key_wait, key_occurrence, key_count = jax.random.split(key, 3)

# today (plumax)
# 𝔼[Δt | t₀] = ∫₀^∞ exp(−Λ(t₀, t₀ + s)) ds, from t₀ = 18:00 [days]
wait = posterior_metric(  # samples (N_E,)
    expected_wait_time, posterior, 0.75, key=key_wait
)

# today (plumax)
# P(N(t, t + h) ≥ 1) = 1 − exp(−Λ(t, t + h)), from 08:00 for h = 4 hours
occurrence = posterior_metric(  # samples (N_E,)
    occurrence_probability, posterior, 8 / 24, 4 / 24, key=key_occurrence
)

# today (plumax)
# 𝔼[N(t, t + h)] = Λ(t, t + h), over a year
count = posterior_metric(  # samples (N_E,)
    expected_count, posterior, 0.0, 365.0, key=key_count
)
```

Each metric takes an intensity (a registry name or any of the 13 `equinox` kernels from `plumax.population.intensity`), a start time and, for a window, a horizon, all in days. `posterior_metric` then maps a metric over the draws of a fitted posterior (Gamma-Poisson, log-linear or the full TMTPP) and returns the samples with their quantiles, so the UQ propagates in full and there are no point estimates. The conditional intensity of section 3 needs a Hawkes kernel, which is still open ([#115](https://github.com/jejjohnson/plumax/issues/115)).

---

(vc-modules)=
## Module layout

*Tier V.C module layout: concern, target module and status.*

| Concern | Module | Status |
| --- | --- | --- |
| Intensity functions | `plumax.population.intensity` | ✓ in tree |
| Wait-time / occurrence / cumulative metrics | `plumax.population.persistency` | ✓ [#112](https://github.com/jejjohnson/plumax/issues/112) |
| Posterior-aware metric wrappers | same module (`posterior_metric`) | ✓ [#112](https://github.com/jejjohnson/plumax/issues/112) |
| Operational dashboard / report templates | out of scope for `plumax`; lives in `plumax-deploy` (future) | - |

The integral over $\lambda(t)$ in the wait-time formula is closed-form for a few intensity choices (constant, exponential decay). Otherwise it is an integral to infinity, which integrix's [An integral to infinity](https://jejjohnson.github.io/integrix/expectations/#an-integral-to-infinity) treats with tanh-sinh quadrature and checks against $1/\lambda_0$; plumax's `expected_wait_time` uses its own log-spaced survival grid today.

---

(vc-validation)=
## Validation strategy

- **Homogeneous limit.** For constant $\lambda$, all four metrics have closed-form formulas; the implementation should match to machine precision.
- **MC self-consistency.** Sample $n$ event times from a known $\lambda(t)$ via thinning, compute the empirical wait time / occurrence frequency, compare to the closed-form metric. Tests both the metric implementation and the simulator.
- **Posterior coverage.** For a synthetic source with known $\lambda_\text{true}(t)$, the 95% credible interval on $\mathbb{E}[\Delta t]$ should contain the truth ~95% of the time across replicates.
- **Diurnal sanity.** A solar-heated tank with peak $\lambda$ at 14:00 should have $\mathbb{E}[\Delta t \mid 14{:}00] \ll \mathbb{E}[\Delta t \mid 02{:}00]$. This is a numerical sanity check rather than a formal test, but it catches sign errors.

---

(vc-open-questions)=
## Open questions

:::{attention} What's 'the' intensity?
A point estimate (posterior mean) or the full posterior over $\lambda$ parameters? Operational dashboards may want the former; researchers want the latter. The API returns posterior samples by default; downstream summarisation is the caller's choice.
:::

:::{attention} Hawkes vs Poisson default
Hawkes is more physically faithful for super-emitters but doubles the parameter count and complicates the wait-time integral. Default to Poisson with a Hawkes opt-in?
:::

:::{attention} Cross-source independence
Persistency metrics are per-source. Aggregating up to "expected events across a basin in 24h" requires the spatial / population point process from Tier V.B's open questions. Out of scope for v1.
:::

:::{attention} Action thresholds
Wait-time and occurrence probability become operational only with a threshold (e.g. "dispatch if $\mathbb{P}(\text{occur}) > 0.7$"). Where do thresholds live? Probably in the dashboard, not in `plumax` core.
:::
