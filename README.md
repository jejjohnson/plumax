<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img alt="plumax" src="docs/assets/logo-light.svg" width="480">
  </picture>
</p>

# plumax: plumes of anything, from pixels to sources

[![Tests](https://github.com/jejjohnson/plumax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/plumax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/plumax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/plumax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Where is the plume, how much is coming out, where does it come from, how often does it happen.**
> Differentiable transport models (Gaussian plume and puff, Lagrangian
> particles, Eulerian finite volumes) and the Bayesian and variational
> machinery to invert them, for any passive tracer: methane, CO₂, NO₂,
> SO₂, smoke or heat. Everything is JAX: `jit`, `vmap` and `grad` through
> the physics.

<p align="center"><img src="docs/assets/diagrams/plumax-stack.png" alt="Where plumax sits: radtrax turns a radiance scene into a column enhancement map; plumax segments it into a mask (PlumeMask, PlumeContours, PlumeFootprint), quantifies an emission rate (IMEEstimate, CrossSectionalFlux, infer_emission_rate), attributes it to a source (compute_footprint, linear_gaussian_inversion, solve_4dvar, fuse_observations) and turns every posterior into a catalogue event for rates, detection probability, persistency metrics and total emission (EmissionCatalog, fit_poisson_rate, population.pod, population.persistency, total_emission), all shipping today; underneath, the transport tiers gauss_plume, gauss_puff, lagrangian, les_fvm and coupled" width="100%"></p>

---

## Four questions

<table align="center">
<tr>
<td align="center" width="25%">
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/glyphs/segment-dark.svg"><img src="docs/assets/glyphs/segment.svg" width="64" alt="segment"></picture><br>
<b>segment</b><br><sub>Where is the plume?<br>mask · contours · footprint</sub><br><sub>✅ masks · contours · footprints</sub>
</td>
<td align="center" width="25%">
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/glyphs/quantify-dark.svg"><img src="docs/assets/glyphs/quantify.svg" width="64" alt="quantify"></picture><br>
<b>quantify</b><br><sub>How much, in kg s⁻¹?<br>model inversion · IME · flux</sub><br><sub>✅ inversion · IME · flux</sub>
</td>
<td align="center" width="25%">
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/glyphs/attribute-dark.svg"><img src="docs/assets/glyphs/attribute.svg" width="64" alt="attribute"></picture><br>
<b>attribute</b><br><sub>Which source?<br>footprints · 4D-Var · fusion</sub><br><sub>✅ known source · multi-source</sub>
</td>
<td align="center" width="25%">
<picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/glyphs/persist-dark.svg"><img src="docs/assets/glyphs/persist.svg" width="64" alt="persist"></picture><br>
<b>persist</b><br><sub>How often, how long?<br>event rates · POD · catalogues</sub><br><sub>✅ rates · persistency metrics</sub>
</td>
</tr>
</table>

✅ ships today · 🚧 designed, not shipped (the [roadmap](docs/design/roadmap.md) has the status of every tier, including the predictors and emulators still open).

---

## Where plumax sits

Turning radiance into a column of gas is radiative transfer, which lives in [**radtrax**](https://github.com/jejjohnson/radtrax), so plumax starts from the column (or any concentration field) and reasons about the *plume*: its shape, its transport and its source. The figure above traces one scene through every step.

The dependency only goes one way, as plumax imports radtrax (the `coupled` tier maps a simulated plume to radiance through it) and radtrax never imports plumax. The split, and why the package graph stays acyclic, is set out in GeoStack's
[package boundaries](https://github.com/jejjohnson/geostack/blob/main/docs/geomodels/package_boundaries.md).

| You want to… | Use |
|---|---|
| model a spectrum, build cross-section tables, run a matched filter or a 3D-Var column retrieval | radtrax |
| simulate where a released tracer goes | plumax (`gauss_plume`, `gauss_puff`, `lagrangian`, `les_fvm`) |
| estimate an emission rate or its uncertainty | plumax |
| mask, contour and measure a plume in an image | plumax `[geo]` (`plumax.geo`) |
| count events, model detection probability, extrapolate to a population | plumax (`population`) |

## What it stands on

plumax sits in GeoStack's GeoModels layer beside radtrax. It reaches down to two GeoNumerics packages for its solvers (finitevolX for the Eulerian grid, gaussx for every posterior covariance) and to pipekit for the `Operator` interface, and nothing below imports it back. Inference sits to one side. Every tier exposes the same `forward(params, met) → observations`, so we write the samplers and assimilation schemes once against it: NumPyro does the sampling, vardax (variational DA, the `assim` extra) runs the Tier II optimal interpolation and the Tier III 3D-Var over plumax's source-receptor maps, and filterax (ensemble Kalman methods) is the planned home for EKI.

<p align="center"><img src="docs/assets/diagrams/plumax-deps.png" alt="plumax and its dependencies by GeoStack layer: an Inference column with numpyro (today, behind the inference extra), vardax and argminax (today, behind the assim extra) and filterax (planned); in GeoModels, plumax depends on radtrax through coupled and adapters, and radtrax never imports plumax; in GeoNumerics, les_fvm uses finitevolX (with spectraldiffx underneath) and lagrangian and les_fvm use gaussx; in GeoStack, operators use pipekit, adapters satisfy pipekit-cycle in tests only, and geotoolz-ops and georeader back the geo extra; in GeoML, xtremax backs the population extra and geonnax arrives through radtrax; jax, equinox, diffrax, optimistix, lineax, xarray, netcdf4 and scipy are the third-party foundations" width="100%"></p>

The full stack's graph, and the rules that keep it acyclic, live in GeoStack's
[package boundaries](https://github.com/jejjohnson/geostack/blob/main/docs/geomodels/package_boundaries.md)
page.

---

## One plume, end to end

The running example is a single stack 20 m tall releasing a passive tracer at
$`Q = 0.5`$ kg s⁻¹ into a 4 m s⁻¹ westerly, under Pasquill stability class C.

### 1. Forward and back: simulate it, then recover Q

The steady Gaussian plume gives the concentration $`C`$ [kg m⁻³] at a point
$`(x, y, z)`$, with $`x`$ downwind, $`y`$ crosswind, $`z`$ up, source height $`h`$,
wind speed $`u`$, and the Briggs widths $`\sigma_y(x)`$, $`\sigma_z(x)`$ of the
stability class; the second exponential reflects the plume off the ground:

```math
C(x, y, z) = \frac{Q}{2 \pi u \sigma_y \sigma_z}
\exp\left(-\frac{y^2}{2 \sigma_y^2}\right)
\left[\exp\left(-\frac{(z-h)^2}{2 \sigma_z^2}\right)
+ \exp\left(-\frac{(z+h)^2}{2 \sigma_z^2}\right)\right]
```

$`N_{\Omega_y} = 200`$ sensors at 5 m record $`o_k = C(x_k, y_k, z_k) + b + \varepsilon_k`$, with a background $`b`$ and noise $`\varepsilon_k \sim \mathcal{N}(0, s^2)`$. NUTS samples the posterior over $`(Q, b)`$, and $`N_{\mathcal{E}}`$ is the number of draws.

```python
import jax
import jax.numpy as jnp
from plumax.gauss_plume import infer_emission_rate, simulate_plume

# Shapes: N_x = 100, N_y = 80, N_z = 13 grid points; N_Ωy = 200 sensors;
#         N_E = 500 posterior draws

# today (plumax)
# C(x, y, z) on the grid, with Q = 0.5 kg/s
truth = simulate_plume(  # → Dataset (N_x, N_y, N_z)
    emission_rate=0.5,
    source_location=(0.0, 0.0, 20.0),
    wind_speed=4.0,
    wind_direction=270.0,
    stability_class="C",
    domain_x=(0.0, 1000.0, 100),
    domain_y=(-200.0, 200.0, 80),
    domain_z=(0.0, 60.0, 13),
)

# C(x, y, z = 5 m), the second level of the grid
surface = truth.concentration.isel(z=1)  # (N_x, N_y, N_z) → (N_x, N_y)
# every 40th column of the flattened surface
sensors = surface.stack(k=("x", "y"))[::40]  # (N_x, N_y) → (N_Ωy,)
sensor_xyz = (
    jnp.asarray(sensors.x.values),  # (N_Ωy,)
    jnp.asarray(sensors.y.values),  # (N_Ωy,)
    jnp.full(sensors.shape, 5.0),  # (N_Ωy,)
)

key = jax.random.key(0)
# εₖ ~ 𝒩(0, s²), with s = 2 × 10⁻⁷ kg/m³
obs_noise = 2e-7 * jax.random.normal(key, sensors.shape)  # → (N_Ωy,)
# oₖ = C(xₖ, yₖ, zₖ) + εₖ
obs = jnp.asarray(sensors.values) + obs_noise  # (N_Ωy,) → (N_Ωy,)

# today (plumax)
# p(Q, b | o) ∝ 𝒩(o; C_Q + b, s² I) p(Q) p(b)
posterior = infer_emission_rate(  # → {Q: (N_E,), b: (N_E,)}
    obs,
    sensor_xyz,
    source_location=(0.0, 0.0, 20.0),
    wind_speed=4.0,
    wind_direction=270.0,
    stability_class="C",
    prior_mean=0.3,
    prior_std=0.3,
    obs_noise_std=2e-7,
    num_warmup=300,
    num_samples=500,
)

# Q⁽ᵉ⁾ ~ p(Q | o)
emission_rate = posterior["emission_rate"]  # (N_E,)
# 𝔼[Q | o] and √Var[Q | o]
posterior_mean = emission_rate.mean()  # (N_E,) → ()
posterior_std = emission_rate.std()  # (N_E,) → ()
# Q = 0.5000 ± 0.0004 kg/s
print(f"Q = {posterior_mean:.4f} ± {posterior_std:.4f} kg/s")
```

Swap `gauss_plume` for `gauss_puff` (time-varying wind), `lagrangian`
(particles and backward footprints) or `les_fvm` (3-D finite volumes with a
4D-Var adjoint by autodiff) and the question stays the same; only the
transport changes.

### 2. From an image: mask, then measure

A satellite does not see $`C`$; it sees the column enhancement
$`\Delta\Omega(x, y)`$ [kg m⁻²] that radtrax retrieves. On an image, plumax
segments the plume into a mask $`M`$ of pixels with area $`A_i`$, sums the
integrated mass enhancement $`\mathrm{IME} = \sum_{i \in M} \Delta\Omega_i A_i`$,
and divides by the residence time $`L / U_{\mathrm{eff}}`$, with plume length $`L`$ and effective wind $`U_{\mathrm{eff}}`$. The plume operators came over from geotoolz into `plumax.geo` (behind the `[geo]` extra), whereas the radtrax call is sketched (the `obs_model` that sets the target, and so the units of the column, is left out). `scene` is a radiance GeoTensor in a projected CRS and `u_eff` the effective wind $`U_{\mathrm{eff}}`$ [m s⁻¹] at the source:

```python
import plumax.geo as pgeo
import radtrax.geo as rgeo

# Shapes: N_Sy = 285 channels; N_y × N_x = 512 × 512 pixels

# today* (radtrax)
# ΔΩ = tᵀ Σ⁻¹ (r − μ) / (tᵀ Σ⁻¹ t)
matched_filter = rgeo.ColumnEnhancement(gas="CO2", sensor="EMIT")
column = matched_filter(scene)  # (N_Sy, N_y, N_x) → (N_y, N_x)

# today (plumax)
# M = {i : ΔΩᵢ > τ}, with τ from Otsu and components of at least 50 pixels
mask = pgeo.PlumeMask(threshold="otsu")(column)  # (N_y, N_x) → (N_y, N_x) bool

# today (plumax)
# IME = ∑_{i∈M} ΔΩᵢ Aᵢ and Q = U_eff IME / L
ime = pgeo.IMEEstimate(wind_speed=u_eff, length_method="convex_hull")
estimate = ime(column, mask)  # (N_y, N_x) → dict
emission_rate = estimate["emission_rate_kg_s"]  # → ()
```

### 3. Over time: how often does it fire?

One plume is one event. A site seen on many overpasses gives an event
catalogue, and a Gamma-Poisson update turns $`n`$ events over an exposure
$`A \cdot T`$ (area times duration) into a posterior on the event rate
$`\lambda \sim \mathrm{Gamma}(\alpha_0 + n, \beta_0 + A T)`$.

```python
from plumax.population import EmissionCatalog, EmissionEvent, fit_poisson_rate

# Shapes: n = 5 events

# (Q [kg/s], day of detection)
detections = [(0.62, 3.0), (0.48, 17.0), (0.55, 41.0), (0.71, 58.0), (0.39, 90.0)]

# today (plumax)
events = tuple(
    EmissionEvent(
        emission_rate=rate, emission_std=0.1, x=0.0, y=0.0, time=day, tier="I"
    )
    for rate, day in detections
)
# today (plumax)
catalog = EmissionCatalog(events=events)

# today (plumax)
# λ | n ~ Gamma(α₀ + n, β₀ + A T)
rate_posterior = fit_poisson_rate(catalog, area=1.0, duration=120.0)
# 0.042 events/day
print(f"{rate_posterior.rate_mean:.3f} events/day")
```

The same catalogue feeds the probability-of-detection models
(`population.pod`), the size-distribution fit and the missing-mass
simulator (`population.paradox`).

---

## The model ladder

Each tier keeps one forward interface, `forward(params, met) → observations`,
so an emulator or an amortised predictor slots in where the physics was.

| Tier | Transport | Sub-package | Status |
|---|---|---|---|
| 0 | meteorology · averaging kernels | `met` | ✅ WRF and ERA5 loaders · L2 ingest · priors |
| I | Gaussian plume and puff | `gauss_plume`, `gauss_puff` | ✅ forward · MAP / MCMC |
| II | Lagrangian particles, backward footprints | `lagrangian` | ✅ forward · Gaussian / lognormal inversion · footprint emulator |
| III | Eulerian finite volumes (Arakawa C-grid) | `les_fvm` | ✅ forward · 4D-Var · Laplace posterior |
| IV | transport + radiative transfer | `coupled` (+ radtrax) | ✅ linear fusion · nonlinear inversion · Tier II / III backends |
| V | event populations and forecasting | `population` | ✅ rates · sizes · POD · TMTPP likelihood · persistency · totals · 🚧 Hawkes / LGCP |

Only Tier II has an emulator (the footprint emulator), and no tier has an amortised predictor yet. The [design pages](docs/design/) set out the plan, the maths and the open questions for each.

---

## Not just methane

The transport and the inference treat a passive tracer: an emission rate in
kg s⁻¹ goes in, a concentration in kg m⁻³ comes out. Three places still
assume methane and are the work left to make that true everywhere:

- `coupled.rtm` converts column mass to ΔVMR with `CH4_MOLAR_MASS_KG_PER_MOL`.
- `les_fvm` labels its outputs and docstrings "methane".
- The `population` intensity kernels are named for methane sectors (coal
  mines, landfills, feedlots); the maths is sector-agnostic.

---

## Install

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/plumax.git
cd plumax
make install      # uv sync --all-groups + pre-commit hooks
```

- The core install pulls JAX, xarray, diffrax/Equinox, einx, finitevolX, gaussx, optimistix, pipekit and radtrax.
- finitevolX, gaussx, pipekit and radtrax are not yet on PyPI, and neither are most packages behind the extras; from a checkout they resolve from git via `[tool.uv.sources]`.
- Extras: `inference` (NumPyro), `assim` (vardax and argminax: the Tier II and Tier III source inversions and the Tier IV nonlinear inversion), `population` (NumPyro and xtremax), `geo` (the plume operators on GeoTensors), `emulate` (pipekit-train, for the footprint emulator), `io` (readers) and `notebooks` (matplotlib, IPython): `uv add "plumax[inference,assim]"`.

## Development

| Command | Description |
|---|---|
| `make install` | Install all dependency groups + pre-commit hooks |
| `make test` | Run the fast tests (everything not marked `slow` or `validation`) |
| `make test-slow` | Run only the tests marked `slow` |
| `make test-validation` | Run only the tests marked `validation`: statistical validation and other tests too long for every PR, which CI runs nightly on main |
| `make test-all` | Run everything: fast, slow and validation |
| `make test-cov` | Run the fast and slow tests with the 80 % coverage gate |
| `make lint` / `make format` | Lint / format with ruff (entire repo) |
| `make typecheck` | Type-check `src/plumax` with ty |
| `make docs` / `make docs-serve` | Build (and preview) the docs: mystmd prose + MkDocs API reference |

The pre-commit checklist (tests, `ruff check .`, `ruff format --check .`,
`ty check src/plumax`) must pass before every commit.

The README diagrams are HTML sources in
[`docs/assets/diagrams/`](docs/assets/diagrams) (the stack's shared
stylesheet, one colour per question); edit the `.html` and re-render the PNGs
with `uv run --no-project --with playwright python docs/assets/diagrams/render.py`. See
[`CLAUDE.md`](CLAUDE.md) and [`AGENTS.md`](AGENTS.md) for the full
contributor and agent contract.

<p align="center">
  <img src="docs/assets/favicon.svg" width="40" alt="">
</p>

## License

MIT, see [LICENSE](LICENSE).
