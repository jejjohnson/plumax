# plumax

> **Where is the plume, how much is coming out, where does it come from, how often does it happen.**
> Differentiable transport models (Gaussian plume and puff, Lagrangian
> particles, Eulerian finite volumes) and the Bayesian and variational
> machinery to invert them, for any passive tracer: methane, CO₂, NO₂,
> SO₂, smoke or heat. Everything is JAX: `jit`, `vmap` and `grad` through
> the physics.

:::{figure} assets/diagrams/plumax-stack.png
:label: fig-plumax-stack
:alt: Where plumax sits: radtrax turns a radiance scene into a column enhancement map; plumax segments it into a mask (PlumeMask, proposed), quantifies an emission rate (IMEEstimate proposed; infer_emission_rate today), attributes it to a source (compute_footprint, linear_gaussian_inversion, solve_4dvar, fuse_observations, all today) and turns every posterior into a catalogue event for rates and detection probability (EmissionCatalog, fit_poisson_rate, population.pod, today); underneath, the transport tiers gauss_plume, gauss_puff, lagrangian, les_fvm and coupled.

From a column map to an emission catalogue. Solid cards ship today; dashed
cards are moving in from geotoolz or are designed but not built.
:::

## Four questions

| Question | Verb | Status |
|---|---|---|
| Where is the plume? | **segment**: mask, contours, footprint | 🚧 moving in from geotoolz |
| How much, in kg s⁻¹? | **quantify**: model inversion, IME, cross-sectional flux | ✅ inversion · 🚧 IME / flux |
| Which source? | **attribute**: footprints, 4D-Var, multi-instrument fusion | ✅ known source · 🚧 multi-source |
| How often, how long? | **persist**: event rates, detection probability, catalogues | ✅ rates · 🚧 persistency metrics |

✅ ships today · 🚧 designed, not shipped. The [roadmap](design/roadmap.md)
has the status of every tier.

## Where plumax sits

Turning radiance into a column of gas is radiative transfer, and that lives
in [radtrax](https://github.com/jejjohnson/radtrax). plumax takes the column,
or any concentration field, and reasons about the *plume*: its shape, its
transport and its source. plumax imports radtrax (the `coupled` tier maps a
simulated plume to radiance through it); radtrax never imports plumax.

| You want to… | Use |
|---|---|
| model a spectrum, build cross-section tables, run a matched filter or a 3D-Var column retrieval | radtrax |
| simulate where a released tracer goes | plumax (`gauss_plume`, `gauss_puff`, `lagrangian`, `les_fvm`) |
| estimate an emission rate or its uncertainty | plumax |
| mask, contour and measure a plume in an image | plumax `[geo]`, moving from geotoolz |
| count events, model detection probability, extrapolate to a population | plumax (`population`) |

## What it stands on

plumax sits in GeoStack's GeoModels layer beside radtrax. It reaches down to
finitevolX for the Eulerian grid, to gaussx for every posterior covariance,
and to pipekit for the `Operator` interface. Inference sits to one side:
every tier exposes the same `forward(params, met) → observations`, so the
samplers and assimilation schemes are written once against it. NumPyro does
that for sampling; vardax (variational, the `assim` extra) runs the Tier II
optimal interpolation and the Tier III 3D-Var; filterax (ensemble Kalman) is
planned.

:::{figure} assets/diagrams/plumax-deps.png
:label: fig-plumax-deps
:alt: plumax and its dependencies by GeoStack layer: an Inference column with numpyro (today, behind the inference extra), vardax (today, behind the assim extra) and filterax (planned); in GeoModels, plumax depends on radtrax through coupled and adapters, and radtrax never imports plumax; in GeoNumerics, les_fvm uses finitevolX (with spectraldiffx underneath) and lagrangian and les_fvm use gaussx; in GeoStack, operators use pipekit, adapters satisfy pipekit-cycle in tests only, and geotoolz is a planned geo extra; in GeoML, xtremax is planned for population and geonnax arrives through radtrax; jax, equinox, diffrax, optimistix, lineax, xarray, netcdf4 and scipy are the third-party foundations.

plumax's declared dependencies by GeoStack layer. Each arrow is labelled with
the plumax subpackage that imports the dependency.
:::

## The modeling cycle

plumax is organised around a five-tier **data-driven modeling cycle**. Each
tier shares a fixed forward interface, so emulation and amortized inference
are substitutions rather than rewrites:

1. **Simple model**: a generative story you can simulate from.
2. **Model-based inference**: slow but exact ground truth.
3. **Model emulator**: a fast surrogate (skip it if the model is cheap).
4. **Emulator-based inference**: step 2 in seconds.
5. **Amortized inference**: a predictor that learns the posterior map.
6. **Improve**: upgrade any component; the previous step is its ground truth.

The [Roadmap & Architecture](design/roadmap.md) page has the tier breakdown
(Gaussian → Lagrangian → Eulerian → Coupled → Population).

## Installation

```bash
uv add plumax
```

`gaussx`, `pipekit` and `radtrax` are not yet on PyPI; from a checkout they
resolve from git via `[tool.uv.sources]`. NumPyro (Bayesian inference),
vardax (the Tier II and Tier III source inversions) and the notebook
dependencies are optional extras:

```bash
uv add "plumax[inference,assim]"
```

## Quickstart: simulate a plume, then recover its emission rate

A single stack 20 m tall releases a passive tracer at $Q = 0.5$ kg s⁻¹ into
a 4 m s⁻¹ westerly, under Pasquill stability class C. The steady Gaussian
plume gives the concentration $C$ [kg m⁻³] at a point $(x, y, z)$, with $x$
downwind, $y$ crosswind, $z$ up, source height $h$, wind speed $u$, and the
Briggs widths $\sigma_y(x)$, $\sigma_z(x)$ of the stability class; the
second exponential reflects the plume off the ground:

$$
C(x, y, z) = \frac{Q}{2 \pi u \sigma_y \sigma_z}
\exp\left(-\frac{y^2}{2 \sigma_y^2}\right)
\left[\exp\left(-\frac{(z-h)^2}{2 \sigma_z^2}\right)
+ \exp\left(-\frac{(z+h)^2}{2 \sigma_z^2}\right)\right]
$$

$K$ sensors at 5 m record $o_k = C(x_k, y_k, z_k) + b + \varepsilon_k$, with
a background $b$ and noise $\varepsilon_k \sim \mathcal{N}(0, s^2)$. NUTS
samples the posterior over $(Q, b)$; $S$ is the number of draws.

```python
import jax
import jax.numpy as jnp
from plumax.gauss_plume import infer_emission_rate, simulate_plume

# Shapes: X = 100, Y = 80, Z = 13 grid points; K = 200 sensors; S = 500 draws

# today (plumax)
# C(x, y, z) on the grid, Q = 0.5 kg/s
truth = simulate_plume(                                  # → Dataset (X, Y, Z)
    emission_rate=0.5,
    source_location=(0.0, 0.0, 20.0),
    wind_speed=4.0,
    wind_direction=270.0,
    stability_class="C",
    domain_x=(0.0, 1000.0, 100),
    domain_y=(-200.0, 200.0, 80),
    domain_z=(0.0, 60.0, 13),
)

# every 40th grid column at z = 5 m
sensors = truth.concentration.isel(z=1).stack(k=("x", "y"))[::40]  # → (K,)

# oₖ = C(xₖ, yₖ, zₖ) + εₖ,  εₖ ~ 𝒩(0, s²)
noise = 2e-7 * jax.random.normal(jax.random.key(0), sensors.shape)  # (K,)
observed = jnp.asarray(sensors.values) + noise           # (K,) → (K,)
sensor_xyz = (
    jnp.asarray(sensors.x.values),
    jnp.asarray(sensors.y.values),
    jnp.full(sensors.shape, 5.0),
)

# today (plumax)
# p(Q, b | o) ∝ 𝒩(o; C_Q + b, s²I) · p(Q) · p(b)
posterior = infer_emission_rate(                         # → {Q: (S,), b: (S,)}
    observed,
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

# 𝔼[Q | o], √Var[Q | o]
q = posterior["emission_rate"]                           # (S,)
print(f"Q = {q.mean():.4f} ± {q.std():.4f} kg/s")      # → Q = 0.5000 ± 0.0004 kg/s
```

[`simulate_plume`](xref:api#plumax.gauss_plume.simulate_plume) and
[`infer_emission_rate`](xref:api#plumax.gauss_plume.infer_emission_rate) in
the API reference list every parameter.

## Not just methane

The transport and the inference treat a passive tracer: an emission rate in
kg s⁻¹ goes in, a concentration in kg m⁻³ comes out. Three places still
assume methane: `coupled.rtm` converts column mass to ΔVMR with
`CH4_MOLAR_MASS_KG_PER_MOL`; `les_fvm` labels its outputs "methane"; and the
`population` intensity kernels are named for methane sectors, although their
maths is sector-agnostic.

## Links

- [Roadmap & Architecture](design/roadmap.md)
- [API Reference](xref:api#plumax)
- [Changelog](changelog.md)
- [GitHub](https://github.com/jejjohnson/plumax)
