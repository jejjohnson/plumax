# plumax

> **Where is the plume, how much is coming out, where does it come from, how often does it happen.**
> Differentiable transport models (Gaussian plume and puff, Lagrangian
> particles, Eulerian finite volumes) and the Bayesian and variational
> machinery to invert them, for any passive tracer: methane, CO₂, NO₂,
> SO₂, smoke or heat. Everything is JAX: `jit`, `vmap` and `grad` through
> the physics.

:::{figure} assets/diagrams/plumax-stack.png
:label: fig-plumax-stack
:alt: Where plumax sits: radtrax turns a radiance scene into a column enhancement map; plumax segments it into a mask (PlumeMask, PlumeContours, PlumeFootprint), quantifies an emission rate (IMEEstimate, CrossSectionalFlux, infer_emission_rate), attributes it to a source (compute_footprint, linear_gaussian_inversion, solve_4dvar, fuse_observations) and turns every posterior into a catalogue event for rates, detection probability, persistency metrics and total emission (EmissionCatalog, fit_poisson_rate, population.pod, population.persistency, total_emission), all shipping today; underneath, the transport tiers gauss_plume, gauss_puff, lagrangian, les_fvm and coupled.

From a column map to an emission catalogue. Every card ships today.
:::

## Four questions

| Question | Verb | Status |
|---|---|---|
| Where is the plume? | **segment**: mask, contours, footprint | ✅ masks, contours, footprints |
| How much, in kg s⁻¹? | **quantify**: model inversion, IME, cross-sectional flux | ✅ inversion · IME · flux |
| Which source? | **attribute**: footprints, 4D-Var, multi-instrument fusion | ✅ known source · multi-source |
| How often, how long? | **persist**: event rates, detection probability, catalogues | ✅ rates · persistency metrics |

✅ ships today. The [roadmap](design/roadmap.md) has the status of every tier, including the predictors and emulators still open.

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
| mask, contour and measure a plume in an image | plumax `[geo]` (`plumax.geo`) |
| count events, model detection probability, extrapolate to a population | plumax (`population`) |

## What it stands on

plumax sits in GeoStack's GeoModels layer beside radtrax, and nothing below imports it back. It reaches down to finitevolX in GeoNumerics for the Eulerian grid, and to Core for the rest: gaussx for every posterior covariance, pipekit for the `Operator` interface and, behind extras, vardax and argminax for the source inversions and xtremax for the point processes. Every tier exposes the same `forward(params, met) → observations`, so we write the samplers and assimilation schemes once against it. NumPyro does the sampling, vardax (the `assim` extra) runs the Tier II optimal interpolation and the Tier III 3D-Var over plumax's source-receptor maps, and filterax (ensemble Kalman methods) is the planned home for EKI. The [stack layer map](https://github.com/jejjohnson/geostack/blob/main/docs/design/stack_layer_map.md) also plans to move some of plumax out, as the particle machinery goes to tracerax, the per-pixel geometry to radiomax and the physical constants to geophysix.

:::{figure} assets/diagrams/plumax-deps.png
:label: fig-plumax-deps
:alt: plumax and its dependencies in the layers of the geostack stack layer map: in GeoModels, plumax imports radtrax and radtrax never imports plumax; in Core, pipekit and gaussx are base dependencies, vardax and argminax come with the assim extra, xtremax with the population extra and pipekit-train with the emulate extra, pipekit-cycle is used in tests only, geonnax arrives through radtrax, and integrix, functax and filterax are planned; in GeoNumerics, finitevolX is a base dependency with spectraldiffx underneath, and tracerax, radiomax and geophysix are planned homes for code that moves out of plumax; in GeoStack, geotoolz-ops comes with the geo extra and xrtoolz with the io extra; jax, equinox, einx, diffrax, optimistix, lineax, numpyro, xarray, netcdf4, scipy, georeader and scikit-learn are the third-party foundations.

plumax's declared dependencies in the layers of the geostack stack layer map. Each card names the plumax subpackages that import it, and planned moves are dashed.
:::

## The modeling cycle

plumax is organised around a five-tier data-driven modeling cycle. Each tier shares a fixed forward interface, so swapping in an emulator or an amortized predictor does not mean rewriting the tier:

1. **Simple model**: a generative story you can simulate from.
2. **Model-based inference**: slow but exact ground truth.
3. **Model emulator**: a fast surrogate (skip it if the model is cheap).
4. **Emulator-based inference**: step 2 in seconds.
5. **Amortized inference**: a predictor that learns the posterior map.
6. **Improve**: upgrade any component; the previous step is its ground truth.

The [Roadmap & Architecture](design/roadmap.md) page has the tier breakdown, going from Gaussian to Lagrangian, Eulerian, coupled and finally population models.

## Installation

```bash
uv add plumax
```

finitevolX, `gaussx`, `pipekit` and `radtrax` are not yet on PyPI; from a checkout they resolve from git via `[tool.uv.sources]`. NumPyro (Bayesian inference), vardax and argminax (the source inversions), xtremax (populations), the plume operators on GeoTensors and the notebook dependencies are optional extras:

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

$N_{\Omega_y} = 200$ sensors at 5 m record $o_k = C(x_k, y_k, z_k) + b + \varepsilon_k$, with a background $b$ and noise $\varepsilon_k \sim \mathcal{N}(0, s^2)$. NUTS samples the posterior over $(Q, b)$, and $N_{\mathcal{E}}$ is the number of draws.

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
- [Notation](theory/notation.md)
- [API Reference](xref:api#plumax)
- [Changelog](changelog.md)
- [GitHub](https://github.com/jejjohnson/plumax)
