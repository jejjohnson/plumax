# plumax

> Differentiable, probabilistic atmospheric plume dispersion and methane
> retrieval models, built on JAX.

`plumax` is a library of **forward models** for atmospheric methane —
Gaussian plume / puff, Eulerian finite-volume transport, line-by-line
radiative transfer — together with the **inference** machinery (Bayesian,
variational, matched-filter) to invert them for source location and
emission rate. Every model is JAX-native (JIT / vmap / autodiff) and
composes through the carrier-agnostic
[`pipekit`](https://github.com/jejjohnson/pipekit) `Operator` primitives.

## The modeling cycle

`plumax` is organised around a five-tier **data-driven modeling cycle** —
each tier shares a fixed forward interface so emulation and amortized
inference are natural substitutions rather than rewrites:

1. **Simple model** — a generative story you can simulate from.
2. **Model-based inference** — slow but exact ground truth.
3. **Model emulator** — a fast surrogate (skip if the model is cheap).
4. **Emulator-based inference** — step 2 in seconds.
5. **Amortized inference** — a predictor that learns the posterior map.
6. **Improve** — upgrade any component; the previous step is its ground truth.

See the [Roadmap & Architecture](design/index.md) for the full tier
breakdown (Gaussian → Lagrangian → Eulerian → Coupled E2E → Population).

## Installation

```bash
uv add plumax
```

`gaussx` and `pipekit` are not yet on PyPI; from a checkout they resolve
from git via `[tool.uv.sources]`. NumPyro (Bayesian inference) and
`hitran-api` (LUT generation) are optional extras:

```bash
uv add "plumax[inference,hapi]"
```

## Quickstart

```python
from plumax.gauss_plume import simulate_plume

ds = simulate_plume(
    emission_rate=1.0,                 # kg/s
    source_location=(0.0, 0.0, 10.0),  # x, y, z [m]
    wind_speed=5.0,                    # m/s
    wind_direction=270.0,              # degrees from North (from the west)
    stability_class="D",
    domain_x=(-100.0, 1000.0, 110),
    domain_y=(-300.0, 300.0, 60),
    domain_z=(0.0, 100.0, 20),
)
ds["column_concentration"].plot()
```

## Links

- [Roadmap & Architecture](design/index.md)
- [API Reference](api/reference.md)
- [Changelog](CHANGELOG.md)
- [GitHub](https://github.com/jejjohnson/plumax)
