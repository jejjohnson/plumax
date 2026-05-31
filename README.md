# plumax

[![Tests](https://github.com/jejjohnson/plumax/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/plumax/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/plumax/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/plumax/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/plumax/actions/workflows/pages.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

> Differentiable, probabilistic atmospheric plume dispersion and methane
> retrieval models, built on JAX.

Author: J. Emmanuel Johnson · Repo: <https://github.com/jejjohnson/plumax>

`plumax` is a library of **forward models** for atmospheric methane —
Gaussian plume / puff, Eulerian finite-volume transport, and line-by-line
radiative transfer — together with the **inference** machinery (Bayesian,
variational, matched-filter) to invert them for source location and
emission rate. Every model is JAX-native (JIT / vmap / autodiff) and
composes through the carrier-agnostic
[`pipekit`](https://github.com/jejjohnson/pipekit) `Operator` primitives,
plugging into data assimilation via the `pipekit_cycle` protocols.

---

## The data-driven modeling cycle

`plumax` is organised around a five-tier modeling cycle. Each tier shares
a fixed forward interface (`forward(params, met) → observations`) so that
emulation and amortized inference are natural substitutions, not rewrites:

| Tier | Forward model | Complexity | Sub-package |
|------|---------------|------------|-------------|
| 0 (prereq) | Met field + averaging-kernel operator | data interface | — |
| I | Gaussian plume / puff | analytical | `gauss_plume`, `gauss_puff` |
| II | Lagrangian particle / footprint | stochastic ODE | _planned_ |
| III | Eulerian finite-volume PDE | PDE | `les_fvm` |
| — | Radiative transfer (parallel track) | multi-physics | `hapi_lut`, `radtran`, `matched_filter` |
| IV | Coupled transport + RTM | end-to-end | _planned_ |
| V | Population & forecasting (TMTPP) | point process | _planned_ |

The full architecture — math, module layout, validation strategy, and
open questions per tier — lives in [`docs/design/`](docs/design/) and
renders under the **Design** tab of the docs site.

---

## Sub-packages

- **`gauss_plume`** — steady-state Gaussian plume with Briggs dispersion
  (stability classes A–F), ground reflection, wind-frame rotation, and a
  NumPyro model for Bayesian emission-rate inference.
- **`gauss_puff`** — time-resolved Gaussian puff with Pasquill–Gifford
  dispersion, `diffrax`-driven time-varying wind, optional Ornstein–
  Uhlenbeck sub-grid turbulence, and NumPyro models for constant-Q and
  random-walk Q inference.
- **`les_fvm`** — Eulerian 3-D advection–diffusion on an Arakawa C-grid
  via [`finitevolX`](https://github.com/jejjohnson/finitevolX): WENO5
  horizontal advection, K-theory diffusion, per-face boundary conditions,
  `diffrax`-compatible time stepping.
- **`hapi_lut`** — HITRAN line-by-line Voigt absorption LUTs plus a
  single-layer Beer–Lambert forward model and its differential-ratio form.
- **`radtran`** — band-integrated radiative transfer + matched-filter
  retrieval; SRF as a linear operator (forward / Jacobian / adjoint), and
  a structured low-rank covariance solve via
  [`gaussx`](https://github.com/jejjohnson/gaussx) Woodbury dispatch.
- **`matched_filter`** — hyperspectral matched-filter detection pipeline
  (robust backgrounds, clustering, streaming).
- **`assimilation`** — 3D/4D-Var cost / control / solve scaffolding on
  `optimistix`.

---

## Quick start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/plumax.git
cd plumax
make install      # uv sync --all-groups + pre-commit hooks
make test         # run tests
make docs-serve   # preview docs locally
```

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
print(ds)
```

---

## Installation notes

- Core install pulls JAX, xarray, diffrax/equinox, finitevolX, optimistix,
  scikit-learn, and pipekit.
- `gaussx` and `pipekit` are not yet on PyPI; from a checkout they resolve
  from git via `[tool.uv.sources]`.
- Optional extras: `inference` (NumPyro), `hapi` (HITRAN LUT generation),
  `notebooks` (matplotlib / IPython). Install with
  `uv add "plumax[inference,hapi]"`.

---

## Development

| Command | Description |
|---------|-------------|
| `make install` | Install all dependency groups + pre-commit hooks |
| `make test` | Run tests (no coverage) |
| `make test-cov` | Run tests with the 80% coverage gate |
| `make lint` / `make format` | Lint / format with ruff (entire repo) |
| `make typecheck` | Type-check `src/plumax` with ty |
| `make docs-serve` | Preview the docs site locally |

The pre-commit checklist (tests, `ruff check .`, `ruff format --check .`,
`ty check src/plumax`) must pass before every commit. See
[`CLAUDE.md`](CLAUDE.md) and [`AGENTS.md`](AGENTS.md) for the full
contributor / agent contract.

## License

MIT — see [LICENSE](LICENSE).
