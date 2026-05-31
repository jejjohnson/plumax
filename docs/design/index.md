# `plumax` — Roadmap & Architecture

> Mathematical models for plume simulation, methane retrieval, source identification, and emission estimation.

This page is the **index** for the architecture roadmap. The detail for each tier lives in its own file so they can grow independently as design decisions land. The high-level overview (philosophy, tier table, principles) stays here; each tier page expands the math, module layout, validation strategy, and open questions.

---

## The Data-Driven Modeling Cycle {#cycle-overview}

Every tier in `plumax` follows the same loop:

```text
┌─────────────────────────────────────────────────────────────────┐
│   (1) Simple Model                                              │
│       ↓                                                         │
│   (2) Model-Based Inference                                     │
│       ↓                                                         │
│   (3) Model Emulator          ← skip if model is cheap          │
│       ↓                                                         │
│   (4) Emulator-Based Inference                                  │
│       ↓                                                         │
│   (5) Amortized Inference (Predictor)                           │
│       ↓                                                         │
│   (6) Improve  ───────────────────────────────────────────────┐ │
│       ↑         upgrade model / data / emulator / posterior   │ │
│       └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

- Step 1 gives you a **generative story** — a known mathematical structure you can simulate from.
- Step 2 gives you **ground truth inference** — slow but exact, used to validate everything downstream.
- Step 3 makes Step 2 **tractable at scale** — replace the expensive forward model with a fast surrogate.
- Step 4 is Step 2 again, but now running in seconds instead of hours.
- Step 5 collapses the inference loop entirely — the predictor learns the posterior map directly.
- Step 6 closes the loop — every component is independently upgradable, with the previous step as ground truth.

---

## Tier overview {#tier-overview}

*plumax tier table — forward models, complexity, and links to detail pages.*

| Tier | Forward model | Complexity | When to use | Detail |
| --- | --- | --- | --- | --- |
| 0 (prereq) | Met field + AK operator | Data interface | All tiers depend on it | [Prerequisites](00_prerequisites.md) |
| I | Gaussian plume / puff | Analytical | Fast prototyping, validation | [Tier I — Gaussian family](01_tier1_gaussian.md) |
| II | Lagrangian particle / footprint | Stochastic ODE | Wind-realistic transport | [Tier II — Lagrangian](02_tier2_lagrangian.md) |
| III | Eulerian finite-volume PDE | PDE | High-fidelity spatial fields | [Tier III — Eulerian FV](03_tier3_eulerian.md) |
| — | Radiative transfer (parallel track) | Multi-physics | Connects any tier to radiances | [RTM stack](04_rtm_stack.md) |
| IV | Coupled transport + RTM | End-to-end | Operational satellite → source posterior | [Tier IV — Coupled E2E](05_tier4_coupled.md) |
| V | Population & forecasting (TMTPP) | Stochastic point process | Aggregate per-event posteriors → wait times, totals | [Tier V — Population](06_tier5_population.md) (and [V.A](06a_instantaneous.md), [V.B](06b_point_process.md), [V.C](06c_persistency.md), [V.D](06d_total_emission.md)) |

The build order is roughly: **Prerequisites → Tier I → RTM stack (parallel) → Tier II → Tier III → Tier IV → Tier V.** RTM is independent of transport tier, so it can be developed in parallel by a different person without coordination cost. Tier V depends on at least Tier I being usable end-to-end (per-event posteriors are the input), but does not need Tiers II–IV — it can launch with Tier I posteriors and absorb richer ones later.

---

## Architectural principles {#architectural-principles}

!!! important "1. The cycle is the architecture"
    Don't treat emulation and amortization as afterthoughts. Design the forward-model API at Step 1 so Steps 3–5 are natural substitutions, not rewrites.

!!! important "2. Each step validates the next"
    The model-based posterior (Step 2) is the ground truth for the emulator posterior (Step 4), which is the ground truth for the amortized predictor (Step 5). Never skip validation; otherwise emulator bugs become posterior bugs.

!!! important "3. The forward-model interface is fixed across tiers"
    All four tiers implement the same shape: `forward(params, met) → observations`. Inference code (`vardaX`, `filterax`, NumPyro) is written once and reused. See [Prerequisites — fixed forward interface](00_prerequisites.md#prereqs-forward-interface) for the concrete signature.

!!! important "4. WRF is a data source, not a competitor"
    WRF provides met forcing and benchmark concentration fields. `plumax` learns to be **fast, differentiable, and probabilistic** — properties WRF doesn't have.

!!! important "5. Improvement is structured"
    Step 6 is not vague iteration. Each improvement targets a specific component — better physics, more training data, richer posterior family, tighter observation operator — and the cycle structure tells you which component to upgrade and how to validate it.

---

## Status snapshot (2026-04-29) {#status-snapshot}

Module-level status is tracked per tier. Quick overview:

- **Tier I — Gaussian:** ✓ plume + puff forward models, ✓ MAP/MCMC inversion. Emulator + amortized predictor not yet started.
- **Tier II — Lagrangian:** 🚧 forward model + model-based inference in [`plumax.lagrangian`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/lagrangian/): Markov-1 Langevin particles, turbulence (homogeneous + Hanna), forward residence-time concentration, backward footprint, and closed-form Gaussian / lognormal source inversion with a Matérn-3/2 prior. Footprint emulator + amortized predictor (Steps 3–5) not started.
- **Tier III — Eulerian FV:** 🚧 [`les_fvm`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/) advection/diffusion/dynamics + the strong-constraint **4D-Var loop wired end-to-end** ([`les_fvm/fourdvar.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/fourdvar.py)): a differentiable emission→column-obs forward, the source-prior + time-summed-observation cost in whitened (temporal Matérn-3/2) control space, and an L-BFGS solve whose gradient is the exact discrete adjoint via reverse-mode AD through the diffrax FV solver. v1 inverts a time-resolved scalar emission rate at a known source, with a Gauss-Newton Laplace **posterior covariance** around the MAP (`posterior_covariance` / `laplace_sample`, Hessian inverted via `gaussx`). The IC/background cost term, spatial source prior, incremental inner solve, En4D-Var posterior, emulator (Step 3) and predictor (Step 5) are future work.
- **RTM stack:** 🚧 [`hapi_lut`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/hapi_lut/) LUT generator + Beer–Lambert in place; [`radtran`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/radtran/) instrument/SRF/forward modules present; [`matched_filter`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/matched_filter/) detection pipeline in place. Optimal-estimation retrieval not wired; neural RTM not started.
- **Tier IV — Coupled:** 🚧 v1 multi-instrument fusion in [`plumax.coupled`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/): the Tier I plume + averaging-kernel coupled forward kept per-instrument at native resolution (`CoupledForward`, `Instrument`), and a closed-form joint posterior over `(Q, bias_inst)` across a list of satellites (`fuse_observations`) — exploiting the plume's linearity in `Q` (the design's linear-conditional-Gaussian limit), with per-instrument bias as a first-class state. An additive RTM-based observation operator ([`coupled/rtm.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/rtm.py): `RadianceObservationOperator`, `radiance_response`) now maps the plume column enhancement → gas `ΔVMR` → band-integrated normalised radiance via the `radtran` Beer–Lambert + SRF stack, as the build-order step toward L1-radiance fusion. Full Tier II/III + RTM coupling, the `Q(t)` stochastic process, trans-dimensional source count, coupled emulator and operational predictor are future work.
- **Tier V — Population & forecasting:** 🚧 standalone [`methane_pod`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/) library is feature-complete (intensity catalog, POD catalog, paradox simulator, NUTS fitter, synthetic-data validation). The in-tree [`plumax.population`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/population/) subpackage now lands the v1 core: the tier-agnostic cross-tier posterior catalog (`catalog.py` — `event_from_posterior` over `GaussianPosterior` / `LognormalPosterior` / `FusionPosterior`), V.A hierarchical lognormal size-distribution fit with per-event uncertainty propagation (`size_distribution.py`), and the V.B point-process core (`point_process.py` — closed-form Gamma-Poisson rate + log-linear inhomogeneous intensity). Still missing: the importance-corrected TMTPP mark likelihood, real-data CSV ingestion, multi-satellite POD fusion, and the LGCP intensity. See [Tier V index](06_tier5_population.md).

See each tier page for module-level breakdown.
