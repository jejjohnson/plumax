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

## Status snapshot (2026-09-15) {#status-snapshot}

Module-level status is tracked per tier; every scheduled ☐ / 🚧 row on a tier page links to its GitHub issue, grouped under one epic per tier ([#69](https://github.com/jejjohnson/plumax/issues/69)–[#75](https://github.com/jejjohnson/plumax/issues/75)); the unlinked ☐ rows — Step 3 / 5 / 6 items in Tiers III–IV and the RTM stack, plus Tier III diagnostics and multi-species coupling — are later-phase work deliberately not filed yet. Since the previous snapshot (2026-04-29) the July correctness epics ([#19](https://github.com/jejjohnson/plumax/issues/19) `les_fvm` BCs / staggering / CFL, [#20](https://github.com/jejjohnson/plumax/issues/20) plume / puff, [#21](https://github.com/jejjohnson/plumax/issues/21) gaussx / finitevolx primitives, [#22](https://github.com/jejjohnson/plumax/issues/22) retrieval robustness, [#23](https://github.com/jejjohnson/plumax/issues/23) Lagrangian / assimilation) closed and v0.1.1 shipped. The overall picture: **Tiers I–III have Steps 1–2 (forward model + model-based inference) at v1; the RTM stack and Tier IV have Step 1 plus a *linear* Step 2 (matched filter, closed-form column fusion) with the iterated optimal-estimation retrieval and the nonlinear end-to-end inversion still open; Tier V has its v1 core with the importance-corrected likelihood open; no tier has reached Step 3 (emulator) or Step 5 (amortized predictor); of the Tier 0 data layer only the WRF loader exists.**

| Tier | Landed | Epic | Next |
| --- | --- | --- | --- |
| 0 — Prerequisites | `MetField` + WRF loader (`plumax.met`); partial PG stability; radiance observation model | [#69](https://github.com/jejjohnson/plumax/issues/69) | ERA5 loader [#77](https://github.com/jejjohnson/plumax/issues/77) on the same `MetField`; L2 ingest with AKs [#80](https://github.com/jejjohnson/plumax/issues/80) |
| I — Gaussian | Plume + puff forward, turbulence, MAP / MCMC for both | [#70](https://github.com/jejjohnson/plumax/issues/70) | Shared column + AK pipeline [#83](https://github.com/jejjohnson/plumax/issues/83) |
| II — Lagrangian | Particles, Hanna + homogeneous turbulence, forward concentration, backward footprint, Gaussian / lognormal inversion | [#71](https://github.com/jejjohnson/plumax/issues/71) | C-grid wind interpolator [#88](https://github.com/jejjohnson/plumax/issues/88) |
| III — Eulerian | Full FV stack, end-to-end 4D-Var with exact AD adjoint, Gauss-Newton Laplace covariance, PG eddy diffusivity | [#72](https://github.com/jejjohnson/plumax/issues/72) | Spatial per-cell control vector [#91](https://github.com/jejjohnson/plumax/issues/91) |
| RTM | LUTs, Beer–Lambert, SRF / instrument / background, matched filter, gaussx linear solve | [#73](https://github.com/jejjohnson/plumax/issues/73) | Optimal-estimation retrieval [#96](https://github.com/jejjohnson/plumax/issues/96) |
| IV — Coupled | Tier I + AK multi-instrument closed-form fusion with per-instrument bias; nonlinear + linearised radiance operators | [#74](https://github.com/jejjohnson/plumax/issues/74) | Tier II / III backends [#101](https://github.com/jejjohnson/plumax/issues/101), then nonlinear inversion [#102](https://github.com/jejjohnson/plumax/issues/102) |
| V — Population | Cross-tier catalog, hierarchical lognormal size distribution, Gamma-Poisson + log-linear intensity | [#75](https://github.com/jejjohnson/plumax/issues/75) | `methane_pod` primitives in tree [#106](https://github.com/jejjohnson/plumax/issues/106), payload v2 [#107](https://github.com/jejjohnson/plumax/issues/107), importance correction [#108](https://github.com/jejjohnson/plumax/issues/108) |

- **Tier 0 — Prerequisites:** ✓ `MetField` container + `wrfout` loader in [`plumax.met`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/met/) (per-column η→z regrid, bilinear onto the `les_fvm` stagger, `to_prescribed_wind` feeds the Eulerian solver). ☐ ERA5 loader, PBL / Monin–Obukhov diagnostics, coordinate frames, inventory prior and L2 ingest. Every tier's real-data benchmark is blocked here.
- **Tier I — Gaussian:** ✓ plume + puff forward models, ✓ MAP / MCMC inversion. ☐ plume rise, column + AK pipeline, likelihoods module, masked-K multi-source, MO dispersion, NPE predictor. Emulator deliberately skipped (model is cheap).
- **Tier II — Lagrangian:** ✓ Steps 1–2 in [`plumax.lagrangian`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/lagrangian/): Markov-1 Langevin particles, turbulence (homogeneous + Hanna adapter), forward residence-time concentration, backward footprint on the exact reverse clock, closed-form Gaussian / lognormal inversion with a Matérn-3/2 prior. ☐ gridded wind interpolation, met ensembles, footprint emulator (Steps 3–4).
- **Tier III — Eulerian FV:** ✓ [`les_fvm`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/) advection / diffusion / dynamics with open-wall + periodic BCs, C-grid wind staggering, CFL guard, and the strong-constraint 4D-Var loop ([`les_fvm/fourdvar.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/fourdvar.py)) — emission → column-obs forward, whitened (temporal Matérn-3/2) control space, L-BFGS with the exact discrete adjoint via reverse-mode AD, convergence surfaced, Gauss-Newton Laplace posterior (`posterior_covariance` / `laplace_sample` via `gaussx`). v1 inverts a time-resolved scalar emission rate at a known source. ☐ spatial control vector, IC / background / BC-scaling term, incremental Gauss-Newton solve (the `assimilation.solve` optimisers exist but are not wired), MO + Smagorinsky diffusivity, FV emulator.
- **RTM stack:** ✓ [`hapi_lut`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/hapi_lut/) LUT generator + Beer–Lambert, [`radtran`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/radtran/) instrument / SRF / forward, [`matched_filter`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/matched_filter/) detection pipeline, gaussx linear solve. ☐ optimal-estimation retrieval, quality flags + information content, SWIR / TIR surface models, factorised LUT, neural RTM.
- **Tier IV — Coupled:** ✓ v1 multi-instrument fusion in [`plumax.coupled`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/): Tier I plume + averaging-kernel forward per instrument (`CoupledForward`, `Instrument`), closed-form joint posterior over `(Q, bias_inst)` (`fuse_observations`), and the additive RTM observation operator ([`coupled/rtm.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/rtm.py)) mapping column enhancement → ΔVMR → band-integrated radiance. ☐ Tier II / III transport backends, nonlinear end-to-end inversion through the radiance operator, `Q(t)` OU process, masked-K multi-source, quality-flag aggregator; emulators and the operational predictor follow those.
- **Tier V — Population & forecasting:** ✓ v1 core in [`plumax.population`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/population/): tier-agnostic posterior catalog (`event_from_posterior` over `GaussianPosterior` / `LognormalPosterior` / `FusionPosterior`), V.A hierarchical lognormal size-distribution fit, V.B Gamma-Poisson rate + log-linear inhomogeneous intensity. ☐ **The `methane_pod` library the Tier V pages cite is not in this repository** — the first Tier V task is to adopt the point-process machinery in [`xtremax`](https://github.com/jejjohnson/xtremax) (git pin; intensities, compensators, thinning samplers, Hawkes, marked processes, detection thinning, goodness-of-fit, survival / wait-time primitives — 369 passing tests) and port only the methane-domain primitives (POD registry, mark families, missing-mass simulator, fit drivers) in tree. Then: extended per-event payload (samples + prior recall), importance-corrected TMTPP likelihood, mark registry, multi-satellite POD union, catalog ingest, persistency metrics, total-emission estimator, Hawkes / LGCP.

See each tier page for the module-level breakdown and the linked issues.
