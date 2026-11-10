# `plumax`: Roadmap & Architecture

> Mathematical models for plume simulation, methane retrieval, source identification, and emission estimation.

This page is the index for the architecture roadmap. The detail for each tier lives in its own file so they can grow independently as design decisions land. The high-level overview (philosophy, tier table, principles) stays here; each tier page expands the math, module layout, validation strategy, and open questions.

---

(cycle-overview)=
## The Data-Driven Modeling Cycle

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

- Step 1 gives us a generative story (a known mathematical structure we can simulate from).
- Step 2 gives us ground truth inference (slow but exact), used to validate everything downstream.
- Step 3 makes Step 2 tractable at scale by replacing the expensive forward model with a fast surrogate.
- Step 4 is Step 2 again, but now running in seconds instead of hours.
- Step 5 collapses the inference loop entirely, as the predictor learns the posterior map directly.
- Step 6 closes the loop: every component is independently upgradable, with the previous step as ground truth.

---

(tier-overview)=
## Tier overview

*plumax tier table: forward models, complexity, and links to detail pages.*

| Tier | Forward model | Complexity | When to use | Detail |
| --- | --- | --- | --- | --- |
| 0 (prereq) | Met field + AK operator | Data interface | All tiers depend on it | [Prerequisites](00_prerequisites.md) |
| I | Gaussian plume / puff | Analytical | Fast prototyping, validation | [Tier I: Gaussian family](01_tier1_gaussian.md) |
| II | Lagrangian particle / footprint | Stochastic ODE | Wind-realistic transport | [Tier II: Lagrangian](02_tier2_lagrangian.md) |
| III | Eulerian finite-volume PDE | PDE | High-fidelity spatial fields | [Tier III: Eulerian FV](03_tier3_eulerian.md) |
| RTM | Radiative transfer (parallel track) | Multi-physics | Connects any tier to radiances | [RTM stack](04_rtm_stack.md) |
| IV | Coupled transport + RTM | End-to-end | Operational satellite → source posterior | [Tier IV: Coupled E2E](05_tier4_coupled.md) |
| V | Population & forecasting (TMTPP) | Stochastic point process | Aggregate per-event posteriors → wait times, totals | [Tier V: Population](06_tier5_population.md) (and [V.A](06_tier5a_instantaneous.md), [V.B](06_tier5b_point_process.md), [V.C](06_tier5c_persistency.md), [V.D](06_tier5d_total_emission.md)) |

The build order is roughly Prerequisites → Tier I → RTM stack (parallel) → Tier II → Tier III → Tier IV → Tier V. RTM is independent of transport tier, so it can be developed in parallel by a different person without coordination cost. Tier V depends on at least Tier I being usable end-to-end (per-event posteriors are the input), but does not need Tiers II-IV, as it can launch with Tier I posteriors and absorb richer ones later.

---

(architectural-principles)=
## Architectural principles

:::{important} 1. The cycle is the architecture
Don't treat emulation and amortization as afterthoughts. Design the forward-model API at Step 1 so Steps 3-5 are natural substitutions rather than rewrites.
:::

:::{important} 2. Each step validates the next
The model-based posterior (Step 2) is the ground truth for the emulator posterior (Step 4), which is the ground truth for the amortized predictor (Step 5). Never skip validation; otherwise emulator bugs become posterior bugs.
:::

:::{important} 3. The forward-model interface is fixed across tiers
All four transport tiers (I-IV) implement the same shape: `forward(params, met) → observations`. Inference code (`vardaX`, `filterax`, NumPyro) is written once and reused. See [Prerequisites: fixed forward interface](#prereqs-forward-interface) for the concrete signature.
:::

:::{important} 4. WRF is a data source, not a competitor
WRF provides met forcing and benchmark concentration fields. `plumax` learns to be fast, differentiable and probabilistic, which are properties WRF doesn't have.
:::

:::{important} 5. Improvement is structured
Step 6 is structured rather than vague iteration. Each improvement targets a specific component (better physics, more training data, richer posterior family, tighter observation operator), and the cycle structure tells us which component to upgrade and how to validate it.
:::

---

(status-snapshot)=
## Status snapshot (2026-11-10)

Module-level status is tracked per tier; every scheduled ☐ / 🚧 row on a tier page links to its GitHub issue, grouped under one epic per tier ([#69](https://github.com/jejjohnson/plumax/issues/69)-[#75](https://github.com/jejjohnson/plumax/issues/75)); the unlinked ☐ rows (Step 3 / 5 / 6 items in Tiers III-IV and the RTM stack, plus Tier III diagnostics and multi-species coupling) are later-phase work deliberately not filed yet. Since the previous snapshot (2026-09-15) the Tier 0 data layer, the Tier I-IV follow-ups and most of Tier V have landed, and the RTM work moved to radtrax ([#133](https://github.com/jejjohnson/plumax/issues/133), so the RTM epic [#73](https://github.com/jejjohnson/plumax/issues/73) and its issues were closed there as not planned). The overall picture: Tier 0 is complete except the static-fields carrier; Tiers I-III have Steps 1-2 at v1 plus the Tier II footprint emulator (Steps 3-4) and met-ensemble runner, with the Tier I NPE predictor and the Tier III FV emulator still open; Tier IV has Tier II / III transport backends, the quality-flag aggregator, masked-K multi-source and a nonlinear end-to-end inversion through the radiance operator with a Laplace posterior, with the `Q(t)` process open; Tier V has everything except catalog ingest and Hawkes / LGCP intensities; only the Tier II footprint emulator has reached Step 3, and no tier has reached Step 5 (amortized predictor).

| Tier | Landed | Epic | Next |
| --- | --- | --- | --- |
| 0: Prerequisites | `MetField` + WRF and ERA5 loaders, PBL + Monin-Obukhov diagnostics, coordinate frames and time axis, emission-inventory and background priors, L2 ingest with AKs; partial PG stability | [#69](https://github.com/jejjohnson/plumax/issues/69) (closed) | static-fields carrier (`z0`, land use, terrain), not in tree although [#78](https://github.com/jejjohnson/plumax/issues/78) is closed |
| I: Gaussian | Plume + puff forward, turbulence, MAP / MCMC for both, Briggs plume rise, column + AK pipeline, Student-t / heteroscedastic likelihoods, masked-K multi-source, Monin-Obukhov dispersion | [#70](https://github.com/jejjohnson/plumax/issues/70) | Amortized NPE predictor [#87](https://github.com/jejjohnson/plumax/issues/87) |
| II: Lagrangian | Particles, Hanna + homogeneous turbulence, forward concentration, backward footprint, Gaussian / lognormal inversion, C-grid wind interpolator, met-ensemble runner, footprint emulator with emulator-based inversion | [#71](https://github.com/jejjohnson/plumax/issues/71) | Field predictor (Step 5), not filed yet |
| III: Eulerian | Full FV stack, end-to-end 4D-Var with exact AD adjoint, Gauss-Newton Laplace covariance, PG + Monin-Obukhov + Smagorinsky diffusivity, spatial per-cell control, background / BC-scaling terms | [#72](https://github.com/jejjohnson/plumax/issues/72) | Incremental Gauss-Newton 4D-Var [#93](https://github.com/jejjohnson/plumax/issues/93), then FV emulator [#95](https://github.com/jejjohnson/plumax/issues/95) |
| RTM | LUTs, Beer-Lambert, SRF / instrument / background, matched filter, gaussx linear solve | [#73](https://github.com/jejjohnson/plumax/issues/73) (closed, moved to radtrax) | Optimal-estimation retrieval, now tracked in radtrax (plumax [#96](https://github.com/jejjohnson/plumax/issues/96) closed as superseded) |
| IV: Coupled | Tier I + AK multi-instrument closed-form fusion with per-instrument bias; nonlinear + linearised radiance operators; Tier II / III transport backends; quality-flag aggregator; masked-K multi-source; nonlinear end-to-end inversion with Laplace posterior | [#74](https://github.com/jejjohnson/plumax/issues/74) | `Q(t)` Ornstein-Uhlenbeck process [#103](https://github.com/jejjohnson/plumax/issues/103) |
| V: Population | Cross-tier catalog, hierarchical lognormal size distribution, Gamma-Poisson + log-linear intensity; intensity + POD registries, missing-mass simulator, POD-power-law NUTS fitter; xtremax generative layer, per-event payload v2, mark registry, importance-corrected TMTPP likelihood; multi-satellite POD union, persistency metrics, total-emission estimator, SBC harness | [#75](https://github.com/jejjohnson/plumax/issues/75) | Catalog ingest [#111](https://github.com/jejjohnson/plumax/issues/111), then Hawkes / LGCP [#115](https://github.com/jejjohnson/plumax/issues/115) |

- **Tier 0 (Prerequisites):** ✓ `MetField` container + `wrfout` loader in [`plumax.met`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/met/) (per-column η→z regrid, bilinear onto the `les_fvm` stagger, `to_prescribed_wind` feeds the Eulerian solver). ✓ ERA5 loader ([#77](https://github.com/jejjohnson/plumax/issues/77)), PBL and Monin-Obukhov diagnostics ([#78](https://github.com/jejjohnson/plumax/issues/78)), coordinate frames and time axis ([#79](https://github.com/jejjohnson/plumax/issues/79)), L2 ingest with averaging kernels ([#80](https://github.com/jejjohnson/plumax/issues/80)), inventory and background priors ([#81](https://github.com/jejjohnson/plumax/issues/81)). ☐ the static-fields carrier (`z0`, land use, terrain), which is not in tree.
- **Tier I (Gaussian):** ✓ plume + puff forward models, ✓ MAP / MCMC inversion. ✓ Briggs plume rise, the shared column + AK pipeline, the likelihoods module, masked-K multi-source and Monin-Obukhov dispersion ([#82](https://github.com/jejjohnson/plumax/issues/82)-[#86](https://github.com/jejjohnson/plumax/issues/86)). ☐ NPE predictor ([#87](https://github.com/jejjohnson/plumax/issues/87)). Emulator deliberately skipped (model is cheap).
- **Tier II (Lagrangian):** ✓ Steps 1-2 in [`plumax.lagrangian`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/lagrangian/): Markov-1 Langevin particles, turbulence (homogeneous + Hanna adapter), forward residence-time concentration, backward footprint on the exact reverse clock, closed-form Gaussian / lognormal inversion with a Matérn-3/2 prior. ✓ C-grid wind interpolator ([#88](https://github.com/jejjohnson/plumax/issues/88)), met-ensemble runner ([#89](https://github.com/jejjohnson/plumax/issues/89)) and footprint emulator with emulator-based inversion ([#90](https://github.com/jejjohnson/plumax/issues/90)). ☐ amortized field predictor (Step 5).
- **Tier III (Eulerian FV):** ✓ [`les_fvm`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/) advection / diffusion / dynamics with open-wall + periodic BCs, C-grid wind staggering, CFL guard, and the strong-constraint 4D-Var loop ([`les_fvm/fourdvar.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/les_fvm/fourdvar.py)): emission → column-obs forward, whitened (temporal Matérn-3/2) control space, L-BFGS with the exact discrete adjoint via reverse-mode AD, convergence surfaced, Gauss-Newton Laplace posterior (`posterior_covariance` / `laplace_sample`), all as a `vardax.ThreeDVar` analysis behind the `assim` extra ([#163](https://github.com/jejjohnson/plumax/issues/163)). v1 inverts a time-resolved scalar emission rate at a known source. ✓ spatial per-cell control vector ([#91](https://github.com/jejjohnson/plumax/issues/91)), IC / background / BC-scaling term ([#92](https://github.com/jejjohnson/plumax/issues/92)), MO + Smagorinsky diffusivity ([#94](https://github.com/jejjohnson/plumax/issues/94)). 🚧 incremental Gauss-Newton solve (`solve_4dvar(method="gauss_newton")` is in tree but [#93](https://github.com/jejjohnson/plumax/issues/93) is still open). ☐ FV emulator ([#95](https://github.com/jejjohnson/plumax/issues/95)).
- **RTM stack:** ✓ [`hapi_lut`](https://github.com/jejjohnson/radtrax/tree/main/src/radtrax/spectroscopy/hitran/) LUT generator + Beer-Lambert, [`radtran`](https://github.com/jejjohnson/radtrax/tree/main/src/radtrax/slab/) instrument / SRF / forward, [`matched_filter`](https://github.com/jejjohnson/radtrax/tree/main/src/radtrax/retrieval/matched_filter/) detection pipeline, gaussx linear solve. ☐ optimal-estimation retrieval, quality flags + information content, SWIR / TIR surface models, factorised LUT and neural RTM: these moved with `plumax.radtran` to radtrax ([#133](https://github.com/jejjohnson/plumax/issues/133)), and the plumax issues ([#96](https://github.com/jejjohnson/plumax/issues/96)-[#100](https://github.com/jejjohnson/plumax/issues/100)) were closed as not planned.
- **Tier IV (Coupled):** ✓ v1 multi-instrument fusion in [`plumax.coupled`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/): Tier I plume + averaging-kernel forward per instrument (`CoupledForward`, `Instrument`), closed-form joint posterior over `(Q, bias_inst)` (`fuse_observations`), and the additive RTM observation operator ([`coupled/rtm.py`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/coupled/rtm.py)) mapping column enhancement → ΔVMR → band-integrated radiance. ✓ Tier II / III transport backends ([#101](https://github.com/jejjohnson/plumax/issues/101)), the nonlinear end-to-end inversion through the radiance operator with a Laplace posterior (`solve_coupled`, [#102](https://github.com/jejjohnson/plumax/issues/102)), masked-K multi-source ([#104](https://github.com/jejjohnson/plumax/issues/104)) and the quality-flag aggregator ([#105](https://github.com/jejjohnson/plumax/issues/105)). ☐ `Q(t)` OU process ([#103](https://github.com/jejjohnson/plumax/issues/103)); emulators and the operational predictor follow.
- **Tier V (Population & forecasting):** ✓ v1 core in [`plumax.population`](https://github.com/jejjohnson/plumax/tree/main/src/plumax/population/): tier-agnostic posterior catalog (`event_from_posterior` over `GaussianPosterior` / `LognormalPosterior` / `FusionPosterior`), V.A hierarchical lognormal size-distribution fit, V.B Gamma-Poisson rate + log-linear inhomogeneous intensity. ✓ The former standalone `methane_pod` library is in tree: `plumax.population.intensity` (13 λ(t) kernels), `plumax.population.pod` (10 probability-of-detection models), `plumax.population.paradox` (missing-mass Monte Carlo) and `plumax.population.fitting` (POD-modified power-law NUTS fit), with its [theory pages](../theory/population/01_mttpp_theory.md) and [notebooks](../notebooks/population/01_missing_mass_paradox.ipynb). ✓ The generative layer runs on [`xtremax`](https://github.com/jejjohnson/xtremax) point processes (`plumax.population.processes`, the `population` extra; `ThinningProcess` for POD thinning). ✓ Extended per-event payload (samples + prior recall) from every tier and the mark registry with the heavy-tail families (`plumax.population.marks`). ✓ The importance-corrected TMTPP likelihood (`plumax.population.importance`, `fitting.tmtpp_iw_model`). ✓ Multi-satellite POD union ([#110](https://github.com/jejjohnson/plumax/issues/110)), persistency metrics ([#112](https://github.com/jejjohnson/plumax/issues/112)), the total-emission estimator ([#113](https://github.com/jejjohnson/plumax/issues/113)) and the SBC harness ([#114](https://github.com/jejjohnson/plumax/issues/114)). ☐ Catalog ingest ([#111](https://github.com/jejjohnson/plumax/issues/111)) and Hawkes / LGCP intensities ([#115](https://github.com/jejjohnson/plumax/issues/115)).

See each tier page for the module-level breakdown and the linked issues.
