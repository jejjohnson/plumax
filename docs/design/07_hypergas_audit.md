# HyperGas functionality audit — what we have, where it lives, what is missing

**Status:** audit report (2026-07)
**Subject:** [SRON-ESG/HyperGas](https://github.com/SRON-ESG/HyperGas) v1.0
([preprint](https://egusphere.copernicus.org/preprints/2026/egusphere-2025-6127/))
audited against `plumax` and the `geotoolz` workspace.

## 1. Purpose

HyperGas is SRON's end-to-end pipeline for retrieving trace-gas (CH₄/CO₂)
enhancements from hyperspectral imagers (PRISMA, EnMAP, EMIT) and
quantifying plume emission rates. The question this audit answers: *how
much of HyperGas do we already cover across plumax + geotoolz, and what
core functionality is genuinely missing?*

**Headline answer:** the two *algorithmic* cores of HyperGas — the
matched-filter retrieval and the plume/emission machinery — are already
covered by us, in most respects more rigorously (structured covariances,
Woodbury solves, differentiable forward models, Bayesian inversion vs.
HyperGas's calibrated point estimates). What we are missing is almost
entirely the *operational data layer* around those cores: sensor readers,
per-column (push-broom) retrieval conventions, wind reanalysis ingestion,
sensor-calibrated effective-wind laws, the plume-masking chain, and
scene-geometry ancillaries (per-pixel angles, DEM/RPC geolocation).

## 2. HyperGas module map

~7 300 LOC, xarray/satpy/numpy, not differentiable, no tests of note.

| Module | Function |
|---|---|
| `hyper.py` | `Hyper` orchestrator: satpy readers (`hsi_l1b` EnMAP, `emit_l1b` EMIT, `hyc_l1` PRISMA), VNIR/SWIR merge, water-band drop, RPC+DEM geolocation refinement, wind attach, RGB, quality mask, angle computation |
| `unit_spectrum.py` | Unit absorption spectrum **K**: Beer–Lambert "model" path (6 climatology atmospheres, solar irradiance, absorption cross-sections, Young air-mass factor) or MODTRAN LUT path (mag1c/Foote); Gaussian FWHM convolution to sensor bands; log-linear slope fit; per-column K for smile (PRISMA 2-D central wavelengths) |
| `lut_interp.py` | mag1c 5°-grid MODTRAN LUT spline lookup (zenith, sensor altitude, ground altitude, water vapor, concentration) |
| `retrieve.py` | `MatchedFilter`: **column-wise** or scene-wise; normal (linearized, target = K·μ) or **lognormal** (log-radiance) variants; background stats per segmentation label (land/water or PCA+k-means cluster); cloud skip; plume-mask exclusion from background stats (iterative MF) |
| `denoise.py` | TV-Chambolle denoising with J-invariant auto-calibration (`skimage.calibrate_denoiser`), per segmentation label |
| `a_priori_mask.py` | Plume masking via `tobac` multi-threshold feature detection + segmentation; thresholds = trimmed mean + 2σ/3σ |
| `plume_utils.py` | Connected-mask selection consistent with wind azimuth (max azimuth difference / max distance from source), Carbon Mapper v2 masking, interactive HTML mask picking |
| `plumeline.py` | Weighted polynomial plume centerline fit (rotated frame), arc-length parameterization, perpendicular transect construction |
| `ime_csf.py` | Emission rates: **IME** (Varon 2018, sensor-calibrated `U_eff = α₁ln U₁₀+α₂+α₃U₁₀`), **IME-fetch** (Varon 2019), **Carbon Mapper** method, **CSF** along curved centerline; uncertainty budget = random (plume-relocation sampling) ⊕ wind (Monte Carlo, σ=50 %/1.5 m s⁻¹) ⊕ calibration (α/β swap) ⊕ masking (per-sensor %) |
| `emiss.py` | L2→L3 plume NetCDF export, CSV results with Nominatim geocoding, IPCC sector tagging |
| `wind.py` | u10/v10/surface-pressure from **ERA5**, **GEOS-FP**, **OpenMeteo**, attached to the scene |
| `angles.py` / `tle.py` | Per-pixel SZA/SAA/VZA/VAA/RAA/sun-glint-angle from TLE orbit propagation (space-track.org) |
| `dem.py` / `orthorectification.py` | Copernicus DEM download; GLT / RPC / GCP orthorectification via rasterio |
| `quality_mask.py` / `landmask.py` / `cluster.py` | TOA-reflectance water/cloud/cirrus flags; OSM/GSHHS/Natural-Earth land mask; PCA+k-means scene segmentation |
| `hsi2rgb.py` / `folium_map.py` / `app/` | HSI2RGB true-color; folium PNG/HTML maps; Streamlit plume-marker + emission GUI |
| `unit_conversion.py` | ppm-anchored unit registry (ppb/ppm/ppm·m/mol m⁻²/kg m⁻²…) |
| `scripts/` | L2B/L3B batch (re)processing, wind calibration, pseudo-observations |

## 3. Capability-by-capability comparison

Legend — ✅ have (equal or better) · 🟡 partial · ❌ missing.

### 3.1 Retrieval core

| HyperGas capability | Ours | Where | Notes |
|---|---|---|---|
| Classical matched filter | ✅ | `plumax.matched_filter.core` (JAX, structured Σ via gaussx); `plumax.radtran.matched_filter` (NumPy); `geotoolz.matched_filter` (operators) | Three implementations; ours dispatch through lineax/gaussx (dense, low-rank Woodbury, Kronecker) vs. HyperGas's dense `spectral` calls |
| Background mean/cov estimation | ✅ | `plumax.matched_filter.background` (median/trimmed/Huber, Ledoit–Wolf, OAS, low-rank), `streaming` (Welford); `geotoolz.matched_filter` | Strictly richer than HyperGas (which uses `spectral.calc_stats` only) |
| Cluster-conditioned background | ✅ | `plumax.matched_filter.cluster` (GMM), `geotoolz` `GMMClusterBackground`/`ApplyClusterMF` | HyperGas uses PCA+k-means; equivalent role |
| Plume-mask exclusion from background stats (iterative MF) | ✅ | `geotoolz` operators accept masks; plumax estimators take pixel stacks (caller filters) | Convention, not code gap |
| **Lognormal matched filter** | ❌ | — | HyperGas `rad_dist='lognormal'`: MF in log-radiance space. Absent everywhere in ours |
| **Column-wise MF (per detector column)** | ❌ | — | The standard push-broom convention (per-column background + per-column K for smile). Ours are scene/cluster/window only. This is HyperGas's *default* mode |
| Sparsity/albedo-corrected MF (mag1c-style) | ❌ | — | Also absent in HyperGas; noted for completeness |
| Target/unit-spectrum from RTM | ✅ | `plumax.radtran.target`, `nb_lut` (band-integrated table), `plumax.matched_filter.target` (jvp through PSF∘GSD∘SRF); `geotoolz` `LinearTargetFromObs` | We generate targets three independent ways, incl. exact Jacobian-vector products HyperGas cannot do |
| Absorption data: HITRAN line-by-line | ✅ | `plumax.hapi_lut` (Voigt LUTs: CH₄, CO₂, H₂O, O₂, N₂O, CO over (ν,T,p)) | HyperGas ships static per-climatology cross-section files instead |
| **Climatology atmospheres + solar irradiance + AMF unit-spectrum recipe** | 🟡 | pieces in `plumax.radtran`/`hapi_lut` (single homogeneous slab, scalar AMF, albedo prefactor) | HyperGas selects one of 6 profile climatologies by lat/DOY, propagates layered columns, and fits K = dln L/dc via least squares. Our slab model is cleaner but the *layered-profile* unit spectrum and the auto lat/season model selection are missing |
| **MODTRAN LUT path** (zenith/altitude/H₂O axes) | ❌ | — | Our LUT axes are (ν,T,p) only; no geometry/water-vapor axes. Flagged already in `04_rtm_stack.md` (factorised LUT ☐) |
| Per-column K for spectral smile | ❌ | — | Needs 2-D central-wavelength support in the SRF layer |
| Sensor-band convolution (FWHM Gaussian) | ✅ | `plumax.radtran.srf` (gaussian/rect/tri/custom + adjoint); `geotoolz` `ApplySRF`/`GaussianSRF` | Ours is a linear operator with exact adjoint |

### 3.2 Scene/data layer

| HyperGas capability | Ours | Where | Notes |
|---|---|---|---|
| **PRISMA / EnMAP / EMIT L1B readers** | ❌ | — | Biggest single gap. `geotoolz.readers` has the `SensorReader` ABC + toy reader only ("Track B" irregular-geolocation readers reserved, unimplemented); plumax has no ingest at all |
| **Wind reanalysis ingestion (ERA5/GEOS-FP/OpenMeteo)** | ❌ | — | plumax wind is user-supplied arrays (`WindSchedule`); geotoolz has none. Planned ☐ in `00_prerequisites.md` (met loaders) |
| Per-pixel viewing/solar angle grids (TLE-based) | 🟡 | `geotoolz.radiometry.ComputeSZA` (solar zenith only) | No SAA/VZA/VAA/RAA/sun-glint, no orbit propagation |
| DEM download + RPC orthorectification | 🟡 | `geotoolz.geom.Georeference` (GLT), `SwathToGrid`; `AltitudeMask`/`SlopeMask` consume a provided DEM | No DEM fetch, no RPC/sensor-model geolocation |
| Quality masks (water/cloud/cirrus from TOA) | ✅ | `geotoolz.qa` (bitmask/SCL decoders, semantic Mask* ops) | geotoolz decodes product QA; HyperGas computes flags from radiance thresholds — a small spectral-threshold recipe worth porting for L1B-only sensors |
| Land mask (OSM/GSHHS/Natural Earth) | ✅ | `geotoolz.mask.LandMask`/`OceanMask`/`CountryMask` (Natural Earth) | Equivalent |
| Destriping / denoising | ✅ | `geotoolz.restore` (`DestripeColumn`, `MomentMatching`, MNF, NLM, TV-adjacent family) | HyperGas's auto-calibrated (J-invariant) TV weight selection is a nice trick we lack; low priority |
| RGB composites | ✅ | `geotoolz.viz.TrueColor` + `spectral.GaussianSRF`/`SpectralBinning` for hyperspectral→RGB | HSI2RGB itself not needed |
| Unit conversion registry (ppb↔ppm↔ppm·m↔mol m⁻²↔kg m⁻²) | 🟡 | `geotoolz.plume.ColumnToMass`, `plumax.coupled.rtm.column_mass_to_delta_vmr` | Both cover column↔mass; no ppb/ppm mixing-ratio registry with surface pressure/gravity (HyperGas `_ime_sum` recipe) |

### 3.3 Plume masking and emission quantification

| HyperGas capability | Ours | Where | Notes |
|---|---|---|---|
| Plume masking (threshold + features) | ✅ | `geotoolz.plume` `PlumeMask` (abs/Otsu/percentile), `PlumeContours`, `PlumeShapeFilter`, `PlumeQNDFeatures`; `geotoolz.segment`/`measure` | HyperGas uses tobac; ours is equivalent without the tobac dependency |
| **Wind-azimuth-consistent mask selection** | ❌ | 🟡 `geotoolz.plume.WindAdvectionCone` is the geometric primitive | The HyperGas `select_connect_masks` logic (keep components whose bearing from source agrees with wind azimuth within `az_max`, within `dist_max`) is not implemented; it composes naturally from `WindAdvectionCone` + `PlumeContours` |
| IME | ✅ | `geotoolz.plume.IMEEstimate` (Varon 2018; max-axis/hull/skeleton length; uncertainty dict) | plumax has none by design (inverse-model tiers instead) |
| CSF | ✅ | `geotoolz.plume.CrossSectionalFlux` (straight downwind transects) | |
| **Curved-centerline CSF (polynomial plumeline)** | ❌ | — | HyperGas fits a weighted polynomial centerline, equal-arc-length transects perpendicular to the curve. Ours only does straight transects |
| IME-fetch (Varon 2019 circular-radius) | ❌ | — | Small, self-contained estimator |
| Carbon Mapper method (hull-length IME) | ❌ | — | Small; mostly a masking-and-length convention |
| **Sensor-calibrated effective wind U_eff(U₁₀)** | ❌ | — | HyperGas ships per-sensor, per-source-type α/β coefficients from LES calibration. `IMEEstimate` takes `U_eff` as an input — the calibration *law + coefficient tables* are the missing piece |
| **Uncertainty budget (random ⊕ wind ⊕ calibration ⊕ masking)** | 🟡 | `IMEEstimate` returns a simple uncertainty | HyperGas's plume-relocation random error (move mask around scene, trimmed σ of background IME) and wind Monte Carlo are not implemented. plumax's Bayesian tiers give *better* uncertainty for model-based inversion, but the mass-balance error budget is a separate, needed thing |
| Emission-rate inference (inverse modelling) | ✅➕ | `plumax` gauss_plume/gauss_puff NumPyro NUTS, Lagrangian BLUE/lognormal inversion, 4D-Var w/ exact adjoint, Tier IV fusion, Tier V population stats | Entirely beyond HyperGas's scope — our differentiator |
| L2/L3 product export, geocoding, IPCC sectors | ❌ | — | Product/packaging layer; out of library scope (belongs to a pipeline built on pipekit, if ever) |
| Folium/HTML maps, Streamlit GUI | ❌ | — | Deliberately out of scope for both libraries |

## 4. Where functionality lives (division of labour)

- **geotoolz** owns the *scene/image* side: readers, QA/masks, geometry,
  destriping, matched-filter operators over carriers, plume
  masking + mass-balance quantification (`geotoolz.plume`).
- **plumax** owns the *physics/inference* side: RTM + LUTs + unit
  spectra, differentiable forward transport (Tiers I–III), variational &
  Bayesian inversion (3D/4D-Var, NUTS, fusion, population).
- Overlap in the matched filter is intentional: geotoolz has the
  carrier-aware operator pipeline; plumax has the JAX/gaussx kernel for
  differentiable end-to-end studies. Both consume the same math.

## 5. Core missing functionality (prioritized)

### P1 — blocks any real-data end-to-end run

1. **Sensor readers: EMIT, EnMAP, PRISMA L1B** → `geotoolz.readers`
   (Track B). EMIT first (NetCDF + GLT already matches
   `geom.Georeference`; no RPC needed). EnMAP/PRISMA need VNIR/SWIR
   merge, water-band drop, per-column wavelengths.
2. **Column-wise matched filter mode** → `geotoolz.matched_filter` (a
   `ColumnwiseBackground`/`ApplyColumnMF` operator) and/or a `plumax`
   vmapped-over-columns kernel. Without it, push-broom retrievals stripe.
3. **Wind reanalysis ingestion (ERA5, GEOS-FP)** → new
   `plumax` met module per `00_prerequisites.md` ☐ (`met.era5`), or a
   geocatalog source + small extraction op. Needed by both mass-balance
   quantification and the inversion tiers.

### P2 — needed for credible emission numbers

4. **Sensor-calibrated `U_eff` laws + coefficient tables** (IME
   `α₁ln U+α₂+α₃U`, CSF `β₁U+β₂`; point/area source variants) →
   `geotoolz.plume`, as a small calibration registry feeding
   `IMEEstimate`/`CrossSectionalFlux`.
5. **Mass-balance uncertainty budget**: plume-relocation random error,
   wind Monte Carlo, calibration-swap, masking-% terms →
   `geotoolz.plume`.
6. **Wind-azimuth-consistent plume-component selection** → compose
   `WindAdvectionCone` + `PlumeContours` into a
   `SelectPlumeComponents` operator.
7. **Curved-centerline CSF** (weighted polynomial fit, equal-arc
   transects) → `geotoolz.plume` alongside `CrossSectionalFlux`.
8. **Lognormal MF variant** → both MF stacks (log-transform + stats is
   ~30 lines each).

### P3 — completes parity

9. **Mixing-ratio unit registry** (ppb/ppm ↔ column via surface
   pressure & gravity) → extend `geotoolz.plume.ColumnToMass` /
   `plumax` conversions.
10. **Per-pixel viewing/solar angle grids** (SZA/SAA/VZA/VAA/RAA);
    TLE propagation optional if readers expose product angles (EMIT
    does; EnMAP/PRISMA need computation) → `geotoolz.radiometry`.
11. **Layered-climatology unit-spectrum path + geometry/H₂O LUT axes**
    → `plumax.radtran`/`hapi_lut`; already designed as the factorised
    LUT ☐ in `04_rtm_stack.md`.
12. **IME-fetch + Carbon Mapper estimators** → `geotoolz.plume`.
13. **DEM fetch + RPC geolocation** → `geotoolz.geom`; only needed for
    EnMAP-grade geolocation refinement.
14. **Radiance-threshold quality masks** (water/cloud/cirrus for
    L1B-only sensors) → `geotoolz.qa`.

### Explicit non-goals (in HyperGas, not wanted here)

- Streamlit GUI, folium/HTML/PNG map export, Nominatim geocoding, IPCC
  sector CSV packaging — application/product layer, not library
  functionality. If needed, build as a thin pipeline on pipekit.
- satpy/tobac/spectral dependencies — our stacks already cover the
  underlying functionality without them.
- HSI2RGB — `viz.TrueColor` + spectral binning suffices.

## 6. What we have that HyperGas does not

For balance — reasons *not* to adopt HyperGas wholesale:

- Differentiable everything: exact Jacobians/adjoints through
  RTM ∘ PSF ∘ GSD ∘ SRF, 4D-Var with discrete adjoints, `jax.jvp`
  targets. HyperGas is not differentiable anywhere.
- Structured covariance MF (Woodbury `O(B·k)`, Kronecker, streaming
  Welford) vs. dense per-column stats.
- Bayesian emission inference with real posteriors (NUTS, random-walk
  `Q(t)`, BLUE/lognormal, multi-instrument fusion, population tiers) vs.
  point estimates with parametric error bars.
- HITRAN line-by-line LUT generation on demand vs. shipped static
  cross-section files.
- Carrier-agnostic operator composition (pipekit), patch/catalog
  infrastructure (geotoolz-patcher/-catalog), test suites and CI.
