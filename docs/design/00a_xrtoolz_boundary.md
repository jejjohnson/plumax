# Prerequisites — the `xrtoolz` boundary

> Where plumax stops and [`xrtoolz`](https://github.com/jejjohnson/xrtoolz) starts, for every pre- and post-processing step in the roadmap.

The tier pages describe *models*: JAX forward operators, their adjoints, and the inference loops around them. Around every model sits a ring of work that is neither JAX nor differentiable — reading files, decoding time axes, destaggering, regridding, vertical remapping, subsetting, masking, binning, integrating columns, scoring fields, plotting. Today that ring is hand-rolled inside plumax (the WRF loader alone carries its own bilinear sampler, per-column vertical interpolator and `Times` decoder), and every open Tier 0 / Tier V issue proposes more of it. `xrtoolz` already owns that ring for the rest of the stack, so this page fixes the boundary once and lists what has to land upstream before plumax builds on it.

---

## The rule {#xrtoolz-rule}

!!! important "Gradient path decides ownership"
    If a function must run under `jax.jit` / `jax.grad` / `jax.vmap`, it lives in plumax. If it maps `xarray` in to `xarray` out (or NumPy in, NumPy out) and sits *outside* every trace, it belongs in `xrtoolz` — used from there, and added there first when missing.

    **Why:** the transport kernels, Langevin steps, matched filter, Beer–Lambert LUT path and every adjoint are the value plumax adds. Regridding and CF hygiene are not; duplicating them costs review time and drifts from the ocean/atmosphere tooling that already has tests for them.
    **How to apply:** a docstring that says *"All of this is NumPy: it is IO-side, not on any gradient path"* (as `met/wrf.py` does) is the tell. Move it.

Two corollaries:

1. **`xarray` at the seam, `equinox` inside.** Every loader returns a CF-conformant `xr.Dataset` on `(time, z, y, x)` produced by an `xrtoolz` pipeline; plumax converts it to the JAX PyTree (`MetField`, `Instrument`, `EmissionCatalog`) in one `from_dataset` step and back with `to_dataset`. This settles the open `coordax` question on the [Prerequisites page](00_prerequisites.md#prereqs-open-questions): dimension naming is `xarray`'s job at the boundary; the PyTrees stay raw.
2. **`xrtoolz` primitives are the test oracles for plumax kernels.** Where the same quantity must exist twice — a NumPy `xarray` version for pipelines and a JAX version inside a trace (column integrals, ΔVMR conversions, averaging-kernel application) — the JAX kernel is tested against the `xrtoolz` function, never re-derived in a test.

---

## What `xrtoolz` provides today {#xrtoolz-inventory}

Verified against `xrtoolz` `main` (2026-09-18; `xrtoolz` 0.0.2, `xrtoolz-reader` 0.0.3). The workspace is six packages; the light ones (`xrtoolz-core` → `xrcore`, `xrtoolz-reader` → `xrreader`) can be core dependencies, the main `xrtoolz` package (cartopy, rioxarray, regionmask, xrft, xskillscore, dask) is an optional extra — see [Packaging](#xrtoolz-packaging).

*xrtoolz surface relevant to plumax, by concern.*

| Concern | `xrtoolz` API | plumax consumer |
| --- | --- | --- |
| Acquisition (ERA5 / CDS) | `xrreader.CDSSource` with the `REANALYSIS` / `REANALYSIS_PRESSURE` form profiles, `CATALOG` short names `era5.single_levels` / `era5.pressure_levels`, `Request`, `BBox`, `TimeRange`, `PressureLevels`; CF `Variable` registry (`u10`, `v10`, `t2m`, `sp`, `msl` already registered) | ERA5 loader [#77](https://github.com/jejjohnson/plumax/issues/77) |
| Coordinate validation / CF renaming | `geo.validate_latitude`, `validate_longitude`, `validate_time`, `decode_cf_time`, `rename_to_cf_standard_names`, `check_dataset_coords`; `xrreader.apply_cf_attrs` | every loader; the `t0` / UTC invariant of [#79](https://github.com/jejjohnson/plumax/issues/79) |
| CRS and local frames | `geo.lonlat_to_xy`, `xy_to_lonlat` (pyproj `Transformer`, `always_xy`), `assign_crs`, `calc_latlon` (2-D lat/lon on an x/y grid), `reproject`, `reproject_match` (rioxarray) | `LocalFrame` [#79](https://github.com/jejjohnson/plumax/issues/79); inventory rasters [#81](https://github.com/jejjohnson/plumax/issues/81) |
| Subsetting / screening | `geo.subset_bbox` (handles 2-D lat/lon swaths), `subset_where`, `subset_time`, `select_variables`, `apply_mask`, `add_land_mask` (regionmask) | L2 ingest [#80](https://github.com/jejjohnson/plumax/issues/80), quality flags [#97](https://github.com/jejjohnson/plumax/issues/97) / [#105](https://github.com/jejjohnson/plumax/issues/105) |
| Horizontal regrid | `interpolate.regrid_like` (`xr.interp` on shared coords), `sample_at_points` (RegularGridInterpolator at scattered points), `coarsen`, `coarsen_conservative` (cos-lat weighted), `refine`, `bin_2d` / `histogram_2d` on a `Grid` | WRF/ERA5 → analysis grid; column field → receptor pixels; inventory → grid; per-tile background |
| Vertical remap | `transforms.remap_axis`, `interpolate.RemapAxis`, presets `ToHeight`, `ToPressureLevels` (1-D source levels, NaN outside range) | η→z [gap 2](#xrtoolz-gaps) |
| Time resampling | `interpolate.resample_time`, `ResampleTime` | hourly met ↔ overpass cadence |
| Mask morphology | `transforms.clean_mask`, `remove_small_objects_2d`, `binary_opening_2d`, `CleanMask` | matched-filter detection masks |
| Smoothing / gap-fill | `gaussian_smooth`, `moving_average`, `fillnan_*` | scene preprocessing before background estimation |
| sklearn marshalling | `xrsklearn.XarrayEstimator` (stack → delegate → unstack, `NanPolicy`), `SklearnOp` | matched-filter background estimators |
| Named-tensor ops | `xreinx.einsum`, `rearrange`, `pack_dataset` / `unpack_dataset` | band-last cube packing in `matched_filter/io.py` |
| Field verification | `metrics.pixel` (`rmse`, `bias`, `r2_score`), `spectral` (`psd_score`, `resolved_scale`), `structural` (`ssim`, `centroid_displacement`), `probabilistic` (`crps_ensemble`, `rank_histogram`, `spread_skill_ratio`, `ensemble_coverage`), `instance` (`mask_iou_matrix`, `instance_f1_at_iou`), `EvaluateByRegion`, `MaskedMetric` | every "Step 2 validates Step 4" check; emulator gates; plume-object verification |
| Visualisation | `viz.make_axes` (cartopy presets), `cmap_for`, `shared_norm`, `viz.validation` panels | docs notebooks once frames land |
| Composition | `xrcore.Operator` (DataTree dispatch), `combinators.Augment` / `ApplyToEach`; `pipekit.Sequential` / `Graph` shared with `plumax.operators` | `Sequential(GaussianPlume(...), Coarsen(...), RMSE(...))` |

`plumax.operators` already subclasses `pipekit.Operator` with a mapping carrier in and a `Dataset` out, so a plumax forward model is a valid **head node** of an `xrtoolz` `Sequential`; everything after it is post-processing and comes from `xrtoolz`.

---

## Existing plumax code that re-implements it {#xrtoolz-duplicates}

*Hand-rolled pre/post-processing in tree and its `xrtoolz` replacement.*

| plumax code | What it does | Replace with | Notes |
| --- | --- | --- | --- |
| `met/wrf.py::_bilinear` | bilinear sample of a `(…, ny, nx)` field at analysis T/U/V points | `regrid_like` / `sample_at_points` | plumax keeps only the half-cell stagger offsets |
| `met/wrf.py::_interpolate_vertical` | per-column linear interpolation from terrain-following mass levels to `z_agl`, nearest-value extrapolation | `RemapAxis` / `ToHeight` | blocked by [gap 2](#xrtoolz-gaps) (N-D source levels, extrapolation policy) |
| `met/wrf.py::_time_axis`, `_decode_times` | `XTIME` minutes / `Times` char rows → seconds since `t0` | a `wrfout` opener in `xrtoolz` + `validate_time` | [gap 1](#xrtoolz-gaps) |
| `met/wrf.py` destagger + `T`, `P+PB`, `PH+PHB` conversions | WRF-specific physical fields | same `wrfout` opener | the whole loader shrinks to `MetField.from_dataset` + restagger |
| `gauss_plume.simulate_plume`, `gauss_puff.simulate_puff`, `lagrangian.simulate_lagrangian`, `les_fvm._to_dataset` | build `xr.Dataset`, integrate the column (`np.trapezoid` in three places, `sum · dz` in `les_fvm`) | `xrtoolz.atm.column_integral` ([gap 3](#xrtoolz-gaps)) + `apply_cf_attrs` | also removes the trapezoid-vs-sum inconsistency between tiers |
| `matched_filter/io.py::apply_image_xarray`, `open_multi_scene` | `apply_ufunc` over a band-last cube; glob → `(n_pixels, n_bands)` batches | `xreinx.rearrange` / `pack_dataset`, `xr.open_mfdataset`; `xrtoolz.inference.JaxModelOp` for the operator form | the JAX matched filter itself stays |
| `matched_filter/background.py`, `cluster.py`, `radtran/background.py` | trimmed / Huber means, Ledoit–Wolf, GMM clusters, local-window stats over `(H, W, B)` | `XarrayEstimator` marshalling + sklearn estimators; `gaussian_smooth` for local windows | plumax keeps the `gaussx` / `lineax` operator wrapping and the Woodbury solve; two of these modules duplicate each other today |
| `coupled/rtm.py::_band_integrate`, `radtran/nb_lut.py`, `hapi_lut/beers.py` | spectral-axis `np.interp` / `ds.interp(temperature=, pressure=)` | **keep** | spectral, not geospatial; `xarray` is already doing the work |
| `assimilation/diagnostics.py` | χ², DFS, posterior-covariance probes | **keep** | inversion-specific, needs the Hessian |

---

## Planned work that must build on `xrtoolz` {#xrtoolz-planned}

Each open issue below keeps its plumax deliverable (the PyTree, the registry, the JAX kernel) and drops the proposed re-implementation of the processing around it.

*Open issues, the part that stays in plumax, and the part that comes from (or goes to) `xrtoolz`.*

| Issue | Stays in plumax | Comes from `xrtoolz` | Goes to `xrtoolz` first |
| --- | --- | --- | --- |
| [#79](https://github.com/jejjohnson/plumax/issues/79) frames + time | `LocalFrame` as a frozen dataclass carrying an EPSG string as static metadata; `TimeAxis` | `lonlat_to_xy` / `xy_to_lonlat`, `calc_latlon`, `validate_time` | UTM-zone-from-origin helper ([gap 4](#xrtoolz-gaps)) |
| [#77](https://github.com/jejjohnson/plumax/issues/77) ERA5 loader | `MetField` assembly, U/V restagger | `CDSSource` download, `ToHeight` (hypsometric), `regrid_like`, `resample_time`; parity test = `coarsen_conservative` + `metrics.rmse` | `blh`, pressure-level `u/v/t/z/q` in the `Variable` registry ([gap 5](#xrtoolz-gaps)); N-D vertical remap ([gap 2](#xrtoolz-gaps)) |
| WRF loader (landed) | `MetField`, restagger | same pipeline as ERA5 with a different opener | `wrfout` opener ([gap 1](#xrtoolz-gaps)) |
| [#78](https://github.com/jejjohnson/plumax/issues/78) PBL / MO | Businger–Dyer `phi_m` / `phi_h`, `u*`, `L`, `w*` as JAX (consumed inside the Langevin step); `HannaTurbulence` factory | static fields (`z0`, land use, terrain) via `reproject_match` + `add_land_mask` | `potential_temperature`, `wind_speed` / `wind_direction`, bulk-Richardson PBL height as `xrtoolz.atm` diagnostics ([gap 3](#xrtoolz-gaps)) |
| [#80](https://github.com/jejjohnson/plumax/issues/80) L2 ingest | `Instrument` PyTree, AK-convention registry (`x_a`, `h`, `R_retr` defaults) | `subset_bbox` (2-D swath lat/lon), `subset_where` on `qa_value`, `lonlat_to_xy` for pixel centres, `rename_to_cf_standard_names` | product openers (TROPOMI / EMIT / GHGSat groups → flat CF dataset) as an `xrreader`-style local `DataSource` ([gap 6](#xrtoolz-gaps)); `column_averaging_kernel`, `dry_air_column`, `mixing_ratio_to_column` in `xrtoolz.atm.gas.ch4` ([gap 3](#xrtoolz-gaps)) |
| [#83](https://github.com/jejjohnson/plumax/issues/83) column + AK pipeline | the traced JAX version used by the forward | the `xrtoolz.atm.gas.ch4` version as its oracle | — |
| [#81](https://github.com/jejjohnson/plumax/issues/81) inventory prior | `Inventory` enum, wiring into `build_problem` / `linear_gaussian_inversion` | `assign_crs` + `reproject_match` (rasterio `sum` / `average` resampling), `coarsen_conservative`, `xrgrad.grid_metrics_from_coords` for basin totals, `Coarsen` / `bin_2d` for the per-tile background | conservative regrid to an arbitrary target ([gap 7](#xrtoolz-gaps)) — rasterio resampling suffices for v1 |
| [#111](https://github.com/jejjohnson/plumax/issues/111) catalog ingest | `EmissionCatalog` / `EmissionEvent` provenance fields, the TMTPP-facing coverage table | `sample_at_points` to pull `U_ERA5` at each `(lat, lon, t)` for wind rescaling; `SpaceTimeGrid` / `histogram_2d` for coverage; sklearn DBSCAN through `xrsklearn` for the 5 km / 12 h de-dup | carry the catalog as an `xr.Dataset` on an `event` dim (the `xrreader` station long-format), so every op above applies without a pandas detour |
| [#89](https://github.com/jejjohnson/plumax/issues/89) met ensembles, [#107](https://github.com/jejjohnson/plumax/issues/107) payload v2 | ensemble runner, posterior samples | `crps_ensemble`, `rank_histogram`, `spread_skill_ratio`, `ensemble_coverage` for calibration of field posteriors | — |
| [#97](https://github.com/jejjohnson/plumax/issues/97) / [#105](https://github.com/jejjohnson/plumax/issues/105) quality flags | flag semantics, `R` assembly | `subset_where`, `ApplyMask`, `clean_mask` | — |
| Emulator gates (Step 3, every tier) | emulator training ([`pipekit-train`](https://github.com/jejjohnson/pipekit)) | `psd_score`, `resolved_scale`, `ssim`, `EvaluateByRegion` as the acceptance metrics | — |

!!! warning "Two different PODs"
    `xrtoolz.metrics.object.ProbabilityOfDetection` is an event-verification score (hits / (hits + misses)) and is still a stub upstream. Tier V's POD is a *detection-thinning function* $P_d(Q)$ ([V.B](06b_point_process.md)). They share a name and nothing else; do not route the Tier V one through `xrtoolz`.

---

## Gaps to land in `xrtoolz` first {#xrtoolz-gaps}

Ordered by how many plumax issues each unblocks. Everything here is `xarray` in, `xarray` out, and sits outside every trace, so it belongs upstream by the [rule](#xrtoolz-rule). All seven are filed under the `xrtoolz` theme epic [xrtoolz#293](https://github.com/jejjohnson/xrtoolz/issues/293).

1. **`wrfout` opener** ([xrtoolz#294](https://github.com/jejjohnson/xrtoolz/issues/294)) — `Times` / `XTIME` decoding to a CF `time`, `U` / `V` / `W` destagger to mass points, `T + 300` and `(P + PB)` to temperature and pressure, `(PH + PHB) / g − HGT` to height above ground, `PBLH` carried through. Generic WRF hygiene that every atmospheric consumer of `xrtoolz` needs; plumax's `_decode_times` and the conversion block move up verbatim. Unblocks: WRF loader simplification, [#77](https://github.com/jejjohnson/plumax/issues/77) parity test.
2. **N-D source levels and an extrapolation policy for `remap_axis`** ([xrtoolz#295](https://github.com/jejjohnson/xrtoolz/issues/295)) — today the source coordinate must be 1-D and out-of-range targets become NaN. Terrain-following (WRF η) and hybrid (ERA5 model-level) grids have a per-column height, and a plume grid whose first level sits below the lowest mass level needs nearest-value holding, not NaN. Unblocks: `_interpolate_vertical` removal, [#77](https://github.com/jejjohnson/plumax/issues/77).
3. **`xrtoolz.atm` and `xrtoolz.atm.gas.ch4` content** ([xrtoolz#296](https://github.com/jejjohnson/xrtoolz/issues/296)) — the stubs already list `potential_temperature`, `wind_speed`, `wind_direction`, `column_averaging_kernel`, `dry_air_column`, `mixing_ratio_to_column`. Add `column_integral(da, dim, method="trapezoid" | "sum")`, `destagger(da, dim)`, `hypsometric_height`, bulk-Richardson PBL height, and the mass-column ↔ ΔVMR conversion (`coupled/rtm.py::column_mass_to_delta_vmr` is the JAX twin). Where these land in the `xrtoolz` tree (`atm/` per its CLAUDE.md layout or `kinematics/` per its D9 decision) is `xrtoolz`'s call. Unblocks: [#78](https://github.com/jejjohnson/plumax/issues/78), [#80](https://github.com/jejjohnson/plumax/issues/80), [#83](https://github.com/jejjohnson/plumax/issues/83), the four `xr.Dataset` builders.
4. **UTM-from-origin helper in `xrtoolz.geo`** ([xrtoolz#297](https://github.com/jejjohnson/xrtoolz/issues/297)) — `utm_crs_for(lon, lat) -> str` (or a `local_frame(origin_lon, origin_lat)` returning the EPSG code) so `LocalFrame` is a wrapper, not a pyproj client. Unblocks: [#79](https://github.com/jejjohnson/plumax/issues/79).
5. **`xrreader` registry entries** ([xrtoolz#298](https://github.com/jejjohnson/xrtoolz/issues/298)) — ERA5 `boundary_layer_height`, pressure-level `u_component_of_wind` / `v_component_of_wind` / `temperature` / `geopotential` / `specific_humidity`, and the methane L2 fields (`methane_mixing_ratio_bias_corrected`, `column_averaging_kernel`, `qa_value`) with CF standard names. Unblocks: [#77](https://github.com/jejjohnson/plumax/issues/77), [#80](https://github.com/jejjohnson/plumax/issues/80).
6. **Local-file `DataSource` for satellite L2 products** ([xrtoolz#299](https://github.com/jejjohnson/xrtoolz/issues/299)) — TROPOMI (NetCDF groups), EMIT, GHGSat: open, flatten groups, rename to CF. The `xrreader.DataSource` ABC already has the `open(dataset_id, Request)` shape; a local backend is the missing piece. Unblocks: [#80](https://github.com/jejjohnson/plumax/issues/80), [#111](https://github.com/jejjohnson/plumax/issues/111).
7. **Conservative regrid to an arbitrary target** ([xrtoolz#300](https://github.com/jejjohnson/xrtoolz/issues/300)) — on the `xrtoolz` roadmap (v0.4 "conservative approximation"); `coarsen_conservative` is integer-factor only. Needed for inventory rasters onto a rotated metric grid; rasterio `sum` resampling via `reproject_match` is the v1 stand-in. Unblocks: [#81](https://github.com/jejjohnson/plumax/issues/81) v2.

---

## Packaging {#xrtoolz-packaging}

- **Optional extra, lazy imports.** The main `xrtoolz` package pulls cartopy, rioxarray, regionmask, xrft, xskillscore and dask — far heavier than plumax's core. Follow the `[inference]` / `[hapi]` precedent: a `plumax[data]` extra that installs `xrtoolz`, with the loaders and post-processing helpers importing it lazily (PEP 562, as `gauss_plume.inference` does for NumPyro). `xrtoolz-core` and `xrtoolz-reader` are light enough for core if `MetField.from_dataset` wants `xrreader.apply_cf_attrs` unconditionally.
- **Git pins, one `pipekit`.** `xrtoolz` is a uv workspace; every member has to be named as a `subdirectory` source, pinned to a rev like `finitevolx` and `gaussx` are. Both `xrtoolz` and plumax build on `pipekit.Operator`; plumax pins `pipekit` at its 0.0.2 release commit and `xrtoolz` requires `>= 0.0.1`, so the graph resolves today — keep it that way by bumping both pins together.
- **Cache goes through `pipekit`, not plumax.** The Prerequisites page asks for a zarr cache of pre-resampled met keyed on `(source, native_grid_hash, target_grid_hash, time_window)`. Every `xrtoolz` operator serialises its configuration (`get_config`, `pipekit.hashing.stable_json`), and `pipekit.Cache` / `Memoize` exist; the cache key is the pipeline config, and plumax should not grow its own.

---

## Validation strategy {#xrtoolz-validation}

- **Loader equivalence.** Before deleting `_bilinear` / `_interpolate_vertical`, the `xrtoolz`-backed `load_wrf` must reproduce the current one on the synthetic `wrfout` fixture to round-off (`tests/met/test_wrf.py::test_wind_lands_on_les_fvm_stagger_points` already pins the stagger semantics).
- **Oracle tests.** Every JAX twin of an `xrtoolz.atm` primitive (`column_integral`, ΔVMR, AK application) gets one test that compares it to the `xrtoolz` function on random input — the JAX side has no independent derivation in its tests.
- **Parity test as a pipeline.** The WRF-vs-ERA5 parity check on the Prerequisites page is `Sequential(load, coarsen_conservative, rmse)`; write it once as an `xrtoolz` pipeline so it runs unchanged on any future loader.

---

## Open questions {#xrtoolz-open-questions}

!!! attention "Where does `EmissionCatalog` live?"
    If the catalog is an `xr.Dataset` on an `event` dim at the boundary, de-duplication, coverage and wind rescaling are all `xrtoolz` ops and only the TMTPP-facing PyTree is plumax. If it stays a dataclass of arrays, [#111](https://github.com/jejjohnson/plumax/issues/111) re-implements them. **Leaning:** `Dataset` at the boundary, PyTree inside, same as `MetField`.

!!! attention "Receptor sampling on the gradient path"
    `coupled.forward.column_response` evaluates the plume at receptor pixels inside a trace, so it cannot call `sample_at_points`. The split is: `xrtoolz` samples *fields* at pixels for diagnostics and emulator training data; the traced forward keeps its own analytic evaluation. Document this on `Instrument` so nobody "fixes" it.

!!! attention "`xrtoolz.atm` vs `xrtoolz.kinematics`"
    `xrtoolz`'s D9 decision collapses the per-domain packages into `kinematics/`, but the tree still ships `atm/` and `atm/gas/ch4/` stubs. plumax should not depend on the final module path until that is settled upstream; import through the top-level re-exports where possible.
