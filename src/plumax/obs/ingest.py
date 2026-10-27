"""L2 methane products → ``coupled.Instrument`` plus observed enhancements.

``coupled.Instrument`` carries receptors in analysis metres, a per-receptor
scalar averaging kernel and the retrieval variance; ``fuse_observations``
takes those instruments with an observed column-mass enhancement each
[kg m⁻²], the unit ``coupled.column_response`` integrates the plume to.
These loaders build both from a product file:

1. read the file with xrreader's opener (QA screening included),
2. place pixel centres in the analysis frame with ``LocalFrame.to_xy``,
3. convert the product's quantity to a column-mass enhancement ``ΔΩ``,
4. reduce the averaging kernel to the scalar the plume sees.

Unit conversions (``M`` the CH₄ molar mass, ``N_A`` Avogadro's number):

- XCH₄ (TROPOMI, GHGSat) in ppb above a background ``b``:
  ``ΔΩ = (XCH₄ − b) · 10⁻⁹ · N_dry · M / N_A``, with ``N_dry`` the dry-air
  column [molecules m⁻²] from :func:`xrtoolz.atm.gas.ch4.dry_air_column`
  (TROPOMI's surface pressure; a standard 101 325 Pa column for GHGSat).
- Enhancement in ppm·m (EMIT): ``ΔΩ = E · 10⁻⁶ · (p / (k_B T)) · M / N_A``,
  the ideal-gas number density of a near-surface layer at ``(p, T)``.

Averaging kernel: a point-source plume stays in the boundary layer, so its
column sees the kernel of the lowest retrieval layer. TROPOMI's layers are
ordered top of atmosphere first, so that is ``layer = -1`` (the default
``surface_layer``). EMIT and GHGSat give no kernel; their enhancements
are treated as unsmoothed (AK = 1).

Needs the ``io`` extra (``pip install 'plumax[io]'``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import xarray as xr

from plumax.coupled import Instrument
from plumax.met.frames import LocalFrame
from plumax.met.time import TimeAxis


#: Molar mass of CH₄ [kg mol⁻¹].
M_CH4 = 0.0160425
#: Avogadro's number [mol⁻¹].
AVOGADRO = 6.02214076e23
#: Boltzmann constant [J K⁻¹].
BOLTZMANN = 1.380649e-23
#: Standard surface pressure [Pa] for products that report none.
P_STANDARD = 101_325.0
#: Near-surface temperature [K] for the ppm·m conversion when none is given.
T_STANDARD = 288.15

#: (lon_min, lat_min, lon_max, lat_max) [degrees].
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ColumnObservation:
    """One product's instrument and its observed enhancement.

    Attributes:
        instrument: Receptors, scalar averaging kernel and retrieval
            variance, ready for ``CoupledForward``.
        enhancement: Observed column-mass enhancement ``ΔΩ`` [kg m⁻²],
            shape ``(n_obs,)``, aligned with ``instrument.receptors``.
        time: Observation timestamp (UTC).
    """

    instrument: Instrument
    enhancement: jnp.ndarray
    time: np.datetime64


def _readers():
    try:
        import xrreader
        from xrtoolz.atm.gas.ch4 import dry_air_column
    except ImportError as err:
        raise ImportError(
            "the L2 loaders read through xrreader / xrtoolz; install them with "
            "`pip install 'plumax[io]'`"
        ) from err
    return xrreader, dry_air_column


def ppb_to_column_mass(
    delta_ppb: np.ndarray, dry_air_molecules: np.ndarray
) -> np.ndarray:
    """``ΔΩ = Δχ · 10⁻⁹ · N_dry · M / N_A`` [kg m⁻²].

    Args:
        delta_ppb: Mixing-ratio enhancement [ppb].
        dry_air_molecules: Dry-air column ``N_dry`` [molecules m⁻²].

    Returns:
        Column-mass enhancement [kg m⁻²].
    """
    return delta_ppb * 1.0e-9 * dry_air_molecules * M_CH4 / AVOGADRO


def ppm_m_to_column_mass(
    enhancement_ppm_m: np.ndarray, *, pressure_pa: float, temperature_k: float
) -> np.ndarray:
    """``ΔΩ = E · 10⁻⁶ · (p / (k_B T)) · M / N_A`` [kg m⁻²].

    Args:
        enhancement_ppm_m: Path-integrated enhancement [ppm m].
        pressure_pa: Layer pressure [Pa].
        temperature_k: Layer temperature [K].

    Returns:
        Column-mass enhancement [kg m⁻²].
    """
    n_air = pressure_pa / (BOLTZMANN * temperature_k)
    return enhancement_ppm_m * 1.0e-6 * n_air * M_CH4 / AVOGADRO


def load_tropomi_l2(
    path: str | Path,
    *,
    frame: LocalFrame,
    bbox: BBox | None = None,
    background_ppb: float | None = None,
    qa_min: float = 0.5,
    surface_layer: int = -1,
    clock: TimeAxis | None = None,
    name: str = "TROPOMI",
) -> ColumnObservation:
    """A TROPOMI CH₄ L2 granule as an instrument and its enhancement.

    Uses the bias-corrected XCH₄, its precision as ``√R_retr``, the
    surface pressure for ``N_dry`` and the column averaging kernel of
    ``surface_layer``. Pixels failing ``qa_min``, outside ``bbox`` or with
    a NaN in any used field are dropped.

    Args:
        path: ``S5P_*_L2__CH4____*.nc`` granule.
        frame: Analysis frame the receptors are placed in.
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        background_ppb: Background XCH₄; the median of the kept pixels
            when ``None``.
        qa_min: Minimum ``qa_value`` (the product guide recommends 0.5).
        surface_layer: Index of the near-surface retrieval layer; layers
            run top of atmosphere first, so the default is ``-1``.
        clock: Sets ``Instrument.observation_time`` as seconds since its
            ``t0``; zero when ``None``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        ValueError: If no pixel survives the screening.
    """
    xrreader, dry_air_column = _readers()
    ds = xrreader.open_tropomi_ch4_l2(path, qa_min=qa_min)
    ds = dry_air_column(ds, surface_pressure="sp")
    xch4 = ds["xch4_bias_corrected"]
    pixel_dims = list(xch4.dims)
    ak = ds["column_averaging_kernel"].isel(layer=surface_layer)
    fields = {
        "xch4": xch4,
        "sigma": ds["xch4_precision"],
        "ak": ak,
        "n_dry": ds["dry_air_column"],
        "lat": ds["lat"],
        "lon": ds["lon"],
    }
    pixels = {k: v.transpose(*pixel_dims).values.ravel() for k, v in fields.items()}
    keep = _screen(pixels, bbox)
    p = {k: v[keep] for k, v in pixels.items()}
    background = (
        float(np.median(p["xch4"])) if background_ppb is None else background_ppb
    )
    enhancement = ppb_to_column_mass(p["xch4"] - background, p["n_dry"])
    sigma = ppb_to_column_mass(p["sigma"], p["n_dry"])
    when = _scene_time(ds)
    return _observation(name, frame, p, enhancement, sigma, p["ak"], when, clock)


def load_emit_l2b(
    path: str | Path,
    *,
    frame: LocalFrame,
    bbox: BBox | None = None,
    glt_path: str | Path | None = None,
    retrieval_sigma_ppm_m: float | None = None,
    pressure_pa: float = P_STANDARD,
    temperature_k: float = T_STANDARD,
    clock: TimeAxis | None = None,
    name: str = "EMIT",
) -> ColumnObservation:
    """An EMIT L2B CH₄ enhancement scene as an instrument and its enhancement.

    The enhancement (ppm·m) is already above background. Its uncertainty
    comes from ``ch4_uncertainty`` when the file has it, else from
    ``retrieval_sigma_ppm_m``. AK = 1.

    Args:
        path: NetCDF raster (see :func:`xrreader.open_emit_ch4_l2b`).
        frame: Analysis frame.
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        glt_path: Optional geometric lookup table for orthorectification.
        retrieval_sigma_ppm_m: Per-pixel 1-σ [ppm m] for files without
            ``ch4_uncertainty``.
        pressure_pa: Near-surface pressure for the ppm·m conversion.
        temperature_k: Near-surface temperature for the ppm·m conversion.
        clock: Sets ``Instrument.observation_time``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        ValueError: If neither an uncertainty field nor
            ``retrieval_sigma_ppm_m`` is available, or no pixel survives.
    """
    xrreader, _ = _readers()
    ds = xrreader.open_emit_ch4_l2b(path, glt_path=glt_path)
    if "ch4_uncertainty" in ds:
        sigma_field = ds["ch4_uncertainty"]
    elif retrieval_sigma_ppm_m is not None:
        sigma_field = xr.full_like(ds["ch4_enhancement"], retrieval_sigma_ppm_m)
    else:
        raise ValueError(
            "load_emit_l2b: the file has no `ch4_uncertainty`; pass "
            "`retrieval_sigma_ppm_m`"
        )
    return _scene_observation(
        ds,
        "ch4_enhancement",
        sigma_field,
        lambda e: ppm_m_to_column_mass(
            e, pressure_pa=pressure_pa, temperature_k=temperature_k
        ),
        name=name,
        frame=frame,
        bbox=bbox,
        clock=clock,
    )


def load_ghgsat_l2(
    path: str | Path,
    *,
    frame: LocalFrame,
    retrieval_sigma_ppb: float,
    bbox: BBox | None = None,
    background_ppb: float | None = None,
    surface_pressure_pa: float = P_STANDARD,
    clock: TimeAxis | None = None,
    name: str = "GHGSat",
) -> ColumnObservation:
    """A GHGSat per-plume XCH₄ file as an instrument and its enhancement.

    GHGSat files carry ``xch4`` [ppb] without a kernel or a per-pixel
    precision, so the precision is an argument and AK = 1. ``N_dry`` uses
    ``surface_pressure_pa``.

    Args:
        path: Per-plume NetCDF (see :func:`xrreader.open_ghgsat_ch4_l2`).
        frame: Analysis frame.
        retrieval_sigma_ppb: Per-pixel 1-σ [ppb].
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        background_ppb: Background XCH₄; the median of the kept pixels when
            ``None``.
        surface_pressure_pa: Surface pressure for ``N_dry``.
        clock: Sets ``Instrument.observation_time``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        KeyError: If the file has no ``xch4``.
        ValueError: If no pixel survives.
    """
    xrreader, dry_air_column = _readers()
    ds = xrreader.open_ghgsat_ch4_l2(path)
    if "xch4" not in ds:
        raise KeyError(f"load_ghgsat_l2: {path!s} has no `xch4`")
    n_dry = float(
        dry_air_column(xr.Dataset({"sp": surface_pressure_pa}))["dry_air_column"]
    )
    xch4 = ds["xch4"]
    valid = xch4.values[np.isfinite(xch4.values)]
    background = float(np.median(valid)) if background_ppb is None else background_ppb
    return _scene_observation(
        ds,
        "xch4",
        xr.full_like(xch4, retrieval_sigma_ppb),
        lambda x: ppb_to_column_mass(x - background, n_dry),
        name=name,
        frame=frame,
        bbox=bbox,
        clock=clock,
        sigma_to_mass=lambda s: ppb_to_column_mass(s, n_dry),
    )


def _scene_observation(
    ds: xr.Dataset,
    variable: str,
    sigma_field: xr.DataArray,
    to_mass,
    *,
    name: str,
    frame: LocalFrame,
    bbox: BBox | None,
    clock: TimeAxis | None,
    sigma_to_mass=None,
) -> ColumnObservation:
    values = ds[variable]
    dims = [d for d in values.dims if d != "time"]
    first = {"time": 0} if "time" in values.dims else {}
    pick = lambda da: da.isel(first).transpose(*dims).values.ravel()
    lat, lon = ds["lat"], ds["lon"]
    pixels = {
        "value": pick(values),
        "sigma": pick(sigma_field),
        "lat": lat.transpose(*[d for d in dims if d in lat.dims]).values.ravel(),
        "lon": lon.transpose(*[d for d in dims if d in lon.dims]).values.ravel(),
    }
    keep = _screen(pixels, bbox)
    p = {k: v[keep] for k, v in pixels.items()}
    sigma = (sigma_to_mass or to_mass)(p["sigma"])
    when = _scene_time(ds)
    return _observation(
        name,
        frame,
        p,
        to_mass(p["value"]),
        sigma,
        np.ones_like(p["value"]),
        when,
        clock,
    )


def _screen(pixels: dict[str, np.ndarray], bbox: BBox | None) -> np.ndarray:
    keep = np.ones(pixels["lat"].shape, dtype=bool)
    for values in pixels.values():
        keep &= np.isfinite(values)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        lat, lon = pixels["lat"], pixels["lon"]
        keep &= (
            (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        )
    if not keep.any():
        raise ValueError("no pixel survives the QA, NaN and bbox screening")
    return keep


def _scene_time(ds: xr.Dataset) -> np.datetime64:
    """The middle scanline time of a swath, else the scene's timestamp."""
    if "scanline_time" in ds.coords:
        stamps = np.sort(ds["scanline_time"].values.ravel())
        return stamps[stamps.size // 2]
    if "time" in ds.coords:
        return np.asarray(ds["time"].values).ravel()[0]
    return np.datetime64("NaT")


def _observation(
    name: str,
    frame: LocalFrame,
    p: dict[str, np.ndarray],
    enhancement: np.ndarray,
    sigma: np.ndarray,
    ak: np.ndarray,
    when: np.datetime64,
    clock: TimeAxis | None,
) -> ColumnObservation:
    x, y = frame.to_xy(p["lat"], p["lon"])
    seconds = 0.0 if clock is None or np.isnat(when) else float(clock.to_seconds(when))
    instrument = Instrument(
        name=name,
        receptors=jnp.asarray(np.column_stack([x, y])),
        averaging_kernel=jnp.asarray(ak),
        retrieval_variance=jnp.asarray(sigma**2),
        observation_time=seconds,
    )
    return ColumnObservation(
        instrument=instrument, enhancement=jnp.asarray(enhancement), time=when
    )
