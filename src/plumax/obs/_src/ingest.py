"""L2 methane products → ``coupled.Instrument`` plus the observed enhancement.

``coupled.Instrument`` carries receptors in analysis metres, a per-receptor
scalar averaging kernel and the retrieval variance; ``fuse_observations``
takes those instruments with an observed column-mass enhancement each
[kg m⁻²], the unit ``coupled.column_response`` integrates the plume to.
These loaders build both from a product file:

1. read the file with xrproducts's opener (QA screening included),
2. place pixel centres in the analysis frame with ``LocalFrame.to_xy``,
3. convert the product's quantity to a column-mass enhancement ``ΔΩ``,
4. reduce the averaging kernel to the scalar the plume sees.

Unit conversions (``M`` the CH₄ molar mass, ``N_A`` Avogadro's number):

- XCH₄ (TROPOMI, GHGSat) in ppb above a background ``b``:
  ``ΔΩ = (XCH₄ − b) · 10⁻⁹ · N_dry · M / N_A``, with ``N_dry`` the dry-air
  column [molecules m⁻²] from :func:`xrops.atm.gas.ch4.dry_air_column`
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

from collections.abc import Callable
from pathlib import Path

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from geophysix.columns import column_from_mixing_ratio
from jaxtyping import Array, Float, Shaped
from pipekit_ops import import_optional

from plumax._src.constants import (
    AVOGADRO,
    BOLTZMANN as BOLTZMANN,
    M_CH4,
    P_STANDARD,
    T_STANDARD,
)
from plumax._src.errors import (
    MissingExtraError,
    MissingVariableError,
    ScreeningError,
)
from plumax._src.units import ppm_m_to_column_mass
from plumax.coupled import Instrument
from plumax.met._src.frames import LocalFrame
from plumax.met._src.time import TimeAxis


#: (lon_min, lat_min, lon_max, lat_max) [degrees].
BBox = tuple[float, float, float, float]
#: Kept pixel values (ppb or ppm·m) → column-mass enhancement [kg m⁻²].
ToMass = Callable[[Float[np.ndarray, " pixel"]], Float[np.ndarray, " pixel"]]


class ColumnObservation(eqx.Module):
    """One product's instrument and its observed enhancement.

    Attributes:
        instrument: Receptors, scalar averaging kernel and retrieval variance,
            ready for ``CoupledForward``.
        enhancement: Observed column-mass enhancement ΔΩ [kg m⁻²], aligned
            with ``instrument.receptors``.
        time: Observation timestamp (UTC); ``NaT`` when the file has none.
    """

    instrument: Instrument
    enhancement: Float[Array, " pixel"]
    time: np.datetime64 = eqx.field(static=True)


def _readers():
    xrops_ch4 = import_optional(
        "xrops.atm.gas.ch4",
        install="plumax[io]",
        needed_by="the L2 reader",
        error=MissingExtraError,
    )
    xrproducts = import_optional(
        "xrproducts",
        install="plumax[io]",
        needed_by="the L2 reader",
        error=MissingExtraError,
    )
    return xrproducts, xrops_ch4.dry_air_column


def ppb_to_column_mass(
    delta_ppb: Float[np.ndarray, "*#pixel"] | float,
    dry_air_molecules: Float[np.ndarray, "*#pixel"] | float,
) -> Float[np.ndarray, " ..."]:
    """Column-mass enhancement of a mixing-ratio enhancement.

    ΔΩ = Δχ · 10⁻⁹ · N_dry · M / N_A.

    Args:
        delta_ppb: Mixing-ratio enhancement Δχ [ppb].
        dry_air_molecules: Dry-air column N_dry [molecules m⁻²].

    Returns:
        ΔΩ [kg m⁻²].

    Examples:
        1 ppb over a 1013.25 hPa dry column is about 5.7 mg m⁻²:

        >>> import numpy as np
        >>> from geophysix.columns import dry_air_column
        >>> from geophysix.constants import STANDARD
        >>> dry_air = dry_air_column(101_325.0) * STANDARD.avogadro
        >>> bool(np.isclose(ppb_to_column_mass(1.0, dry_air), 5.72e-6, rtol=1e-3))
        True

        Linear in Δχ, elementwise over pixels:

        >>> ppb_to_column_mass(np.array([0.0, 10.0, -10.0]), dry_air).shape
        (3,)
    """
    # dry-air column in mol m⁻²: N_dry / N_A
    dry_air_moles = np.asarray(dry_air_molecules) / AVOGADRO
    # ΔΩ = M · Δχ · 10⁻⁹ · N_dry / N_A
    return np.asarray(
        M_CH4 * column_from_mixing_ratio(np.asarray(delta_ppb) * 1.0e-9, dry_air_moles)
    )


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

    ΔΩ = (XCH₄ − b) · 10⁻⁹ · N_dry(p_s) · M / N_A,  σ_Ω = σ_χ · 10⁻⁹ · N_dry · M / N_A,
    A = A_{surface layer}.

    Algorithm:
        1. Open the granule with xrproducts (QA screening at ``qa_min``) and add
           the dry-air column from the surface pressure.
        2. Flatten the swath to pixels; drop NaNs and pixels outside ``bbox``.
        3. Take the background (default: median of the kept XCH₄), convert
           XCH₄ and its precision to column mass, and pick the near-surface
           kernel layer.
        4. Place the kept pixel centres in the frame.

    Args:
        path: ``S5P_*_L2__CH4____*.nc`` granule.
        frame: Analysis frame the receptors are placed in.
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        background_ppb: Background XCH₄ b [ppb]; the median of the kept
            pixels when ``None``.
        qa_min: Minimum ``qa_value`` (the product guide recommends 0.5).
        surface_layer: Index of the near-surface retrieval layer; layers run
            top of atmosphere first, so the default is ``-1``.
        clock: Sets ``Instrument.observation_time`` as seconds since its
            ``t0``; zero when ``None``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        MissingExtraError: If the ``io`` extra is not installed.
        ScreeningError: If no pixel survives the screening.
    """
    xrproducts, dry_air_column = _readers()

    # --- read ---
    dataset = xrproducts.open_tropomi_ch4_l2(path, qa_min=qa_min)
    dataset = dry_air_column(dataset, surface_pressure="sp")
    xch4 = dataset["xch4_bias_corrected"]
    pixel_dims = list(xch4.dims)
    kernel = dataset["column_averaging_kernel"].isel(layer=surface_layer)
    fields = {
        "xch4": xch4,
        "sigma": dataset["xch4_precision"],
        "kernel": kernel,
        "dry_air": dataset["dry_air_column"],
        "lat": dataset["lat"],
        "lon": dataset["lon"],
    }

    # --- screen ---
    pixels = {
        key: _flatten(field.transpose(*pixel_dims).values)
        for key, field in fields.items()
    }
    kept = _screen(pixels, bbox)

    # --- convert ---
    if background_ppb is None:
        background_ppb = float(np.median(kept["xch4"]))
    # Δχ = XCH₄ − b
    mixing_ratio_enhancement = kept["xch4"] - background_ppb
    enhancement = ppb_to_column_mass(mixing_ratio_enhancement, kept["dry_air"])
    # σ_Ω = σ_χ · 10⁻⁹ · N_dry · M / N_A
    obs_noise_std = ppb_to_column_mass(kept["sigma"], kept["dry_air"])
    when = _scene_time(dataset)
    return _observation(
        name, frame, kept, enhancement, obs_noise_std, kept["kernel"], when, clock
    )


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

    ΔΩ = E · 10⁻⁶ · (p / (k_B T)) · M / N_A,  A = 1.

    The enhancement E (ppm·m) is already above background. Its uncertainty
    comes from ``ch4_uncertainty`` when the file has it, else from
    ``retrieval_sigma_ppm_m``.

    Args:
        path: NetCDF raster (see :func:`xrproducts.open_emit_ch4_l2b`).
        frame: Analysis frame.
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        glt_path: Optional geometric lookup table for orthorectification.
        retrieval_sigma_ppm_m: Per-pixel 1-σ [ppm m] for files without
            ``ch4_uncertainty``.
        pressure_pa: Near-surface pressure p for the ppm·m conversion.
        temperature_k: Near-surface temperature T for the ppm·m conversion.
        clock: Sets ``Instrument.observation_time``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        MissingExtraError: If the ``io`` extra is not installed.
        MissingVariableError: If neither an uncertainty field nor
            ``retrieval_sigma_ppm_m`` is available.
        ScreeningError: If no pixel survives the screening.
    """
    xrproducts, _ = _readers()
    dataset = xrproducts.open_emit_ch4_l2b(path, glt_path=glt_path)
    if "ch4_uncertainty" in dataset:
        noise_field = dataset["ch4_uncertainty"]
    elif retrieval_sigma_ppm_m is not None:
        noise_field = xr.full_like(dataset["ch4_enhancement"], retrieval_sigma_ppm_m)
    else:
        raise MissingVariableError(
            "load_emit_l2b: the file has no `ch4_uncertainty`; pass "
            "`retrieval_sigma_ppm_m`"
        )

    def to_mass(kept: Float[np.ndarray, " pixel"]) -> Float[np.ndarray, " pixel"]:
        return ppm_m_to_column_mass(
            kept, pressure_pa=pressure_pa, temperature_k=temperature_k
        )

    return _scene_observation(
        dataset,
        "ch4_enhancement",
        noise_field,
        to_mass,
        noise_to_mass=to_mass,
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

    ΔΩ = (XCH₄ − b) · 10⁻⁹ · N_dry(p_s) · M / N_A,  A = 1.

    GHGSat files carry ``xch4`` [ppb] without a kernel or a per-pixel
    precision, so the precision is an argument.

    Args:
        path: Per-plume NetCDF (see :func:`xrproducts.open_ghgsat_ch4_l2`).
        frame: Analysis frame.
        retrieval_sigma_ppb: Per-pixel 1-σ [ppb].
        bbox: Optional ``(lon_min, lat_min, lon_max, lat_max)`` crop.
        background_ppb: Background XCH₄ b [ppb]; the median of the kept pixels
            when ``None``.
        surface_pressure_pa: Surface pressure p_s for N_dry.
        clock: Sets ``Instrument.observation_time``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`.

    Raises:
        MissingExtraError: If the ``io`` extra is not installed.
        MissingVariableError: If the file has no ``xch4``.
        ScreeningError: If no pixel survives the screening.
    """
    xrproducts, dry_air_column = _readers()
    dataset = xrproducts.open_ghgsat_ch4_l2(path)
    if "xch4" not in dataset:
        raise MissingVariableError(f"load_ghgsat_l2: {path!s} has no `xch4`")
    surface = xr.Dataset({"sp": surface_pressure_pa})
    dry_air = float(dry_air_column(surface)["dry_air_column"])

    def to_mass(kept: Float[np.ndarray, " pixel"]) -> Float[np.ndarray, " pixel"]:
        # `kept` is already screened, so the default background is the median
        # of the pixels that stay (as for TROPOMI), not of the whole file
        background = (
            float(np.median(kept)) if background_ppb is None else background_ppb
        )
        # Δχ = XCH₄ − b
        mixing_ratio_enhancement = kept - background
        return ppb_to_column_mass(mixing_ratio_enhancement, dry_air)

    def noise_to_mass(kept: Float[np.ndarray, " pixel"]) -> Float[np.ndarray, " pixel"]:
        return ppb_to_column_mass(kept, dry_air)

    return _scene_observation(
        dataset,
        "xch4",
        xr.full_like(dataset["xch4"], retrieval_sigma_ppb),
        to_mass,
        noise_to_mass=noise_to_mass,
        name=name,
        frame=frame,
        bbox=bbox,
        clock=clock,
    )


def _flatten(values: Shaped[np.ndarray, " ..."]) -> Shaped[np.ndarray, " pixel"]:
    return einx.id("... -> (...)", values)


def _scene_observation(
    dataset: xr.Dataset,
    variable: str,
    noise_field: xr.DataArray,
    to_mass: ToMass,
    *,
    noise_to_mass: ToMass,
    name: str,
    frame: LocalFrame,
    bbox: BBox | None,
    clock: TimeAxis | None,
) -> ColumnObservation:
    values = dataset[variable]
    raster_dims = [dim for dim in values.dims if dim != "time"]

    def pixels_of(field: xr.DataArray) -> Shaped[np.ndarray, " pixel"]:
        # a scene file holds one time step
        if "time" in field.dims:
            field = field.isel(time=0)
        dims = [dim for dim in raster_dims if dim in field.dims]
        return _flatten(field.transpose(*dims).values)

    pixels = {
        "value": pixels_of(values),
        "sigma": pixels_of(noise_field),
        "lat": pixels_of(dataset["lat"]),
        "lon": pixels_of(dataset["lon"]),
    }
    kept = _screen(pixels, bbox)
    enhancement = to_mass(kept["value"])
    # σ_Ω from the screened noise field
    obs_noise_std = noise_to_mass(kept["sigma"])
    unit_kernel = np.ones_like(kept["value"])
    when = _scene_time(dataset)
    return _observation(
        name, frame, kept, enhancement, obs_noise_std, unit_kernel, when, clock
    )


def _screen(
    pixels: dict[str, Float[np.ndarray, " pixel"]], bbox: BBox | None
) -> dict[str, Float[np.ndarray, " kept"]]:
    """The pixels with every field finite and, given a ``bbox``, inside it."""
    keep = np.ones(pixels["lat"].shape, dtype=bool)
    for values in pixels.values():
        keep &= np.isfinite(values)
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        lat, lon = pixels["lat"], pixels["lon"]
        inside_lon = (lon >= lon_min) & (lon <= lon_max)
        inside_lat = (lat >= lat_min) & (lat <= lat_max)
        keep &= inside_lon & inside_lat
    if not keep.any():
        raise ScreeningError("no pixel survives the QA, NaN and bbox screening")
    return {key: values[keep] for key, values in pixels.items()}


def _scene_time(dataset: xr.Dataset) -> np.datetime64:
    """The middle scanline time of a swath, else the scene's timestamp."""
    if "scanline_time" in dataset.coords:
        stamps = np.sort(_flatten(dataset["scanline_time"].values))
        return stamps[stamps.size // 2]
    if "time" in dataset.coords:
        return _flatten(np.asarray(dataset["time"].values))[0]
    return np.datetime64("NaT")


def _observation_time(clock: TimeAxis | None, when: np.datetime64) -> float:
    """Seconds of ``when`` on ``clock``, or ``0.0`` without a clock or for NaT.

    Args:
        clock: The time axis, or ``None``.
        when: The observation time.

    Returns:
        ``clock.to_seconds(when)`` as a float, ``0.0`` when there is no clock
        or ``when`` is NaT.
    """
    no_clock = clock is None or np.isnat(when)
    return 0.0 if no_clock else float(clock.to_seconds(when))


def _column_observation(
    name: str,
    receptors: Float[np.ndarray, "kept 2"],
    kernel: Float[np.ndarray, " kept"],
    retrieval_variance: Float[np.ndarray, " kept"] | float,
    enhancement: Float[np.ndarray, " kept"],
    when: np.datetime64,
    clock: TimeAxis | None,
    *,
    flags: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> ColumnObservation:
    """Build the :class:`ColumnObservation` of one scene's kept pixels.

    Args:
        name: Instrument name.
        receptors: Receptor ``(x, y)`` per kept pixel.
        kernel: Averaging kernel per kept pixel.
        retrieval_variance: Retrieval variance R_retr.
        enhancement: Observed column enhancement per kept pixel.
        when: Observation time.
        clock: Time axis for ``observation_time``, or ``None``.
        flags: Optional per-pixel quality flags.
        mask: Optional per-pixel observation mask.

    Returns:
        The observation, with the instrument's ``observation_time`` set by
        :func:`_observation_time`.
    """
    instrument = Instrument(
        name=name,
        receptors=jnp.asarray(receptors),
        averaging_kernel=jnp.asarray(kernel),
        retrieval_variance=jnp.asarray(retrieval_variance),
        observation_time=_observation_time(clock, when),
        flags=None if flags is None else jnp.asarray(flags),
        mask=None if mask is None else jnp.asarray(mask),
    )
    return ColumnObservation(
        instrument=instrument, enhancement=jnp.asarray(enhancement), time=when
    )


def _observation(
    name: str,
    frame: LocalFrame,
    kept: dict[str, Float[np.ndarray, " kept"]],
    enhancement: Float[np.ndarray, " kept"],
    obs_noise_std: Float[np.ndarray, " kept"],
    kernel: Float[np.ndarray, " kept"],
    when: np.datetime64,
    clock: TimeAxis | None,
) -> ColumnObservation:
    easting, northing = frame.to_xy(kept["lat"], kept["lon"])
    receptors = einx.id("p, p -> p (1 + 1)", easting, northing)
    # R_retr = σ_Ω²
    retrieval_variance = obs_noise_std**2
    return _column_observation(
        name, receptors, kernel, retrieval_variance, enhancement, when, clock
    )
