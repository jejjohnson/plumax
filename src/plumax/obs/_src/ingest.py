"""L2 methane products → ``coupled.Instrument`` plus the observed enhancement.

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

from collections.abc import Callable
from pathlib import Path

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Float, Shaped

from plumax._src.errors import (
    MissingExtraError,
    MissingVariableError,
    ScreeningError,
)
from plumax.coupled import Instrument
from plumax.met._src.frames import LocalFrame
from plumax.met._src.time import TimeAxis


#: Molar mass of CH₄, M [kg mol⁻¹].
M_CH4 = 0.0160425
#: Avogadro's number N_A [mol⁻¹].
AVOGADRO = 6.02214076e23
#: Boltzmann constant k_B [J K⁻¹].
BOLTZMANN = 1.380649e-23
#: Standard surface pressure [Pa] for products that report none.
P_STANDARD = 101_325.0
#: Near-surface temperature [K] for the ppm·m conversion when none is given.
T_STANDARD = 288.15

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
    try:
        import xrreader
        import xrtoolz.atm.gas.ch4
    except ImportError as err:
        raise MissingExtraError(
            "the L2 loaders read through xrreader / xrtoolz; install them with "
            "`pip install 'plumax[io]'`"
        ) from err
    return xrreader, xrtoolz.atm.gas.ch4.dry_air_column


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
        >>> dry_air = 101_325.0 * 6.02214076e23 / (9.80665 * 0.0289644)
        >>> bool(np.isclose(ppb_to_column_mass(1.0, dry_air), 5.72e-6, rtol=1e-3))
        True

        Linear in Δχ, elementwise over pixels:

        >>> ppb_to_column_mass(np.array([0.0, 10.0, -10.0]), dry_air).shape
        (3,)
    """
    # mol of CH₄ per molecule of air per ppb: 10⁻⁹ / N_A
    per_molecule = 1.0e-9 / AVOGADRO
    # ΔΩ = Δχ · 10⁻⁹ · N_dry · M / N_A
    return np.asarray(delta_ppb * per_molecule * dry_air_molecules * M_CH4)


def ppm_m_to_column_mass(
    enhancement_ppm_m: Float[np.ndarray, "*#pixel"] | float,
    *,
    pressure_pa: float,
    temperature_k: float,
) -> Float[np.ndarray, " ..."]:
    """Column-mass enhancement of a path-integrated enhancement.

    ΔΩ = E · 10⁻⁶ · (p / (k_B T)) · M / N_A.

    Args:
        enhancement_ppm_m: Path-integrated enhancement E [ppm m].
        pressure_pa: Layer pressure p [Pa].
        temperature_k: Layer temperature T [K].

    Returns:
        ΔΩ [kg m⁻²].

    Examples:
        1000 ppm·m at 1013.25 hPa and 288.15 K: 42.3 mol m⁻³ · 1 mm · M:

        >>> import numpy as np
        >>> mass = ppm_m_to_column_mass(
        ...     1000.0, pressure_pa=101_325.0, temperature_k=288.15
        ... )
        >>> bool(np.isclose(mass, 42.29e-3 * 0.0160425, rtol=1e-3))
        True

        Thinner air at altitude carries less mass for the same E:

        >>> high = ppm_m_to_column_mass(
        ...     1000.0, pressure_pa=70_000.0, temperature_k=288.15
        ... )
        >>> bool(high < mass)
        True
    """
    # n = p / (k_B T)  [molecules m⁻³]
    number_density = pressure_pa / (BOLTZMANN * temperature_k)
    # ΔΩ = E · 10⁻⁶ · n · M / N_A
    return np.asarray(enhancement_ppm_m * 1.0e-6 * number_density * M_CH4 / AVOGADRO)


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
        1. Open the granule with xrreader (QA screening at ``qa_min``) and add
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
    xrreader, dry_air_column = _readers()

    # --- read ---
    ds = xrreader.open_tropomi_ch4_l2(path, qa_min=qa_min)
    ds = dry_air_column(ds, surface_pressure="sp")
    xch4 = ds["xch4_bias_corrected"]
    pixel_dims = list(xch4.dims)
    kernel = ds["column_averaging_kernel"].isel(layer=surface_layer)
    fields = {
        "xch4": xch4,
        "sigma": ds["xch4_precision"],
        "kernel": kernel,
        "dry_air": ds["dry_air_column"],
        "lat": ds["lat"],
        "lon": ds["lon"],
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
    delta_ppb = kept["xch4"] - background_ppb
    enhancement = ppb_to_column_mass(delta_ppb, kept["dry_air"])
    sigma = ppb_to_column_mass(kept["sigma"], kept["dry_air"])
    when = _scene_time(ds)
    return _observation(
        name, frame, kept, enhancement, sigma, kept["kernel"], when, clock
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
        path: NetCDF raster (see :func:`xrreader.open_emit_ch4_l2b`).
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
    xrreader, _ = _readers()
    ds = xrreader.open_emit_ch4_l2b(path, glt_path=glt_path)
    if "ch4_uncertainty" in ds:
        sigma_field = ds["ch4_uncertainty"]
    elif retrieval_sigma_ppm_m is not None:
        sigma_field = xr.full_like(ds["ch4_enhancement"], retrieval_sigma_ppm_m)
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
        ds,
        "ch4_enhancement",
        sigma_field,
        to_mass,
        sigma_to_mass=to_mass,
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
        path: Per-plume NetCDF (see :func:`xrreader.open_ghgsat_ch4_l2`).
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
    xrreader, dry_air_column = _readers()
    ds = xrreader.open_ghgsat_ch4_l2(path)
    if "xch4" not in ds:
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
        delta_ppb = kept - background
        return ppb_to_column_mass(delta_ppb, dry_air)

    def sigma_to_mass(kept: Float[np.ndarray, " pixel"]) -> Float[np.ndarray, " pixel"]:
        return ppb_to_column_mass(kept, dry_air)

    return _scene_observation(
        ds,
        "xch4",
        xr.full_like(ds["xch4"], retrieval_sigma_ppb),
        to_mass,
        sigma_to_mass=sigma_to_mass,
        name=name,
        frame=frame,
        bbox=bbox,
        clock=clock,
    )


def _flatten(values: Shaped[np.ndarray, " ..."]) -> Shaped[np.ndarray, " pixel"]:
    return einx.id("... -> (...)", values)


def _scene_observation(
    ds: xr.Dataset,
    variable: str,
    sigma_field: xr.DataArray,
    to_mass: ToMass,
    *,
    sigma_to_mass: ToMass,
    name: str,
    frame: LocalFrame,
    bbox: BBox | None,
    clock: TimeAxis | None,
) -> ColumnObservation:
    values = ds[variable]
    raster_dims = [dim for dim in values.dims if dim != "time"]

    def pixels_of(field: xr.DataArray) -> Shaped[np.ndarray, " pixel"]:
        # a scene file holds one time step
        if "time" in field.dims:
            field = field.isel(time=0)
        dims = [dim for dim in raster_dims if dim in field.dims]
        return _flatten(field.transpose(*dims).values)

    pixels = {
        "value": pixels_of(values),
        "sigma": pixels_of(sigma_field),
        "lat": pixels_of(ds["lat"]),
        "lon": pixels_of(ds["lon"]),
    }
    kept = _screen(pixels, bbox)
    enhancement = to_mass(kept["value"])
    sigma = sigma_to_mass(kept["sigma"])
    unit_kernel = np.ones_like(kept["value"])
    when = _scene_time(ds)
    return _observation(name, frame, kept, enhancement, sigma, unit_kernel, when, clock)


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


def _scene_time(ds: xr.Dataset) -> np.datetime64:
    """The middle scanline time of a swath, else the scene's timestamp."""
    if "scanline_time" in ds.coords:
        stamps = np.sort(_flatten(ds["scanline_time"].values))
        return stamps[stamps.size // 2]
    if "time" in ds.coords:
        return _flatten(np.asarray(ds["time"].values))[0]
    return np.datetime64("NaT")


def _observation(
    name: str,
    frame: LocalFrame,
    kept: dict[str, Float[np.ndarray, " kept"]],
    enhancement: Float[np.ndarray, " kept"],
    sigma: Float[np.ndarray, " kept"],
    kernel: Float[np.ndarray, " kept"],
    when: np.datetime64,
    clock: TimeAxis | None,
) -> ColumnObservation:
    x, y = frame.to_xy(kept["lat"], kept["lon"])
    receptors = einx.id("p, p -> p (1 + 1)", x, y)
    no_clock = clock is None or np.isnat(when)
    seconds = 0.0 if no_clock else float(clock.to_seconds(when))
    # R_retr = σ_Ω²
    retrieval_variance = sigma**2
    instrument = Instrument(
        name=name,
        receptors=jnp.asarray(receptors),
        averaging_kernel=jnp.asarray(kernel),
        retrieval_variance=jnp.asarray(retrieval_variance),
        observation_time=seconds,
    )
    return ColumnObservation(
        instrument=instrument, enhancement=jnp.asarray(enhancement), time=when
    )
