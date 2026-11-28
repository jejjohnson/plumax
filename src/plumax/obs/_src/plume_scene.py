"""One plume from a matched-filter product, ready for the image fit.

The image fit of ``gauss_plume.ColumnPlume`` sees a window of pixels around
one plume through the observation model

    yₙ = Aₙ ΔΩₙ + εₙ,  εₙ ~ 𝒩(0, σₙ²),

with yₙ the column-mass enhancement [kg m⁻²], Aₙ the matched filter's
sensitivity (the averaging kernel), σₙ the per-pixel noise and ωₙ a mask.
:func:`load_emit_plume_complex` builds that window from an operational EMIT
plume product (enhancement, sensitivity, uncertainty, the L2A quality mask
and the plume-complex outline), and :func:`matched_filter_scene` builds the
observation from radtrax's own score, noise and sensitivity maps.

The loaders run on the host: they read files once and hand back arrays.
Nothing is ever divided by the sensitivity, which can be near zero on dark
pixels; the model is multiplied by it instead.

The procedure follows JPL's SPIIRE (``load_MF_from_plume_ID`` in
``load_data.py``, ``emit-sds/SPIIRE`` @ ``ae9a54d``).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Bool, Float, Int
from pipekit_ops import import_optional

from plumax._src.errors import (
    GridMismatchError,
    MissingExtraError,
    MissingVariableError,
    ParameterError,
    ScreeningError,
    UnitsError,
)
from plumax._src.geometry import direction_from_centroid
from plumax._src.tracing import require_positive
from plumax.coupled import Instrument, QualityFlag
from plumax.met._src.frames import LocalFrame
from plumax.met._src.time import TimeAxis
from plumax.obs._src.ingest import (
    M_CH4,
    P_STANDARD,
    T_STANDARD,
    ColumnObservation,
    _column_observation,
    _readers,
    _scene_time,
    ppm_m_to_column_mass,
)


# Shapes: N_Ωy pixels kept in the window (dim "pixel");
#         V outline vertices (dim "vertex");
#         N_row × N_col image pixels (dims "row", "col")

#: EMIT's raw-swath dims, renamed to the raster's.
_SWATH_DIMS = {"downtrack": "y", "crosstrack": "x"}
#: The L2A mask flags SPIIRE masks out: clouds and surface water.
_L2A_TARGETS = ("cloud", "cirrus", "water")
#: Dilation of the saturation mask [pixels], SPIIRE's.
_SATURATION_DILATION = 10


class PlumeScene(eqx.Module):
    """One plume: its observation, its outline and where the fit starts.

    yₙ = Aₙ ΔΩₙ + εₙ, εₙ ~ 𝒩(0, σₙ²), with yₙ, Aₙ, σₙ² and the mask ωₙ
    held by ``observation``.

    Attributes:
        observation: Instrument (receptors, Aₙ, σₙ², ωₙ) and enhancement yₙ.
        origin: Pseudo-origin s₀ in the local frame [m].
        outline: Plume outline in the local frame [m], closed.
        in_plume: Whether each kept pixel lies inside the outline.
        direction: Starting direction φ₀ [rad], origin to plume centroid,
            counter-clockwise from the frame's x axis.
        pixel_index: ``(row, col)`` of each kept pixel in the window, to put
            the pixels back on the image.
        plume_id: Identifier of the plume in its product.
        pixel_size: Pixel side g [m].
        window: ``(row_start, row_stop, col_start, col_stop)`` of the window
            in the scene's raster.
    """

    observation: ColumnObservation
    origin: Float[Array, " axis"]
    outline: Float[Array, "vertex axis"]
    in_plume: Bool[Array, " pixel"]
    direction: Float[Array, ""]
    pixel_index: Int[Array, "pixel 2"]
    plume_id: str = eqx.field(static=True)
    pixel_size: float = eqx.field(static=True)
    window: tuple[int, int, int, int] = eqx.field(static=True)

    @property
    def window_shape(self) -> tuple[int, int]:
        """``(N_row, N_col)`` of the window."""
        row_start, row_stop, col_start, col_stop = self.window
        return row_stop - row_start, col_stop - col_start

    @property
    def instrument(self) -> Instrument:
        """The instrument of the observation."""
        return self.observation.instrument

    @property
    def enhancement(self) -> Float[Array, " pixel"]:
        """Observed column-mass enhancement yₙ [kg m⁻²]."""
        return self.observation.enhancement


def load_emit_plume_complex(
    enhancement_path: str | Path,
    outlines_path: str | Path,
    *,
    plume_id: str,
    frame: LocalFrame,
    sensitivity_path: str | Path | None = None,
    uncertainty_path: str | Path | None = None,
    quality_path: str | Path | None = None,
    glt_path: str | Path | None = None,
    margin_px: int = 10,
    uncertainty_of: Literal["truth", "score"] = "truth",
    retrieval_sigma_ppm_m: float | None = None,
    pressure_pa: float = P_STANDARD,
    temperature_k: float = T_STANDARD,
    clock: TimeAxis | None = None,
    name: str = "EMIT",
) -> PlumeScene:
    """One catalogued plume of an EMIT scene as a ``PlumeScene``.

    yₙ = κ Eₙ, Aₙ from the sensitivity raster, and σₙ = κ Aₙ uₙ
    (``"truth"``) or σₙ = κ uₙ (``"score"``), with κ the ppm m to kg m⁻²
    factor of :func:`ppm_m_to_column_mass`.

    Algorithm:
        1. Open the enhancement with its companions, and the outlines.
        2. Select the plume; crop to its bounding box plus the margin.
        3. Convert the enhancement and its noise to column mass.
        4. Build the mask from the fill values, the sensitivity and the
           quality raster.
        5. Project receptors, origin and outline into the local frame.
        6. Take the pixel side and the starting direction from them.

    The stored enhancement is the raw matched-filter score, so it is
    converted but never divided by the sensitivity. A pixel is masked
    (ωₙ = 0) when its enhancement, sensitivity or uncertainty is missing,
    its sensitivity is not positive, or the quality raster flags it as
    cloud, cirrus, water (``geoops.qa.EMITL2AMask``) or saturated
    (``geoops.qa.EMITBandMask``, dilated by 10 pixels). A masked pixel stays
    in the arrays with yₙ = 0, Aₙ = 1 and the median σₙ² of the kept pixels,
    so the window keeps its pixels for plotting; a pixel with no location,
    or with none of the three fields, is dropped. The DAAC plume files carry
    no pseudo-origin; the brightest in-plume pixel stands in for it then.

    Args:
        enhancement_path: Enhancement raster E [ppm m] (NetCDF, see
            ``xrproducts.open_emit_ch4_l2b``).
        outlines_path: Plume-complex GeoJSON.
        plume_id: Identifier of the plume to load.
        frame: Local metre frame of the analysis.
        sensitivity_path: Sensitivity raster A; ``None`` sets A = 1.
        uncertainty_path: Uncertainty raster u [ppm m].
        quality_path: EMIT L2A MASK NetCDF (``mask`` and, if present,
            ``band_mask``) in swath geometry; flagged pixels get ω = 0.
            Needs the ``geo`` extra.
        glt_path: Geometric lookup table, for swath-geometry rasters; the
            quality flags are decoded in swath geometry and go through it too.
        margin_px: Margin around the outline's bounding box, in pixels.
        uncertainty_of: Whether u is the noise of the enhancement divided
            by the sensitivity (``"truth"``) or of the raw score.
        retrieval_sigma_ppm_m: Constant u for files without an
            uncertainty raster.
        pressure_pa: Near-surface pressure p [Pa] for the conversion.
        temperature_k: Near-surface temperature T [K] for the conversion.
        clock: Sets the instrument's observation time.
        name: Instrument name.

    Returns:
        The :class:`PlumeScene`.

    Raises:
        MissingExtraError: If the ``io`` extra is not installed, or the
            ``geo`` extra when a quality raster is given.
        MissingVariableError: If ``plume_id`` is not in the file (the
            message lists the identifiers it has), or there is neither an
            uncertainty raster nor ``retrieval_sigma_ppm_m``.
        GridMismatchError: If the outline's bounding box leaves the scene,
            or the quality raster is not on the scene's grid.
        ScreeningError: If no pixel survives the mask.
        ParameterError: If ``margin_px`` is negative or ``uncertainty_of``
            is unknown.
        DomainError: If the plume's centroid is within a pixel of its origin.

    Examples:
        On the files a catalogue lists for one plume::

            scene = load_emit_plume_complex(
                "scene_CH4ENH.nc",
                "plume_complexes.json",
                plume_id="CH4_PlumeComplex-1234",
                frame=frame,
                sensitivity_path="scene_CH4SENS.nc",
                uncertainty_path="scene_CH4UNCERT.nc",
            )
            scene.instrument.averaging_kernel  # Aₙ, shape (N_Ωy,)
    """
    require_positive("margin_px", margin_px, allow_zero=True)
    if uncertainty_of not in ("truth", "score"):
        raise ParameterError(
            f"uncertainty_of must be 'truth' or 'score', got {uncertainty_of!r}"
        )
    xrproducts, _ = _readers()

    # --- read ---
    dataset = xrproducts.open_emit_ch4_l2b(
        enhancement_path,
        sensitivity_path=sensitivity_path,
        uncertainty_path=uncertainty_path,
        glt_path=glt_path,
    )
    plumes = xrproducts.open_ch4_plume_complexes(outlines_path)
    ids = [str(value) for value in plumes["plume"].values]
    if plume_id not in ids:
        raise MissingVariableError(
            f"load_emit_plume_complex: {plume_id!r} is not in {outlines_path!s}; "
            f"it has {ids}"
        )
    if "ch4_uncertainty" not in dataset and retrieval_sigma_ppm_m is None:
        raise MissingVariableError(
            "load_emit_plume_complex: no uncertainty raster; pass "
            "`uncertainty_path` or `retrieval_sigma_ppm_m`"
        )
    enhancement = _raster(dataset["ch4_enhancement"])
    lat, lon = _raster(dataset["lat"]), _raster(dataset["lon"])
    dtype = enhancement.dtype
    if "ch4_sensitivity" in dataset:
        sensitivity = _raster(dataset["ch4_sensitivity"])
    else:
        sensitivity = np.ones_like(enhancement)
    if "ch4_uncertainty" in dataset:
        uncertainty = _raster(dataset["ch4_uncertainty"])
    else:
        uncertainty = np.full_like(enhancement, retrieval_sigma_ppm_m)
    flags = np.zeros(enhancement.shape, dtype=np.int32)
    if quality_path is not None:
        flags = _quality_flags(quality_path, glt_path, enhancement.shape)

    # --- window ---
    # the outline as (lon, lat) rows
    ring = xrproducts.plume_outline(plumes, plume_id)
    rows, cols = _window(ring, lat, lon, margin_px)
    window = (slice(*rows), slice(*cols))
    fields = {
        "enhancement": enhancement[window],
        "sensitivity": sensitivity[window],
        "uncertainty": uncertainty[window],
        "lat": lat[window],
        "lon": lon[window],
        "flags": flags[window],
    }

    # --- convert ---
    # κ, ppm m → kg m⁻²
    mass_per_ppm_m = float(
        ppm_m_to_column_mass(1.0, pressure_pa=pressure_pa, temperature_k=temperature_k)
    )
    # yₙ = κ Eₙ
    column_mass = (mass_per_ppm_m * fields["enhancement"]).astype(dtype)
    if uncertainty_of == "truth":
        # σₙ = κ Aₙ uₙ
        obs_noise_std = mass_per_ppm_m * fields["sensitivity"] * fields["uncertainty"]
    else:
        # σₙ = κ uₙ
        obs_noise_std = mass_per_ppm_m * fields["uncertainty"]

    # --- mask ---
    keep, mask = _pixel_mask(
        column_mass,
        fields["sensitivity"],
        obs_noise_std,
        fields["lat"],
        fields["lon"],
    )
    mask &= fields["flags"] == 0
    kept = _kept_pixels(
        column_mass,
        fields["sensitivity"],
        obs_noise_std,
        mask,
        keep,
        fields["flags"],
    )

    # --- geometry ---
    # sₙ = (xₙ, yₙ) in the frame
    easting, northing = frame.to_xy(fields["lat"], fields["lon"])
    pixel_size = _pixel_size(easting, northing)
    receptors = einx.id("p, p -> p (1 + 1)", easting[keep], northing[keep])
    outline_x, outline_y = frame.to_xy(ring[:, 1], ring[:, 0])
    outline = einx.id("v, v -> v (1 + 1)", outline_x, outline_y)
    in_plume = _inside(receptors, outline)
    origin = _origin(plumes, plume_id, frame, receptors, kept, in_plume)
    direction = _direction(receptors, kept, in_plume, origin, pixel_size)
    row_index, col_index = np.nonzero(keep)
    pixel_index = einx.id("p, p -> p (1 + 1)", row_index, col_index)

    observation = _column_observation(
        name,
        receptors,
        kept["kernel"],
        kept["variance"],
        kept["enhancement"],
        _scene_time(dataset),
        clock,
        flags=kept["flags"],
        mask=kept["mask"].astype(dtype),
    )
    return PlumeScene(
        observation=observation,
        origin=jnp.asarray(origin),
        outline=jnp.asarray(outline),
        in_plume=jnp.asarray(in_plume),
        direction=jnp.asarray(direction),
        pixel_index=jnp.asarray(pixel_index),
        plume_id=plume_id,
        pixel_size=pixel_size,
        window=(*rows, *cols),
    )


def matched_filter_scene(
    score: Float[np.ndarray, "row col"],
    score_std: Float[np.ndarray, "row col"] | float,
    *,
    lat: Float[np.ndarray, "row col"],
    lon: Float[np.ndarray, "row col"],
    frame: LocalFrame,
    sensitivity: Float[np.ndarray, "row col"] | None = None,
    exclude: Bool[np.ndarray, "row col"] | None = None,
    unit: Literal["mol_m2", "ppm_m", "kg_m2"] = "mol_m2",
    pressure_pa: float = P_STANDARD,
    temperature_k: float = T_STANDARD,
    name: str = "EMIT",
) -> ColumnObservation:
    """A ``ColumnObservation`` from radtrax's score, noise and sensitivity maps.

    yₙ = M α̂ₙ and σₙ = M σ_α,n for a score in mol m⁻², with M the molar
    mass of methane; Aₙ is the sensitivity, or 1. A scalar ``score_std`` is
    the noise of every pixel.

    The noise is that of the raw score (radtrax's ``score_std``), so it is
    not multiplied by the sensitivity. For ``unit="ppm_m"`` the factor is κ
    of :func:`ppm_m_to_column_mass`, and for ``"kg_m2"`` it is 1. Pixels
    with a missing value or location, a non-positive sensitivity or set in
    ``exclude`` get ω = 0 (with y = 0, A = 1 and the median σ² of the kept
    pixels); pixels with no location or none of the fields are dropped.

    Args:
        score: Matched-filter score α̂ in ``unit``.
        score_std: Its standard deviation σ_α, per pixel or one for all.
        lat: Pixel-centre latitudes [degrees].
        lon: Pixel-centre longitudes [degrees].
        frame: Local metre frame of the analysis.
        sensitivity: Sensitivity A; ``None`` sets A = 1.
        exclude: Pixels to mask out, such as ``geoops.qa`` masks.
        unit: Unit of ``score`` and ``score_std``.
        pressure_pa: Near-surface pressure p [Pa], for ``"ppm_m"``.
        temperature_k: Near-surface temperature T [K], for ``"ppm_m"``.
        name: Instrument name.

    Returns:
        The :class:`ColumnObservation`, with ``time`` ``NaT``.

    Raises:
        UnitsError: If ``unit`` is not one of the three.
        ScreeningError: If no pixel survives the mask.

    Examples:
        >>> import numpy as np
        >>> from plumax.met import LocalFrame
        >>> frame = LocalFrame.at(31.95, -102.15)
        >>> lat, lon = frame.to_latlon(*np.meshgrid([0.0, 60.0], [0.0, 60.0]))
        >>> observation = matched_filter_scene(
        ...     np.full((2, 2), 0.01), 0.002, lat=lat, lon=lon, frame=frame
        ... )
        >>> observation.enhancement.tolist()  # M α̂ = 0.0160425 × 0.01
        [0.000160425, 0.000160425, 0.000160425, 0.000160425]
    """
    if unit == "mol_m2":
        # yₙ = M α̂ₙ
        factor = M_CH4
    elif unit == "ppm_m":
        # yₙ = κ α̂ₙ
        factor = float(
            ppm_m_to_column_mass(
                1.0, pressure_pa=pressure_pa, temperature_k=temperature_k
            )
        )
    elif unit == "kg_m2":
        factor = 1.0
    else:
        raise UnitsError(
            f"matched_filter_scene: unit must be 'mol_m2', 'ppm_m' or 'kg_m2', "
            f"got {unit!r}"
        )
    values = np.asarray(score)
    dtype = np.result_type(values.dtype, np.float32)
    column_mass = (factor * values).astype(dtype)
    # σₙ = M σ_α,n
    obs_noise_std = factor * np.broadcast_to(
        np.asarray(score_std, dtype=dtype), values.shape
    )
    gain = (
        np.ones_like(column_mass)
        if sensitivity is None
        else np.asarray(sensitivity, dtype=dtype)
    )
    lat, lon = np.asarray(lat), np.asarray(lon)
    keep, mask = _pixel_mask(column_mass, gain, obs_noise_std, lat, lon)
    flags = np.zeros(values.shape, dtype=np.int32)
    if exclude is not None:
        excluded = np.asarray(exclude, dtype=bool)
        flags = np.where(excluded, int(QualityFlag.CLOUD), 0).astype(np.int32)
        mask &= ~excluded
    kept = _kept_pixels(column_mass, gain, obs_noise_std, mask, keep, flags)
    easting, northing = frame.to_xy(lat, lon)
    receptors = einx.id("p, p -> p (1 + 1)", easting[keep], northing[keep])
    return _column_observation(
        name,
        receptors,
        kept["kernel"],
        kept["variance"],
        kept["enhancement"],
        np.datetime64("NaT"),
        None,
        flags=kept["flags"],
        mask=kept["mask"].astype(dtype),
    )


def _raster(field: xr.DataArray) -> np.ndarray:
    """The ``(y, x)`` values of one raster field, its one time step taken."""
    if "time" in field.dims:
        field = field.isel(time=0)
    return np.asarray(field.transpose("y", "x").values)


def _quality_flags(
    quality_path: str | Path,
    glt_path: str | Path | None,
    shape: tuple[int, ...],
) -> Int[np.ndarray, "row col"]:
    """QualityFlag bits of each pixel from the EMIT L2A MASK file.

    CLOUD where ``EMITL2AMask`` flags cloud, cirrus or water, SATURATED
    where ``EMITBandMask`` flags saturation; both decoded in swath geometry
    and then sent through the GLT when there is one.
    """
    geoops_qa = import_optional(
        "geoops.qa",
        install="plumax[geo]",
        needed_by="the quality raster",
        error=MissingExtraError,
    )
    EMITBandMask, EMITL2AMask = geoops_qa.EMITBandMask, geoops_qa.EMITL2AMask
    quality = xr.open_dataset(quality_path)
    quality = quality.rename(
        {
            swath: raster
            for swath, raster in _SWATH_DIMS.items()
            if swath in quality.dims
        }
    )
    if "mask" not in quality:
        raise MissingVariableError(f"{quality_path!s} has no `mask` variable")
    # (band, y, x), the band dim first as geoops reads it
    flag_bands = _band_first(quality["mask"])
    obscured = np.asarray(EMITL2AMask(targets=_L2A_TARGETS)(flag_bands))
    saturated = np.zeros_like(obscured)
    if "band_mask" in quality:
        packed = _band_first(quality["band_mask"]).astype(np.uint8)
        saturated = np.asarray(EMITBandMask(dilation=_SATURATION_DILATION)(packed))
    bits = np.where(obscured, int(QualityFlag.CLOUD), 0) | np.where(
        saturated, int(QualityFlag.SATURATED), 0
    )
    if glt_path is not None:
        glt = xr.open_dataset(glt_path)
        glt = glt.rename(
            {
                swath: raster
                for swath, raster in _SWATH_DIMS.items()
                if swath in glt.dims
            }
        )
        glt_x = np.asarray(glt["glt_x"].values, dtype=np.int64)
        glt_y = np.asarray(glt["glt_y"].values, dtype=np.int64)
        filled = (glt_x > 0) & (glt_y > 0)
        # the swath pixel (1-based) that fills each map cell
        bits = np.where(
            filled, bits[np.maximum(glt_y - 1, 0), np.maximum(glt_x - 1, 0)], 0
        )
    if bits.shape != tuple(shape):
        raise GridMismatchError(
            f"the quality raster has shape {bits.shape}, the scene {tuple(shape)}"
        )
    return bits.astype(np.int32)


def _band_first(field: xr.DataArray) -> np.ndarray:
    """``(band, y, x)`` values of a mask variable stored as ``(y, x, band)``."""
    others = [dim for dim in field.dims if dim not in ("y", "x")]
    return np.asarray(field.transpose(*others, "y", "x").values)


def _window(
    ring: Float[np.ndarray, "vertex 2"],
    lat: Float[np.ndarray, "row col"],
    lon: Float[np.ndarray, "row col"],
    margin_px: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Row and column ranges of the outline's bounding box plus the margin."""
    lon_min, lat_min = einx.min("[vertex] coord", ring)
    lon_max, lat_max = einx.max("[vertex] coord", ring)
    located = np.isfinite(lat) & np.isfinite(lon)
    # half a pixel in degrees, the slack at the scene's edge
    # Δλ = λ[:, c + 1] − λ[:, c]
    lon_step = lon[:, 1:] - lon[:, :-1]
    # Δφ = φ[r + 1, :] − φ[r, :]
    lat_step = lat[1:, :] - lat[:-1, :]
    # half_λ = ½ medianₙ |Δλₙ|
    half_lon = 0.5 * float(np.nanmedian(np.abs(lon_step)))
    # half_φ = ½ medianₙ |Δφₙ|
    half_lat = 0.5 * float(np.nanmedian(np.abs(lat_step)))
    inside_scene = (
        lon_min >= np.nanmin(lon) - half_lon
        and lon_max <= np.nanmax(lon) + half_lon
        and lat_min >= np.nanmin(lat) - half_lat
        and lat_max <= np.nanmax(lat) + half_lat
    )
    if not inside_scene:
        raise GridMismatchError(
            "load_emit_plume_complex: the outline's bounding box leaves the scene"
        )
    # pixel centres inside the bounding box
    in_box = (
        located
        & (lon >= lon_min)
        & (lon <= lon_max)
        & (lat >= lat_min)
        & (lat <= lat_max)
    )
    if not bool(in_box.any()):
        raise GridMismatchError(
            "load_emit_plume_complex: no pixel centre lies in the outline's "
            "bounding box"
        )
    rows, cols = np.nonzero(in_box)
    num_rows, num_cols = lat.shape
    # [r₀ − margin, r₁ + margin], clipped to the scene
    row_range = (
        max(int(rows.min()) - margin_px, 0),
        min(int(rows.max()) + margin_px + 1, num_rows),
    )
    col_range = (
        max(int(cols.min()) - margin_px, 0),
        min(int(cols.max()) + margin_px + 1, num_cols),
    )
    return row_range, col_range


def _pixel_mask(
    column_mass: np.ndarray,
    sensitivity: np.ndarray,
    obs_noise_std: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[Bool[np.ndarray, "row col"], Bool[np.ndarray, "row col"]]:
    """Pixels kept in the arrays, and those of them the fit uses."""
    located = np.isfinite(lat) & np.isfinite(lon)
    finite = (
        np.isfinite(column_mass),
        np.isfinite(sensitivity),
        np.isfinite(obs_noise_std),
    )
    # dropped: no location, or none of the three fields
    keep = located & (finite[0] | finite[1] | finite[2])
    # ωₙ = 1: every field finite, Aₙ > 0, σₙ > 0
    usable = finite[0] & finite[1] & finite[2]
    positive = np.where(usable, sensitivity, 0.0) > 0.0
    noisy = np.where(usable, obs_noise_std, 0.0) > 0.0
    return keep, keep & usable & positive & noisy


def _kept_pixels(
    column_mass: np.ndarray,
    sensitivity: np.ndarray,
    obs_noise_std: np.ndarray,
    mask: np.ndarray,
    keep: np.ndarray,
    flags: np.ndarray,
) -> dict[str, np.ndarray]:
    """The kept pixels' fields, with finite fillers where ω = 0."""
    if not bool(mask.any()):
        raise ScreeningError(
            "no pixel survives the fill, sensitivity and quality screening"
        )
    variance = np.square(obs_noise_std)
    # σ² of a masked pixel: the median of the used ones, so it stays valid
    filler = float(np.median(variance[mask]))
    return {
        # yₙ = 0, Aₙ = 1 where ωₙ = 0
        "enhancement": np.where(mask, column_mass, 0.0)[keep].astype(column_mass.dtype),
        "kernel": np.where(mask, sensitivity, 1.0)[keep].astype(column_mass.dtype),
        "variance": np.where(mask, variance, filler)[keep].astype(column_mass.dtype),
        "mask": mask[keep],
        "flags": np.asarray(flags)[keep].astype(np.int32),
    }


def _pixel_size(
    easting: Float[np.ndarray, "row col"], northing: Float[np.ndarray, "row col"]
) -> float:
    """g = medianₙ ‖sₙ₊₁ − sₙ‖₂ over row and column neighbours."""
    # ‖sₙ₊₁ − sₙ‖ along each row, between neighbouring columns
    along_row = np.hypot(
        easting[:, 1:] - easting[:, :-1], northing[:, 1:] - northing[:, :-1]
    )
    # ‖sₙ₊₁ − sₙ‖ down each column, between neighbouring rows
    along_col = np.hypot(
        easting[1:, :] - easting[:-1, :], northing[1:, :] - northing[:-1, :]
    )
    # a one-pixel side has no neighbours, and einx.id rejects an empty axis
    neighbours = [dist for dist in (along_row, along_col) if dist.size > 0]
    if not neighbours:
        raise ScreeningError("the window has no two neighbouring located pixels")
    spacings = np.concatenate(
        [einx.id("row col -> (row col)", dist) for dist in neighbours]
    )
    spacings = spacings[np.isfinite(spacings)]
    if spacings.size == 0:
        raise ScreeningError("the window has no two neighbouring located pixels")
    return float(np.median(spacings))


def _inside(
    points: Float[np.ndarray, "pixel 2"], ring: Float[np.ndarray, "vertex 2"]
) -> Bool[np.ndarray, " pixel"]:
    """Point-in-polygon by the even-odd rule, for a closed ring."""
    easting, northing = points[:, 0], points[:, 1]
    # edges (x₁, y₁) → (x₂, y₂) of the ring
    start, end = ring[:-1], ring[1:]
    # the edges with one end above each point's y and one not
    above_start = einx.greater("edge, pixel -> pixel edge", start[:, 1], northing)
    above_end = einx.greater("edge, pixel -> pixel edge", end[:, 1], northing)
    straddles = above_start != above_end
    # (x₂ − x₁) / (y₂ − y₁), finite where an edge straddles
    rise = end[:, 1] - start[:, 1]
    slope = (end[:, 0] - start[:, 0]) / np.where(rise == 0.0, 1.0, rise)
    # crossing x of each edge at the point's height: x₁ + (y − y₁)(x₂ − x₁)/(y₂ − y₁)
    height = einx.subtract("pixel, edge -> pixel edge", northing, start[:, 1])
    crossing = einx.add("edge, pixel edge -> pixel edge", start[:, 0], height * slope)
    # crossings to the right of the point
    to_the_right = einx.less("pixel, pixel edge -> pixel edge", easting, crossing)
    crossings = einx.sum("pixel [edge]", (straddles & to_the_right).astype(np.int64))
    # inside when the count is odd
    return crossings % 2 == 1


def _origin(
    plumes: xr.Dataset,
    plume_id: str,
    frame: LocalFrame,
    receptors: np.ndarray,
    kept: dict[str, np.ndarray],
    in_plume: np.ndarray,
) -> Float[np.ndarray, " axis"]:
    """s₀: the pseudo-origin, or the brightest in-plume pixel without one."""
    entry = plumes.sel(plume=plume_id)
    origin_lon = float(entry["origin_lon"].values)
    origin_lat = float(entry["origin_lat"].values)
    if math.isfinite(origin_lon) and math.isfinite(origin_lat):
        easting, northing = frame.to_xy(origin_lat, origin_lon)
        return np.array([float(easting), float(northing)])
    candidates = in_plume & kept["mask"]
    if not bool(candidates.any()):
        raise ScreeningError(
            f"plume {plume_id!r} has no pseudo-origin and no usable in-plume pixel"
        )
    brightest = np.argmax(np.where(candidates, kept["enhancement"], -np.inf))
    return np.asarray(receptors[brightest], dtype=float)


def _direction(
    receptors: np.ndarray,
    kept: dict[str, np.ndarray],
    in_plume: np.ndarray,
    origin: np.ndarray,
    pixel_size: float,
) -> float:
    """φ₀ = atan2(s̄₂ − s₀₂, s̄₁ − s₀₁), s̄ weighted by max(yₙ, 0) in the plume."""
    # ωₙ max(yₙ, 0) over the in-plume pixels
    weight = np.where(
        in_plume & kept["mask"], np.maximum(kept["enhancement"], 0.0), 0.0
    )
    total = float(einx.sum("[pixel]", weight))
    if total <= 0.0:
        # no enhancement above zero: the plain centroid of the outline's pixels
        weight = (in_plume & kept["mask"]).astype(float)
        total = float(einx.sum("[pixel]", weight))
    if total <= 0.0:
        raise ScreeningError("no usable pixel lies inside the plume outline")
    return direction_from_centroid(
        receptors,
        weight,
        origin,
        pixel_size,
        undefined_message=(
            "the plume's centroid is within a pixel of its origin, so the "
            "starting direction is undefined"
        ),
    )


__all__ = ["PlumeScene", "load_emit_plume_complex", "matched_filter_scene"]
