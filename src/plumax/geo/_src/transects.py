"""Cross-plume transects of a column image: profiles, moments, fits and smoothing.

Profiles of the enhancement across the plume at regular downwind distances,
by spline interpolation of the image, and what each profile gives: its line
density q̂_k (kg m⁻¹), centre m̂_k and width ŵ_k. A steady plume has a flat
line density, which times a wind speed is the cross-sectional flux
(Krings et al. 2011). The centres and widths are the starting values of an
image fit: the centreline knots and the width law of
``gauss_plume.ColumnPlume``.

The procedure is SPIIRE's ``compute_cross_plume_slices`` (``emit-sds/SPIIRE``
@ ``ae9a54d``), with the line density from the signed sum and rejected
centres replaced by the smoothed line rather than by zero.

Everything here is NumPy and SciPy code on the host, outside any traced
path: there is no gradient, ``jit`` or ``vmap`` to take.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import einx
import numpy as np
from jaxtyping import Bool, Float
from scipy import ndimage, optimize, signal

from plumax._src.errors import DomainError, ParameterError
from plumax._src.tracing import require_positive
from plumax.errors import MissingExtraError
from plumax.geo._src.plume_array import angle_frame, transect_line
from plumax.geo.plume_array import pixel_area, squeeze_single_band


try:
    import geopandas as gpd
    from functax.geometry import to_pixel
    from geoops.carrier import (
        affine_of,
        as_tuple,
        mask_invalid_to_nan,
        require_geoarray,
        require_ndim,
        require_projected_crs,
    )
    from pipekit_ops import Operator
except ModuleNotFoundError as err:  # pragma: no cover - the base-install gate
    raise MissingExtraError(
        "plumax.geo.transects needs the [geo] extra: pip install 'plumax[geo]'"
    ) from err


if TYPE_CHECKING:
    from geoarray import GeoArray

# Shapes: N_row × N_col image pixels (dims "row", "col");
#         T transects (dim "transect"); S samples across each (dim "offset")

#: Samples within this many pixels of a nodata pixel are set to NaN.
_NODATA_MARGIN = 2
#: Zeros put before the centres, so the smoothed line starts at the source.
_SOURCE_PAD = 100


@dataclasses.dataclass(frozen=True)
class TransectProfiles:
    """Interpolated enhancement across the plume at regular downwind distances.

    Attributes:
        distance: Downwind distance d_k of each transect [m].
        offset: Crosswind offset c_j of each sample [m], δ apart and
            symmetric about the plume axis.
        values: Enhancement v_kj [kg m⁻²]; NaN outside the image or near
            nodata.
    """

    distance: Float[np.ndarray, " transect"]
    offset: Float[np.ndarray, " offset"]
    values: Float[np.ndarray, "transect offset"]

    @property
    def sample_spacing(self) -> float:
        """δ = c_{j+1} − c_j [m]."""
        return float(self.offset[1] - self.offset[0])


def cross_plume_transects(
    enhancement: Float[np.ndarray, "row col"],
    transform: Any,
    *,
    source: tuple[float, float],
    direction: float,
    n_transects: int,
    spacing: float,
    half_length: float,
    sample_spacing: float | None = None,
    order: int = 3,
) -> TransectProfiles:
    """Profiles across the plume by spline interpolation of the image.

    v_kj = 𝓘_ΔΩ(p_kj) with p_kj = s₀ + d_k e_∥ + c_j e_⊥, d_k = k Δ_d and
    c_j = (j − (S − 1)/2) δ.

    Algorithm:
        1. Build the sample points p_kj in the carrier's coordinates.
        2. Map them to fractional pixel indices with the inverse transform.
        3. Interpolate with a spline of the given order, nodata filled with
           zero; samples outside the image or within two pixels of nodata
           become NaN.
        4. Drop transects whose centre point is outside the image.

    The S = 2 round(h / δ) + 1 samples run from −h to h, rounded to whole
    samples. Transects closer than a pixel would reuse the same pixels, so
    their line densities would be correlated.

    Args:
        enhancement: Column enhancement ΔΩ [kg m⁻²], nodata as NaN.
        transform: Affine geotransform from ``(col, row)`` to metres.
        source: Source position s₀ [m] in the carrier's CRS.
        direction: Plume direction φ [rad], counter-clockwise from east.
        n_transects: Number of transects T.
        spacing: Downwind spacing Δ_d [m].
        half_length: Half the transect length h [m].
        sample_spacing: Sample spacing δ [m]; defaults to the pixel side.
        order: Spline order of the interpolation, 0 to 5.

    Returns:
        The :class:`TransectProfiles`, in the dtype of ``enhancement``.

    Raises:
        ParameterError: If ``spacing`` is below one pixel, a length is not
            positive, ``half_length`` is below one sample, ``n_transects`` is
            below one, or ``order`` is outside 0 to 5.
        DomainError: If no transect lies inside the image.

    Examples:
        A uniform ridge of 1 kg m⁻² three pixels wide along x, on a 60 m
        grid, has a line density of 3 × 60 = 180 kg m⁻¹:

        >>> import numpy as np
        >>> from affine import Affine
        >>> transform = Affine(60.0, 0.0, 0.0, 0.0, -60.0, 1200.0)
        >>> image = np.zeros((20, 20))
        >>> image[9:12, :] = 1.0
        >>> profiles = cross_plume_transects(
        ...     image, transform, source=(30.0, 570.0), direction=0.0,
        ...     n_transects=4, spacing=120.0, half_length=300.0,
        ... )
        >>> profiles.values.shape
        (4, 11)
        >>> np.round(transect_moments(profiles)["line_density"], 9).tolist()
        [180.0, 180.0, 180.0, 180.0]
    """
    dtype = np.result_type(np.asarray(enhancement).dtype, np.float32)
    image = np.asarray(enhancement, dtype=np.float64)
    if image.ndim != 2:
        raise ParameterError(
            f"cross_plume_transects: the image must be 2-D, got {image.ndim}-D"
        )
    # g = √|det A|, the pixel side
    pixel_side = math.sqrt(pixel_area(transform))
    # δ, the sample spacing along a transect
    spacing_sample = pixel_side if sample_spacing is None else float(sample_spacing)
    _check_sampling(
        n_transects, spacing, half_length, spacing_sample, order, pixel_side
    )

    # --- sample points ---
    # d_k = k Δ_d, k = 1, …, T
    distance = spacing * np.arange(1, n_transects + 1, dtype=np.float64)
    # c_j = (j − (S − 1)/2) δ, S = 2 round(h / δ) + 1
    half_samples = round(half_length / spacing_sample)
    offset = spacing_sample * np.arange(
        -half_samples, half_samples + 1, dtype=np.float64
    )
    # e_∥ = (cos φ, sin φ),  e_⊥ = (−sin φ, cos φ)
    along, across = angle_frame(direction)
    # p_kj = s₀ + d_k e_∥ + c_j e_⊥
    downwind = einx.multiply("transect, axis -> transect axis", distance, along)
    crosswind = einx.multiply("offset, axis -> offset axis", offset, across)
    points = einx.add(
        "transect axis, offset axis, axis -> transect offset axis",
        downwind,
        crosswind,
        np.asarray(source, dtype=np.float64),
    )

    # --- fractional pixel indices ---
    rows, cols = _pixel_indices(points, transform)

    # --- interpolation ---
    nodata = ~np.isfinite(image)
    filled = np.where(nodata, 0.0, image)
    coordinates = np.stack([rows, cols])
    # v_kj = 𝓘_ΔΩ(p_kj)
    values = ndimage.map_coordinates(filled, coordinates, order=order, mode="nearest")
    # a sample is outside when its nearest pixel is
    num_rows, num_cols = image.shape
    inside = (
        (rows >= -0.5)
        & (rows <= num_rows - 0.5)
        & (cols >= -0.5)
        & (cols <= num_cols - 0.5)
    )
    near_nodata = _near_nodata(nodata, coordinates)
    values = np.where(inside & ~near_nodata, values, np.nan)

    # --- transects inside the image ---
    centre = inside[:, half_samples]
    if not bool(centre.any()):
        raise DomainError(
            "cross_plume_transects: no transect has its centre inside the image."
        )
    return TransectProfiles(
        distance=distance[centre].astype(dtype),
        offset=offset.astype(dtype),
        values=values[centre].astype(dtype),
    )


def transect_moments(
    profiles: TransectProfiles,
) -> dict[str, Float[np.ndarray, " transect"]]:
    """Line density, centre and width of each transect from its moments.

    q̂_k = δ ∑_j v_kj;  m̂_k = ∑_j c_j v⁺_kj / ∑_j v⁺_kj;
    ŵ_k² = ∑_j (c_j − m̂_k)² v⁺_kj / ∑_j v⁺_kj, with v⁺ = max(v, 0).

    The line density is the signed sum, unbiased under zero-mean noise. A
    positive-part sum would add σ / √(2π) per sample for noise of standard
    deviation σ, a bias of S δ σ / √(2π) that grows with the transect. The
    same noise floor pulls ŵ_k towards the half-length, so the width of
    :func:`fit_transect_gaussians` is the one to use on noisy images. NaN
    samples carry no weight.

    Args:
        profiles: The transects.

    Returns:
        ``line_density`` q̂ [kg m⁻¹], ``centre`` m̂ [m] and ``width`` ŵ [m],
        one per transect; the centre and width are NaN where no sample is
        positive.

    Examples:
        >>> import numpy as np
        >>> profiles = TransectProfiles(
        ...     distance=np.array([60.0]),
        ...     offset=np.array([-60.0, 0.0, 60.0]),
        ...     values=np.array([[1.0, 2.0, 1.0]]),
        ... )
        >>> moments = transect_moments(profiles)
        >>> moments["line_density"].tolist(), moments["centre"].tolist()
        ([240.0], [0.0])
    """
    # v⁺ = max(v, 0), NaN samples carry no weight
    finite = np.isfinite(profiles.values)
    values = np.where(finite, profiles.values, 0.0)
    positive = np.maximum(values, 0.0)
    # q̂_k = δ ∑_j v_kj
    signed_sum = einx.sum("transect [offset]", values)
    line_density = profiles.sample_spacing * signed_sum
    # ∑_j v⁺_kj
    weight = einx.sum("transect [offset]", positive)
    # m̂_k = ∑_j c_j v⁺_kj / ∑_j v⁺_kj
    first_moment = einx.dot(
        "transect offset, offset -> transect", positive, profiles.offset
    )
    centre = _divide(first_moment, weight)
    # (c_j − m̂_k)²
    deviation = einx.subtract(
        "offset, transect -> transect offset", profiles.offset, centre
    )
    # ŵ_k² = ∑_j (c_j − m̂_k)² v⁺_kj / ∑_j v⁺_kj
    second_moment = einx.sum("transect [offset]", np.square(deviation) * positive)
    width = np.sqrt(_divide(second_moment, weight))
    return {"line_density": line_density, "centre": centre, "width": width}


def fit_transect_gaussians(
    profiles: TransectProfiles,
    *,
    min_width: float,
    max_offset: float | None = None,
) -> dict[str, np.ndarray]:
    """Least-squares Gaussian per transect.

    (q̂_k, m̂_k, ŵ_k) = argmin ∑_j (v_kj − q / (√(2π) w) exp(−(c_j − m)² / 2w²))²,
    with q ≥ 0, |m| ≤ m_max and w ≥ w_min.

    Algorithm:
        1. Start from (the moment q̂_k, the offset of the largest sample, 2δ).
        2. Solve the bounded problem by trust-region reflective least squares
           over the finite samples, transect by transect.
        3. Mark a fit converged when the solver reports success and no
           parameter sits on a bound.

    The fitted q is the area under the Gaussian, so it is a line density.

    Args:
        profiles: The transects.
        min_width: Smallest width w_min [m], the sample spacing in SPIIRE.
        max_offset: Largest centre offset m_max [m]; defaults to twice the
            half-length.

    Returns:
        ``line_density`` [kg m⁻¹], ``centre`` [m], ``width`` [m] and
        ``converged``, one per transect; NaN and not converged where a
        transect has fewer than three finite samples or none above zero.

    Raises:
        ParameterError: If ``min_width`` or ``max_offset`` is not positive.

    Examples:
        >>> import numpy as np
        >>> offset = np.arange(-600.0, 601.0, 60.0)
        >>> values = 0.1 / (np.sqrt(2 * np.pi) * 150.0) * np.exp(
        ...     -0.5 * ((offset - 30.0) / 150.0) ** 2
        ... )
        >>> profiles = TransectProfiles(
        ...     distance=np.array([60.0]), offset=offset, values=values[None]
        ... )
        >>> fits = fit_transect_gaussians(profiles, min_width=60.0)
        >>> [round(float(fits[key][0]), 6) for key in ("line_density", "centre", "width")]
        [0.1, 30.0, 150.0]
        >>> bool(fits["converged"][0])
        True
    """
    require_positive("min_width", min_width)
    offset = np.asarray(profiles.offset, dtype=np.float64)
    spacing_sample = profiles.sample_spacing
    # m_max = 2 h
    bound = 2.0 * float(np.max(np.abs(offset))) if max_offset is None else max_offset
    require_positive("max_offset", bound)
    lower = np.array([0.0, -bound, min_width])
    upper = np.array([np.inf, bound, np.inf])
    moments = transect_moments(profiles)

    num_transects = profiles.values.shape[0]
    estimates = np.full((num_transects, 3), np.nan)
    converged = np.zeros(num_transects, dtype=bool)
    for transect in range(num_transects):
        row = np.asarray(profiles.values[transect], dtype=np.float64)
        finite = np.isfinite(row)
        if int(finite.sum()) < 3:
            continue
        # no positive sample: nothing to fit, a failed fit
        if not bool(np.any(row[finite] > 0.0)):
            continue
        positions = offset[finite]
        # v_kj / s_k with s_k = max_j |v_kj|, so the solver's tolerances see
        # residuals of order one
        scale = float(np.max(np.abs(row[finite])))
        samples = row[finite] / scale
        start = _gaussian_start(
            float(moments["line_density"][transect]) / scale,
            samples,
            positions,
            spacing_sample,
            lower,
            upper,
        )
        fit = optimize.least_squares(
            _gaussian_residual,
            start,
            jac=_gaussian_jacobian,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            args=(positions, samples),
        )
        # q = s_k q̃
        estimates[transect] = fit.x * np.array([scale, 1.0, 1.0])
        converged[transect] = bool(fit.success) and not bool(
            np.any(fit.active_mask != 0)
        )
    dtype = profiles.values.dtype
    # q, m, w per transect, parameters first
    params = einx.id("transect param -> param transect", estimates.astype(dtype))
    line_density, centre, width = params
    return {
        "line_density": line_density,
        "centre": centre,
        "width": width,
        "converged": converged,
    }


def smooth_centreline(
    centre: Float[np.ndarray, " transect"],
    width: Float[np.ndarray, " transect"],
    converged: Bool[np.ndarray, " transect"],
    *,
    window: int = 19,
    polyorder: int = 2,
    n_iter: int = 3,
    threshold: float = 1.5,
    rejected: Literal["smoothed", "zero"] = "smoothed",
) -> Float[np.ndarray, " transect"]:
    """Centreline with failed fits interpolated and outliers rejected.

    m̃ = SG(m̂); m̂_k ← m̃_k where |m̂_k − m̃_k| > 1.5 ŵ_k; repeated, with zeros
    put before the centres so the line starts at the source.

    Algorithm:
        1. Interpolate m̂_k and ŵ_k linearly across the unconverged
           transects.
        2. Put 100 zeros before them, which are never rejected.
        3. Repeat ``n_iter`` times: smooth with a Savitzky–Golay filter to
           get m̃_k, and replace m̂_k wherever |m̂_k − m̃_k| > threshold · ŵ_k.
        4. Smooth once more and drop the zeros.

    SPIIRE sets the rejected centres to zero (``rejected="zero"``), which
    pulls a bent plume back towards a straight one; the default replaces
    them by the smoothed line. The window is shortened to the largest odd
    number not above the series length.

    Args:
        centre: Centres m̂_k [m], one per transect.
        width: Widths ŵ_k [m], the scale of the rejection test.
        converged: Which fits converged.
        window: Savitzky–Golay window, odd.
        polyorder: Savitzky–Golay polynomial order, below ``window``.
        n_iter: Rounds of rejection.
        threshold: Rejection threshold, in widths.
        rejected: What a rejected centre becomes: ``"smoothed"`` or ``"zero"``.

    Returns:
        The smoothed centreline m̃_k [m], one per transect.

    Raises:
        ParameterError: If ``window`` is not odd, ``polyorder`` is not below
            it, ``n_iter`` is negative, ``threshold`` is not positive or
            ``rejected`` is unknown.
        DomainError: If no fit converged.

    Examples:
        A straight line with one failed fit and one outlier:

        >>> import numpy as np
        >>> centre = np.zeros(30)
        >>> centre[10], centre[20] = 500.0, 400.0
        >>> converged = np.ones(30, dtype=bool)
        >>> converged[10] = False
        >>> smoothed = smooth_centreline(centre, np.full(30, 60.0), converged)
        >>> bool(np.max(np.abs(smoothed)) < 60.0)
        True
    """
    if window < 1 or window % 2 == 0:
        raise ParameterError(f"window must be a positive odd number, got {window}")
    if not 0 <= polyorder < window:
        raise ParameterError(
            f"polyorder must be in [0, window), got {polyorder} for window {window}"
        )
    require_positive("n_iter", n_iter, allow_zero=True)
    require_positive("threshold", threshold)
    if rejected not in ("smoothed", "zero"):
        raise ParameterError(f"rejected must be 'smoothed' or 'zero', got {rejected!r}")
    dtype = np.result_type(np.asarray(centre).dtype, np.float32)
    centre = np.asarray(centre, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    good = np.asarray(converged, dtype=bool) & np.isfinite(centre) & np.isfinite(width)
    if not bool(good.any()):
        raise DomainError("smooth_centreline: no fit converged.")

    # --- fill the failed fits ---
    index = np.arange(centre.shape[0], dtype=np.float64)
    # m̂_k and ŵ_k, linear between converged neighbours
    filled_centre = np.interp(index, index[good], centre[good])
    filled_width = np.interp(index, index[good], width[good])

    # --- start at the source ---
    # (0, …, 0, m̂_1, …, m̂_T), the zeros never rejected
    series = np.concatenate([np.zeros(_SOURCE_PAD), filled_centre])
    scale = np.concatenate([np.full(_SOURCE_PAD, np.inf), filled_width])
    length = series.shape[0]
    # the largest odd window not above the length, and above polyorder
    window = min(window, length if length % 2 == 1 else length - 1)
    if polyorder >= window:
        raise ParameterError(
            f"polyorder {polyorder} needs a window above it; the series has "
            f"{length} points"
        )

    # --- reject the outliers ---
    for _ in range(n_iter):
        # m̃ = SG(m̂)
        smoothed = signal.savgol_filter(series, window, polyorder)
        # |m̂_k − m̃_k| > threshold · ŵ_k
        outlier = np.abs(series - smoothed) > threshold * scale
        replacement = smoothed if rejected == "smoothed" else 0.0
        series = np.where(outlier, replacement, series)
    # m̃ = SG(m̂), the zeros dropped
    smoothed = signal.savgol_filter(series, window, polyorder)
    return smoothed[_SOURCE_PAD:].astype(dtype)


def fit_width_law(
    distance: Float[np.ndarray, " transect"],
    width: Float[np.ndarray, " transect"],
    *,
    reference_distance: float,
    initial_width: float = 0.0,
) -> tuple[float, float]:
    """Reference width and exponent from a line through the log widths.

    ½ log(ŵ_k² − w₀²) = log w_ref + b log(d_k / d_ref), by least squares.

    Widths at or below w₀, and non-finite ones, are left out. The result is
    the ``reference_width`` and ``width_exponent`` of
    ``gauss_plume.PlumeGeometry``.

    Args:
        distance: Downwind distances d_k [m].
        width: Widths ŵ_k [m], as fitted by :func:`fit_transect_gaussians`.
        reference_distance: Reference distance d_ref [m].
        initial_width: Initial width w₀ [m].

    Returns:
        ``(w_ref, b)``: the width at d_ref [m] and the exponent.

    Raises:
        ParameterError: If ``reference_distance`` is not positive or
            ``initial_width`` is negative.
        DomainError: If fewer than two transects at distinct distances keep a
            width above w₀.

    Examples:
        >>> import numpy as np
        >>> distance = np.array([250.0, 500.0, 1000.0, 2000.0])
        >>> width = 100.0 * (distance / 1000.0) ** 0.8
        >>> reference_width, exponent = fit_width_law(
        ...     distance, width, reference_distance=1000.0
        ... )
        >>> round(reference_width, 6), round(exponent, 6)
        (100.0, 0.8)
    """
    require_positive("reference_distance", reference_distance)
    require_positive("initial_width", initial_width, allow_zero=True)
    distance = np.asarray(distance, dtype=np.float64)
    width = np.asarray(width, dtype=np.float64)
    kept = np.isfinite(width) & (width > initial_width) & (distance > 0.0)
    if np.unique(distance[kept]).shape[0] < 2:
        raise DomainError(
            "fit_width_law: fewer than two distances keep a width above the "
            "initial width."
        )
    # x_k = log(d_k / d_ref)
    log_distance = np.log(distance[kept] / reference_distance)
    # y_k = ½ log(ŵ_k² − w₀²)
    log_width = 0.5 * np.log(np.square(width[kept]) - initial_width**2)
    # x_k − x̄,  y_k − ȳ
    centred_x = log_distance - np.mean(log_distance)
    centred_y = log_width - np.mean(log_width)
    # b = ∑ (x − x̄)(y − ȳ) / ∑ (x − x̄)²
    exponent = float(einx.dot("[k], [k] ->", centred_x, centred_y)) / float(
        einx.dot("[k], [k] ->", centred_x, centred_x)
    )
    # log w_ref = ȳ − b x̄
    log_reference = float(np.mean(log_width)) - exponent * float(np.mean(log_distance))
    return math.exp(log_reference), exponent


class CrossPlumeProfile(Operator):
    """Transect table of a plume image: one row per downwind distance.

    Called as ``op(enhancement)`` on a GeoArray in kg m⁻² and a projected
    CRS in metres. Returns a GeoDataFrame with ``geometry`` (the transect
    LineString), ``distance_m``, ``line_density_kg_m``, ``centre_m``,
    ``width_m``, ``centre_smooth_m`` and ``converged``.

    The line density is the signed sum of :func:`transect_moments`; the
    centre and width are those of :func:`fit_transect_gaussians` with
    w_min = δ, and the smoothed centre is :func:`smooth_centreline` of them.
    Samples are a pixel apart. Times a wind speed U, the line density is the
    flux Q_k = U q̂_k that ``CrossSectionalFlux`` reports, with interpolated
    samples in place of whole pixels.

    Geo-dependent: the input must be a georeferenced ``GeoArray`` (plain
    arrays raise ``TypeError``) in a projected CRS with metre units (a
    geographic CRS raises ``ValueError``), as for ``CrossSectionalFlux``.
    Nodata pixels (non-finite or equal to the fill value) become NaN.

    Args:
        source: Source position s₀ [m] in the carrier's CRS.
        direction: Plume direction φ [rad], counter-clockwise from east.
        n_transects: Number of transects T.
        transect_spacing_m: Downwind spacing Δ_d [m].
        half_length_m: Half the transect length [m].
        order: Spline order of the interpolation.

    Examples:
        Profiles every 120 m out to 2.4 km, each ±1.5 km across::

            table = CrossPlumeProfile(
                source=(x0, y0),
                direction=np.deg2rad(30.0),
                n_transects=20,
                transect_spacing_m=120.0,
                half_length_m=1500.0,
            )(column_mass)
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        source: Sequence[float],
        direction: float,
        n_transects: int,
        transect_spacing_m: float,
        half_length_m: float,
        order: int = 3,
    ) -> None:
        self.source = as_tuple(source)
        self.direction = float(direction)
        self.n_transects = int(n_transects)
        self.transect_spacing_m = float(transect_spacing_m)
        self.half_length_m = float(half_length_m)
        self.order = int(order)

    def get_config(self) -> dict[str, Any]:
        return {
            "source": list(self.source),
            "direction": self.direction,
            "n_transects": self.n_transects,
            "transect_spacing_m": self.transect_spacing_m,
            "half_length_m": self.half_length_m,
            "order": self.order,
        }

    def _apply(self, geo_array: GeoArray) -> gpd.GeoDataFrame:
        require_ndim(geo_array, (2, 3), "CrossPlumeProfile")
        require_geoarray(geo_array, "CrossPlumeProfile")
        require_projected_crs(geo_array, "CrossPlumeProfile")
        enhancement = squeeze_single_band(mask_invalid_to_nan(geo_array))
        transform = affine_of(geo_array)
        profiles = cross_plume_transects(
            enhancement,
            transform,
            source=self.source,
            direction=self.direction,
            n_transects=self.n_transects,
            spacing=self.transect_spacing_m,
            half_length=self.half_length_m,
            order=self.order,
        )
        moments = transect_moments(profiles)
        fits = fit_transect_gaussians(profiles, min_width=profiles.sample_spacing)
        smoothed = smooth_centreline(fits["centre"], fits["width"], fits["converged"])

        # e_∥, e_⊥
        along, across = angle_frame(self.direction)
        first, last = float(profiles.offset[0]), float(profiles.offset[-1])
        rows: list[dict[str, Any]] = []
        for transect, distance in enumerate(profiles.distance.tolist()):
            # s₀ + d_k e_∥ + c e_⊥ at the two ends
            middle = np.asarray(self.source) + distance * along
            line = transect_line(middle, across, first, last)
            rows.append(
                {
                    "geometry": line,
                    "distance_m": distance,
                    "line_density_kg_m": float(moments["line_density"][transect]),
                    "centre_m": float(fits["centre"][transect]),
                    "width_m": float(fits["width"][transect]),
                    "centre_smooth_m": float(smoothed[transect]),
                    "converged": bool(fits["converged"][transect]),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=geo_array.crs)


def _check_sampling(
    n_transects: int,
    spacing: float,
    half_length: float,
    spacing_sample: float,
    order: int,
    pixel_side: float,
) -> None:
    if n_transects < 1:
        raise ParameterError(f"n_transects must be ≥ 1, got {n_transects}")
    require_positive("spacing", spacing)
    require_positive("half_length", half_length)
    require_positive("sample_spacing", spacing_sample)
    if spacing < pixel_side * (1.0 - 1e-9):
        raise ParameterError(
            f"spacing {spacing} m is below one pixel ({pixel_side} m): closer "
            "transects reuse the same pixels"
        )
    if round(half_length / spacing_sample) < 1:
        raise ParameterError(
            f"half_length {half_length} m is below one sample ({spacing_sample} m)"
        )
    if not 0 <= order <= 5:
        raise ParameterError(f"order must be in 0 to 5, got {order}")


def _pixel_indices(
    points: Float[np.ndarray, "transect offset axis"], transform: Any
) -> tuple[Float[np.ndarray, "transect offset"], Float[np.ndarray, "transect offset"]]:
    """Fractional ``(row, col)`` of each point, pixel centres at integers."""
    # [col + ½, row + ½] = A⁻¹ (x − c, y − f), with A = [[a, b], [d, e]]
    row_col = to_pixel(
        (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ),
        points,
    )
    return row_col[..., 0], row_col[..., 1]


def _near_nodata(
    nodata: Bool[np.ndarray, "row col"],
    coordinates: Float[np.ndarray, "2 transect offset"],
) -> Bool[np.ndarray, "transect offset"]:
    """Samples whose nearest pixel is within two pixels of a nodata pixel."""
    if not bool(nodata.any()):
        return np.zeros(coordinates.shape[1:], dtype=bool)
    near = ndimage.binary_dilation(
        nodata, structure=np.ones((3, 3), dtype=bool), iterations=_NODATA_MARGIN
    )
    sampled = ndimage.map_coordinates(
        near.astype(np.float64), coordinates, order=0, mode="nearest"
    )
    return sampled > 0.5


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    safe = np.where(denominator > 0.0, denominator, 1.0)
    return np.where(denominator > 0.0, numerator / safe, np.nan)


def _gaussian_start(
    line_density: float,
    samples: np.ndarray,
    positions: np.ndarray,
    spacing_sample: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """(q̂_k, c_argmax, 2δ), moved inside the bounds."""
    peak = float(np.max(np.maximum(samples, 0.0)))
    # q₀ = q̂_k, or a pixel of the peak when the signed sum is not positive
    rate = line_density if line_density > 0.0 else max(spacing_sample * peak, 1e-30)
    start = np.array([rate, float(positions[np.argmax(samples)]), 2.0 * spacing_sample])
    # strictly inside [lower, upper]
    margin = 1e-6 * np.maximum(np.abs(lower), 1.0)
    return np.clip(
        start, lower + margin, np.where(np.isfinite(upper), upper - margin, np.inf)
    )


def _gaussian(
    params: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rate, centre, width = params
    # u_j = (c_j − m) / w
    scaled = (positions - centre) / width
    # f_j = q / (√(2π) w) exp(−u_j² / 2)
    profile = rate / (math.sqrt(2.0 * math.pi) * width) * np.exp(-0.5 * scaled**2)
    return profile, scaled


def _gaussian_residual(
    params: np.ndarray, positions: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    profile, _ = _gaussian(params, positions)
    # f_j − v_kj
    return profile - samples


def _gaussian_jacobian(
    params: np.ndarray, positions: np.ndarray, samples: np.ndarray
) -> np.ndarray:
    rate, _, width = params
    profile, scaled = _gaussian(params, positions)
    # ∂f/∂q = f / q, ∂f/∂m = f u / w, ∂f/∂w = f (u² − 1) / w
    by_rate = (
        profile / rate
        if rate > 0.0
        else np.exp(-0.5 * scaled**2) / (math.sqrt(2.0 * math.pi) * width)
    )
    by_centre = profile * scaled / width
    by_width = profile * (scaled**2 - 1.0) / width
    return einx.id(
        "position, position, position -> position (1 + 1 + 1)",
        by_rate,
        by_centre,
        by_width,
    )


__all__ = [
    "CrossPlumeProfile",
    "TransectProfiles",
    "cross_plume_transects",
    "fit_transect_gaussians",
    "fit_width_law",
    "smooth_centreline",
    "transect_moments",
]
