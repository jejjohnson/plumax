"""The plume operators on GeoArrays (behind the ``[geo]`` extra).

Ten operators ported from geotoolz's ``plume`` module (#134) with their
behaviour unchanged; its eleventh, ``SBMP``, moved to radtrax as
``radtrax.geo.MBMP``. They wrap the array functions in
:mod:`plumax.geo.plume_array` and are pipekit Operators built with geoops's
public carrier helpers, as radtrax's ``[geo]`` operators are.

Per-pixel operators accept a ``geoarray.GeoArray`` or a plain
``np.ndarray`` and return the same carrier kind; geo-dependent operators
(advection geometry, pixel-area maths, footprints) require a georeferenced
``GeoArray`` and raise ``TypeError`` otherwise. The area and distance
operators (``PlumeFootprint``, ``WindAdvectionCone``, ``IMEEstimate``,
``CrossSectionalFlux``) work in metres and raise ``ValueError`` on a
geographic CRS (see :func:`geoops.carrier.require_projected_crs`). Vector
outputs are ``geopandas.GeoDataFrame``. Algorithms follow the literature
cited in :mod:`plumax.geo.plume_array`:

- ``PlumeMask`` / ``PlumeContours`` / ``PlumeFootprint``: Frankenberg
  et al. (2016) detection and segmentation conventions.
- ``IMEEstimate``: Varon et al. (2018) integrated mass enhancement.
- ``CrossSectionalFlux``: Krings et al. (2011) / Varon (2018) appendix.
- ``PlumeShapeFilter`` / ``PlumeColumnStats`` / ``PlumeQNDFeatures``:
  Pérez Carrasco et al. (2026), the false-plume classifier's features.

The IME and cross-section operators expect the enhancement carrier to be
in kg/m^2. Use :class:`ColumnToMass` to convert from ppm m or mol/m^2.

Nodata: pixels that are non-finite or equal the carrier's
``fill_value`` (see :func:`geoops.carrier.valid_pixels`) never
enter a threshold, statistic or sum, and are never part of a plume mask or
contour. Output fill values follow the output's meaning, not the
input's: boolean masks declare ``fill_value=False``, float
products (scores, converted columns) hold ``NaN`` at nodata pixels and
declare ``fill_value=NaN``, and label images use ``0`` (see
:class:`PlumeContours` for its one documented exception).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import jax.numpy as jnp
import numpy as np
from integrix import rule as integrix_rule
from pipekit_ops import import_optional

from plumax._src.constants import P_STANDARD, T_STANDARD
from plumax._src.tracing import require_positive
from plumax.errors import MissingExtraError
from plumax.geo._src.plume_array import transect_line, wind_frame
from plumax.geo.plume_array import (
    ColumnUnit,
    Connectivity,
    ThresholdMode,
    convert_column_units,
    label_components,
    pixel_area,
    pixel_centers,
    plume_length,
    plume_mask,
    squeeze_single_band,
    wind_advection_cone,
)


try:
    import geopandas as gpd
    import pandas as pd
    from geoops.carrier import (
        Coupling,
        affine_of,
        as_tuple,
        callable_name,
        carried_fill,
        mask_invalid_to_nan,
        over_frames,
        reject_config_summary,
        require_geoarray,
        require_grid_match,
        require_ndim,
        require_projected_crs,
        valid_pixels,
        wrap_filled,
        wrap_like,
    )
    from geoops.measure import DEFAULT_REGIONPROPS, regionprops_frame, skeleton_length
    from georeader import vectorize
    from pipekit_ops import Operator
    from pyproj import CRS, Transformer
    from shapely.ops import unary_union
except ModuleNotFoundError as err:  # pragma: no cover - the base-install gate
    raise MissingExtraError(
        "plumax.geo.plume needs the [geo] extra: pip install 'plumax[geo]'"
    ) from err


if TYPE_CHECKING:
    from geoarray import GeoArray


__all__ = [
    "ColumnToMass",
    "CrossSectionalFlux",
    "IMEEstimate",
    "PlumeColumnStats",
    "PlumeContours",
    "PlumeFootprint",
    "PlumeMask",
    "PlumeQNDFeatures",
    "PlumeShapeFilter",
    "WindAdvectionCone",
]


def _single_band_nan(array: GeoArray | np.ndarray) -> np.ndarray:
    """Single-band ``(H, W)`` float map of ``array`` with nodata pixels set to NaN."""
    return squeeze_single_band(mask_invalid_to_nan(array))


def _mask_and_enhancement(
    geo_array: GeoArray | np.ndarray, plume_mask: GeoArray | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean ``(H, W)`` plume mask and NaN-filled ``(H, W)`` enhancement map.

    Shared by :class:`IMEEstimate` and :class:`CrossSectionalFlux`.

    Args:
        geo_array: Column-enhancement carrier.
        plume_mask: Plume mask on the same pixel grid.

    Returns:
        ``(mask, enhancement)``.
    """
    mask = squeeze_single_band(np.asarray(plume_mask)).astype(bool)
    enhancement = _single_band_nan(geo_array)
    return mask, enhancement


def _masked_mass(enhancement: np.ndarray, mask: np.ndarray, area: float) -> float:
    """∑ᵢ mᵢ Ωᵢ ΔA over an ``(H, W)`` map, with NaN pixels left out.

    The mass of the column enhancement Ω (kg m⁻²) over the pixels where
    ``mask`` holds, each of area ΔA (m²): ``np.nansum(Ω[mask] * ΔA)``, as one
    traceable integrix sum (plumax#250).
    """
    # ΔA per pixel, in the image's shape
    pixel_areas = jnp.full(enhancement.shape, area)
    # ∑ᵢ mᵢ Ωᵢ ΔA, NaN pixels omitted
    total = integrix_rule.integrate(
        pixel_areas, enhancement, axis=(0, 1), where=mask, nan_policy="omit"
    )
    return float(total.value)


def _instance_labels(array: GeoArray | np.ndarray) -> np.ndarray:
    """``(H, W)`` int64 label image of ``array`` with nodata pixels as background.

    A positive fill (a uint8 raster's ``255``) would otherwise read as one
    more instance.
    """
    labels = squeeze_single_band(np.asarray(array)).astype(np.int64, copy=False)
    valid = squeeze_single_band(valid_pixels(array))
    return np.where(valid, labels, 0)


def _label_fields(
    geo_labels: GeoArray | np.ndarray,
    column: GeoArray | np.ndarray,
    name: str,
    albedo: GeoArray | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and extract the label map, column and optional albedo fields.

    Shared preamble of :class:`PlumeColumnStats` and :class:`PlumeQNDFeatures`.

    Args:
        geo_labels: Instance label map.
        column: Column field on the label map's grid.
        name: Operator name used in error messages.
        albedo: Optional albedo field on the label map's grid.

    Returns:
        ``(labels, column, albedo)`` as ``(H, W)`` arrays (``albedo`` is
        ``None`` when not given).

    Raises:
        ValueError: If a field's grid or shape does not match the labels.
    """
    require_ndim(geo_labels, (2, 3), name)
    require_grid_match(geo_labels, column, name, names=("labels", "column"))
    if albedo is not None:
        require_grid_match(geo_labels, albedo, name, names=("labels", "albedo"))
    labels = _instance_labels(geo_labels)
    col_arr = _single_band_nan(column).astype(float, copy=False)
    if col_arr.shape != labels.shape:
        raise ValueError(
            f"column shape {col_arr.shape} does not match label shape {labels.shape}"
        )
    if albedo is None:
        return labels, col_arr, None
    alb_arr = _single_band_nan(albedo).astype(float, copy=False)
    if alb_arr.shape != labels.shape:
        raise ValueError(
            f"albedo shape {alb_arr.shape} does not match label shape {labels.shape}"
        )
    return labels, col_arr, alb_arr


def _instances_full(
    labels: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, int]]:
    """Yield ``(label, full_mask, area)`` per non-zero label.

    ``full_mask`` is the instance's boolean mask on the full ``labels`` grid
    and ``area`` its pixel count. One full-size mask exists at a time, so
    memory stays O(H W) however many instances the raster holds.
    """
    for lbl, compact, bbox in _iter_instances(labels):
        row_start, col_start, row_stop, col_stop = bbox
        full_mask = np.zeros(labels.shape, dtype=bool)
        full_mask[row_start:row_stop, col_start:col_stop] = compact
        yield lbl, full_mask, int(compact.sum())


def _frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """DataFrame of ``rows``, or an empty frame with ``columns`` when none."""
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


class PlumeMask(Operator):
    """Binary plume mask from a single-band score or enhancement map.

    Thresholds the input (absolute number, Otsu, or percentile) and
    drops connected components smaller than ``min_area_px`` pixels — the
    Frankenberg et al. (2016) detection convention, also used by Varon
    et al. (2018, 2021) for S2/AVIRIS plume identification.

    Args:
        threshold: ``float`` (absolute), ``"otsu"``, or
            ``"percentile:<p>"`` with ``p`` in [0, 100].
        min_area_px: Minimum component size in pixels.
        connectivity: 4 or 8 connectivity for component labelling.
        nbins: Histogram bins for the ``"otsu"`` threshold (see
            :func:`geoops.segment.otsu_threshold`); ignored by the other modes.
            Default ``256``.

    Nodata pixels (non-finite or equal to the input's fill value) are
    excluded from the Otsu / percentile threshold and are always
    ``False`` in the mask, which declares ``fill_value=False``.

    Examples:
        >>> mask = PlumeMask(
        ...     threshold="percentile:99.5", min_area_px=50,
        ... )(enhancement)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    # a scene threshold, then connected components
    coupling: ClassVar[Coupling] = Coupling(couples={"space"}, reach=None)

    def __init__(
        self,
        *,
        threshold: ThresholdMode = "otsu",
        min_area_px: int = 50,
        connectivity: Connectivity = 8,
        nbins: int = 256,
    ) -> None:
        self.threshold = threshold
        self.min_area_px = min_area_px
        self.connectivity = connectivity
        self.nbins = nbins

    @over_frames
    def _apply(self, enhancement: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        mask = plume_mask(
            _single_band_nan(enhancement),
            threshold=self.threshold,
            min_area_px=self.min_area_px,
            connectivity=self.connectivity,
            nbins=self.nbins,
        )
        return wrap_like(enhancement, mask, fill_value=False)


class PlumeContours(Operator):
    """Connected-component labelling of a plume mask.

    Returns either an int32 label image (default) or a boolean mask of
    the retained components. Component size threshold and connectivity
    follow :func:`label_components`.

    Nodata pixels (non-finite -- e.g. a ``NaN`` score -- or equal to the
    input's fill value) are never part of a contour. In the boolean
    output they are ``False`` (``fill_value=False``). In the
    label image they hold the input's fill value when it is a
    non-positive int32 value (e.g. ``-9999``, so it cannot collide with
    a label); otherwise (``NaN``, no fill, or a
    positive fill) they are ``0`` (background), and a GeoArray output
    then carries ``fill_value=0``.

    Examples:
        >>> labels = PlumeContours(min_area_px=50)(mask)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    # connected components
    coupling: ClassVar[Coupling] = Coupling(couples={"space"}, reach=None)

    def __init__(
        self,
        *,
        min_area_px: int = 50,
        return_labels: bool = True,
        connectivity: Connectivity = 8,
    ) -> None:
        self.min_area_px = min_area_px
        self.return_labels = return_labels
        self.connectivity = connectivity

    @over_frames
    def _apply(self, mask: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        valid = valid_pixels(mask)
        labels = label_components(
            squeeze_single_band(np.asarray(mask)).astype(bool) & valid,
            min_area_px=self.min_area_px,
            connectivity=self.connectivity,
        )
        if not self.return_labels:
            return wrap_like(mask, labels > 0, fill_value=False)
        # ``carried_fill`` is None unless the fill is an int32 value; a
        # positive fill would collide with a component label.
        fill = carried_fill(mask, labels.dtype)
        if fill is None or isinstance(fill, bool | np.bool_) or fill > 0:
            fill = 0
        return wrap_filled(mask, labels, fill_value=fill, valid=valid)


class PlumeFootprint(Operator):
    """Vectorize a plume mask into polygons with per-plume metadata.

    The region properties come from the same machinery as
    :class:`geoops.measure.RegionProps` (defaults to
    :data:`geoops.measure.DEFAULT_REGIONPROPS`, pixel units) and the
    polygons from :func:`georeader.vectorize.get_polygons`. Returns a
    ``GeoDataFrame`` with one row per surviving plume: ``geometry``, ``area_m2``,
    ``centroid``,
    ``mean_enhancement`` / ``max_enhancement`` (if ``enhancement`` is
    supplied), ``n_pixels``, ``label_id``, and skimage region properties such as
    ``major_axis_length``, ``orientation``, ``eccentricity``, ``solidity``,
    ``perimeter``, ``bbox-*``, and ``inertia_tensor_eigvals-*``.

    Nodata pixels never form a polygon (of a boolean mask, or of a label
    image whose fill is positive, such as a uint8 raster's ``255``),
    non-positive labels are background, and nodata ``enhancement`` pixels are ignored
    by the per-polygon statistics.

    Geo-dependent: polygonisation needs the carrier's transform and CRS,
    so the input must be a georeferenced ``GeoArray`` (plain arrays
    raise ``TypeError``). Areas are in m^2 and the simplify tolerance in
    m, so the CRS must be projected with metre units; a geographic CRS
    raises ``ValueError`` (reproject with :class:`geoops.geom.Reproject`
    first).

    Args:
        min_area_m2: Drop polygons smaller than this area (m^2).
        simplify_tolerance: Douglas-Peucker tolerance (m) for polygon
            simplification; ``None`` to skip.
        properties: Region properties forwarded to ``regionprops_table``.
        extra_properties: User-defined ``regionprops_table`` callables.

    Called as ``op(mask, enhancement=None)``. The optional ``enhancement``
    carrier must sit on the mask's pixel grid (``ValueError`` otherwise);
    it enables the ``mean_enhancement`` / ``max_enhancement`` statistics
    per polygon.

    Examples:
        >>> gdf = PlumeFootprint(min_area_m2=500.0)(mask, kg_m2)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling(
        couples={"space"}, reach=None, kind="conversion"
    )

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        min_area_m2: float = 500.0,
        simplify_tolerance: float | None = 15.0,
        properties: Sequence[str] | None = None,
        extra_properties: Sequence[Callable[..., Any]] | None = None,
    ) -> None:
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.properties = tuple(
            DEFAULT_REGIONPROPS if properties is None else properties
        )
        self.extra_properties = (
            None
            if extra_properties is None
            else tuple(
                reject_config_summary(extra_property, "extra_properties")
                for extra_property in extra_properties
            )
        )

    def _apply(
        self, mask: GeoArray, enhancement: GeoArray | np.ndarray | None = None
    ) -> gpd.GeoDataFrame:
        require_ndim(mask, (2, 3), "PlumeFootprint")
        require_geoarray(mask, "PlumeFootprint")
        require_projected_crs(mask, "PlumeFootprint")
        if enhancement is not None:
            require_grid_match(
                mask, enhancement, "PlumeFootprint", names=("mask", "enhancement")
            )
        mask_arr = squeeze_single_band(np.asarray(mask))
        valid = squeeze_single_band(valid_pixels(mask))
        if mask_arr.dtype == bool:
            labels = label_components(
                mask_arr & valid,
                min_area_px=1,
                connectivity=8,
            )
        else:
            # Non-positive labels (background, or a label image's fill
            # value such as -9999) are not plumes, and neither are nodata
            # pixels under a positive fill such as a uint8 raster's 255.
            is_plume = (mask_arr > 0) & valid
            labels = np.where(is_plume, mask_arr, 0).astype(np.int32, copy=False)
        enh = None if enhancement is None else _single_band_nan(enhancement)
        properties = list(self.properties)
        if enh is not None:
            properties.extend(
                prop
                for prop in ("centroid_weighted", "mean_intensity", "max_intensity")
                if prop not in properties
            )
        props = regionprops_frame(
            labels,
            intensity_image=enh,
            properties=properties,
            extra_properties=self.extra_properties,
        )
        # Cast keys to int defensively: skimage returns numpy-int labels and
        # pandas to_dict("index") propagates that dtype. Python int hashes
        # equal in CPython but the explicit cast makes the lookup contract
        # obvious and survives dtype changes upstream.
        #
        # When the caller's ``properties`` omits ``"label"`` we still need a
        # per-component lookup, so synthesise it from ``regionprops_table``'s
        # row order (which mirrors ``sorted(unique(labels) - {0})``).
        if props.empty:
            props_by_label = {}
        else:
            if "label" in props.columns:
                indexed = props.set_index("label", drop=False)
            else:
                synthetic_labels = sorted(
                    int(value) for value in np.unique(labels) if value != 0
                )
                indexed = props.copy()
                indexed.index = pd.Index(synthetic_labels[: len(indexed)])
            props_by_label = {
                int(label): record for label, record in indexed.to_dict("index").items()
            }

        rows: list[dict[str, Any]] = []
        for label_id in sorted(int(value) for value in np.unique(labels) if value != 0):
            component = labels == label_id
            num_pixels = int(component.sum())
            # One component can still vectorise to several 4-connected
            # polygons (diagonal-only contacts); union them.
            geometries = vectorize.get_polygons(
                component, min_area=0.0, tolerance=0.0, transform=affine_of(mask)
            )
            if not geometries:
                continue
            geometry = unary_union(geometries)
            if self.simplify_tolerance is not None:
                geometry = geometry.simplify(
                    self.simplify_tolerance, preserve_topology=True
                )
            area_m2 = float(geometry.area)
            if area_m2 < self.min_area_m2:
                continue
            component_values = None if enh is None else enh[component]
            region_props = props_by_label.get(label_id, {})

            # Bind region_props / component_values via default args to avoid
            # the late-binding closure trap that ruff B023 flags.
            def _nan_safe(
                reducer: Callable[[np.ndarray], float],
                key: str,
                props: dict[str, Any] = region_props,
                vals: np.ndarray | None = component_values,
            ) -> float:
                # Prefer the regionprops_table value, but fall back to the
                # NaN-aware reducer when skimage propagates NaN through
                # ``mean_intensity`` / ``max_intensity`` (it does not ignore
                # NaNs internally). ``vals`` is non-None here.
                region_value = props.get(key)
                if region_value is None or not np.isfinite(float(region_value)):
                    return float(reducer(vals))
                return float(region_value)

            mean_enhancement = (
                None
                if component_values is None
                else _nan_safe(np.nanmean, "mean_intensity")
            )
            max_enhancement = (
                None
                if component_values is None
                else _nan_safe(np.nanmax, "max_intensity")
            )
            rows.append(
                {
                    **region_props,
                    "geometry": geometry,
                    "area_m2": area_m2,
                    "centroid": geometry.centroid,
                    "mean_enhancement": mean_enhancement,
                    "max_enhancement": max_enhancement,
                    "n_pixels": num_pixels,
                    "label_id": label_id,
                }
            )
        if not rows:
            # No surviving components (empty mask, or all polygons filtered
            # out by ``min_area_m2``). Construct an empty GeoDataFrame with
            # the expected schema so downstream consumers see a stable
            # column layout instead of a constructor error.
            return gpd.GeoDataFrame(
                {
                    "geometry": [],
                    "area_m2": [],
                    "centroid": [],
                    "mean_enhancement": [],
                    "max_enhancement": [],
                    "n_pixels": [],
                    "label_id": [],
                },
                geometry="geometry",
                crs=mask.crs,
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=mask.crs)

    def get_config(self) -> dict[str, Any]:
        return {
            "min_area_m2": self.min_area_m2,
            "simplify_tolerance": self.simplify_tolerance,
            "properties": list(self.properties),
            "extra_properties": None
            if self.extra_properties is None
            else [
                {"callable": callable_name(extra_property)}
                for extra_property in self.extra_properties
            ],
        }


class WindAdvectionCone(Operator):
    """Mask likely downwind plume extent from a source and wind vector.

    Rasterizes a circular sector centered at ``source``, oriented along
    ``(wind_u, wind_v)``, with half-angle ``half_angle_deg`` and radius
    ``max_distance``. The mask is useful as a prior for plume
    segmentation when the source location and wind are known (e.g. from
    a reanalysis product), following the practice of Varon et al. (2018,
    2021).

    Geo-dependent: the cone is rasterised in CRS coordinates via the
    carrier's transform, so the input must be a georeferenced
    ``GeoArray`` (plain arrays raise ``TypeError``). ``max_distance`` is
    in metres and the cone angle is measured in the map plane, so the
    carrier CRS must be projected with metre units; a geographic CRS
    raises ``ValueError`` (reproject with :class:`geoops.geom.Reproject`
    first).

    Args:
        source: ``(x, y)`` source coordinates. If ``crs`` is supplied,
            ``source`` is interpreted in that CRS and reprojected to the
            carrier CRS (which must then be set); otherwise it is in
            carrier CRS coordinates (m).
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).
        half_angle_deg: Half-angle of the cone in degrees, in [0, 180];
            ``180`` is the full disc (see :func:`wind_advection_cone`).
        max_distance: Radius of the cone (m).
        crs: Optional CRS of ``source``.

    Examples:
        >>> cone = WindAdvectionCone(
        ...     source=(x0, y0), wind_u=3.0, wind_v=1.0,
        ...     half_angle_deg=30.0, max_distance=5000.0,
        ... )(enhancement)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    # depends only on the grid, the source and the wind
    coupling: ClassVar[Coupling] = Coupling(reads_values=False)

    def __init__(
        self,
        *,
        source: Sequence[float],
        wind_u: float,
        wind_v: float,
        half_angle_deg: float = 30.0,
        max_distance: float = 5000.0,
        crs: str | None = None,
    ) -> None:
        self.source = as_tuple(source)
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.half_angle_deg = half_angle_deg
        self.max_distance = max_distance
        self.crs = crs

    @over_frames
    def _apply(self, geo_array: GeoArray) -> GeoArray:
        require_geoarray(geo_array, "WindAdvectionCone")
        require_projected_crs(geo_array, "WindAdvectionCone")
        arr = squeeze_single_band(np.asarray(geo_array))
        source = self.source
        if self.crs is not None:
            if geo_array.crs is None:
                raise ValueError(
                    "WindAdvectionCone was given source crs="
                    f"{self.crs!r} but the input GeoArray has no CRS, so "
                    "the source cannot be reprojected onto it. Set the "
                    "carrier CRS, or pass source in carrier coordinates "
                    "with crs=None."
                )
            src_crs = CRS.from_user_input(self.crs)
            dst_crs = CRS.from_user_input(geo_array.crs)
            if src_crs != dst_crs:
                transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
                source = transformer.transform(*source)
        mask = wind_advection_cone(
            arr.shape,
            affine_of(geo_array),
            source=source,
            wind_u=self.wind_u,
            wind_v=self.wind_v,
            half_angle_deg=self.half_angle_deg,
            max_distance=self.max_distance,
        )
        return wrap_like(geo_array, mask, fill_value=False)


class IMEEstimate(Operator):
    r"""Integrated Mass Enhancement (IME) flux estimator.

    Implements the IME method of Varon et al. (2018), AMT
    *Quantifying methane point sources from fine-scale satellite
    observations of atmospheric methane plumes*:

    .. math::

        \mathrm{IME} \;=\; \sum_{i \in \mathrm{plume}}
                            \Omega_i \,\Delta A_i, \qquad
        Q \;=\; \frac{U_{\mathrm{eff}}\,\mathrm{IME}}{L}

    where :math:`\Omega_i` is the column enhancement (kg/m^2) of pixel
    :math:`i`, :math:`\Delta A_i` is its area (m^2), :math:`L` is the
    effective plume length (m) — see :func:`plume_length` — and
    :math:`U_{\mathrm{eff}}` is the effective wind speed (m/s). Varon
    et al. (2018) calibrate :math:`U_{\mathrm{eff}}` against LES
    simulations; this operator takes the calibrated value as
    ``wind_speed``.

    The ``enhancement`` carrier MUST be in kg/m^2. Use
    :class:`ColumnToMass` upstream to convert from ppm m or mol/m^2:
    passing other units silently produces nonsense. Nodata enhancement
    pixels (non-finite or equal to the fill value) add nothing to the IME.

    Geo-dependent: pixel area (m^2) and plume length (m) are derived from
    the carrier's transform, so the input must be a georeferenced
    ``GeoArray`` (plain arrays raise ``TypeError``) in a projected CRS
    with metre units; a geographic CRS raises ``ValueError`` (reproject
    with :class:`geoops.geom.Reproject` first).

    Called as ``op(enhancement, plume_mask)``: ``plume_mask`` is a boolean
    carrier selecting plume pixels on the enhancement's pixel grid (same
    shape, transform and CRS; ``ValueError`` otherwise -- a mask on another
    grid would be indexed against the wrong pixels).

    Args:
        wind_speed: Effective wind speed :math:`U_{\mathrm{eff}}` in m/s.
        length_method: Length estimator: ``"max_axis"``, ``"convex_hull"``,
            or ``"skeleton"``. See :func:`plume_length`; ``"skeleton"``
            raises ``ValueError`` for compact plumes whose skeleton
            collapses to a point.
        pixel_area_m2: Override the pixel area (m^2); default is the
            absolute input transform determinant in the projected CRS's
            metres (exact on an equal-area projection; on UTM it ignores
            the projection's scale factor, a < 0.1 % error near the
            central meridian).
        return_uncertainty: Append ``emission_rate_uncertainty_kg_s``.
        uncertainty_fraction: Fractional 1-sigma uncertainty on Q. The
            default 0.5 follows Varon et al. (2018) Table 3.

    The call returns a dict with keys ``ime_kg``, ``length_m``,
    ``wind_speed_m_s``, ``emission_rate_kg_s``, and optionally
    ``emission_rate_uncertainty_kg_s``.

    Examples:
        Estimate Q in kg/s from a ppm m enhancement map::

            from plumax.geo import ColumnToMass, IMEEstimate, PlumeMask

            kg_m2 = ColumnToMass(units_in="ppm_m")(enhancement)
            mask = PlumeMask(threshold="percentile:99")(kg_m2)
            result = IMEEstimate(
                wind_speed=3.5,
                length_method="convex_hull",
            )(kg_m2, mask)
            print(result["emission_rate_kg_s"])
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling(
        couples={"space"}, reach=None, kind="conversion"
    )

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        wind_speed: float,
        length_method: Literal["max_axis", "convex_hull", "skeleton"] = "max_axis",
        pixel_area_m2: float | None = None,
        return_uncertainty: bool = True,
        uncertainty_fraction: float = 0.5,
    ) -> None:
        self.wind_speed = wind_speed
        self.length_method = length_method
        self.pixel_area_m2 = pixel_area_m2
        self.return_uncertainty = return_uncertainty
        require_positive("uncertainty_fraction", uncertainty_fraction, allow_zero=True)
        self.uncertainty_fraction = uncertainty_fraction

    def _apply(
        self, geo_array: GeoArray, plume_mask: GeoArray | np.ndarray
    ) -> dict[str, float]:
        require_ndim(geo_array, (2, 3), "IMEEstimate")
        require_geoarray(geo_array, "IMEEstimate")
        require_projected_crs(geo_array, "IMEEstimate")
        require_grid_match(
            geo_array, plume_mask, "IMEEstimate", names=("enhancement", "plume_mask")
        )
        mask, enhancement = _mask_and_enhancement(geo_array, plume_mask)
        area = (
            self.pixel_area_m2
            if self.pixel_area_m2 is not None
            else pixel_area(affine_of(geo_array))
        )
        ime_kg = _masked_mass(enhancement, mask, area)
        length_m = plume_length(mask, affine_of(geo_array), method=self.length_method)
        rate = 0.0 if length_m == 0.0 else ime_kg * self.wind_speed / length_m
        out = {
            "ime_kg": ime_kg,
            "length_m": float(length_m),
            "wind_speed_m_s": float(self.wind_speed),
            "emission_rate_kg_s": float(rate),
        }
        if self.return_uncertainty:
            out["emission_rate_uncertainty_kg_s"] = float(
                abs(rate) * self.uncertainty_fraction
            )
        return out


class CrossSectionalFlux(Operator):
    r"""Estimate cross-sectional flux at downwind transects.

    For each transect at downwind distance :math:`d_k`, the operator
    integrates column mass across the plume:

    .. math::

        Q_k \;=\; U \int_{-\infty}^{\infty}
                       \Omega(d_k, y) \, dy
            \;\approx\; U \sum_{i \in \mathrm{strip}_k}
                       \Omega_i \,\Delta y_i

    where :math:`\mathrm{strip}_k` is the set of plume pixels whose
    along-wind distance to the source falls in the half-open interval
    :math:`[d_k - \Delta y / 2,\, d_k + \Delta y / 2)`, so a pixel whose
    centre lies on a strip boundary is counted once. This is the cross-section method of Krings et al. (2011)
    and Varon et al. (2018, appendix); a flat downwind profile of
    :math:`Q_k` versus :math:`d_k` confirms a steady-state estimate.

    The enhancement carrier MUST be in kg/m^2 (see :class:`ColumnToMass`).
    Nodata enhancement pixels (non-finite or equal to the fill value) add
    nothing to a transect's flux.
    The implementation assumes (approximately) square pixels: it uses
    :math:`\Delta y = \sqrt{\mathrm{pixel\_area}}` as the across-wind
    pixel size. For strongly anisotropic pixel grids, reproject first.

    Geo-dependent: transect placement and pixel size come from the
    carrier's transform and CRS, so the input must be a georeferenced
    ``GeoArray`` (plain arrays raise ``TypeError``) in a projected CRS
    with metre units; a geographic CRS raises ``ValueError`` (reproject
    with :class:`geoops.geom.Reproject` first).

    Called as ``op(enhancement, plume_mask)``: ``plume_mask`` is a boolean
    plume carrier on the enhancement's pixel grid (``ValueError``
    otherwise).

    Args:
        source: ``(x, y)`` source coordinates in the carrier CRS (m).
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).
        n_transects: Number of downwind transects to evaluate.
        transect_spacing_m: Along-wind distance between transects (m).

    The call returns a ``geopandas.GeoDataFrame`` with one row per
    transect: ``geometry`` (across-wind ``LineString``), ``transect_id``,
    ``distance_m``, ``flux_kg_s``, ``n_pixels``.

    Examples:
        Evaluate fluxes at 100 m, 200 m, 300 m downwind::

            gdf = CrossSectionalFlux(
                source=(x0, y0),
                wind_u=3.0, wind_v=1.0,
                n_transects=3, transect_spacing_m=100.0,
            )(kg_m2, mask)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling(
        couples={"space"}, reach=None, kind="conversion"
    )

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        source: Sequence[float],
        wind_u: float,
        wind_v: float,
        n_transects: int = 5,
        transect_spacing_m: float = 100.0,
    ) -> None:
        self.source = as_tuple(source)
        self.wind_u = wind_u
        self.wind_v = wind_v
        self.n_transects = n_transects
        self.transect_spacing_m = transect_spacing_m

    def _apply(
        self, geo_array: GeoArray, plume_mask: GeoArray | np.ndarray
    ) -> gpd.GeoDataFrame:
        require_ndim(geo_array, (2, 3), "CrossSectionalFlux")
        require_geoarray(geo_array, "CrossSectionalFlux")
        require_projected_crs(geo_array, "CrossSectionalFlux")
        require_grid_match(
            geo_array,
            plume_mask,
            "CrossSectionalFlux",
            names=("enhancement", "plume_mask"),
        )
        wind_norm = float(np.hypot(self.wind_u, self.wind_v))
        # e_∥, e_⊥ of the wind; raises on a zero wind vector
        e_par, e_perp = wind_frame(self.wind_u, self.wind_v)
        mask, enhancement = _mask_and_enhancement(geo_array, plume_mask)
        area = pixel_area(affine_of(geo_array))
        width = float(np.sqrt(area))
        easting, northing = pixel_centers(mask.shape, affine_of(geo_array))
        # Δx = x − x_source
        offset_x = easting - self.source[0]
        # Δy = y − y_source
        offset_y = northing - self.source[1]
        # a = (Δx u + Δy v) / ‖w‖, written out and divided last: a matmul, or a
        # dot with the unit axes e_∥, rounds differently and moves a pixel centre
        # on a strip edge into the next strip
        along = (offset_x * self.wind_u + offset_y * self.wind_v) / wind_norm
        # c = (−Δx v + Δy u) / ‖w‖
        across = (-offset_x * self.wind_v + offset_y * self.wind_u) / wind_norm
        rows: list[dict[str, Any]] = []
        for idx in range(self.n_transects):
            distance = (idx + 1) * self.transect_spacing_m
            # Half-open strips [d − w/2, d + w/2) tile the along-wind axis,
            # so a pixel centre on a strip boundary counts in one strip only.
            offset = along - distance
            in_strip = (offset >= -width / 2.0) & (offset < width / 2.0)
            on_transect = mask & in_strip
            # Q_k = (U / w) ∑ over the strip of Ωᵢ ΔA, which is U ∑ Ωᵢ w
            flux = _masked_mass(enhancement, on_transect, area) * wind_norm / width
            half_len = max(
                float(np.max(np.abs(across[mask]))) if mask.any() else width,
                width,
            )
            # s_c = s₀ + d e_∥
            centre = np.asarray(self.source, dtype=float) + distance * e_par
            line = transect_line(centre, e_perp, -half_len, half_len)
            rows.append(
                {
                    "geometry": line,
                    "transect_id": idx + 1,
                    "distance_m": float(distance),
                    "flux_kg_s": flux,
                    "n_pixels": int(on_transect.sum()),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=geo_array.crs)


class ColumnToMass(Operator):
    """Convert column enhancement units (ppm m / mol m^-2 / kg m^-2).

    Thin wrapper over :func:`convert_column_units`. The ppm m conversion
    uses the ideal-gas air density at the layer's pressure and temperature,
    the ISA sea-level 101 325 Pa and 288.15 K unless the scene's are given;
    see that function's docstring for the exact relation. Nodata
    pixels (non-finite or equal to the fill value) are not converted:
    they hold ``NaN`` and the output declares ``fill_value=NaN``
    (a converted column is a new quantity, so the input's fill -- e.g.
    ``-9999`` ppm m -- is not carried over).

    Args:
        gas: ``"CH4"`` (default) or ``"CO2"``.
        units_in: Units of the input carrier.
        units_out: Desired output units.
        pressure_pa: Layer pressure p [Pa] for the ppm m conversion.
        temperature_k: Layer temperature T [K] for the ppm m conversion.

    Examples:
        >>> kg_m2 = ColumnToMass(
        ...     gas="CH4", units_in="ppm_m", units_out="kg_m2",
        ... )(enhancement)
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling()

    def __init__(
        self,
        *,
        gas: str = "CH4",
        units_in: ColumnUnit = "ppm_m",
        units_out: ColumnUnit = "kg_m2",
        pressure_pa: float = P_STANDARD,
        temperature_k: float = T_STANDARD,
    ) -> None:
        self.gas = gas
        self.units_in = units_in
        self.units_out = units_out
        self.pressure_pa = pressure_pa
        self.temperature_k = temperature_k

    def _apply(self, column: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        out = convert_column_units(
            np.asarray(column),
            gas=self.gas,
            units_in=self.units_in,
            units_out=self.units_out,
            pressure_pa=self.pressure_pa,
            temperature_k=self.temperature_k,
        )
        if np.ndim(out) < 2:
            return wrap_like(column, out)
        out = np.asarray(out, dtype=np.result_type(out, np.float32))
        return wrap_filled(column, out, fill_value=np.nan)


def _iter_instances(
    labels: np.ndarray,
) -> list[tuple[int, np.ndarray, tuple[int, int, int, int]]]:
    """Return ``(label, compact_mask, bbox)`` per non-zero label."""
    from scipy import ndimage

    work = np.where(labels > 0, labels, 0).astype(np.int64, copy=False)
    ids = np.unique(work)
    ids = ids[ids > 0]
    if ids.size == 0:
        return []
    slices = ndimage.find_objects(work)
    out: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    for lbl_value in ids:
        lbl = int(lbl_value)
        bounds = slices[lbl - 1]
        if bounds is None:
            continue
        row_slice, col_slice = bounds
        compact = work[bounds] == lbl
        bbox = (
            int(row_slice.start),
            int(col_slice.start),
            int(row_slice.stop),
            int(col_slice.stop),
        )
        out.append((lbl, compact, bbox))
    return out


class PlumeShapeFilter(Operator):
    """Drop instance labels that fail a fiber-shape morphology gate.

    Adapted from Pérez Carrasco et al. (2026), ``physical_constraints.
    apply_physical_constraints`` — a post-Mask R-CNN gate that distinguishes
    elongated plume features from compact false positives. Per instance:

    1. ``area = mask.sum()``; drop when below ``min_area_px``.
    2. ``fiber_length`` = longest Euclidean path through the skeleton, in
       pixel widths (diagonal steps count ``sqrt(2)``; see
       :func:`geoops.measure.skeleton_length`).
    3. ``major_axis`` = ``regionprops.axis_major_length`` — the PCA-based
       major axis of the equivalent ellipse. The original paper uses
       ``cv2.minAreaRect`` (rotated minimum-area rectangle), avoided here
       to keep the optional-dependency surface unchanged; the two measures
       agree closely for typical plume morphologies.
    4. ``fiber_width = area / fiber_length``.
    5. Keep iff
       ``min_fiber_to_major_ratio <= fiber_length / major_axis
       <= max_fiber_to_major_ratio`` AND
       ``min_fiber_width <= fiber_width <= max_fiber_width``.

    Surviving instances keep their original integer labels; rejected
    instances are mapped to background (``0``).

    Args:
        min_area_px: Minimum pixel count to consider an instance.
        min_fiber_to_major_ratio: Lower bound on ``fiber/major``.
        max_fiber_to_major_ratio: Upper bound on ``fiber/major``.
        min_fiber_width: Lower bound on ``area / fiber_length``.
        max_fiber_width: Upper bound on ``area / fiber_length``.
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    # shape measures of connected components
    coupling: ClassVar[Coupling] = Coupling(couples={"space"}, reach=None)

    def __init__(
        self,
        *,
        min_area_px: int = 50,
        min_fiber_to_major_ratio: float = 1.0,
        max_fiber_to_major_ratio: float = 10.0,
        min_fiber_width: float = 1.0,
        max_fiber_width: float = 100.0,
    ) -> None:
        require_positive("min_area_px", min_area_px, allow_zero=True)
        if min_fiber_to_major_ratio < 0 or max_fiber_to_major_ratio <= 0:
            raise ValueError("fiber-to-major ratio bounds must be non-negative")
        if min_fiber_to_major_ratio > max_fiber_to_major_ratio:
            raise ValueError("min_fiber_to_major_ratio must be <= max")
        if min_fiber_width < 0 or max_fiber_width <= 0:
            raise ValueError("fiber width bounds must be non-negative")
        if min_fiber_width > max_fiber_width:
            raise ValueError("min_fiber_width must be <= max")
        self.min_area_px = int(min_area_px)
        self.min_fiber_to_major_ratio = float(min_fiber_to_major_ratio)
        self.max_fiber_to_major_ratio = float(max_fiber_to_major_ratio)
        self.min_fiber_width = float(min_fiber_width)
        self.max_fiber_width = float(max_fiber_width)

    @over_frames
    def _apply(self, geo_labels: GeoArray | np.ndarray) -> GeoArray | np.ndarray:
        labels = _instance_labels(geo_labels)
        kept: set[int] = set()
        for lbl, compact, _bbox in _iter_instances(labels):
            area = int(compact.sum())
            if area < self.min_area_px:
                continue
            fiber_length = skeleton_length(compact)
            if fiber_length <= 0:
                continue
            major_axis = _major_axis_length(compact)
            if major_axis <= 0:
                continue
            ratio = fiber_length / major_axis
            fiber_width = area / fiber_length
            if not (
                self.min_fiber_to_major_ratio <= ratio <= self.max_fiber_to_major_ratio
            ):
                continue
            if not self.min_fiber_width <= fiber_width <= self.max_fiber_width:
                continue
            kept.add(lbl)

        if not kept:
            out = np.zeros_like(labels, dtype=np.int32)
        else:
            keep_mask = np.isin(labels, list(kept))
            out = np.where(keep_mask, labels, 0).astype(np.int32, copy=False)
        return wrap_like(geo_labels, out, fill_value=0)


def _major_axis_length(compact_mask: np.ndarray) -> float:
    """PCA-based major-axis length of a compact binary mask (in pixels)."""
    props = regionprops_frame(compact_mask, properties=("axis_major_length",))
    if props.empty:
        return 0.0
    return float(props["axis_major_length"].iloc[0])


def _in_out_mask_stats(
    values: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """``(mean_plume, background_mean, background_std, contrast, z_score, ipa)`` from
    Pérez Carrasco et al. (2026), ``rfc_utils.xch4_metrics``."""
    plume_data = values[mask]
    plume_data = plume_data[np.isfinite(plume_data)]
    if plume_data.size == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan, nan
    background = values[~mask]
    background = background[np.isfinite(background)]
    if background.size == 0:
        # Fallback used in the paper: synthesise a background by shifting
        # the plume distribution down by one std. Keeps the metric
        # well-defined when the mask covers the entire valid area.
        background = plume_data - float(np.std(plume_data))
    # μ_b = mean of the background
    background_mean = float(np.mean(background))
    # σ_b = std of the background
    background_std = float(np.std(background))
    # μ_p = mean of the plume
    mean_plume = float(np.mean(plume_data))
    # C = μ_p / μ_b
    contrast = mean_plume / background_mean if background_mean != 0 else float("nan")
    # z = (μ_p − μ_b) / σ_b
    z_score = (
        (mean_plume - background_mean) / background_std
        if background_std > 0
        else float("nan")
    )
    # IPA = (Σ_p − μ_b N_p) / N_p
    ipa = (
        float(np.sum(plume_data) - background_mean * plume_data.size) / plume_data.size
    )
    return mean_plume, background_mean, background_std, contrast, z_score, ipa


class PlumeColumnStats(Operator):
    """Per-instance in-mask vs background statistics of a column field.

    Adapted from Pérez Carrasco et al. (2026), ``rfc_utils.xch4_metrics``.
    For every non-zero label in the input map, returns a row with the
    in-mask mean, the out-of-mask (background) mean / std, and three
    derived metrics used by the paper's Random-Forest false-plume
    classifier:

    - ``contrast`` = ``mean_plume / mean_background``
    - ``z_score`` = ``(mean_plume - mean_background) / std_background``
    - ``intensity_per_area`` = ``(sum_plume - mean_background * n_plume)
      / n_plume``

    Nodata column pixels (non-finite or equal to the column carrier's
    fill value) are left out of every statistic.

    Called as ``op(labels, column)``: ``column`` is the column field
    (single-band ``GeoArray`` or plain array) on the label map's pixel
    grid (``ValueError`` otherwise).
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling(
        couples={"space"}, reach=None, kind="conversion"
    )

    _terminal: ClassVar[bool] = True

    def _apply(
        self, geo_labels: GeoArray | np.ndarray, column: GeoArray | np.ndarray
    ) -> pd.DataFrame:
        labels, col_arr, _ = _label_fields(geo_labels, column, "PlumeColumnStats")
        rows: list[dict[str, Any]] = []
        for lbl, full_mask, area in _instances_full(labels):
            mean_plume, background_mean, background_std, contrast, z_score, ipa = (
                _in_out_mask_stats(col_arr, full_mask)
            )
            rows.append(
                {
                    "label": lbl,
                    "area": area,
                    "mean_enhancement": mean_plume,
                    "background_mean": background_mean,
                    "background_std": background_std,
                    "contrast": contrast,
                    "z_score": z_score,
                    "intensity_per_area": ipa,
                }
            )
        return _frame(
            rows,
            [
                "label",
                "area",
                "mean_enhancement",
                "background_mean",
                "background_std",
                "contrast",
                "z_score",
                "intensity_per_area",
            ],
        )


def _qnd_polynomial(
    values: np.ndarray,
    mask: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Quantile-Normality-Deviation polynomial coefficients.

    Mirrors Pérez Carrasco et al. (2026), ``rfc_utils.get_QND_poly``: at
    each percentile ``q in [0.01, 0.99]``, take the absolute difference
    between the empirical CDF of the in-mask values and the CDF of a
    Gaussian fit to *all finite values in ``values``*, then fit a polynomial
    of the given degree. Returns coefficients in ``np.polyfit`` order
    (highest power first). Returns ``np.full(degree + 1, nan)`` when fewer
    than 10 finite in-mask samples are available.
    """
    from scipy import stats

    in_mask = values[mask]
    in_mask = in_mask[np.isfinite(in_mask)]
    if in_mask.size < 10:
        return np.full(degree + 1, np.nan)
    scene = values[np.isfinite(values)]
    if scene.size == 0:
        return np.full(degree + 1, np.nan)
    quantiles = np.linspace(0.01, 0.99, 100)
    q_values = np.percentile(in_mask, quantiles * 100.0)
    theoretical_cdf = stats.norm(loc=scene.mean(), scale=scene.std()).cdf(q_values)
    sorted_in = np.sort(in_mask)
    empirical_cdf = np.searchsorted(sorted_in, q_values, side="right") / sorted_in.size
    # d = |F_emp − F_theory|
    ks_distance = np.abs(empirical_cdf - theoretical_cdf)
    return np.polyfit(quantiles, ks_distance, degree)


def _dbscan_clump_count(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    neighbourhood_radius: float,
    min_samples: int,
    perc_threshold: float,
) -> int:
    """Number of DBSCAN clumps among in-mask pixels above ``perc_threshold``.

    Wraps ``rfc_utils.DBSCAN_clusters`` from the reference implementation.
    Returns ``0`` when no in-mask pixels clear the percentile gate.

    Counts unique non-noise DBSCAN cluster labels directly rather than
    re-rasterising to a 4-connected canvas — DBSCAN's ``eps`` already
    encodes the proximity rule the paper relies on, and a separate
    connected-component pass would split a single density-connected
    DBSCAN cluster into multiple "clumps" whenever its members are within
    ``eps`` but not 4-adjacent.
    """
    DBSCAN = import_optional(
        "sklearn.cluster",
        install="plumax[geo]",
        needed_by="PlumeQNDFeatures",
        error=MissingExtraError,
    ).DBSCAN

    in_mask = values[mask]
    in_mask = in_mask[np.isfinite(in_mask)]
    if in_mask.size == 0:
        return 0
    threshold = float(np.percentile(in_mask, perc_threshold))
    high_mask = (values > threshold) & mask & np.isfinite(values)
    coords = np.argwhere(high_mask)
    if coords.shape[0] == 0:
        return 0
    labels_db = DBSCAN(eps=neighbourhood_radius, min_samples=min_samples).fit_predict(
        coords
    )
    non_noise = labels_db[labels_db != -1]
    return int(np.unique(non_noise).size)


class PlumeQNDFeatures(Operator):
    """Per-instance Quantile-Normality-Deviation features for plume QA.

    Adapted from Pérez Carrasco et al. (2026) — the feature pipeline that
    feeds the Random-Forest false-plume classifier
    (``rfc_utils.get_QND_df_cms`` / ``get_plume_info``). For each non-zero
    label in the input map this operator emits one row containing:

    - ``qnd_column_*`` — polynomial coefficients of the
      Kolmogorov-Smirnov distance between the in-mask CDF of the column
      field and a Gaussian fit to the full scene's column distribution,
      sampled across 100 percentiles in ``[0.01, 0.99]``.
    - ``qnd_albedo_*`` — same construction over the optional albedo field.
    - ``n_clumps`` — DBSCAN cluster count of in-mask pixels above the
      ``perc_threshold`` percentile (per the paper's clump-count signal).
    - ``contrast`` / ``z_score`` / ``intensity_per_area`` — the same stats
      that :class:`PlumeColumnStats` emits, included so a single call
      assembles the full feature vector used by the paper.

    Nodata column / albedo pixels (non-finite or equal to the carrier's
    fill value) are left out of every feature.

    Called as ``op(labels, column, albedo=None)``: ``column`` is the
    column-enhancement carrier and ``albedo`` an optional albedo carrier,
    both on the label map's pixel grid (``ValueError`` otherwise). Without
    ``albedo`` the ``qnd_albedo_*`` coefficients are NaN.

    Args:
        degree: Degree of the QND polynomial (paper default: 6).
        eps: DBSCAN ``eps`` (pixels).
        min_samples: DBSCAN ``min_samples``.
        perc_threshold: Percentile (within the instance mask) above which
            pixels are eligible for clustering.
    """

    backends: ClassVar[tuple[str, ...] | None] = ("numpy",)
    coupling: ClassVar[Coupling] = Coupling(
        couples={"space"}, reach=None, kind="conversion"
    )

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        *,
        degree: int = 6,
        eps: float = 50.0,
        min_samples: int = 50,
        perc_threshold: float = 98.0,
    ) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1")
        require_positive("eps", eps)
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        if not 0.0 < perc_threshold < 100.0:
            raise ValueError("perc_threshold must be in (0, 100)")
        self.degree = int(degree)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.perc_threshold = float(perc_threshold)

    def _apply(
        self,
        geo_labels: GeoArray | np.ndarray,
        column: GeoArray | np.ndarray,
        albedo: GeoArray | np.ndarray | None = None,
    ) -> pd.DataFrame:
        labels, col_arr, alb_arr = _label_fields(
            geo_labels, column, "PlumeQNDFeatures", albedo
        )

        col_keys = [f"qnd_column_c{power}" for power in range(self.degree + 1)]
        alb_keys = [f"qnd_albedo_c{power}" for power in range(self.degree + 1)]
        rows: list[dict[str, Any]] = []
        for lbl, full_mask, area in _instances_full(labels):
            qnd_col = _qnd_polynomial(col_arr, full_mask, self.degree)
            qnd_alb = (
                _qnd_polynomial(alb_arr, full_mask, self.degree)
                if alb_arr is not None
                else np.full(self.degree + 1, np.nan)
            )
            num_clumps = _dbscan_clump_count(
                col_arr,
                full_mask,
                neighbourhood_radius=self.eps,
                min_samples=self.min_samples,
                perc_threshold=self.perc_threshold,
            )
            _mp, _bm, _bs, contrast, z_score, ipa = _in_out_mask_stats(
                col_arr, full_mask
            )
            row: dict[str, Any] = {
                "label": lbl,
                "area": area,
                "n_clumps": num_clumps,
                "contrast": contrast,
                "z_score": z_score,
                "intensity_per_area": ipa,
            }
            row.update(dict(zip(col_keys, qnd_col.tolist(), strict=True)))
            row.update(dict(zip(alb_keys, qnd_alb.tolist(), strict=True)))
            rows.append(row)

        return _frame(
            rows,
            [
                "label",
                "area",
                "n_clumps",
                "contrast",
                "z_score",
                "intensity_per_area",
                *col_keys,
                *alb_keys,
            ],
        )
