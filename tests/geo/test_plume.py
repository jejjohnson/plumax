"""The ten plume operators ported from geotoolz's ``plume`` (#134).

Ported from geotoolz's ``tests/test_plume.py`` at the pinned geotoolz-ops rev,
minus the SBMP cases (SBMP moved to radtrax as ``MBMP``). Needs the ``[geo]``
extra (in the dev group).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest
import rasterio
from geoops.segment import resolve_threshold
from georeader.geotensor import GeoTensor
from pipekit import Operator

from plumax import geo
from plumax.geo import convert_column_units
from tests.geo._carriers import fill_pixel_mask, time_stack, toy_geotensor


def test_plumax_geo_exports_the_plume_operators() -> None:
    from plumax.geo import plume, plume_array

    assert geo.PlumeMask is plume.PlumeMask
    assert geo.ColumnToMass is plume.ColumnToMass
    assert geo.wind_advection_cone is plume_array.wind_advection_cone


def _gt(values: np.ndarray) -> GeoTensor:
    # Explicit -9999 fill: GeoTensor's default fill of 0 would mark the
    # zero-valued backgrounds used throughout these tests as nodata.
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )


def test_plume_mask_otsu_matches_resolved_threshold() -> None:
    values = np.r_[np.zeros(50), np.full(50, 10.0)].reshape(10, 10)
    gt = _gt(values)

    out = geo.PlumeMask(threshold="otsu", min_area_px=1)(gt)

    assert np.array_equal(np.asarray(out), values > resolve_threshold(values, "otsu"))
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"


def test_plume_mask_percentile_keeps_top_half_percent() -> None:
    values = np.arange(10_000, dtype=float).reshape(100, 100)

    out = geo.PlumeMask(threshold="percentile:99.5", min_area_px=1)(_gt(values))

    assert int(np.asarray(out).sum()) == 50


def test_plume_contours_labels_connected_components() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0:2, 0:2] = True
    mask[4, 4] = True

    labels = geo.PlumeContours(min_area_px=2)(_gt(mask))

    assert set(np.unique(np.asarray(labels))) == {0, 1}
    assert np.asarray(labels)[0, 0] == 1
    assert np.asarray(labels)[4, 4] == 0


def test_plume_footprint_area_and_enhancement_stats() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:4] = True
    enhancement = np.arange(16, dtype=float).reshape(4, 4)

    gdf = geo.PlumeFootprint(min_area_m2=1.0, simplify_tolerance=None)(
        _gt(mask), _gt(enhancement)
    )

    assert len(gdf) == 1
    row = gdf.iloc[0]
    assert row.area_m2 == pytest.approx(6 * 100.0)
    assert row.n_pixels == 6
    assert row.area_m2 == pytest.approx(row.area * 100.0)
    assert row.mean_enhancement == pytest.approx(float(enhancement[mask].mean()))
    assert row.max_enhancement == pytest.approx(float(enhancement[mask].max()))
    assert row.area == pytest.approx(6.0)
    assert row.major_axis_length > 0.0
    assert row["bbox-0"] == 1
    assert row["bbox-1"] == 1
    assert row["bbox-2"] == 3
    assert row["bbox-3"] == 4
    assert gdf.crs == "EPSG:32629"

    no_stats = geo.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
    )(_gt(mask))
    assert no_stats.iloc[0].mean_enhancement is None
    assert no_stats.iloc[0].max_enhancement is None


def test_plume_footprint_ignores_a_positive_label_fill() -> None:
    """A uint8 label image's 255 nodata is not a plume (#177 review)."""
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1:3, 1:3] = 1
    labels[4:, :] = 255  # nodata
    gt = GeoTensor(
        values=labels,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0),
        crs="EPSG:32629",
        fill_value_default=255,
    )
    gdf = geo.PlumeFootprint(min_area_m2=1.0, simplify_tolerance=None)(gt)
    assert gdf["label_id"].tolist() == [1]
    assert gdf["n_pixels"].tolist() == [4]


def test_plume_footprint_empty_mask_returns_empty_geodataframe() -> None:
    """Empty masks (or all-filtered components) must yield a valid empty GDF.

    Regression test: previously ``gpd.GeoDataFrame(rows, geometry="geometry")``
    raised ``ValueError`` when ``rows`` was empty because the ``geometry``
    column did not exist.
    """
    empty_mask = _gt(np.zeros((4, 4), dtype=bool))

    gdf = geo.PlumeFootprint(
        min_area_m2=1.0,
        simplify_tolerance=None,
    )(empty_mask)

    assert len(gdf) == 0
    assert gdf.crs == "EPSG:32629"
    expected_cols = {
        "geometry",
        "area_m2",
        "centroid",
        "mean_enhancement",
        "max_enhancement",
        "n_pixels",
        "label_id",
    }
    assert expected_cols.issubset(set(gdf.columns))
    assert gdf.geometry.name == "geometry"

    # Also covers the "all components filtered by min_area_m2" path: a
    # single-pixel mask filtered out by a huge ``min_area_m2``.
    one_pixel = np.zeros((4, 4), dtype=bool)
    one_pixel[0, 0] = True
    filtered = geo.PlumeFootprint(
        min_area_m2=1.0e9,
        simplify_tolerance=None,
    )(_gt(one_pixel))
    assert len(filtered) == 0
    assert filtered.crs == "EPSG:32629"


def test_wind_advection_cone_follows_wind_orientation() -> None:
    transform = rasterio.Affine.translation(-5.5, 5.5) * rasterio.Affine.scale(
        1.0, -1.0
    )
    gt = GeoTensor(values=np.zeros((11, 11)), transform=transform, crs="EPSG:32629")

    east = geo.WindAdvectionCone(
        source=(0.0, 0.0),
        wind_u=1.0,
        wind_v=0.0,
        half_angle_deg=20.0,
        max_distance=5.0,
    )(gt)
    north = geo.WindAdvectionCone(
        source=(0.0, 0.0),
        wind_u=0.0,
        wind_v=1.0,
        half_angle_deg=20.0,
        max_distance=5.0,
    )(gt)

    assert bool(np.asarray(east)[5, 8])
    assert not bool(np.asarray(east)[2, 5])
    assert bool(np.asarray(north)[2, 5])
    assert not bool(np.asarray(north)[5, 8])


_CONE_TRANSFORM = rasterio.Affine.translation(-5.5, 5.5) * rasterio.Affine.scale(
    1.0, -1.0
)


def test_cone_full_disc_at_180() -> None:
    from plumax.geo import wind_advection_cone

    cone = wind_advection_cone(
        (11, 11),
        _CONE_TRANSFORM,
        source=(0.0, 0.0),
        wind_u=1.0,
        wind_v=0.0,
        half_angle_deg=180.0,
        max_distance=4.0,
    )
    rows, cols = np.indices((11, 11))
    disc = np.hypot(cols - 5.0, rows - 5.0) <= 4.0
    np.testing.assert_array_equal(cone, disc)
    assert cone[5, 1]  # directly upwind of the source


@pytest.mark.parametrize("half_angle", [0.0, 30.0, 89.0, 90.0, 120.0, 150.0])
@pytest.mark.parametrize(("wind_u", "wind_v"), [(1.0, 0.0), (-2.0, 1.5), (0.3, -1.0)])
def test_cone_matches_brute_force_angle(
    half_angle: float, wind_u: float, wind_v: float
) -> None:
    from plumax.geo import wind_advection_cone

    shape, source, radius = (15, 15), (0.3, -0.2), 6.0
    transform = _CONE_TRANSFORM * rasterio.Affine.scale(15 / 11)
    cone = wind_advection_cone(
        shape,
        transform,
        source=source,
        wind_u=wind_u,
        wind_v=wind_v,
        half_angle_deg=half_angle,
        max_distance=radius,
    )
    wind_bearing = np.degrees(np.arctan2(wind_v, wind_u))
    expected = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        for col in range(shape[1]):
            x, y = transform * (col + 0.5, row + 0.5)
            dx, dy = x - source[0], y - source[1]
            if np.hypot(dx, dy) > radius:
                continue
            bearing = np.degrees(np.arctan2(dy, dx))
            angle = abs((bearing - wind_bearing + 180.0) % 360.0 - 180.0)
            expected[row, col] = angle <= half_angle + 1e-9
    np.testing.assert_array_equal(cone, expected)
    if half_angle > 90.0:
        # Wider than a half-plane: some pixels behind the crosswind line.
        xs, ys = np.meshgrid(np.arange(15) + 0.5, np.arange(15) + 0.5)
        x, y = transform * (xs, ys)
        assert ((x - source[0]) * wind_u + (y - source[1]) * wind_v < 0)[cone].any()


def test_contours_ignore_nan() -> None:
    score = np.array([[np.nan, 0.0], [0.0, 0.0]])

    labels = geo.PlumeContours(min_area_px=1)(score)
    mask = geo.PlumeContours(min_area_px=1, return_labels=False)(score)

    np.testing.assert_array_equal(np.asarray(labels), np.zeros((2, 2)))
    assert not np.asarray(mask).any()


def test_plume_mask_threads_otsu_nbins() -> None:
    from geoops.segment import otsu_threshold

    rng = np.random.default_rng(0)
    values = np.concatenate(
        [rng.normal(0.0, 1.0, 300), rng.normal(3.0, 1.0, 100), [40.0]]
    )
    values = np.pad(values, (0, 441 - values.size)).reshape(21, 21)
    coarse = otsu_threshold(values, nbins=4)
    assert coarse != otsu_threshold(values)
    assert resolve_threshold(values, "otsu", nbins=4) == coarse

    op = geo.PlumeMask(threshold="otsu", min_area_px=1, nbins=4)
    np.testing.assert_array_equal(np.asarray(op(_gt(values))), values > coarse)
    config = op.get_config()
    assert config["nbins"] == 4
    rebuilt = geo.PlumeMask(**config)
    np.testing.assert_array_equal(
        np.asarray(rebuilt(_gt(values))), np.asarray(op(_gt(values)))
    )


def test_ime_estimate_uses_integrated_mass_and_max_axis_length() -> None:
    mask = _gt(np.array([[True, True, True]]))
    enhancement = _gt(np.array([[2.0, 2.0, 2.0]]))

    estimate = geo.IMEEstimate(wind_speed=2.0, return_uncertainty=False)(
        enhancement, mask
    )

    assert estimate["ime_kg"] == pytest.approx(600.0)
    assert estimate["length_m"] == pytest.approx(20.0)
    assert estimate["emission_rate_kg_s"] == pytest.approx(60.0)


def test_ime_skeleton_length_and_uncertainty_fraction() -> None:
    # Bent 1-pixel-wide plume: along the top row, then down the right
    # column. The skeleton follows the bend (cutting the corner with one
    # diagonal step), so it is longer than the endpoint chord.
    mask_arr = np.zeros((5, 5), dtype=bool)
    mask_arr[0, :] = True
    mask_arr[:, 4] = True
    mask = _gt(mask_arr)
    enhancement = _gt(np.ones((5, 5), dtype=float))

    estimate = geo.IMEEstimate(
        wind_speed=2.0, length_method="skeleton", uncertainty_fraction=0.2
    )(enhancement, mask)
    max_axis = geo.IMEEstimate(wind_speed=2.0, length_method="max_axis")(
        enhancement, mask
    )

    # 3 + 3 orthogonal 10 m steps plus one diagonal step at the corner.
    expected_length = 60.0 + 10.0 * np.sqrt(2.0)
    assert estimate["ime_kg"] == pytest.approx(900.0)
    assert estimate["length_m"] == pytest.approx(expected_length)
    assert max_axis["length_m"] == pytest.approx(40.0 * np.sqrt(2.0))
    assert estimate["emission_rate_kg_s"] == pytest.approx(
        900.0 * 2.0 / expected_length
    )
    assert estimate["emission_rate_uncertainty_kg_s"] == pytest.approx(
        0.2 * 900.0 * 2.0 / expected_length
    )


def test_skeleton_length_diagonal_plume() -> None:
    from plumax.geo.plume_array import plume_length

    # An 8-connected diagonal plume is one component for PlumeMask, so it
    # must have a positive skeleton length (was 0 with 4-connectivity).
    mask = np.eye(10, dtype=bool)
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    assert plume_length(mask, transform, method="skeleton") == pytest.approx(
        90.0 * np.sqrt(2.0)
    )
    estimate = geo.IMEEstimate(wind_speed=1.0, length_method="skeleton")(
        _gt(np.ones((10, 10))), _gt(mask)
    )
    assert estimate["length_m"] == pytest.approx(127.279, abs=1e-3)
    assert estimate["emission_rate_kg_s"] > 0.0


def test_skeleton_length_honours_anisotropic_pixels() -> None:
    from plumax.geo.plume_array import plume_length

    mask = np.eye(10, dtype=bool)
    # 10 m wide, 20 m tall pixels: each diagonal step is hypot(10, 20).
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -20.0, 200.0)

    assert plume_length(mask, transform, method="skeleton") == pytest.approx(
        9.0 * np.hypot(10.0, 20.0)
    )
    # A vertical line measures 20 m per step, a horizontal one 10 m.
    vertical = np.zeros((10, 10), dtype=bool)
    vertical[:, 3] = True
    assert plume_length(vertical, transform, method="skeleton") == pytest.approx(180.0)
    assert plume_length(vertical.T, transform, method="skeleton") == pytest.approx(90.0)


def test_skeleton_length_does_not_exceed_chord_for_straight_thick_plume() -> None:
    from plumax.geo.plume_array import plume_length

    # A straight 5-pixel-thick plume: the old Manhattan diameter (330 m)
    # over-estimated L beyond the chord; the centreline cannot.
    mask = np.zeros((9, 34), dtype=bool)
    mask[2:7, 2:32] = True
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    skeleton = plume_length(mask, transform, method="skeleton")
    chord = plume_length(mask, transform, method="max_axis")
    assert 0.0 < skeleton <= chord


def test_skeleton_length_single_pixel_and_compact_blob() -> None:
    from plumax.geo.plume_array import plume_length

    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    # Single pixel: documented one-pixel-size proxy, same as other methods.
    single = np.zeros((4, 4), dtype=bool)
    single[1, 1] = True
    assert plume_length(single, transform, method="skeleton") == pytest.approx(10.0)
    # A compact blob whose skeleton collapses to one pixel raises rather
    # than silently returning L = 0 (which zeroed Q in IMEEstimate).
    blob = np.array([[True, True], [False, True]])
    with pytest.raises(ValueError, match="skeleton plume length is 0"):
        plume_length(blob, transform, method="skeleton")
    with pytest.raises(ValueError, match="max_axis"):
        geo.IMEEstimate(wind_speed=1.0, length_method="skeleton")(
            _gt(np.ones((2, 2))), _gt(blob)
        )


def test_skeleton_plume_length_is_deterministic_for_multi_component_mask() -> None:
    """Skeleton length must not depend on set iteration order.

    Regression test: previously the longest-path helper seeded the
    double-BFS from ``next(iter(nodes))`` where ``nodes`` was a Python
    set. Across two disconnected components, the arbitrary starting node
    made the result depend on insertion order (effectively the hash of
    the tuple coordinates).
    """
    from geoops.measure import skeleton_length

    from plumax.geo.plume_array import plume_length

    # Two disconnected components: a long horizontal arm (10 px) and a
    # short isolated blob (1 px). The longest path should come from the
    # long arm, regardless of which component the algorithm visits first.
    mask = np.zeros((6, 12), dtype=bool)
    mask[1, 1:11] = True  # 10-pixel horizontal arm
    mask[4, 10] = True  # isolated single pixel
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)

    length_a = skeleton_length(mask, step=transform)
    length_b = skeleton_length(mask, step=transform)
    length_via_api = plume_length(mask, transform, method="skeleton")

    # 10 pixels at 10 m spacing => 9 * 10 m between endpoint centroids.
    assert length_a == pytest.approx(90.0)
    assert length_b == pytest.approx(90.0)
    assert length_via_api == pytest.approx(90.0)

    # Permuting the mask via row/col reflection still picks the same
    # connected component as the dominant one.
    reflected = mask[::-1, ::-1].copy()
    assert skeleton_length(reflected, step=transform) == pytest.approx(90.0)


_GEO_OPS = ("PlumeFootprint", "IMEEstimate", "WindAdvectionCone", "CrossSectionalFlux")


def _geo_case(name: str, make: Callable[[np.ndarray], GeoTensor]) -> tuple:
    """Build ``(operator, inputs)`` for a metre-assuming plume operator."""
    mask_arr = np.zeros((10, 15), dtype=bool)
    mask_arr[4:6, :] = True
    mask = make(mask_arr)
    enhancement = make(mask_arr.astype(float))
    if name == "PlumeFootprint":
        return geo.PlumeFootprint(min_area_m2=0.0), (mask,)
    if name == "IMEEstimate":
        return geo.IMEEstimate(wind_speed=2.0), (enhancement, mask)
    if name == "WindAdvectionCone":
        op = geo.WindAdvectionCone(
            source=(0.0, 0.0), wind_u=1.0, wind_v=0.0, max_distance=200.0
        )
        return op, (mask,)
    op = geo.CrossSectionalFlux(source=(0.0, 0.0), wind_u=1.0, wind_v=0.0)
    return op, (enhancement, mask)


@pytest.mark.parametrize(
    ("crs", "match"),
    [("EPSG:4326", "geographic"), ("EPSG:2263", "linear units")],
    ids=["geographic", "us-feet"],
)
@pytest.mark.parametrize("name", _GEO_OPS)
def test_geo_ops_reject_geographic_crs(crs: str, match: str, name: str) -> None:
    def make(values: np.ndarray) -> GeoTensor:
        return GeoTensor(
            values=values,
            transform=rasterio.Affine(1e-4, 0.0, 0.0, 0.0, -1e-4, 0.001),
            crs=crs,
        )

    op, inputs = _geo_case(name, make)
    with pytest.raises(ValueError, match=rf"{name}.*{match}.*Reproject"):
        op(*inputs)


@pytest.mark.parametrize("name", _GEO_OPS)
def test_geo_ops_accept_projected_crs(name: str) -> None:
    op, inputs = _geo_case(name, _gt)
    op(*inputs)


def test_wind_cone_source_crs_requires_carrier_crs() -> None:
    gt = GeoTensor(
        values=np.zeros((5, 5)),
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 50.0),
        crs=None,
    )
    op = geo.WindAdvectionCone(
        source=(-8.0, 53.0), wind_u=1.0, wind_v=0.0, crs="EPSG:4326"
    )
    with pytest.raises(ValueError, match="has no CRS"):
        op(gt)
    # Without a source CRS the crs-less carrier is still accepted (metres).
    out = geo.WindAdvectionCone(
        source=(0.0, 25.0), wind_u=1.0, wind_v=0.0, max_distance=30.0
    )(gt)
    assert np.asarray(out).any()


def test_ime_rejects_negative_uncertainty_fraction() -> None:
    with pytest.raises(ValueError, match="uncertainty_fraction"):
        geo.IMEEstimate(wind_speed=1.0, uncertainty_fraction=-0.1)


def test_ime_convex_hull_length_agrees_with_max_axis_on_convex_plume() -> None:
    mask = _gt(np.ones((2, 2), dtype=bool))
    enhancement = _gt(np.ones((2, 2), dtype=float))

    convex_hull = geo.IMEEstimate(
        wind_speed=1.0, length_method="convex_hull", return_uncertainty=False
    )(enhancement, mask)
    max_axis = geo.IMEEstimate(
        wind_speed=1.0, length_method="max_axis", return_uncertainty=False
    )(enhancement, mask)

    assert convex_hull["length_m"] == pytest.approx(max_axis["length_m"])


def test_cross_sectional_flux_returns_transect_geodataframe() -> None:
    mask = _gt(np.ones((3, 3), dtype=bool))
    enhancement = _gt(np.ones((3, 3), dtype=float))

    gdf = geo.CrossSectionalFlux(
        source=(0.0, 100.0),
        wind_u=1.0,
        wind_v=0.0,
        n_transects=2,
        transect_spacing_m=10.0,
    )(enhancement, mask)

    assert list(gdf["transect_id"]) == [1, 2]
    assert (gdf["flux_kg_s"] >= 0.0).all()
    assert gdf.crs == "EPSG:32629"


def test_column_to_mass_round_trip_from_ppm_m() -> None:
    values = np.array([[100.0, 250.0]])
    kg_m2 = convert_column_units(values, gas="CH4", units_in="ppm_m", units_out="kg_m2")
    recovered = convert_column_units(
        kg_m2, gas="CH4", units_in="kg_m2", units_out="ppm_m"
    )

    assert np.allclose(recovered, values)


def test_column_to_mass_operator_preserves_metadata() -> None:
    gt = _gt(np.array([[1.0, 2.0]]))

    out = geo.ColumnToMass(gas="CO2", units_in="mol_m2")(gt)

    assert np.allclose(np.asarray(out), np.array([[0.04401, 0.08802]]))
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"


def test_ime_recovers_injected_mass_on_synthetic_plume() -> None:
    """Synthetic plume on a flat background: IME equals injected mass."""
    enhancement = np.zeros((20, 20), dtype=float)
    # Linear plume: 1 px tall, 10 px long, uniform 0.5 kg/m^2 column.
    enhancement[10, 5:15] = 0.5
    mask_arr = enhancement > 0
    # 10 m pixels -> pixel area 100 m^2 -> per-pixel mass 50 kg ->
    # total injected mass over 10 pixels = 500 kg.
    expected_ime = 10 * 100.0 * 0.5
    # Length: 10 pixels at 10 m spacing => 9 * 10 m between centroids.
    expected_length = 9 * 10.0
    wind = 4.0

    result = geo.IMEEstimate(wind_speed=wind, return_uncertainty=False)(
        _gt(enhancement), _gt(mask_arr)
    )

    assert result["ime_kg"] == pytest.approx(expected_ime)
    assert result["length_m"] == pytest.approx(expected_length)
    assert result["emission_rate_kg_s"] == pytest.approx(
        wind * expected_ime / expected_length
    )


def test_ime_empty_mask_yields_zero_emission() -> None:
    mask = _gt(np.zeros((4, 4), dtype=bool))
    enhancement = _gt(np.ones((4, 4), dtype=float))

    result = geo.IMEEstimate(wind_speed=5.0, return_uncertainty=False)(
        enhancement, mask
    )

    assert result["ime_kg"] == 0.0
    assert result["length_m"] == 0.0
    assert result["emission_rate_kg_s"] == 0.0


def test_cross_sectional_flux_matches_uniform_plume_analytical() -> None:
    """Uniform Omega along a 1-pixel-tall plume gives Q = U * Omega * W * N.

    For a 1-pixel-tall row at the source y, the across-wind integral of
    Omega = const is Omega * (n_active_pixels) * dx. Each transect picks
    up exactly one pixel along the row, so flux = U * Omega * dx.
    """
    enhancement = np.zeros((11, 21), dtype=float)
    enhancement[5, :] = 1.0  # kg/m^2, full row
    mask_arr = enhancement > 0

    # Place source at the row centre.
    source_x = 5.5 * 10.0  # column 5 centre
    source_y = 100.0 - 5.5 * 10.0  # row 5 centre (transform.f=100, e=-10)
    wind_u = 1.0
    wind_v = 0.0

    gdf = geo.CrossSectionalFlux(
        source=(source_x, source_y),
        wind_u=wind_u,
        wind_v=wind_v,
        n_transects=3,
        transect_spacing_m=10.0,
    )(_gt(enhancement), _gt(mask_arr))

    # Each transect crosses exactly one pixel (1 kg/m^2) of along-wind
    # half-width 5 m. Flux = Omega * |U| * dx_across = 1 * 1 * 10 = 10.
    for flux in gdf["flux_kg_s"]:
        assert flux == pytest.approx(10.0)
    assert (gdf["n_pixels"] == 1).all()


def test_cross_sectional_flux_counts_a_boundary_pixel_once() -> None:
    """A transect on a pixel boundary takes one column, not both (#177 review).

    With the source at x = 0 on the 10 m grid, the transect at 50 m lies
    between the pixel centres at 45 m and 55 m. The old inclusive test took
    both and returned 20 kg/s for a one-row plume of 1 kg/m² in a 1 m/s wind.
    """
    enhancement = np.zeros((11, 21), dtype=float)
    enhancement[5, :] = 1.0
    mask_arr = enhancement > 0
    source_y = 100.0 - 5.5 * 10.0  # row 5 centre
    gdf = geo.CrossSectionalFlux(
        source=(0.0, source_y),
        wind_u=1.0,
        wind_v=0.0,
        n_transects=3,
        transect_spacing_m=50.0,
    )(_gt(enhancement), _gt(mask_arr))
    assert gdf["n_pixels"].tolist() == [1, 1, 1]
    assert gdf["flux_kg_s"].tolist() == pytest.approx([10.0, 10.0, 10.0])


def test_cross_sectional_flux_rejects_zero_wind() -> None:
    mask = _gt(np.ones((3, 3), dtype=bool))
    enhancement = _gt(np.ones((3, 3), dtype=float))
    op = geo.CrossSectionalFlux(
        source=(0.0, 0.0),
        wind_u=0.0,
        wind_v=0.0,
    )
    with pytest.raises(ValueError, match="wind vector"):
        op(enhancement, mask)


def test_column_unit_round_trip_through_mol_m2() -> None:
    """ppm_m -> mol_m2 -> kg_m2 -> ppm_m round-trips to within float eps."""
    values = np.array([[10.0, 100.0, 1000.0]])
    mol = convert_column_units(values, gas="CH4", units_in="ppm_m", units_out="mol_m2")
    kg = convert_column_units(mol, gas="CH4", units_in="mol_m2", units_out="kg_m2")
    back = convert_column_units(kg, gas="CH4", units_in="kg_m2", units_out="ppm_m")
    assert np.allclose(back, values)


def test_plume_operators_get_config_is_json_safe() -> None:
    """get_config dicts must be JSON-serialisable (no GeoTensors)."""
    ops = [
        geo.PlumeMask(threshold="otsu", min_area_px=10),
        geo.PlumeContours(min_area_px=5),
        geo.PlumeFootprint(min_area_m2=100.0),
        geo.WindAdvectionCone(
            source=(0.0, 0.0),
            wind_u=1.0,
            wind_v=0.0,
        ),
        geo.ColumnToMass(gas="CO2", units_in="ppm_m"),
        geo.IMEEstimate(wind_speed=3.0),
        geo.CrossSectionalFlux(
            source=(0.0, 0.0),
            wind_u=1.0,
            wind_v=0.0,
        ),
        geo.PlumeColumnStats(),
        geo.PlumeQNDFeatures(),
        geo.PlumeShapeFilter(min_area_px=10, max_fiber_width=50.0),
    ]
    for op in ops:
        # Must round-trip through JSON without raising.
        config = op.get_config()
        reloaded = json.loads(json.dumps(config))
        # ... and rebuild an operator with the same configuration (#134).
        rebuilt = type(op)(**reloaded)
        assert json.loads(json.dumps(rebuilt.get_config())) == reloaded


def test_label_components_drops_small_and_renumbers_contiguously() -> None:
    """label_components must yield contiguous 1..K labels after dropping."""
    from plumax.geo.plume_array import label_components

    mask = np.zeros((7, 7), dtype=bool)
    # Three isolated components separated by >=2 pixels on every side so
    # 8-connectivity does not merge them.
    mask[0:2, 0:2] = True  # 4 px component
    mask[6, 6] = True  # 1 px component (dropped at min_area_px=2)
    mask[5:7, 0:3] = True  # 6 px component (top-right diagonal of [6,6]
    # is (5,5)=False, so [6,6] stays isolated)
    mask[5, 5] = False

    labels = label_components(mask, min_area_px=2, connectivity=4)
    unique = sorted(set(np.unique(labels).tolist()))
    assert unique == [0, 1, 2]
    assert int((labels == 1).sum()) + int((labels == 2).sum()) == 4 + 6


def _make_long_thin_plume(label_id: int = 1) -> np.ndarray:
    labels = np.zeros((20, 60), dtype=np.int32)
    labels[10, 5:55] = label_id  # 50-px long, 1-px wide: clearly fiber-like
    return labels


def _make_blob(label_id: int = 1) -> np.ndarray:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = label_id  # 10x10 blob: fiber/major ~ 1
    return labels


def test_plume_shape_filter_keeps_long_thin_features() -> None:
    labels = _make_long_thin_plume()
    # skimage's PCA-based major-axis length slightly overshoots the
    # geodesic fiber length for axis-aligned thin strips (the paper used
    # cv2.minAreaRect which is exact), so soften the default lower ratio
    # bound for this faithfulness test.
    out = geo.PlumeShapeFilter(min_area_px=10, min_fiber_to_major_ratio=0.5)(
        _gt(labels)
    )
    out_arr = np.asarray(out)
    assert set(np.unique(out_arr[out_arr > 0])) == {1}


def test_plume_shape_filter_rejects_compact_blob_with_min_ratio() -> None:
    labels = _make_blob()
    # min_fiber_to_major_ratio=2.0 forces elongated shapes only; a square
    # blob has skeleton/major <~ 1 so it is dropped.
    out = geo.PlumeShapeFilter(min_area_px=10, min_fiber_to_major_ratio=2.0)(
        _gt(labels)
    )
    assert (np.asarray(out) == 0).all()


def test_plume_shape_filter_drops_tiny_instances_by_area() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1, 1] = 1
    labels[5:9, 5:9] = 2  # 16 px

    out = geo.PlumeShapeFilter(min_area_px=10)(_gt(labels))
    out_arr = np.asarray(out)
    assert 1 not in out_arr  # tiny single-pixel instance filtered
    # The 4x4 blob survives the area gate but may still be dropped by the
    # default fiber/major ratio of >=1.0 — only check the area filter here.
    assert set(np.unique(out_arr)) <= {0, 2}


def test_plume_column_stats_in_out_contrast_matches_xch4_metrics() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[3:7, 3:7] = 1
    values = np.ones((10, 10), dtype=float)
    values[3:7, 3:7] = 5.0

    df = geo.PlumeColumnStats()(_gt(labels), _gt(values))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["label"] == 1
    assert row["area"] == 16
    assert row["mean_enhancement"] == pytest.approx(5.0)
    assert row["background_mean"] == pytest.approx(1.0)
    assert row["contrast"] == pytest.approx(5.0)
    # ipa = (sum_plume - bg_mean * n_plume) / n_plume = (80 - 1*16)/16 = 4
    assert row["intensity_per_area"] == pytest.approx(4.0)


def test_plume_column_stats_empty_label_map_returns_empty_frame() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    df = geo.PlumeColumnStats()(_gt(labels), _gt(np.zeros((5, 5))))
    assert df.empty
    assert "intensity_per_area" in df.columns


def test_plume_qnd_features_emits_polynomial_columns() -> None:
    rng = np.random.default_rng(0)
    column = rng.normal(loc=0.0, scale=1.0, size=(40, 40))
    column[15:25, 15:25] += 3.0  # an obvious enhancement
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[15:25, 15:25] = 1

    op = geo.PlumeQNDFeatures(degree=4, eps=5.0, min_samples=3, perc_threshold=90.0)
    df = op(_gt(labels), _gt(column))

    assert len(df) == 1
    row = df.iloc[0]
    assert row["label"] == 1
    # 5 polynomial coefficients (degree 4 + 1), all finite for a well-populated
    # in-mask sample.
    coeffs = [row[f"qnd_column_c{i}"] for i in range(5)]
    assert all(np.isfinite(coeffs))
    # Albedo not supplied -> NaN coefficients.
    assert all(not np.isfinite(row[f"qnd_albedo_c{i}"]) for i in range(5))
    # Plume z-score against the background should be clearly positive.
    assert row["z_score"] > 1.0


def test_plume_qnd_features_n_clumps_uses_dbscan_labels_not_raster_connectivity() -> (
    None
):
    """Regression (PR #87 review): ``n_clumps`` must count unique DBSCAN
    cluster labels, not re-run a 4-connected component pass on the
    rasterised non-noise pixels. A single density-connected cluster
    whose members sit within ``eps`` but are not 4-adjacent would
    otherwise be over-counted."""
    # Two pixels diagonally adjacent: 4-connectivity sees two components,
    # 8-connectivity (and any eps >= sqrt(2)) sees one. The QND feature
    # should report one cluster.
    column = np.full((10, 10), -5.0, dtype=float)
    column[4, 4] = 10.0
    column[5, 5] = 10.0
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[3:8, 3:8] = 1

    df = geo.PlumeQNDFeatures(degree=3, eps=2.0, min_samples=2, perc_threshold=80.0)(
        _gt(labels), _gt(column)
    )
    assert int(df.iloc[0]["n_clumps"]) == 1


def test_plume_qnd_features_handles_too_few_samples() -> None:
    column = np.zeros((20, 20), dtype=float)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[0, 0:5] = 1  # only 5 pixels: below the 10-sample threshold

    df = geo.PlumeQNDFeatures(degree=3)(_gt(labels), _gt(column))

    row = df.iloc[0]
    # All polynomial coefficients should fall through to NaN when fewer
    # than 10 valid samples are inside the mask.
    assert all(not np.isfinite(row[f"qnd_column_c{i}"]) for i in range(4))


def _ndarray_case_plume_mask() -> tuple[object, np.ndarray]:
    values = np.r_[np.zeros(8), np.full(8, 10.0)].reshape(4, 4)
    return geo.PlumeMask(threshold=5.0, min_area_px=1), values


def _ndarray_case_plume_contours() -> tuple[object, np.ndarray]:
    mask = np.zeros((4, 4), dtype=bool)
    mask[0:2, 0:2] = True
    return geo.PlumeContours(min_area_px=1), mask


def _ndarray_case_column_to_mass() -> tuple[object, np.ndarray]:
    values = np.array([[100.0, 250.0], [1.0, 2.0]])
    return geo.ColumnToMass(gas="CH4", units_in="ppm_m"), values


def _ndarray_case_shape_filter() -> tuple[object, np.ndarray]:
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[3, 1:7] = 1  # thin 6-px strip
    return (
        geo.PlumeShapeFilter(min_area_px=1, min_fiber_to_major_ratio=0.0),
        labels,
    )


@pytest.mark.parametrize(
    "case",
    [
        _ndarray_case_plume_mask,
        _ndarray_case_plume_contours,
        _ndarray_case_column_to_mass,
        _ndarray_case_shape_filter,
    ],
    ids=["PlumeMask", "PlumeContours", "ColumnToMass", "PlumeShapeFilter"],
)
def test_plain_ndarray_in_plain_ndarray_out(
    case: Callable[[], tuple[object, np.ndarray]],
) -> None:
    """Per-pixel plume ops: ndarray in -> ndarray out, values match GeoTensor."""
    op, values = case()

    out_arr = op(values)
    out_gt = op(_gt(values))

    assert type(out_arr) is np.ndarray
    assert np.array_equal(out_arr, np.asarray(out_gt))


def test_geo_dependent_plume_ops_reject_plain_arrays() -> None:
    """Transform/CRS-dependent ops must fail loudly on plain arrays."""
    arr = np.ones((4, 4), dtype=float)
    mask = np.ones((4, 4), dtype=bool)
    cases = [
        (geo.PlumeFootprint(min_area_m2=1.0), ()),
        (geo.WindAdvectionCone(source=(0.0, 0.0), wind_u=1.0, wind_v=0.0), ()),
        (geo.IMEEstimate(wind_speed=1.0), (mask,)),
        (
            geo.CrossSectionalFlux(source=(0.0, 0.0), wind_u=1.0, wind_v=0.0),
            (mask,),
        ),
    ]
    for op, extra in cases:
        with pytest.raises(TypeError, match="georeferenced GeoTensor"):
            op(arr, *extra)


def test_plume_column_stats_accepts_plain_arrays() -> None:
    import pandas as pd

    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:5, 2:5] = 1
    values = np.ones((6, 6), dtype=float)
    values[2:5, 2:5] = 4.0

    df_arr = geo.PlumeColumnStats()(labels, values)
    df_gt = geo.PlumeColumnStats()(_gt(labels), _gt(values))

    pd.testing.assert_frame_equal(df_arr, df_gt)


def test_ime_convex_hull_handles_collinear_points() -> None:
    """Degenerate convex hull (1-pixel row) must not crash."""
    mask = _gt(np.array([[True, True, True, True]]))
    enhancement = _gt(np.ones((1, 4), dtype=float))

    result = geo.IMEEstimate(
        wind_speed=1.0, length_method="convex_hull", return_uncertainty=False
    )(enhancement, mask)

    # Four pixels at 10 m spacing -> length = 3 * 10 = 30 m.
    assert result["length_m"] == pytest.approx(30.0)


@pytest.mark.parametrize(
    "case",
    [
        "contours_fill",
        "contours_nan",
        "contours_nan_fill",
        "mask_percentile",
        "ime",
        "column_to_mass",
    ],
)
def test_fill_pixels_are_excluded(case: str) -> None:
    """Fill / NaN pixels are never plume and never enter a statistic (#145)."""
    shape = (5, 5)
    fill = fill_pixel_mask(shape)
    valid = ~fill
    if case == "contours_fill":
        # -9999 is truthy: it used to be labelled as plume.
        gt = toy_geotensor(np.ones(shape), with_fill_pixels=True)
        labels = np.asarray(geo.PlumeContours(min_area_px=1)(gt))
        assert (labels[fill] == -9999).all()
        assert (labels[valid] == 1).all()
        mask = geo.PlumeContours(min_area_px=1, return_labels=False)(gt)
        np.testing.assert_array_equal(np.asarray(mask), valid)
        assert mask.fill_value_default is False
    elif case == "contours_nan":
        # Issue reproduction: NaN scores used to be treated as plume.
        score = np.zeros(shape)
        score[fill] = np.nan
        labels = geo.PlumeContours(min_area_px=1)(score)
        assert not labels.any()
        mask = geo.PlumeContours(min_area_px=1, return_labels=False)(score)
        assert not mask.any()
    elif case == "contours_nan_fill":
        gt = toy_geotensor(np.ones(shape), fill_value_default=np.nan)
        gt.values[fill] = np.nan
        out = geo.PlumeContours(min_area_px=1)(gt)
        labels = np.asarray(out)
        # NaN cannot live in int32 labels: fill pixels are background (0).
        assert (labels[fill] == 0).all()
        assert (labels[valid] == 1).all()
        assert out.fill_value_default == 0
    elif case == "mask_percentile":
        values = np.arange(25, dtype=float).reshape(shape)
        gt = toy_geotensor(values, with_fill_pixels=True)
        result = geo.PlumeMask(threshold="percentile:50", min_area_px=1)(gt)
        # A boolean mask declares False, not the score's -9999 (#146).
        assert result.fill_value_default is False
        out = np.asarray(result)
        cutoff = np.percentile(values[valid], 50)
        np.testing.assert_array_equal(out, (values > cutoff) & valid)
        otsu = np.asarray(geo.PlumeMask(threshold="otsu", min_area_px=1)(gt))
        assert not otsu[fill].any()
        expected = (values > resolve_threshold(values[valid], "otsu")) & valid
        np.testing.assert_array_equal(otsu, expected)
    elif case == "ime":
        enhancement = toy_geotensor(np.ones(shape), with_fill_pixels=True)
        plume = toy_geotensor(np.ones(shape, dtype=bool))
        result = geo.IMEEstimate(wind_speed=1.0, return_uncertainty=False)(
            enhancement, plume
        )
        assert result["ime_kg"] == pytest.approx(valid.sum() * 100.0)
    elif case == "column_to_mass":
        # Issue reproduction (#146): the fill pixel used to be scaled to
        # -6.6e-3 kg/m^2 while the output still declared -9999.
        gt = toy_geotensor(np.full(shape, 1000.0), with_fill_pixels=True)
        result = geo.ColumnToMass(units_in="ppm_m")(gt)
        out = np.asarray(result)
        assert np.isnan(result.fill_value_default)
        assert np.isnan(out[fill]).all()
        assert (out[valid] > 0).all()


def test_4d_time_stack() -> None:
    """Per-frame operators map a stack; table outputs reject it (#147)."""
    stack = time_stack()
    masks = geo.PlumeMask(threshold=0.5, min_area_px=1)(time_stack((2, 1, 4, 4)))
    assert masks.shape == (2, 1, 4, 4)
    with pytest.raises(ValueError, match="PlumeFootprint accepts 2-D"):
        geo.PlumeFootprint()(stack)


# ---------------------------------------------------------------------------
# Non-default grids (``_carriers.TOY_GRIDS``)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grid", "area_ratio"),
    [("non_square", 2.0), ("rotated", 1.0), ("sheared", 1.0)],
)
def test_ime_mass_scales_with_the_true_pixel_area(grid: str, area_ratio: float) -> None:
    """IME sums column x pixel area, and the area is |det(transform)|.

    10 m x 20 m pixels double the mass of the 10 m grid; rotation and
    shear keep the 100 m^2 area even though ``a`` / ``e`` change.
    """
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:6] = True
    column = np.full((1, 8, 8), 0.5)

    def ime(g: str) -> float:
        plume = toy_geotensor(mask, grid=g, fill_value_default=None)
        op = geo.IMEEstimate(wind_speed=1.0)
        return op(toy_geotensor(column, grid=g), plume)["ime_kg"]

    assert ime(grid) == pytest.approx(area_ratio * ime("utm"))


# ---------------------------------------------------------------------------
# Multi-input convention (#141)
# ---------------------------------------------------------------------------


def test_ime_rejects_a_plume_mask_on_another_grid() -> None:
    """A 10 m mask against a 20 m enhancement is an error, not a wrong IME.

    With the mask pinned in the constructor, ``IMEEstimate`` indexed the
    20 m enhancement with the 10 m mask's pixels and returned an ``ime_kg``
    computed with the enhancement's pixel area -- a silently wrong answer.
    """
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    mask_10m = _gt(mask)
    enh_20m = GeoTensor(
        values=np.ones((8, 8)),
        transform=rasterio.Affine(20.0, 0.0, 0.0, 0.0, -20.0, 100.0),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )
    op = geo.IMEEstimate(wind_speed=1.0)
    with pytest.raises(ValueError, match=r"IMEEstimate: the plume_mask pixel grid"):
        op(enh_20m, mask_10m)
    # On the enhancement's own grid the estimate goes through.
    enh_10m = _gt(np.ones((8, 8)))
    assert op(enh_10m, mask_10m)["ime_kg"] == pytest.approx(16 * 100.0)


def test_plume_carrier_kwargs_are_gone() -> None:
    """Second carriers are call arguments; the old kwargs raise (#141)."""
    mask = _gt(np.ones((2, 2), dtype=bool))
    with pytest.raises(TypeError, match="plume_mask"):
        geo.IMEEstimate(plume_mask=mask, wind_speed=1.0)  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError, match="takes no arguments"):
        geo.PlumeColumnStats(column=mask)  # ty: ignore[unknown-argument]
    # Without runtime carriers, the operators round-trip through their state.
    for op in (
        geo.IMEEstimate(wind_speed=2.0),
        geo.CrossSectionalFlux(source=(0.0, 0.0), wind_u=1.0, wind_v=0.0),
        geo.PlumeColumnStats(),
        geo.PlumeQNDFeatures(degree=3),
    ):
        assert not op.forbid_in_yaml
        clone = Operator.from_state(json.loads(json.dumps(op.state)))
        assert clone.get_config() == op.get_config()


@pytest.mark.parametrize("seed", range(5))
def test_max_axis_length_equals_the_brute_force_diameter(seed: int) -> None:
    """The hull-vertex diameter is the all-pairs one, exactly (#177 review)."""
    from plumax.geo.plume_array import pixel_centers, plume_length

    rng = np.random.default_rng(seed)
    mask = rng.random((15, 12)) < 0.3
    transform = rasterio.Affine(10.0, 5.0, 500_000.0, 0.0, -20.0, 4_000_000.0)
    xs, ys = pixel_centers(mask.shape, transform)
    points = np.column_stack([xs[mask], ys[mask]])
    diff = points[:, None, :] - points[None, :, :]
    brute = float(np.sqrt(np.max(np.sum(diff**2, axis=-1))))
    assert plume_length(mask, transform, method="max_axis") == brute


@pytest.mark.parametrize("line", ["row", "column", "diagonal"])
def test_max_axis_length_of_collinear_pixels(line: str) -> None:
    from plumax.geo.plume_array import plume_length

    mask = np.zeros((9, 9), dtype=bool)
    if line == "row":
        mask[4, 1:8] = True
    elif line == "column":
        mask[1:8, 4] = True
    else:
        np.fill_diagonal(mask, True)
    transform = rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 100.0)
    expected = {"row": 60.0, "column": 60.0, "diagonal": 80.0 * np.sqrt(2.0)}
    length = plume_length(mask, transform, method="max_axis")
    assert length == pytest.approx(expected[line], rel=1e-12)


def _uint8_labels_with_fill() -> GeoTensor:
    """Two 3 x 6 instances on a 12 x 12 grid, and a 255 nodata block."""
    labels = np.zeros((12, 12), dtype=np.uint8)
    labels[1:4, 1:7] = 1
    labels[6:9, 1:7] = 2
    labels[10:, :] = 255  # nodata
    return GeoTensor(
        values=labels,
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 120.0),
        crs="EPSG:32629",
        fill_value_default=255,
    )


@pytest.mark.parametrize("op_name", ["PlumeColumnStats", "PlumeQNDFeatures"])
def test_instance_features_ignore_a_positive_label_fill(op_name: str) -> None:
    """A uint8 label image's 255 nodata is not an instance (#177 review)."""
    column = GeoTensor(
        values=np.random.default_rng(0).random((12, 12)),
        transform=rasterio.Affine(10.0, 0.0, 0.0, 0.0, -10.0, 120.0),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )
    op = getattr(geo, op_name)()
    df = op(_uint8_labels_with_fill(), column)
    assert df["label"].tolist() == [1, 2]


def test_shape_filter_ignores_a_positive_label_fill() -> None:
    out = geo.PlumeShapeFilter(min_area_px=1, min_fiber_to_major_ratio=0.0)(
        _uint8_labels_with_fill()
    )
    assert set(np.unique(np.asarray(out))) <= {0, 1, 2}
    assert not np.asarray(out)[10:].any()
