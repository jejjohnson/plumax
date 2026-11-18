"""One plume from a matched-filter product: ``PlumeScene`` and its loaders (#272).

The fixture is written at test time: a 40 × 40 scene of 60 m pixels on the
analysis frame with a plume along 30° from a source, its sensitivity,
uncertainty and L2A mask rasters, and a two-plume GeoJSON. The loaders read
files on the host, so there is no gradient, ``jit`` or ``vmap`` to take of
them; the arrays of the ``PlumeScene`` they return go through
``eqx.filter_jit``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings, strategies as st
from jaxtyping import TypeCheckError

from plumax.coupled import QualityFlag
from plumax.errors import (
    GridMismatchError,
    MissingVariableError,
    ScreeningError,
    UnitsError,
)
from plumax.met import LocalFrame
from plumax.obs import (
    PlumeScene,
    load_emit_plume_complex,
    matched_filter_scene,
    ppm_m_to_column_mass,
)


pytest.importorskip("xrproducts")
pytest.importorskip("pyproj")

PIXEL = 60.0
N = 40
DIRECTION = math.radians(30.0)
#: the source at pixel (row 25, col 8) of the north-up grid
SOURCE_ROW, SOURCE_COL = 25, 8
KAPPA = float(ppm_m_to_column_mass(1.0, pressure_pa=101_325.0, temperature_k=288.15))


def _grid(frame):
    """Frame metres and lat/lon of the pixel centres, row 0 to the north."""
    cols = np.arange(N, dtype=float)
    rows = np.arange(N, dtype=float)
    x = PIXEL * (cols - SOURCE_COL)
    y = PIXEL * (SOURCE_ROW - rows)
    xx, yy = np.meshgrid(x, y)
    lat, lon = frame.to_latlon(xx, yy)
    return xx, yy, lat, lon


def _plume(xx, yy):
    """E [ppm m]: a ridge from the source along 30°, widening downwind."""
    # d = sᵀ e_∥,  c = sᵀ e_⊥
    downwind = xx * math.cos(DIRECTION) + yy * math.sin(DIRECTION)
    crosswind = -xx * math.sin(DIRECTION) + yy * math.cos(DIRECTION)
    width = 60.0 + 0.1 * np.maximum(downwind, 0.0)
    return 800.0 * np.exp(-0.5 * (crosswind / width) ** 2) * (downwind > 0.0)


def _outline(frame, corners_xy):
    xs, ys = np.asarray(corners_xy, dtype=float).T
    lat, lon = frame.to_latlon(xs, ys)
    ring = np.column_stack([lon, lat]).tolist()
    return [*ring, ring[0]]


def write_scene(tmp: Path, frame: LocalFrame, *, dtype=np.float64) -> dict[str, Path]:
    xx, yy, lat, lon = _grid(frame)
    rng = np.random.default_rng(0)
    enhancement = (_plume(xx, yy) + 5.0 * rng.normal(size=(N, N))).astype(dtype)
    sensitivity = rng.uniform(0.6, 1.4, size=(N, N)).astype(dtype)
    sensitivity[3, 30] = 0.0
    sensitivity[4, 31] = -0.2
    uncertainty = rng.uniform(20.0, 40.0, size=(N, N)).astype(dtype)
    enhancement[10, 20] = -9999.0
    coords = {"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)}
    paths = {
        "enhancement": tmp / "scene_CH4ENH.nc",
        "sensitivity": tmp / "scene_CH4SENS.nc",
        "uncertainty": tmp / "scene_CH4UNCERT.nc",
        "quality": tmp / "scene_L2A_MASK.nc",
        "outlines": tmp / "plume_complexes.json",
    }
    xr.Dataset(
        {"ch4_enhancement": (("y", "x"), enhancement)},
        coords=coords,
        attrs={"time_coverage_start": "2024-06-01T18:30:00Z"},
    ).to_netcdf(paths["enhancement"])
    xr.Dataset({"ch4_sensitivity": (("y", "x"), sensitivity)}, coords=coords).to_netcdf(
        paths["sensitivity"]
    )
    xr.Dataset({"ch4_uncertainty": (("y", "x"), uncertainty)}, coords=coords).to_netcdf(
        paths["uncertainty"]
    )

    # the L2A mask in its own layout: (downtrack, crosstrack, band)
    flags = np.zeros((N, N, 8), dtype=np.float32)
    flags[20:23, 26:29, 0] = 1.0  # a cloud on the plume
    flags[30:, :5, 2] = 1.0  # water
    flags[..., 5] = 0.1  # AOD, not a flag
    band_mask = np.zeros((N, N, 36), dtype=np.uint8)
    band_mask[..., 0] = 0b0000_0001  # one static bad band everywhere
    band_mask[5, 35, 10:15] = 0xFF  # a flare
    xr.Dataset(
        {
            "mask": (("downtrack", "crosstrack", "bands"), flags),
            "band_mask": (("downtrack", "crosstrack", "packed_bands"), band_mask),
        }
    ).to_netcdf(paths["quality"])

    # two plumes: ours along 30°, and a small one in the north-west corner
    # a trapezoid symmetric about the plume axis, (d, c) → (x, y)
    along = np.array([math.cos(DIRECTION), math.sin(DIRECTION)])
    across = np.array([-math.sin(DIRECTION), math.cos(DIRECTION)])
    corners = [(-60.0, -120.0), (1500.0, -400.0), (1500.0, 400.0), (-60.0, 120.0)]
    ours = _outline(frame, [d * along + c * across for d, c in corners])
    other = _outline(frame, [(-420.0, 1320.0), (-300.0, 1320.0), (-300.0, 1440.0)])
    origin_lat, origin_lon = frame.to_latlon(0.0, 0.0)
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [ours]},
            "properties": {
                "Plume ID": "CH4_PlumeComplex-1",
                "Psuedo-Origin": json.dumps(
                    {
                        "type": "Point",
                        "coordinates": [float(origin_lon), float(origin_lat)],
                    }
                ),
            },
        },
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [other]},
            "properties": {"Plume ID": "CH4_PlumeComplex-2"},
        },
    ]
    paths["outlines"].write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )
    return paths


@pytest.fixture(scope="module")
def frame() -> LocalFrame:
    return LocalFrame.at(31.95, -102.15)


@pytest.fixture(scope="module")
def paths(tmp_path_factory, frame):
    return write_scene(tmp_path_factory.mktemp("plume_scene"), frame)


def _load(paths, frame, **options):
    defaults = {
        "plume_id": "CH4_PlumeComplex-1",
        "frame": frame,
        "sensitivity_path": paths["sensitivity"],
        "uncertainty_path": paths["uncertainty"],
    }
    return load_emit_plume_complex(
        paths["enhancement"], paths["outlines"], **{**defaults, **options}
    )


def _window_fields(paths, scene):
    """The fixture's rasters at the scene's kept pixels."""
    rows, cols = np.asarray(scene.pixel_index).T
    row_start, _, col_start, _ = scene.window
    fields = {}
    for key in ("enhancement", "sensitivity", "uncertainty"):
        with xr.open_dataset(paths[key]) as ds:
            values = ds[f"ch4_{key}"].values
        fields[key] = values[rows + row_start, cols + col_start]
    return fields


# ── oracles ─────────────────────────────────────────────────────────────────


def test_kernel_and_variance_follow_the_rasters(paths, frame):
    truth = _load(paths, frame)
    score = _load(paths, frame, uncertainty_of="score")
    fields = _window_fields(paths, truth)
    mask = np.asarray(truth.instrument.mask) > 0
    sensitivity = fields["sensitivity"][mask]
    uncertainty = fields["uncertainty"][mask]
    # Aₙ is the sensitivity raster on the kept pixels
    np.testing.assert_allclose(
        np.asarray(truth.instrument.averaging_kernel)[mask], sensitivity, rtol=1e-12
    )
    # σₙ² = (κ Aₙ uₙ)² for "truth", (κ uₙ)² for "score"
    np.testing.assert_allclose(
        np.asarray(truth.instrument.retrieval_variance)[mask],
        (KAPPA * sensitivity * uncertainty) ** 2,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(score.instrument.retrieval_variance)[mask],
        (KAPPA * uncertainty) ** 2,
        rtol=1e-12,
    )
    # yₙ = κ Eₙ, never divided by Aₙ
    np.testing.assert_allclose(
        np.asarray(truth.enhancement)[mask],
        KAPPA * fields["enhancement"][mask],
        rtol=1e-12,
    )


def test_matched_filter_scene_multiplies_by_the_molar_mass(frame):
    xx, _, lat, lon = _grid(frame)
    score = np.full((N, N), 0.02)
    observation = matched_filter_scene(score, 0.004, lat=lat, lon=lon, frame=frame)
    np.testing.assert_allclose(observation.enhancement, 0.0160425 * 0.02, rtol=1e-12)
    np.testing.assert_allclose(
        observation.instrument.retrieval_variance, (0.0160425 * 0.004) ** 2, rtol=1e-12
    )
    np.testing.assert_allclose(observation.instrument.averaging_kernel, 1.0)
    # ppm m uses κ, kg m⁻² nothing
    ppm = matched_filter_scene(
        score, 0.004, lat=lat, lon=lon, frame=frame, unit="ppm_m"
    )
    np.testing.assert_allclose(ppm.enhancement, KAPPA * 0.02, rtol=1e-12)
    mass = matched_filter_scene(
        score, 0.004, lat=lat, lon=lon, frame=frame, unit="kg_m2"
    )
    np.testing.assert_allclose(mass.enhancement, 0.02, rtol=1e-12)
    receptors = np.asarray(observation.instrument.receptors)
    np.testing.assert_allclose(receptors[:, 0], xx.ravel(), atol=1e-6)


def test_matched_filter_scene_masks_what_it_is_told(frame):
    _, _, lat, lon = _grid(frame)
    score = np.full((N, N), 0.02)
    sensitivity = np.ones((N, N))
    sensitivity[0, 0] = 0.0
    exclude = np.zeros((N, N), dtype=bool)
    exclude[1, 1] = True
    score[2, 2] = np.nan
    observation = matched_filter_scene(
        score,
        np.full((N, N), 0.004),
        lat=lat,
        lon=lon,
        frame=frame,
        sensitivity=sensitivity,
        exclude=exclude,
    )
    mask = np.asarray(observation.instrument.mask).reshape(N, N)
    assert mask[0, 0] == 0.0 and mask[1, 1] == 0.0 and mask[2, 2] == 0.0
    assert mask.sum() == N * N - 3
    assert np.isfinite(np.asarray(observation.enhancement)).all()
    flags = np.asarray(observation.instrument.flags).reshape(N, N)
    assert flags[1, 1] == int(QualityFlag.CLOUD) and flags[0, 0] == 0


@pytest.mark.parametrize("latitude", [0.0, 60.0])
def test_pixel_size_and_direction(tmp_path, latitude):
    frame = LocalFrame.at(latitude, 10.0)
    paths = write_scene(tmp_path, frame)
    scene = _load(paths, frame)
    assert scene.pixel_size == pytest.approx(PIXEL, rel=1e-3)
    assert math.degrees(abs(float(scene.direction) - DIRECTION)) < 1.0
    np.testing.assert_allclose(scene.origin, 0.0, atol=1e-3)


def test_quality_raster_masks_cloud_water_and_saturation(paths, frame):
    plain = _load(paths, frame)
    screened = _load(paths, frame, quality_path=paths["quality"])
    flags = np.asarray(screened.instrument.flags)
    mask = np.asarray(screened.instrument.mask)
    cloud = (flags & int(QualityFlag.CLOUD)) > 0
    saturated = (flags & int(QualityFlag.SATURATED)) > 0
    assert cloud.any() and saturated.any()
    assert (mask[cloud | saturated] == 0.0).all()
    # only flagged pixels lose their weight
    lost = (np.asarray(plain.instrument.mask) > 0) & (mask == 0)
    np.testing.assert_array_equal(
        lost, (cloud | saturated) & (np.asarray(plain.instrument.mask) > 0)
    )


# ── properties ──────────────────────────────────────────────────────────────


@settings(max_examples=10, deadline=None)
@given(margin=st.integers(0, 5))
def test_window_is_the_bounding_box_plus_the_margin(paths, frame, margin):
    tight = _load(paths, frame, margin_px=0)
    grown = _load(paths, frame, margin_px=margin)
    row_start, row_stop, col_start, col_stop = tight.window
    # the fixture's outline sits more than five pixels inside the scene
    assert grown.window == (
        row_start - margin,
        row_stop + margin,
        col_start - margin,
        col_stop + margin,
    )
    rows, cols = grown.window_shape
    assert np.asarray(grown.pixel_index).max(axis=0).tolist() == [rows - 1, cols - 1]


def test_bad_sensitivity_and_fill_are_always_masked(paths, frame):
    scene = _load(paths, frame, margin_px=40)
    fields = _window_fields(paths, scene)
    mask = np.asarray(scene.instrument.mask)
    assert (mask[fields["sensitivity"] <= 0.0] == 0.0).all()
    assert (mask[fields["enhancement"] == -9999.0] == 0.0).all()
    assert np.isfinite(np.asarray(scene.enhancement)).all()
    assert (np.asarray(scene.instrument.retrieval_variance) > 0.0).all()


@settings(max_examples=10, deadline=None)
@given(scale=st.floats(0.1, 10.0))
def test_scaling_the_score_scales_the_observation(frame, scale):
    _, _, lat, lon = _grid(frame)
    rng = np.random.default_rng(1)
    score = rng.normal(size=(N, N))
    sensitivity = rng.uniform(0.5, 1.5, size=(N, N))
    base = matched_filter_scene(
        score, 0.1, lat=lat, lon=lon, frame=frame, sensitivity=sensitivity
    )
    scaled = matched_filter_scene(
        scale * score, 0.1, lat=lat, lon=lon, frame=frame, sensitivity=sensitivity
    )
    np.testing.assert_allclose(scaled.enhancement, scale * base.enhancement, rtol=1e-12)
    np.testing.assert_array_equal(
        scaled.instrument.averaging_kernel, base.instrument.averaging_kernel
    )


# ── the scene through JAX ───────────────────────────────────────────────────


def test_plume_scene_arrays_pass_through_filter_jit(paths, frame):
    # coupled.Instrument is a frozen dataclass, not a pytree (it checks its
    # shapes with NumPy on construction), so the scene's arrays go in with
    # the instrument filtered out
    scene = _load(paths, frame)
    assert isinstance(scene, PlumeScene)
    arrays = eqx.filter(scene, eqx.is_array)

    @eqx.filter_jit
    def weighted_sum(s: PlumeScene) -> jnp.ndarray:
        # ∑ₙ [in plume] yₙ, and the direction carried through
        return jnp.sum(jnp.where(s.in_plume, s.enhancement, 0.0)) + 0.0 * s.direction

    expected = np.sum(
        np.where(np.asarray(scene.in_plume), np.asarray(scene.enhancement), 0.0)
    )
    assert float(weighted_sum(arrays)) == pytest.approx(float(expected), rel=1e-12)
    assert scene.window_shape == arrays.window_shape


# ── errors and dtypes ───────────────────────────────────────────────────────


def test_unknown_plume_lists_the_identifiers(paths, frame):
    with pytest.raises(MissingVariableError, match="CH4_PlumeComplex-2"):
        _load(paths, frame, plume_id="CH4_PlumeComplex-9")


def test_missing_uncertainty(paths, frame):
    with pytest.raises(MissingVariableError, match="retrieval_sigma_ppm_m"):
        _load(paths, frame, uncertainty_path=None)
    scene = _load(paths, frame, uncertainty_path=None, retrieval_sigma_ppm_m=30.0)
    mask = np.asarray(scene.instrument.mask) > 0
    kernel = np.asarray(scene.instrument.averaging_kernel)[mask]
    np.testing.assert_allclose(
        np.asarray(scene.instrument.retrieval_variance)[mask],
        (KAPPA * kernel * 30.0) ** 2,
        rtol=1e-12,
    )


def test_outline_outside_the_scene(tmp_path, frame):
    paths = write_scene(tmp_path, frame)
    doc = json.loads(paths["outlines"].read_text())
    far = _outline(frame, [(5000.0, 5000.0), (5100.0, 5000.0), (5100.0, 5100.0)])
    doc["features"][1]["geometry"]["coordinates"] = [far]
    paths["outlines"].write_text(json.dumps(doc))
    with pytest.raises(GridMismatchError, match="leaves the scene"):
        _load(paths, frame, plume_id="CH4_PlumeComplex-2")


def test_nothing_survives(tmp_path, frame):
    paths = write_scene(tmp_path, frame)
    with xr.open_dataset(paths["sensitivity"]) as ds:
        zero = ds.load() * 0.0
    zero.to_netcdf(tmp_path / "zero_sens.nc")
    with pytest.raises(ScreeningError, match="no pixel survives"):
        _load(paths, frame, sensitivity_path=tmp_path / "zero_sens.nc")


def test_unknown_unit(frame):
    _, _, lat, lon = _grid(frame)
    # the type-check hook of the test suite rejects the Literal first
    with pytest.raises((UnitsError, TypeCheckError), match="unit"):
        matched_filter_scene(
            np.zeros((N, N)), 1.0, lat=lat, lon=lon, frame=frame, unit="ppb"
        )


def test_float32_rasters_stay_float32(tmp_path, frame):
    paths = write_scene(tmp_path, frame, dtype=np.float32)
    scene = _load(paths, frame)
    assert np.asarray(scene.enhancement).dtype == np.float32
    _, _, lat, lon = _grid(frame)
    observation = matched_filter_scene(
        np.zeros((N, N), dtype=np.float32), 1.0, lat=lat, lon=lon, frame=frame
    )
    assert np.asarray(observation.enhancement).dtype == np.float32
