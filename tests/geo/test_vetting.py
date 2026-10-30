"""PlumeVetting, plumax.vetting on GeoTensors as one Dataset (#157).

Gate VT7: on the plume-vetting notebook's scene wrapped as carriers, the
operator's scores equal `vet_candidate`'s bit for bit, and ``p_null`` equals
the null run by hand with the same key. Gate VT8 (part): ``plumax.vetting``,
the L1 package, imports no carrier library. Needs the ``[geo]`` extra (in the
dev group).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import xarray as xr
from radtrax import ParameterError, ShapeError

from plumax.geo import PlumeVetting
from plumax.vetting import null_masks, null_pvalue, vet_candidate
from tests.geo._carriers import DEFAULT_CRS, DEFAULT_TRANSFORM, GeoTensor
from tests.vetting._vetting_scene import notebook_scene, vet_scene


BOX = 16  # the notebook's pool margin


def _carriers(scene, *, mf: np.ndarray | None = None) -> tuple:
    """The scene's radiance as a (B, H, W) GeoTensor and its MF as (H, W)."""
    cube = GeoTensor(
        np.moveaxis(scene.radiance, -1, 0),
        transform=DEFAULT_TRANSFORM,
        crs=DEFAULT_CRS,
        fill_value_default=None,
        attrs={"wavelengths": scene.wavelength_nm.tolist()},
    )
    score = GeoTensor(
        scene.mf if mf is None else mf,
        transform=DEFAULT_TRANSFORM,
        crs=DEFAULT_CRS,
        fill_value_default=None,
    )
    return cube, score


def _operator(scene, **kwargs) -> PlumeVetting:
    return PlumeVetting(scene.window, scene.table, scene.delta_grid, box=BOX, **kwargs)


@pytest.fixture(scope="module")
def scene():
    return notebook_scene()


@pytest.fixture(scope="module")
def scores(scene) -> xr.Dataset:
    cube, mf = _carriers(scene)
    return _operator(scene, n_null=0)(cube, mf, scene.masks)


def test_scores_equal_vet_candidate_bit_for_bit(scene, scores: xr.Dataset) -> None:
    """Gate VT7."""
    assert scores.sizes == {"candidate": 2, "band": 9}
    for c, mask in enumerate(scene.masks):
        result = vet_scene(scene, mask, box=BOX)
        row = scores.isel(candidate=c)
        assert row["delta_x_mol_m2"].item() == float(result.delta)
        assert row["d_norm"].item() == float(result.d_norm)
        assert row["n_pairs"].item() == result.n_pairs
        np.testing.assert_array_equal(row["ratio"], np.asarray(result.ratio))
        continuum = np.asarray(result.fit.continuum)
        np.testing.assert_array_equal(row["continuum"], continuum)
        fitted = continuum * np.asarray(result.fit.transmittance)
        np.testing.assert_array_equal(row["fit"], fitted)
    fit_bands = scene.window.fit_bands(scene.wavelength_nm)
    np.testing.assert_array_equal(
        scores["wavelength_nm"], scene.wavelength_nm[fit_bands]
    )


def test_without_a_null_p_null_is_nan(scores: xr.Dataset) -> None:
    assert np.isnan(scores["p_null"]).all()


def test_p_null_equals_the_null_run_by_hand(scene) -> None:
    key = jax.random.key(3)
    cube, mf = _carriers(scene)
    scores = _operator(scene, n_null=3, key=key)(cube, mf, scene.masks)
    expected = []
    for mask in scene.masks:
        key, subkey = jax.random.split(key)
        copies = null_masks(mask, 3, key=subkey, exclude=scene.masks)
        null = [float(vet_scene(scene, copy, box=BOX).d_norm) for copy in copies]
        d_norm = float(vet_scene(scene, mask, box=BOX).d_norm)
        expected.append(null_pvalue(d_norm, np.array(null)))
    np.testing.assert_array_equal(scores["p_null"], expected)


def test_nodata_pixels_leave_the_candidates_and_the_pool(scene) -> None:
    mf = scene.mf.copy()
    hole = np.argwhere(scene.masks[0])[:5]
    mf[tuple(hole.T)] = np.nan
    cube, score = _carriers(scene, mf=mf)
    scores = _operator(scene, n_null=0)(cube, score, scene.masks)
    invalid = np.isnan(mf)
    result = vet_candidate(
        scene.radiance,
        mf,
        scene.masks[0] & ~invalid,
        scene.wavelength_nm,
        scene.table,
        scene.delta_grid,
        scene.window,
        exclude=np.concatenate([scene.masks, invalid[None]]),
        box=BOX,
    )
    assert scores["d_norm"].isel(candidate=0).item() == float(result.d_norm)


def test_the_dataset_round_trips_through_netcdf(
    scores: xr.Dataset, tmp_path: Path
) -> None:
    path = tmp_path / "scores.nc"
    scores.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        xr.testing.assert_identical(reloaded.load(), scores)


def test_attrs_hold_the_window_and_the_key(scene, scores: xr.Dataset) -> None:
    window = json.loads(scores.attrs["window"])
    assert window["gas"] == "CH4"
    assert window["mf_background_mol_m2"] == scene.window.mf_background_mol_m2
    key_data = jax.random.key_data(jax.random.key(0))
    np.testing.assert_array_equal(scores.attrs["key"], np.asarray(key_data))


def test_get_config_round_trips(scene) -> None:
    op = _operator(scene, n_null=7, k=30, keep=0.4, degree=1)
    config = json.loads(json.dumps(op.get_config()))
    rebuilt = PlumeVetting.from_config(
        config, table=scene.table, delta_grid=scene.delta_grid
    )
    assert rebuilt.window == scene.window
    assert rebuilt.get_config() == config


def test_a_cube_without_wavelengths_raises(scene) -> None:
    cube, mf = _carriers(scene)
    cube.attrs = {}
    with pytest.raises(ShapeError, match="wavelengths"):
        _operator(scene, n_null=0)(cube, mf, scene.masks)


def test_wavelengths_that_do_not_match_the_bands_raise(scene) -> None:
    cube, mf = _carriers(scene)
    cube.attrs = {"wavelengths": scene.wavelength_nm[:-1].tolist()}
    with pytest.raises(ShapeError, match="wavelengths"):
        _operator(scene, n_null=0)(cube, mf, scene.masks)


def test_a_table_that_does_not_match_the_fitting_bands_raises(scene) -> None:
    cube, mf = _carriers(scene)
    op = PlumeVetting(scene.window, scene.table[:, :-1], scene.delta_grid, n_null=0)
    with pytest.raises(ShapeError, match="table"):
        op(cube, mf, scene.masks)


def test_masks_off_the_cube_grid_raise(scene) -> None:
    cube, mf = _carriers(scene)
    with pytest.raises(ShapeError, match="masks"):
        _operator(scene, n_null=0)(cube, mf, scene.masks[:, :-1])


def test_a_negative_n_null_raises(scene) -> None:
    with pytest.raises(ParameterError, match="n_null"):
        _operator(scene, n_null=-1)


def test_plumax_vetting_imports_no_carrier_library() -> None:
    """Gate VT8 (part): the L1 package stays on arrays."""
    carriers = {"geoops", "georeader", "geotoolz", "geoarray", "pipekit"}
    root = Path(__file__).resolve().parents[2] / "src" / "plumax" / "vetting"
    found = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            found += [f"{path.name}: {n}" for n in names if n.split(".")[0] in carriers]
    assert not found, found
