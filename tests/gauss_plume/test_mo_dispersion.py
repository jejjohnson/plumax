"""Tests for the Golder (1972) Monin–Obukhov → Pasquill–Gifford dispersion."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import ParameterError
from plumax.gauss_plume import (
    GolderDispersion,
    calculate_briggs_dispersion,
    get_dispersion_params,
    golder_class_boundaries,
    golder_stability_class,
    golder_stability_index,
    plume_concentration,
    simulate_plume,
)
from plumax.gauss_plume.dispersion import STABILITY_CLASSES


def _mmif_pg_golder(obukhov_length: float, z0: float) -> int:
    """Line-by-line port of ``pg_golder`` in EPA MMIF v4.1.1, as a 0-based class."""
    z0 = min(0.5, max(0.01, z0))
    log_z0 = math.log(z0)
    if obukhov_length < 0.0:
        if obukhov_length >= 70.0 / (log_z0 - 4.350):
            return 0
        if obukhov_length >= 85.2 / (log_z0 - 0.502):
            return 1
        if obukhov_length >= 245.0 / (log_z0 - 0.050):
            return 2
        return 3
    if obukhov_length >= -327.0 / (log_z0 - 0.627):
        return 3
    if obukhov_length >= -70.0 / (log_z0 - 0.295):
        return 4
    return 5


# ---- class boundaries -------------------------------------------------------


def test_boundaries_match_golder_curves_at_z0_0_1():
    # L_k = c_k / (ln z₀ − d_k) at z₀ = 0.1 m, hand-computed
    log_z0 = math.log(0.1)
    expected_lengths = [
        70.0 / (log_z0 - 4.350),
        85.2 / (log_z0 - 0.502),
        245.0 / (log_z0 - 0.050),
        -327.0 / (log_z0 - 0.627),
        -70.0 / (log_z0 - 0.295),
    ]
    np.testing.assert_allclose(
        1.0 / np.asarray(golder_class_boundaries(0.1)), expected_lengths, rtol=1e-12
    )
    np.testing.assert_allclose(
        expected_lengths, [-10.5222, -30.3788, -104.141, 111.620, 26.9481], rtol=1e-5
    )


@pytest.mark.parametrize("z0", [0.01, 0.03, 0.1, 0.3, 0.5])
def test_lookup_switches_class_exactly_at_each_boundary(z0):
    boundaries = np.asarray(golder_class_boundaries(z0))
    assert np.all(np.diff(boundaries) > 0.0)
    for k, boundary in enumerate(boundaries):
        # on the boundary: the less stable class, as in MMIF
        on, below, above = boundary, boundary * (1 - 1e-9), boundary * (1 + 1e-9)
        less_stable = below if boundary > 0.0 else above
        more_stable = above if boundary > 0.0 else below
        assert int(golder_stability_class(on, z0)) == k
        assert int(golder_stability_class(less_stable, z0)) == k
        assert int(golder_stability_class(more_stable, z0)) == k + 1


@settings(max_examples=200, deadline=None)
@given(
    obukhov_length=st.one_of(
        st.floats(min_value=-1e4, max_value=-0.5),
        st.floats(min_value=0.5, max_value=1e4),
    ),
    z0=st.floats(min_value=1e-3, max_value=2.0),
)
def test_lookup_matches_the_mmif_reference(obukhov_length, z0):
    ours = int(golder_stability_class(1.0 / obukhov_length, z0))
    assert ours == _mmif_pg_golder(obukhov_length, z0)


def test_z0_is_clamped_to_golders_range():
    np.testing.assert_array_equal(
        golder_class_boundaries(1e-4), golder_class_boundaries(0.01)
    )
    np.testing.assert_array_equal(
        golder_class_boundaries(3.0), golder_class_boundaries(0.5)
    )


# ---- neutral limit ----------------------------------------------------------


@pytest.mark.parametrize("z0", [0.01, 0.1, 0.5])
@pytest.mark.parametrize("obukhov_length", [-1e6, -1e12, 1e6, 1e12, math.inf])
def test_large_abs_l_is_class_d(z0, obukhov_length):
    inverse = 1.0 / obukhov_length
    assert STABILITY_CLASSES[int(golder_stability_class(inverse, z0))] == "D"
    np.testing.assert_allclose(golder_stability_index(inverse, z0), 3.0, atol=1e-3)


# ---- continuous index -------------------------------------------------------


def test_index_rounds_to_the_hard_class():
    z0 = 0.1
    inverse = jnp.linspace(-0.3, 0.3, 4001)
    index = np.asarray(golder_stability_index(inverse, z0))
    hard = np.asarray(golder_stability_class(inverse, z0))
    # ties (index on a half-integer, i.e. a boundary) go to the less stable class
    np.testing.assert_array_equal(np.ceil(index - 0.5), hard)
    assert np.all(np.diff(index) >= 0.0)
    assert index[0] == 0.0 and index[-1] == 5.0


# ---- σ ordering --------------------------------------------------------------


@pytest.mark.parametrize("interpolate", [False, True])
@pytest.mark.parametrize("distance", [100.0, 1000.0, 10000.0])
def test_sigma_monotone_in_stability(interpolate, distance):
    # from very unstable (1/L < 0) to very stable (1/L > 0), σ never grows
    inverse = jnp.linspace(-0.3, 0.3, 601)
    sigma_y, sigma_z = jax.vmap(
        lambda s: GolderDispersion(s, z0=0.1, interpolate=interpolate)(
            jnp.array(distance)
        )
    )(inverse)
    assert np.all(np.diff(np.asarray(sigma_y)) <= 0.0)
    assert np.all(np.diff(np.asarray(sigma_z)) <= 0.0)
    assert sigma_z[0] > sigma_z[300] > sigma_z[-1]


@pytest.mark.parametrize("interpolate", [False, True])
def test_each_class_centre_reproduces_pg(interpolate):
    distance = jnp.array([50.0, 500.0, 5000.0])
    boundaries = np.asarray(golder_class_boundaries(0.1))
    # deep in A, the B and C midpoints, neutral, the E midpoint, deep in F
    inverse = [
        2.0 * boundaries[0],
        0.5 * (boundaries[0] + boundaries[1]),
        0.5 * (boundaries[1] + boundaries[2]),
        0.0,
        0.5 * (boundaries[3] + boundaries[4]),
        2.0 * boundaries[4],
    ]
    for stability_class, value in zip(STABILITY_CLASSES, inverse, strict=True):
        sigma = GolderDispersion(value, z0=0.1, interpolate=interpolate)(distance)
        expected = calculate_briggs_dispersion(
            distance, get_dispersion_params(stability_class)
        )
        np.testing.assert_allclose(sigma, expected, rtol=1e-12)


# ---- gradients ---------------------------------------------------------------


def _sigma_z(inverse, interpolate):
    return GolderDispersion(inverse, z0=0.1, interpolate=interpolate)(
        jnp.array(1000.0)
    )[1]


@pytest.mark.parametrize("interpolate", [False, True])
def test_gradient_wrt_l_is_finite(interpolate):
    for obukhov_length in [-5.0, -20.0, -60.0, 1e9, 60.0, 20.0, 5.0]:
        inverse = 1.0 / obukhov_length
        # d/dL = −(1/L)² d/d(1/L)
        grad_l = jax.grad(lambda length: _sigma_z(1.0 / length, interpolate))(
            obukhov_length
        )
        assert np.isfinite(float(grad_l))
        assert np.isfinite(float(jax.grad(_sigma_z)(inverse, interpolate)))


@pytest.mark.parametrize("obukhov_length", [-20.0, -60.0, -500.0, 500.0, 60.0])
def test_interpolated_gradient_nonzero_inside_a_class(obukhov_length):
    grad = jax.grad(lambda length: _sigma_z(1.0 / length, True))(obukhov_length)
    assert float(grad) != 0.0
    hard = jax.grad(lambda length: _sigma_z(1.0 / length, False))(obukhov_length)
    assert float(hard) == 0.0


def test_gradient_through_the_plume_against_finite_differences():
    def concentration(inverse):
        dispersion = GolderDispersion(inverse, z0=0.1, interpolate=True)
        return plume_concentration(
            jnp.array([800.0]),
            jnp.array([30.0]),
            jnp.array([2.0]),
            0.0,
            0.0,
            10.0,
            5.0,
            0.0,
            1.0,
            dispersion,
        )[0]

    inverse, step = -0.02, 1e-6
    fd = (concentration(inverse + step) - concentration(inverse - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(concentration)(inverse), fd, rtol=1e-5)


# ---- PG stays the default ---------------------------------------------------


@pytest.mark.parametrize("stability_class", STABILITY_CLASSES)
def test_plume_with_golder_class_equals_pg(stability_class):
    boundaries = np.asarray(golder_class_boundaries(0.1))
    centre = {
        "A": 2.0 * boundaries[0],
        "B": 0.5 * (boundaries[0] + boundaries[1]),
        "C": 0.5 * (boundaries[1] + boundaries[2]),
        "D": 0.0,
        "E": 0.5 * (boundaries[3] + boundaries[4]),
        "F": 2.0 * boundaries[4],
    }[stability_class]
    x = jnp.linspace(-100.0, 3000.0, 9)
    y = jnp.full(9, 20.0)
    z = jnp.full(9, 1.5)
    args = (x, y, z, 0.0, 0.0, 15.0, 4.0, 1.0, 0.5)
    pg = plume_concentration(*args, get_dispersion_params(stability_class))
    golder = plume_concentration(*args, GolderDispersion(centre, z0=0.1))
    np.testing.assert_array_equal(np.asarray(golder), np.asarray(pg))


def test_simulate_plume_accepts_golder():
    pytest.importorskip("xarray")
    kwargs = dict(
        domain_x=(0.0, 1000.0, 5), domain_y=(-200.0, 200.0, 5), domain_z=(0.0, 50.0, 3)
    )
    pg = simulate_plume(0.5, (0.0, 0.0, 10.0), 4.0, 270.0, "D", **kwargs)
    golder = simulate_plume(
        0.5, (0.0, 0.0, 10.0), 4.0, 270.0, GolderDispersion(0.0, z0=0.1), **kwargs
    )
    np.testing.assert_array_equal(golder["concentration"], pg["concentration"])
    assert golder.attrs["stability_class"] == "golder"
    assert pg.attrs["stability_class"] == "D"


def test_jit_and_vmap_agree_with_a_loop():
    inverse = jnp.array([-0.08, -0.02, 0.0, 0.02, 0.05])
    distance = jnp.array(700.0)

    def sigma(value):
        return GolderDispersion(value, z0=0.2, interpolate=True)(distance)

    looped = np.array([np.asarray(sigma(v)) for v in inverse])
    vmapped = np.stack([np.asarray(s) for s in jax.vmap(sigma)(inverse)], axis=-1)
    jitted = np.array([np.asarray(jax.jit(sigma)(v)) for v in inverse])
    np.testing.assert_allclose(vmapped, looped, rtol=1e-12)
    np.testing.assert_allclose(jitted, looped, rtol=1e-12)


def test_non_positive_z0_is_rejected():
    with pytest.raises(ParameterError, match="z0"):
        GolderDispersion(0.0, z0=0.0)
