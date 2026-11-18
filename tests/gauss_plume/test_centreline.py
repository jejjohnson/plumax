"""The meandering centreline of the column plume and its prior (#270)."""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from jaxtyping import TypeCheckError

from plumax._src.errors import ParameterError
from plumax.gauss_plume import (
    CentrelinePrior,
    ColumnPlume,
    PlumeGeometry,
    centreline_offset,
    power_law_width,
)


PIXEL = 60.0
# w₀ = g / √12
INITIAL_WIDTH = PIXEL / math.sqrt(12.0)
SPACING = 180.0
KNOTS = jnp.array([40.0, -30.0, 80.0, 120.0, 60.0])


def _geometry(centreline=KNOTS, direction=0.0):
    return PlumeGeometry(
        location=jnp.zeros(2),
        direction=direction,
        reference_width=150.0,
        width_exponent=0.9,
        centreline=centreline,
    )


def _grid(length=1200.0, half_width=600.0):
    xs = jnp.arange(-2 * PIXEL, length + 1.0, PIXEL)
    ys = jnp.arange(-half_width, half_width + 1.0, PIXEL)
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    return einx.rearrange("x y, x y -> (x y) (1 + 1)", gx, gy)


# ── centreline_offset ────────────────────────────────────────────────────────


def test_offset_is_zero_upwind_hits_the_knots_and_holds_beyond():
    n_knots = KNOTS.shape[0]
    # d_k = k Δ_d
    knot_distance = SPACING * jnp.arange(1, n_knots + 1)
    upwind = centreline_offset(
        jnp.array([-500.0, -1.0, 0.0]), KNOTS, knot_spacing=SPACING
    )
    at_knots = centreline_offset(knot_distance, KNOTS, knot_spacing=SPACING)
    beyond = centreline_offset(
        knot_distance[-1] + jnp.array([1.0, 500.0, 5000.0]), KNOTS, knot_spacing=SPACING
    )
    np.testing.assert_array_equal(upwind, 0.0)
    np.testing.assert_allclose(at_knots, KNOTS, rtol=1e-12)
    np.testing.assert_allclose(beyond, KNOTS[-1], rtol=1e-12)


def test_zero_knots_give_the_straight_response():
    receptors = _grid()
    for plume in (
        ColumnPlume(initial_width=INITIAL_WIDTH, knot_spacing=SPACING),
        ColumnPlume(
            initial_width=INITIAL_WIDTH,
            upwind_scale=INITIAL_WIDTH,
            pixel_size=PIXEL,
            knot_spacing=SPACING,
        ),
    ):
        bent = plume(receptors, _geometry(centreline=jnp.zeros(5), direction=0.2))
        straight = plume(receptors, _geometry(centreline=None, direction=0.2))
        np.testing.assert_allclose(bent, straight, rtol=1e-12, atol=0.0)


@settings(max_examples=25, deadline=None)
@given(
    a=st.floats(-3.0, 3.0),
    b=st.floats(-3.0, 3.0),
    seed=st.integers(0, 2**16),
)
def test_offset_is_linear_in_the_knots(a, b, seed):
    first_key, second_key = jax.random.split(jax.random.PRNGKey(seed))
    first = 100.0 * jax.random.normal(first_key, (6,))
    second = 100.0 * jax.random.normal(second_key, (6,))
    downwind = jnp.linspace(-200.0, 1500.0, 97)
    # m(d; a m′ + b m″) = a m(d; m′) + b m(d; m″)
    combined = centreline_offset(downwind, a * first + b * second, knot_spacing=SPACING)
    expected = a * centreline_offset(
        downwind, first, knot_spacing=SPACING
    ) + b * centreline_offset(downwind, second, knot_spacing=SPACING)
    np.testing.assert_allclose(combined, expected, rtol=1e-10, atol=1e-9)


# ── gate G2 with a centreline ────────────────────────────────────────────────


def test_bent_pixel_average_conserves_mass_at_every_distance():
    """Gate G2, exact form, bent: a crosswind row of pixel averages sums to q."""
    plume = ColumnPlume(
        initial_width=INITIAL_WIDTH, pixel_size=PIXEL, knot_spacing=SPACING
    )
    offsets = jnp.arange(-200, 201) * PIXEL
    for distance in (30.0, 100.0, 300.0, 700.0, 3000.0):
        downwind = jnp.full_like(offsets, distance)
        receptors = einx.rearrange("obs, obs -> obs (1 + 1)", downwind, offsets)
        mass = einx.sum("[obs]", plume(receptors, _geometry())) * PIXEL
        np.testing.assert_allclose(mass, 1.0, rtol=1e-10)


def test_bent_point_evaluation_conserves_mass_once_a_pixel_wide():
    """Gate G2, point form, bent: exact up to aliasing where w ≥ g."""
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, knot_spacing=SPACING)
    offsets = jnp.arange(-200, 201) * PIXEL
    distances = jnp.array([500.0, 700.0, 1000.0, 3000.0])
    widths = power_law_width(
        distances,
        reference_width=150.0,
        width_exponent=0.9,
        reference_distance=1000.0,
        initial_width=INITIAL_WIDTH,
    )
    assert bool(jnp.all(widths >= PIXEL))
    for distance in distances:
        downwind = jnp.full_like(offsets, distance)
        receptors = einx.rearrange("obs, obs -> obs (1 + 1)", downwind, offsets)
        mass = einx.sum("[obs]", plume(receptors, _geometry())) * PIXEL
        np.testing.assert_allclose(mass, 1.0, rtol=1e-3)


# ── gradient and transforms ──────────────────────────────────────────────────


def test_knot_gradient_matches_central_differences_and_is_local():
    plume = ColumnPlume(
        initial_width=INITIAL_WIDTH, pixel_size=PIXEL, knot_spacing=SPACING
    )
    receptors = _grid()
    # direction 0: the downwind distance is the receptor's x
    downwind = receptors[:, 0]

    def response(centreline):
        return plume(receptors, _geometry(centreline=centreline))

    jacobian = jax.jacrev(response)(KNOTS)
    step = 1e-4
    # (r(m + h eₖ) − r(m − h eₖ)) / 2h, one column per knot
    differences = [
        (response(KNOTS + step * unit) - response(KNOTS - step * unit)) / (2 * step)
        for unit in jnp.eye(KNOTS.shape[0])
    ]
    expected = einx.rearrange("knot obs -> obs knot", jnp.stack(differences))
    scale = float(jnp.max(jnp.abs(expected)))
    np.testing.assert_allclose(jacobian, expected, rtol=1e-5, atol=1e-5 * scale)

    n_knots = KNOTS.shape[0]
    for k in range(1, n_knots + 1):
        # the hat function of knot k spans d_{k−1} ≤ d ≤ d_{k+1}; the last knot holds
        low = (k - 1) * SPACING
        high = math.inf if k == n_knots else (k + 1) * SPACING
        outside = (downwind < low) | (downwind > high)
        column = jacobian[:, k - 1]
        np.testing.assert_array_equal(column[outside], 0.0)
        assert bool(jnp.any(column[~outside] != 0.0))


def test_eager_jit_vmap_agree_over_centrelines():
    plume = ColumnPlume(
        initial_width=INITIAL_WIDTH, pixel_size=PIXEL, knot_spacing=SPACING
    )
    receptors = _grid(length=600.0, half_width=300.0)
    batch = 50.0 * jax.random.normal(jax.random.PRNGKey(0), (4, KNOTS.shape[0]))

    def response(centreline):
        return plume(receptors, _geometry(centreline=centreline))

    looped = jnp.stack([response(centreline) for centreline in batch])
    jitted = jnp.stack([jax.jit(response)(centreline) for centreline in batch])
    mapped = jax.vmap(response)(batch)
    np.testing.assert_allclose(jitted, looped, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(mapped, looped, rtol=1e-12, atol=1e-18)


def test_float32_in_gives_float32_out():
    plume = ColumnPlume(
        initial_width=INITIAL_WIDTH, pixel_size=PIXEL, knot_spacing=SPACING
    )
    receptors = _grid(length=600.0, half_width=300.0).astype(jnp.float32)
    geometry = PlumeGeometry(
        location=jnp.zeros(2, dtype=jnp.float32),
        direction=jnp.float32(0.1),
        reference_width=jnp.float32(150.0),
        width_exponent=jnp.float32(0.9),
        centreline=KNOTS.astype(jnp.float32),
    )
    assert plume(receptors, geometry).dtype == jnp.float32
    offset = centreline_offset(
        jnp.ones(3, dtype=jnp.float32), KNOTS.astype(jnp.float32), knot_spacing=SPACING
    )
    assert offset.dtype == jnp.float32
    knot_distance = SPACING * jnp.arange(1, 6, dtype=jnp.float32)
    for kind in ("matern32", "random_walk"):
        assert CentrelinePrior(kind=kind)(knot_distance).dtype == jnp.float32


def test_parameter_errors():
    with pytest.raises(ParameterError, match="knot_spacing"):
        ColumnPlume(knot_spacing=0.0)
    with pytest.raises(ParameterError, match="knot_spacing"):
        ColumnPlume()(_grid(length=300.0), _geometry())
    with pytest.raises(ParameterError, match="length_scale"):
        CentrelinePrior(length_scale=0.0)
    with pytest.raises(ParameterError, match="amplitude"):
        CentrelinePrior(amplitude=-1.0)
    with pytest.raises(ParameterError, match="jitter"):
        CentrelinePrior(jitter=-1e-8)
    # the suite's jaxtyping hook rejects the Literal before __check_init__ runs
    with pytest.raises((ParameterError, TypeCheckError)):
        CentrelinePrior(kind="matern52")


def test_knot_spacing_round_trips_through_the_config():
    plume = ColumnPlume(initial_width=INITIAL_WIDTH, knot_spacing=SPACING)
    assert ColumnPlume.from_config(plume.get_config()) == plume


# ── CentrelinePrior ──────────────────────────────────────────────────────────


def test_matern_marginal_variance_is_the_closed_form():
    length_scale, amplitude = 1000.0, 300.0
    knot_distance = SPACING * jnp.arange(1, 34)
    prior = CentrelinePrior(length_scale=length_scale, amplitude=amplitude, jitter=0.0)
    variance = jnp.diagonal(prior(knot_distance))
    # ρ(d) = (1 + √3 d / ℓ) exp(−√3 d / ℓ)
    scaled = math.sqrt(3.0) * knot_distance / length_scale
    correlation = (1.0 + scaled) * jnp.exp(-scaled)
    # 𝕍[m(d)] = σ_m² (1 − ρ(d)²)
    expected = amplitude**2 * (1.0 - jnp.square(correlation))
    np.testing.assert_allclose(variance, expected, rtol=1e-10)


def test_random_walk_is_the_brownian_covariance():
    knot_distance = jnp.array([100.0, 250.0, 900.0])
    prior = CentrelinePrior(
        length_scale=500.0, amplitude=200.0, kind="random_walk", jitter=0.0
    )
    # κ² min(d_j, d_k), κ² = σ_m² / ℓ
    expected = (200.0**2 / 500.0) * jnp.minimum(
        knot_distance[:, None], knot_distance[None, :]
    )
    np.testing.assert_allclose(prior(knot_distance), expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["matern32", "random_walk"])
def test_draws_have_the_prior_covariance(kind):
    knot_distance = SPACING * jnp.arange(1, 7)
    covariance = CentrelinePrior(kind=kind)(knot_distance)
    n_draws = 100_000
    draws = jax.random.multivariate_normal(
        jax.random.PRNGKey(1), jnp.zeros(6), covariance, shape=(n_draws,)
    )
    # S = (1/n) Σ m mᵀ, the mean being known to be zero
    sample = einx.dot("draw row, draw col -> row col", draws, draws) / n_draws
    # se(S_jk) = √((Σ_jk² + Σ_jj Σ_kk) / n)
    diagonal = jnp.diagonal(covariance)
    product = einx.multiply("row, col -> row col", diagonal, diagonal)
    standard_error = jnp.sqrt((jnp.square(covariance) + product) / n_draws)
    assert bool(jnp.all(jnp.abs(sample - covariance) < 3.0 * standard_error))


@settings(max_examples=50, deadline=None)
@given(
    increments=st.lists(st.floats(20.0, 500.0), min_size=1, max_size=30),
    length_scale=st.floats(200.0, 5000.0),
    amplitude=st.floats(10.0, 1000.0),
    kind=st.sampled_from(["matern32", "random_walk"]),
)
def test_covariance_is_symmetric_positive_definite(
    increments, length_scale, amplitude, kind
):
    knot_distance = jnp.cumsum(jnp.asarray(increments))
    prior = CentrelinePrior(length_scale=length_scale, amplitude=amplitude, kind=kind)
    covariance = prior(knot_distance)
    transposed = einx.rearrange("row col -> col row", covariance)
    np.testing.assert_allclose(covariance, transposed, rtol=1e-12, atol=0.0)
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    assert bool(jnp.all(eigenvalues > 0.0))
