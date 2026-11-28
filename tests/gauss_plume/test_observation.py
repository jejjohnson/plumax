"""The shared column + averaging-kernel observation (#83)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import DomainError
from plumax.coupled import Instrument
from plumax.gauss_plume.observation import ColumnObservation
from plumax.les_fvm import make_grid


def _field(seed: int = 0, shape: tuple[int, int, int] = (5, 4, 6)) -> jax.Array:
    return jax.random.uniform(jax.random.PRNGKey(seed), shape)


def test_identity_ak_is_column_average():
    # Normalised pressure weights h_k = Δp_k / p_s turn the column into the
    # pressure-weighted column average.
    pressure_edges = jnp.array([1000.0, 900.0, 750.0, 500.0, 200.0, 0.0])
    layer_weights = -jnp.diff(pressure_edges) / pressure_edges[0]
    field = _field()
    observation = ColumnObservation(layer_weights=layer_weights)
    expected = np.einsum("k,kyx->yx", np.asarray(layer_weights), np.asarray(field))
    np.testing.assert_allclose(
        np.asarray(observation.from_field(field)), expected.reshape(-1), rtol=1e-12
    )


def test_averaging_kernel_scales_each_receptor():
    field = _field()
    index = jnp.array([[0, 0], [3, 5], [1, 2]])
    kernel = jnp.array([1.0, 0.5, 0.0])
    plain = ColumnObservation(jnp.full(5, 2.0), receptor_index=index)
    weighted = ColumnObservation(jnp.full(5, 2.0), kernel, index)
    np.testing.assert_allclose(
        np.asarray(weighted.from_field(field)),
        np.asarray(kernel * plain.from_field(field)),
        rtol=1e-12,
    )


@pytest.mark.slow
@settings(max_examples=25, deadline=None)
@given(
    spacing=st.lists(
        st.floats(min_value=0.1, max_value=100.0), min_size=1, max_size=12
    ),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_trapezoid_weights_match_jnp_trapezoid(spacing, seed):
    levels = jnp.concatenate([jnp.zeros(1), jnp.cumsum(jnp.asarray(spacing))])
    profiles = jax.random.normal(jax.random.PRNGKey(seed), (3, levels.shape[0]))
    observation = ColumnObservation.trapezoid(levels)
    np.testing.assert_allclose(
        np.asarray(observation.column(profiles)),
        np.asarray(jnp.trapezoid(profiles, x=levels, axis=1)),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.slow
def test_for_grid_places_instrument_receptors_in_their_cells():
    grid = make_grid((0.0, 400.0, 8), (0.0, 200.0, 4), (0.0, 100.0, 5))
    # Cell centres sit at x = 25, 75, …; y = 25, 75, …
    receptors = jnp.array([[25.0, 25.0], [374.0, 174.0], [101.0, 99.0]])
    kernel = jnp.array([1.0, 0.5, 0.25])
    instrument = Instrument("toy", receptors, averaging_kernel=kernel)
    observation = ColumnObservation.for_grid(grid, instrument=instrument)
    np.testing.assert_array_equal(
        np.asarray(observation.receptor_index), [[0, 0], [3, 7], [1, 2]]
    )
    field = _field(seed=2, shape=(5, 4, 8))
    column = np.asarray(field).sum(axis=0) * grid.dz
    expected = np.asarray(kernel) * column[[0, 3, 1], [0, 7, 2]]
    np.testing.assert_allclose(
        np.asarray(observation.from_field(field)), expected, rtol=1e-12
    )


def test_for_grid_rejects_bad_receptors():
    grid = make_grid((0.0, 400.0, 8), (0.0, 200.0, 4), (0.0, 100.0, 5))
    outside = Instrument("toy", jnp.array([[401.0, 10.0]]))
    with pytest.raises(DomainError):
        ColumnObservation.for_grid(grid, instrument=outside)
    inside = Instrument("toy", jnp.array([[10.0, 10.0]]))
    with pytest.raises(ValueError, match="not both"):
        ColumnObservation.for_grid(
            grid, instrument=inside, receptor_index=jnp.array([[0, 0]])
        )


def test_for_grid_cells_are_half_open_on_their_edges():
    grid = make_grid((0.0, 400.0, 8), (0.0, 200.0, 4), (0.0, 100.0, 5))
    # an inner edge belongs to the cell above it, the lower edge to cell 0
    on_edges = Instrument("toy", jnp.array([[50.0, 0.0], [0.0, 150.0]]))
    observation = ColumnObservation.for_grid(grid, instrument=on_edges)
    np.testing.assert_array_equal(
        np.asarray(observation.receptor_index), [[0, 1], [3, 0]]
    )
    # the last edge is outside, and outside raises
    with pytest.raises(DomainError, match="outside the grid in x"):
        ColumnObservation.for_grid(
            grid, instrument=Instrument("toy", jnp.array([[400.0, 10.0]]))
        )


def test_gradient_is_the_layer_weights_times_the_kernel():
    # y is linear in c: ∂y_i / ∂c_{k,i} = A_i h_k.
    layer_weights = jnp.array([1.0, 2.0, 3.0])
    kernel = jnp.array([0.5, 2.0])
    observation = ColumnObservation(layer_weights, kernel)
    profiles = jnp.ones((2, 3))
    jacobian = jax.jacrev(observation.column)(profiles)
    expected = np.einsum("ij,k->ijk", np.diag(np.asarray(kernel)), layer_weights)
    np.testing.assert_allclose(np.asarray(jacobian), expected, rtol=1e-12)


def test_jit_and_vmap_agree_with_a_loop():
    index = jnp.array([[0, 0], [3, 5]])
    observation = ColumnObservation(jnp.full(5, 2.0), jnp.ones(2), index)
    fields = jnp.stack([_field(seed=s) for s in range(3)])
    looped = np.stack([np.asarray(observation.from_field(f)) for f in fields])
    jitted = jax.jit(lambda field: observation.from_field(field))
    np.testing.assert_allclose(np.asarray(jax.vmap(jitted)(fields)), looped, rtol=1e-12)
