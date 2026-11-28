"""Tests for les_fvm.grid."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.les_fvm.grid import (
    PlumeGrid3D,
    coord_arrays_from_grid,
    coord_to_numpy,
    make_grid,
)


def test_make_grid_basic_shape():
    g = make_grid(
        domain_x=(0.0, 100.0, 8),
        domain_y=(0.0, 200.0, 4),
        domain_z=(0.0, 50.0, 4),
    )
    assert isinstance(g, PlumeGrid3D)
    # Full shape includes one ghost ring per axis.
    assert g.shape == (6, 6, 10)
    # Interior shape equals the requested counts.
    assert g.interior_shape == (4, 4, 8)
    # Cell spacings come from (L / n_interior).
    np.testing.assert_allclose(g.dx, 100.0 / 8)
    np.testing.assert_allclose(g.dy, 200.0 / 4)
    np.testing.assert_allclose(g.dz, 50.0 / 4)


def test_make_grid_coordinate_alignment():
    g = make_grid(
        domain_x=(10.0, 50.0, 4),
        domain_y=(0.0, 40.0, 4),
        domain_z=(0.0, 20.0, 4),
    )
    # Interior x-coordinates: cell centres at x_min + (i + 0.5) * dx for i=0..n-1.
    dx = g.dx
    expected_x = 10.0 + (np.arange(4) + 0.5) * dx
    np.testing.assert_allclose(np.asarray(g.x), expected_x)


def test_grid_coordinates_are_the_cell_centres_off_the_origin():
    # The gridax grid carries the origin, so its interior T points are the
    # physical cell centres even when the domain does not start at 0.
    g = make_grid(
        domain_x=(-100.0, 300.0, 8),
        domain_y=(50.0, 90.0, 4),
        domain_z=(0.0, 40.0, 4),
        dtype=jnp.float64,
    )
    for axis in "xyz":
        np.testing.assert_array_equal(
            np.asarray(g.grid.coords(axis))[1:-1], np.asarray(getattr(g, axis))
        )
    np.testing.assert_array_equal(np.asarray(g.x)[:3], [-75.0, -25.0, 25.0])


@pytest.mark.parametrize(
    ("domain_x", "domain_y", "domain_z", "match"),
    [
        pytest.param(
            (0.0, 10.0, 3),
            (0.0, 10.0, 4),
            (0.0, 10.0, 4),
            r"at least 4 interior cells",
            id="small_interior",
        ),
        pytest.param(
            (10.0, 10.0, 4),
            (0.0, 10.0, 4),
            (0.0, 10.0, 4),
            r"extent must be positive",
            id="nonpositive_extent",
        ),
        # Regression test for PR #16: the validation error used to always say
        # ``x_max - x_min`` regardless of which axis failed.  The message must
        # now name the actual axis that violated the constraint.
        pytest.param(
            (0.0, 10.0, 4),
            (5.0, 5.0, 4),  # zero y-extent
            (0.0, 10.0, 4),
            r"y_max - y_min",
            id="error_message_names_y_axis",
        ),
        pytest.param(
            (0.0, 10.0, 4),
            (0.0, 10.0, 4),
            (5.0, 5.0, 4),  # zero z-extent
            r"z_max - z_min",
            id="error_message_names_z_axis",
        ),
    ],
)
def test_make_grid_rejects_invalid_domain(domain_x, domain_y, domain_z, match):
    with pytest.raises(ValueError, match=match):
        make_grid(domain_x=domain_x, domain_y=domain_y, domain_z=domain_z)


def test_coord_arrays_from_grid_layout():
    g = make_grid(
        domain_x=(0.0, 40.0, 4),
        domain_y=(0.0, 20.0, 4),
        domain_z=(0.0, 10.0, 4),
    )
    X, Y, Z = coord_arrays_from_grid(g)
    assert X.shape == g.interior_shape
    assert Y.shape == g.interior_shape
    assert Z.shape == g.interior_shape
    # X varies along the last axis (x), Z along the first (z).
    np.testing.assert_allclose(np.asarray(X[0, 0, :]), np.asarray(g.x))
    np.testing.assert_allclose(np.asarray(Y[0, :, 0]), np.asarray(g.y))
    np.testing.assert_allclose(np.asarray(Z[:, 0, 0]), np.asarray(g.z))


def test_coord_to_numpy_matches_jnp_arrays():
    g = make_grid(
        domain_x=(0.0, 40.0, 4),
        domain_y=(0.0, 20.0, 4),
        domain_z=(0.0, 10.0, 4),
    )
    x_np, y_np, z_np = coord_to_numpy(g)
    np.testing.assert_allclose(x_np, np.asarray(g.x))
    np.testing.assert_allclose(y_np, np.asarray(g.y))
    np.testing.assert_allclose(z_np, np.asarray(g.z))


def test_make_grid_supports_float32_jit():
    # Sanity: grid construction returns JAX arrays compatible with jit.
    g = make_grid(
        domain_x=(0.0, 40.0, 4),
        domain_y=(0.0, 20.0, 4),
        domain_z=(0.0, 10.0, 4),
    )
    assert g.x.dtype == jnp.float32
