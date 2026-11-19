"""Tests for les_fvm.source — Gaussian point-source and surface-flux terms."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.les_fvm.grid import make_grid
from plumax.les_fvm.source import (
    SurfaceFluxSource,
    make_gaussian_source,
)


def _build_grid():
    return make_grid(
        domain_x=(0.0, 400.0, 16),
        domain_y=(0.0, 200.0, 8),
        domain_z=(0.0, 80.0, 8),
    )


@pytest.mark.slow
def test_constant_source_integrates_to_emission_rate():
    g = _build_grid()
    q = 0.07
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=q,
        source_location=(50.0, 100.0, 20.0),
    )
    # Integrate the source tendency ∫ S dV.  Only interior cells carry mass,
    # so we sum the interior slice and multiply by the cell volume.
    s = source(jnp.asarray(0.0))
    interior = s[1:-1, 1:-1, 1:-1]
    total = float(interior.sum()) * (g.dx * g.dy * g.dz)
    # The Gaussian density is normalised to integrate to 1, so the result
    # must be exactly ``q`` up to floating-point noise.
    np.testing.assert_allclose(total, q, rtol=1e-5)


def test_time_varying_emission_reflects_rate_function():
    g = _build_grid()
    # q(t) = 0.01 * t → linearly ramping emission.
    rate = lambda t: 0.01 * t
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=rate,
        source_location=(100.0, 100.0, 20.0),
    )
    for t_query in (0.0, 5.0, 10.0):
        s = source(jnp.asarray(t_query, dtype=jnp.float32))
        interior = s[1:-1, 1:-1, 1:-1]
        total = float(interior.sum()) * (g.dx * g.dy * g.dz)
        np.testing.assert_allclose(total, rate(t_query), rtol=1e-5)


def test_zero_rate_accepted_and_yields_zero_source():
    g = _build_grid()
    source = make_gaussian_source(
        plume_grid=g,
        emission_rate=0.0,
        source_location=(100.0, 100.0, 20.0),
    )
    s = source(jnp.asarray(0.0))
    np.testing.assert_allclose(np.asarray(s), 0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"emission_rate": -0.1, "source_location": (100.0, 100.0, 20.0)},
            r"`emission_rate` must be",
            id="constant_negative_rate",
        ),
        pytest.param(
            {"emission_rate": 0.1, "source_location": (-10.0, 100.0, 20.0)},
            r"outside the interior domain",
            id="source_outside_domain",
        ),
        pytest.param(
            {
                "emission_rate": 0.1,
                "source_location": (100.0, 100.0, 20.0),
                "source_radius": -1.0,
            },
            r"`source_radius` must be > 0",
            id="negative_radius",
        ),
    ],
)
def test_make_gaussian_source_rejects_invalid_inputs(kwargs, match):
    g = _build_grid()
    with pytest.raises(ValueError, match=match):
        make_gaussian_source(plume_grid=g, **kwargs)


def test_surface_flux_tendency_closes_the_mass_budget():
    # Σ ∂C/∂t Δx Δy Δz = Σ q Δx Δy, in the lowest layer only
    grid = make_grid((0.0, 100.0, 5), (0.0, 80.0, 4), (0.0, 40.0, 4))
    flux = jnp.arange(1.0, 21.0).reshape(4, 5)
    source = SurfaceFluxSource(lambda t: flux, grid.dz, 4)
    tendency = source(jnp.asarray(0.0))
    assert tendency.shape == grid.shape
    assert float(jnp.abs(tendency[2:]).max()) == 0.0
    injected = jnp.sum(tendency) * grid.dx * grid.dy * grid.dz
    expected = jnp.sum(flux) * grid.dx * grid.dy
    assert float(jnp.abs(injected - expected)) < 1e-10 * float(expected)
