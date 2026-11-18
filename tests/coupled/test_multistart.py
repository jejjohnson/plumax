"""The coupled solve from a given start and from several starts (#271).

Toy scenes only: a column plume on a few hundred 60 m pixels.
"""

from __future__ import annotations

import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import (
    CoupledPrior,
    CoupledProblem,
    Instrument,
    MultistartResult,
    build_coupled_forward,
    solve_coupled,
    solve_coupled_multistart,
)
from plumax.errors import ControlBlockError, ConvergenceError
from plumax.gauss_plume import ColumnPlume, ColumnPlumeSource, PlumeGeometry


PIXEL = 60.0
# w₀ = g / √12
INITIAL_WIDTH = PIXEL / math.sqrt(12.0)
LINE_DENSITY = 0.1
NOISE = 2e-4


def _receptors(length=1800.0, half_width=600.0):
    xs = jnp.arange(-2 * PIXEL, length + 1.0, PIXEL)
    ys = jnp.arange(-half_width, half_width + 1.0, PIXEL)
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    return einx.id("x y, x y -> (x y) (1 + 1)", gx, gy)


def _geometry():
    return PlumeGeometry(
        location=jnp.zeros(2),
        direction=0.1,
        reference_width=120.0,
        width_exponent=0.8,
    )


def _source():
    plume = ColumnPlume(
        reference_distance=900.0,
        initial_width=INITIAL_WIDTH,
        upwind_scale=INITIAL_WIDTH,
        pixel_size=PIXEL,
    )
    return ColumnPlumeSource(plume=plume, geometry=_geometry())


def _problem(noise, seed, location_mean, location_std, length=1800.0):
    source = _source()
    instrument = Instrument(
        "EMIT", _receptors(length=length), retrieval_variance=noise**2
    )
    # y = q r + ε
    clean = LINE_DENSITY * source.column_response(instrument)
    observation = clean + noise * jax.random.normal(
        jax.random.PRNGKey(seed), clean.shape
    )
    geometry = source.geometry
    mean = {
        "log_line_density": jnp.log(LINE_DENSITY),
        "log_reference_width": jnp.log(geometry.reference_width),
        "width_exponent": geometry.width_exponent,
        "direction": geometry.direction,
        "location": jnp.asarray(location_mean, dtype=float),
        "background": 0.0,
    }
    std = {
        "log_line_density": 2.0,
        "log_reference_width": 1.0,
        "width_exponent": 0.5,
        "direction": 0.3,
        "location": location_std,
        "background": 1e-4,
    }
    forward = build_coupled_forward(source, [instrument])
    return CoupledProblem(
        forward=forward,
        prior=CoupledPrior.from_std(mean, std),
        observations={"EMIT": observation},
    )


def _grid(centre, n=3):
    # s₀ + g (i, j) for i, j ∈ {−1, 0, 1}, x first
    offsets = PIXEL * jnp.arange(-(n // 2), n // 2 + 1, dtype=float)
    offset_x, offset_y = jnp.meshgrid(offsets, offsets, indexing="ij")
    grid = einx.id("x y, x y -> (x y) (1 + 1)", offset_x, offset_y)
    return einx.add("start axis, axis -> start axis", grid, jnp.asarray(centre))


# ── solve_coupled(initial=...) ──────────────────────────────────────────────


@pytest.mark.slow
def test_initial_at_the_prior_mean_is_the_default():
    problem = _problem(NOISE, 0, (0.0, 0.0), 60.0, length=600.0)
    default = solve_coupled(problem)
    started = solve_coupled(problem, initial=problem.prior.mean)
    np.testing.assert_allclose(started.mean, default.mean, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(started.covariance, default.covariance, rtol=1e-10)
    assert started.n_steps == default.n_steps


@pytest.mark.slow
def test_initial_changes_the_start_not_the_prior():
    problem = _problem(NOISE, 0, (0.0, 0.0), 60.0, length=600.0)
    default = solve_coupled(problem)
    moved = solve_coupled(problem, initial={"location": jnp.array([60.0, -60.0])})
    assert moved.converged
    assert moved.prior is problem.prior
    # the same minimum, reached by another path
    np.testing.assert_allclose(moved.cost, default.cost, rtol=1e-8)
    np.testing.assert_allclose(moved.mean, default.mean, rtol=1e-4, atol=1e-4)
    assert moved.n_steps != default.n_steps


def test_initial_validation():
    problem = _problem(NOISE, 0, (0.0, 0.0), 60.0, length=300.0)
    with pytest.raises(ControlBlockError, match="does not hold"):
        solve_coupled(problem, initial={"wind": jnp.zeros(2)})
    with pytest.raises(ControlBlockError, match="shape"):
        solve_coupled(problem, initial={"location": jnp.zeros(3)})


# ── solve_coupled_multistart ────────────────────────────────────────────────


def test_starts_validation():
    problem = _problem(NOISE, 0, (0.0, 0.0), 60.0, length=300.0)
    with pytest.raises(ControlBlockError, match="empty"):
        solve_coupled_multistart(problem, {})
    mismatched = {"location": jnp.zeros((3, 2)), "direction": jnp.zeros(2)}
    with pytest.raises(ControlBlockError, match="same"):
        solve_coupled_multistart(problem, mismatched)
    with pytest.raises(ValueError, match="initial"):
        solve_coupled_multistart(
            problem, {"location": jnp.zeros((1, 2))}, initial=problem.prior.mean
        )


@pytest.mark.slow
def test_no_converged_start_raises():
    problem = _problem(NOISE, 0, (0.0, 0.0), 60.0, length=300.0)
    starts = {"location": jnp.array([[60.0, 0.0], [0.0, 60.0]])}
    with pytest.raises(ConvergenceError, match="none of the 2 starts"):
        solve_coupled_multistart(problem, starts, max_steps=1, laplace=False)


@pytest.mark.slow
def test_multistart_escapes_the_minimum_a_single_start_falls_into():
    # three times the noise and a catalogue prior 2 pixels north of the source,
    # 5 pixels wide: a scene, found by a scan of seeds and offsets, where the
    # start at the prior mean ends in a second minimum
    noise = 3 * NOISE
    prior_source = (0.0, 2 * PIXEL)
    problem = _problem(noise, 2, prior_source, 5 * PIXEL)
    fits = solve_coupled_multistart(problem, {"location": _grid(prior_source)})
    assert isinstance(fits, MultistartResult)
    assert fits.costs.shape == fits.converged.shape == (9,)
    assert len(fits.map_estimates) == 9

    # the middle start is the prior mean, the single-start solve
    single = fits.map_estimates[4]["location"]
    assert float(jnp.linalg.norm(single)) > PIXEL

    # the minimum the data put the source at: the fit started from the truth
    oracle = solve_coupled(problem, initial={"location": jnp.zeros(2)})
    best = fits.best.map_estimate["location"]
    assert float(jnp.linalg.norm(best - oracle.map_estimate["location"])) < PIXEL / 2
    assert fits.best.cost < fits.costs[4]
    np.testing.assert_allclose(fits.best.cost, oracle.cost, rtol=1e-6)
    assert fits.index == int(np.argmin(np.where(fits.converged, fits.costs, np.inf)))
