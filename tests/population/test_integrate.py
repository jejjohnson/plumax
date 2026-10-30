"""Log-space mark integration and Pareto-smoothed importance weights."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.population.integrate import (
    log_gauss_hermite,
    log_simpson,
    pareto_smoothed_log_weights,
)
from plumax.population.marks import LognormalParetoMark


pytest.importorskip("xtremax")


def _log_hill(q, q50=1.0, steepness=2.0):
    # log P_d = −log(1 + (Q₅₀/q)ⁿ)
    return -jnp.log1p((q50 / q) ** steepness)


@pytest.mark.parametrize("power", [1.0, 2.0, -1.5])
def test_log_gauss_hermite_lognormal_moments(power):
    # E[Qᵏ] = exp(kμ + k²σ²/2) for Q ~ LN(μ, σ²)
    value = log_gauss_hermite(lambda q: power * jnp.log(q), 0.3, 0.6)
    np.testing.assert_allclose(value, power * 0.3 + 0.5 * power**2 * 0.36, rtol=1e-10)


def test_log_gauss_hermite_stays_finite_where_linear_underflows():
    # f(q) = q⁻⁴⁰⁰ underflows to 0 in float64 at the nodes, its log does not
    value = log_gauss_hermite(lambda q: -400.0 * jnp.log(q), 5.0, 0.005)
    assert float(jnp.exp(value)) == 0.0
    np.testing.assert_allclose(value, -400.0 * 5.0 + 0.5 * 400.0**2 * 0.005**2)


@pytest.mark.parametrize("alpha", [1.5, 2.0, 2.5, 3.0, 4.0])
def test_log_space_vs_pareto_is(alpha):
    mark = LognormalParetoMark(q_break_loc=2.0)
    theta = {"mu": 0.0, "sigma": 0.7, "alpha": alpha, "q_break": 2.0}

    def log_integrand(q):
        return _log_hill(q) + mark.log_prob(q, **theta)

    log_space = float(jnp.exp(log_simpson(log_integrand, 1e-6, 1e10, n=4001)))
    # Monte Carlo reference: the body by its truncated lognormal and the tail
    # by its Pareto, sampled exactly, so the weights are the piece masses
    draws = np.asarray(mark.sample(jax.random.key(1), 4_000_000, **theta))
    detected = np.asarray(jnp.exp(_log_hill(jnp.asarray(draws))))
    reference = detected.mean()
    standard_error = detected.std() / np.sqrt(draws.size)
    assert abs(log_space - reference) < 4.0 * standard_error
    # a uniform grid in linear Q cannot resolve the body and the tail together
    grid = np.linspace(0.0, 2e4, 5001)[1:]
    integrand = np.exp(np.asarray(log_integrand(jnp.asarray(grid))))
    linear = np.trapezoid(integrand, grid)
    if alpha > 2.0:
        assert abs(linear - reference) / reference > 0.1


def test_log_simpson_handles_a_support_edge():
    # ∫ from q_min of a Pareto with index 2.5 is 1
    def log_pareto(q):
        inside = q >= 1.0
        value = jnp.log(2.5) - 3.5 * jnp.log(jnp.where(inside, q, 1.0))
        return jnp.where(inside, value, -jnp.inf)

    value = log_simpson(log_pareto, 1.0, 1e8, n=4001)
    np.testing.assert_allclose(value, 0.0, atol=1e-8)


def test_pareto_smoothing_flags_heavy_tails():
    rng = np.random.default_rng(0)
    light = rng.normal(0.0, 0.3, 4000)
    smoothed, k_light = pareto_smoothed_log_weights(light)
    np.testing.assert_allclose(np.logaddexp.reduce(smoothed), 0.0, atol=1e-12)
    assert k_light < 0.5
    # weights that are themselves Pareto with shape 1 have k̂ ≈ 1
    heavy = np.log(rng.pareto(1.0, 4000) + 1.0)
    _, k_heavy = pareto_smoothed_log_weights(heavy)
    assert k_heavy > 0.7


def test_pareto_smoothing_of_equal_weights_needs_no_fit():
    smoothed, k_hat = pareto_smoothed_log_weights(np.zeros(64))
    np.testing.assert_allclose(np.exp(smoothed), 1.0 / 64)
    assert np.isnan(k_hat)


def test_pareto_smoothing_of_a_single_weight():
    smoothed, k_hat = pareto_smoothed_log_weights(np.array([0.3]))
    np.testing.assert_allclose(smoothed, [0.0])
    assert np.isnan(k_hat)
