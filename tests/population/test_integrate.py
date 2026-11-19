"""Log-space integration of a detection curve against a mark density, against Monte Carlo."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.population.integrate import (
    log_simpson,
)
from plumax.population.marks import LognormalParetoMark


pytest.importorskip("xtremax")


def _log_hill(q, q50=1.0, steepness=2.0):
    # log P_d = −log(1 + (Q₅₀/q)ⁿ)
    return -jnp.log1p((q50 / q) ** steepness)


@pytest.mark.validation
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
