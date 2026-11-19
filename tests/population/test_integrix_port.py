"""The deprecated integration aliases warn and forward to ``integrix.expect``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("xtremax")


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("log_gauss_hermite", lambda f: f(jnp.log, 0.2, 0.5)),
        ("log_simpson", lambda f: f(lambda q: -2.0 * jnp.log(q), 1.0, 1e4)),
        ("pareto_smoothed_log_weights", lambda f: f(np.linspace(0.0, 3.0, 500))[1]),
    ],
)
def test_integrate_aliases_warn_and_forward(name, call):
    import integrix.expect as ix_expect

    import plumax.population.integrate as deprecated

    with pytest.warns(DeprecationWarning, match=f"integrix.expect.{name}"):
        value = call(getattr(deprecated, name))
    expected = call(getattr(ix_expect, name))
    np.testing.assert_allclose(np.asarray(value), np.asarray(expected), rtol=0.0)


def test_effective_sample_size_alias_warns_and_forwards():
    import integrix.expect as ix_expect

    from plumax.population.importance import effective_sample_size

    log_weights = jnp.linspace(-1.0, 1.0, 64)
    with pytest.warns(
        DeprecationWarning, match="integrix.expect.effective_sample_size"
    ):
        value = effective_sample_size(log_weights)
    assert float(value) == float(ix_expect.effective_sample_size(log_weights))
