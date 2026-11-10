"""Port test for plumax#253: the Tier V numbers do not move when integrix takes over.

The values were pinned with plumax's own log-space Simpson, Gauss–Hermite, ESS and
Pareto smoothing (plumax ``feb5e86`` plus #257/#258), before the call sites switched
to ``integrix.expect`` (v0.0.5). Float64 throughout (``tests/conftest.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.population.fitting import log_thinned_rate
from plumax.population.importance import catalog_evidence, ess_report
from plumax.population.marks import LognormalMark
from plumax.population.totals import total_emission
from tests.population.test_importance import MARK, _catalog
from tests.population.test_totals import Q_RANGE, _logistic, _posterior


pytest.importorskip("xtremax")

RTOL = 1e-10


def _narrow(q: jax.Array) -> jax.Array:
    # log g(q) = log LN(q; 0.5, 0.8²), the size distribution the catalog is drawn from
    return MARK.log_prob(q, mu=0.5, sigma=0.8)


def test_total_emission_missing_mass_is_unchanged():
    posterior = _posterior([1.5, 2.0], [2.0, 1.0], [1.5, 0.7])
    total = total_emission(
        posterior,
        horizon=(0.0, 4.0),
        event_duration=3.0,
        log_pod=_logistic(100.0, 0.02),
        **Q_RANGE,
    )
    np.testing.assert_allclose(
        total.missing_kg, [208.88105624855712, 72.349662458571], rtol=RTOL
    )


def test_log_thinned_rate_is_unchanged():
    # log p̄ = log ∫ P_d(q) LN(q; 2, 1.5²) dq, logistic P_d with q₅₀ = 100 kg/s
    value = log_thinned_rate(
        LognormalMark(),
        {"mu": 2.0, "sigma": 1.5},
        lambda q: jax.nn.log_sigmoid(0.02 * (q - 100.0)),
        (1e-4, 1e6),
        n_nodes=2001,
    )
    np.testing.assert_allclose(float(value), -1.6972490428452154, rtol=RTOL)


def test_ess_report_is_unchanged():
    ess = np.asarray(
        ess_report(_catalog(0.5, n_draws=1000), log_integrand=_narrow, warn=False)
    )
    expected_head = [
        736.3077585074004,
        350.058131127812,
        206.2706053201661,
        226.78928620841708,
        618.6555619411671,
        534.2747914841764,
    ]
    np.testing.assert_allclose(ess[:6], expected_head, rtol=RTOL)
    np.testing.assert_allclose(ess.sum(), 21054.1288810667, rtol=RTOL)


def test_pareto_k_is_unchanged():
    import integrix.expect as ix_expect

    log_weights = catalog_evidence(_catalog(0.5, n_draws=1000)).log_terms(_narrow)
    # one k̂ per event, batched over events
    _, k_hat = ix_expect.pareto_smoothed_log_weights(jnp.asarray(log_weights)[:6])
    expected = [
        0.3608758825526091,
        0.4119918872476346,
        0.5473656902241397,
        0.7052620589795802,
        0.47039608595862265,
        0.40333669560490704,
    ]
    np.testing.assert_allclose(np.asarray(k_hat), expected, rtol=RTOL)


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
