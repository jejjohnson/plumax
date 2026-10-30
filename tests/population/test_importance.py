"""The importance-corrected TMTPP mark likelihood (06a / 06b validation list).

Synthetic catalogs: true marks Qᵢ ~ LN(μ*, σ*²), one log-normal measurement
yᵢ = log Qᵢ + ε per event, and a per-event prior π = LN(0, s²), so each
per-event posterior is the conjugate lognormal and the importance correction
can be checked against the population likelihood it must reproduce.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from plumax._src.scalar_emission import LognormalEmission
from plumax.errors import PayloadError
from plumax.population.catalog import EmissionCatalog, EmissionEvent
from plumax.population.importance import (
    Regime,
    catalog_evidence,
    effective_sample_size,
    ess_report,
    event_evidence,
)
from plumax.population.marks import LognormalMark


pytest.importorskip("xtremax")

TRUTH = {"mu": 0.5, "sigma": 0.8}
MARK = LognormalMark()


def _catalog(prior_scale, *, noise=0.4, n_events=40, n_draws=4000, seed=0):
    rng = np.random.default_rng(seed)
    log_q = rng.normal(TRUTH["mu"], TRUTH["sigma"], n_events)
    measurements = log_q + rng.normal(0.0, noise, n_events)
    # conjugate posterior on log Q under the prior N(0, s²)
    precision = 1.0 / noise**2 + 1.0 / prior_scale**2
    post_mean = (measurements / noise**2) / precision
    post_std = 1.0 / math.sqrt(precision)
    prior = LognormalEmission(0.0, prior_scale)
    draw_rng = np.random.default_rng(seed + 1)
    events = []
    for center in post_mean:
        draws = np.exp(center + post_std * draw_rng.standard_normal(n_draws))
        mean = math.exp(center + 0.5 * post_std**2)
        std = mean * math.sqrt(math.expm1(post_std**2))
        events.append(
            EmissionEvent(
                emission_rate=mean,
                emission_std=std,
                x=0.0,
                y=0.0,
                time=0.0,
                tier="I",
                q_samples=jnp.asarray(draws),
                prior_logpdf=prior.log_prob,
            )
        )
    return EmissionCatalog.from_events(events)


MU_GRID = np.linspace(-0.3, 1.3, 33)
SIGMA_GRID = np.linspace(0.4, 1.4, 26)


def _grid_log_likelihood(catalog, regime):
    evidence = catalog_evidence(catalog, regime=regime)

    def total(mu, sigma):
        terms = evidence.log_mark_term(lambda q: MARK.log_prob(q, mu=mu, sigma=sigma))
        return jnp.sum(terms)

    surface = jax.vmap(jax.vmap(total, (None, 0)), (0, None))(
        jnp.asarray(MU_GRID), jnp.asarray(SIGMA_GRID)
    )
    return np.asarray(surface)


def _grid_posterior_mean(surface):
    # flat prior on the grid
    weights = np.exp(surface - surface.max())
    weights /= weights.sum()
    mu = float(np.sum(weights.sum(axis=1) * MU_GRID))
    sigma = float(np.sum(weights.sum(axis=0) * SIGMA_GRID))
    return mu, sigma


def test_prior_swap_invariance_of_the_likelihood():
    broad = _grid_log_likelihood(_catalog(1.5), Regime.FULL)
    narrow = _grid_log_likelihood(_catalog(0.5), Regime.FULL)
    # same data, different per-event priors: the corrected likelihoods differ
    # by a constant in θ, up to importance noise
    offset = broad - narrow
    # weigh the spread of the offset by the likelihood: far from the mode the
    # weights f/π fall in the tails of the draws and carry more IS noise
    weights = np.exp(broad - broad.max())
    weights /= weights.sum()
    centred = offset - np.sum(weights * offset)
    assert float(np.sqrt(np.sum(weights * centred**2))) < 0.05
    for broad_value, narrow_value in zip(
        _grid_posterior_mean(broad), _grid_posterior_mean(narrow), strict=True
    ):
        assert abs(broad_value - narrow_value) < 0.01


def test_uncorrected_likelihood_is_not_prior_invariant():
    # the same fit without dividing out π moves with the per-event prior
    def uncorrected(prior_scale):
        catalog = _catalog(prior_scale)
        draws = jnp.stack([event.q_samples for event in catalog.events])

        def total(mu, sigma):
            log_f = MARK.log_prob(draws, mu=mu, sigma=sigma)
            return jnp.sum(jax.scipy.special.logsumexp(log_f, axis=-1))

        surface = jax.vmap(jax.vmap(total, (None, 0)), (0, None))(
            jnp.asarray(MU_GRID), jnp.asarray(SIGMA_GRID)
        )
        return _grid_posterior_mean(np.asarray(surface))

    broad, narrow = uncorrected(1.5), uncorrected(0.5)
    assert abs(broad[0] - narrow[0]) > 0.05


def test_regimes_agree():
    catalog = _catalog(1.5, noise=0.05)
    means = {
        regime: _grid_posterior_mean(_grid_log_likelihood(catalog, regime))
        for regime in Regime
    }
    reference = means[Regime.FULL]
    for regime, (mu, sigma) in means.items():
        assert abs(mu - reference[0]) < 0.02, regime
        assert abs(sigma - reference[1]) < 0.02, regime


def test_gaussian_regime_matches_full_for_lognormal_posteriors():
    catalog = _catalog(1.5, n_events=5, n_draws=200_000)
    full = catalog_evidence(catalog, regime=Regime.FULL)
    gaussian = catalog_evidence(catalog, regime=Regime.GAUSSIAN)

    def integrand(q):
        return MARK.log_prob(q, mu=0.2, sigma=0.9)

    np.testing.assert_allclose(
        gaussian.log_mark_term(integrand), full.log_mark_term(integrand), atol=0.01
    )


def test_reduces_to_the_plain_average_when_the_prior_is_the_population():
    # π = f(· | θ₀): at θ₀ every weight f/π is one, so the corrected mark term
    # is the plain posterior average of P_d
    population = LognormalEmission(0.2, 0.9)
    draws = population.sample(jax.random.key(0), 1000)
    event = EmissionEvent(
        emission_rate=1.0,
        emission_std=1.0,
        x=0.0,
        y=0.0,
        time=0.0,
        tier="I",
        q_samples=draws,
        prior_logpdf=population.log_prob,
    )
    evidence = event_evidence(event, regime=Regime.FULL)

    def log_pod(q):
        return -jnp.log1p(1.0 / q**2)

    def integrand(q):
        return log_pod(q) + MARK.log_prob(q, mu=0.2, sigma=0.9)

    plain = jnp.log(jnp.mean(jnp.exp(log_pod(draws))))
    np.testing.assert_allclose(evidence.log_mark_term(integrand), plain, rtol=1e-10)
    # without a POD the term is log 1 = 0
    no_pod = evidence.log_mark_term(lambda q: MARK.log_prob(q, mu=0.2, sigma=0.9))
    np.testing.assert_allclose(no_pod, 0.0, atol=1e-10)


def test_point_regime_with_identity_pod_is_the_point_likelihood():
    catalog = _catalog(1.5, n_events=12)
    surface = _grid_log_likelihood(catalog, Regime.POINT)
    points = jnp.asarray(catalog.emission_rate)

    def point_likelihood(mu, sigma):
        return jnp.sum(MARK.log_prob(points, mu=mu, sigma=sigma))

    reference = jax.vmap(jax.vmap(point_likelihood, (None, 0)), (0, None))(
        jnp.asarray(MU_GRID), jnp.asarray(SIGMA_GRID)
    )
    offset = surface - np.asarray(reference)
    np.testing.assert_allclose(offset, offset.mean(), atol=1e-10)


def test_ess_limits():
    assert float(effective_sample_size(jnp.zeros(500))) == pytest.approx(500.0)
    one_dominant = jnp.full(500, -60.0).at[3].set(0.0)
    assert float(effective_sample_size(one_dominant)) == pytest.approx(1.0)
    two_equal = jnp.full(500, -60.0).at[jnp.array([3, 9])].set(0.0)
    assert float(effective_sample_size(two_equal)) == pytest.approx(2.0)


def test_ess_warns_on_collapse():
    # wide per-event posteriors against a narrow population f: the weights
    # f/π concentrate on the few draws near f's bulk
    catalog = _catalog(1.5, noise=1.5, n_events=5, n_draws=2000)

    def narrow(q):
        return MARK.log_prob(q, mu=3.0, sigma=0.05)

    with pytest.warns(RuntimeWarning, match="collapsed"):
        ess = ess_report(catalog, log_integrand=narrow)
    assert np.all(ess < 0.1 * 2000)
    # matched to the posteriors, no warning
    healthy = _catalog(1.5, noise=0.4, n_events=5, n_draws=2000)
    ess = ess_report(
        healthy,
        log_integrand=lambda q: MARK.log_prob(q, **TRUTH),
        warn=False,
    )
    assert np.all(ess > 0.1 * 2000)


def test_evidence_needs_the_payload():
    bare = EmissionEvent(
        emission_rate=1.0, emission_std=0.1, x=0.0, y=0.0, time=0.0, tier="II"
    )
    with pytest.raises(PayloadError, match="per-event prior"):
        event_evidence(bare, regime=Regime.POINT)
    no_draws = EmissionEvent(
        emission_rate=1.0,
        emission_std=0.1,
        x=0.0,
        y=0.0,
        time=0.0,
        tier="II",
        prior_logpdf=LognormalEmission(0.0, 1.0).log_prob,
    )
    with pytest.raises(PayloadError, match="posterior draws"):
        event_evidence(no_draws, regime=Regime.FULL)
    assert event_evidence(no_draws, regime="gaussian").q_samples.shape == (32,)


def test_mark_likelihood_gradient_matches_finite_differences():
    from numpyro.infer.util import log_density

    from plumax.population.fitting import mark_iw_model

    evidence = catalog_evidence(_catalog(1.5, n_events=8, n_draws=500))

    def log_pod(q):
        return -jnp.log1p(1.0 / q**2)

    def value(mu):
        params = {"mark.mu": mu, "mark.sigma": 0.8}
        density, _ = log_density(
            mark_iw_model, (evidence,), {"mark": MARK, "log_pod": log_pod}, params
        )
        return density

    step = 1e-5
    central = (value(0.4 + step) - value(0.4 - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(value)(0.4), central, rtol=1e-6)


def test_tmtpp_model_composes_ground_marks_and_thinned_rate():
    from plumax.population.fitting import log_thinned_rate, tmtpp_iw_model
    from plumax.population.processes import padded_event_buffer

    catalog = _catalog(1.5, n_events=4, n_draws=300)
    evidence = catalog_evidence(catalog)
    window = (0.0, 10.0)
    event_times, mask = padded_event_buffer([1.0, 2.5, 6.0, 9.0], window)

    def log_pod(q):
        return -jnp.log1p(1.0 / q**2)

    params = {"mark.mu": 0.4, "mark.sigma": 0.7, "lambda_0": 0.6}
    support = (1e-4, 1e4)
    model = handlers.substitute(tmtpp_iw_model, data=params)
    trace = handlers.trace(model).get_trace(
        event_times,
        mask,
        evidence,
        mark=MARK,
        intensity="constant",
        window=window,
        log_pod=log_pod,
        support=support,
    )
    factor = trace["tmtpp"]["fn"].log_factor
    theta = {"mu": 0.4, "sigma": 0.7}
    marks = evidence.log_mark_term(
        lambda q: log_pod(q) + MARK.log_prob(q, **theta)
    ).sum()
    thinned = jnp.exp(log_thinned_rate(MARK, theta, log_pod, support))
    # Σ log λ + Σ mᵢ − λ T p̄ for a constant λ = 0.6 over T = 10
    expected = 4 * math.log(0.6) + marks - 0.6 * 10.0 * thinned
    np.testing.assert_allclose(factor, expected, rtol=1e-10)
    # p̄ against an independent dense quadrature
    grid = np.linspace(math.log(1e-4), math.log(1e4), 200_001)
    q = np.exp(grid)
    dense = np.trapezoid(
        np.exp(np.asarray(log_pod(jnp.asarray(q)) + MARK.log_prob(q, **theta))) * q,
        grid,
    )
    np.testing.assert_allclose(thinned, dense, rtol=1e-6)


@pytest.mark.slow
def test_prior_swap_invariance():
    from plumax.population.fitting import fit_size_distribution_iw

    fits = [
        fit_size_distribution_iw(
            _catalog(prior_scale, n_draws=1000), mark="lognormal", seed=1
        )
        for prior_scale in (1.5, 0.5)
    ]
    for name in ("mu", "sigma"):
        broad, narrow = (fit.samples[name] for fit in fits)
        assert abs(broad.mean() - narrow.mean()) < 0.5 * broad.std(), name
        # and both cover the truth
        assert abs(broad.mean() - TRUTH[name]) < 3.0 * broad.std(), name


@pytest.mark.slow
def test_lognormal_size_distribution_delegates_with_a_regime():
    from plumax.population.size_distribution import fit_lognormal_size_distribution

    posterior = fit_lognormal_size_distribution(
        _catalog(1.5, n_draws=500), regime="full", num_warmup=300, num_samples=300
    )
    assert posterior.ess is not None
    assert posterior.ess.shape == (40,)
    assert abs(posterior.mu_mean - TRUTH["mu"]) < 3.0 * float(
        np.std(posterior.mu_samples)
    )


def test_thinned_rate_keeps_the_tail_beyond_the_grid():
    from plumax.population.fitting import log_thinned_rate
    from plumax.population.marks import PowerLawMark

    # α = 1.1: about 40 % of the mass lies above 1e4; an identity POD must
    # still give p̄ = 1
    def identity(q):
        return jnp.zeros_like(q)

    mark = PowerLawMark(q_min=1.0)
    rate = jnp.exp(log_thinned_rate(mark, {"alpha": 1.1}, identity, (1e-4, 1e4)))
    np.testing.assert_allclose(rate, 1.0, atol=1e-3)


def test_gaussian_fit_warns_on_collapse():
    from plumax.population.fitting import fit_size_distribution_iw

    # a population f pinned near Q = 1 by its priors against wide posteriors
    catalog = _catalog(1.5, noise=1.5, n_events=4, n_draws=50)
    with pytest.warns(RuntimeWarning, match="collapsed"):
        fit_size_distribution_iw(
            catalog,
            mark="lognormal",
            mark_fields={"mu_prior_scale": 0.01, "sigma_prior_scale": 0.01},
            regime="gaussian",
            num_warmup=20,
            num_samples=20,
        )
