"""The mark registry: normalisation, moments and heavy tails of each family."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from plumax.population.marks import (
    MARKS,
    LognormalMark,
    LognormalParetoMark,
    Mark,
    MixtureLognormalMark,
    PowerLawMark,
    make_mark,
)


CASES = {
    "powerlaw": (PowerLawMark(q_min=0.5), {"alpha": 6.5}),
    "lognormal": (LognormalMark(), {"mu": 0.3, "sigma": 0.8}),
    "lognormal_pareto": (
        LognormalParetoMark(q_break_loc=2.0),
        {"mu": 0.0, "sigma": 0.7, "alpha": 4.5, "q_break": 2.0},
    ),
    "mixture_lognormal": (
        MixtureLognormalMark(n_components=2),
        {
            "weights": jnp.array([0.7, 0.3]),
            "locs": jnp.array([-0.5, 1.5]),
            "scales": jnp.array([0.4, 0.6]),
        },
    ),
}


def _log_grid(lower, upper, size=400_001):
    return np.linspace(np.log(lower), np.log(upper), size)


def _integrate(mark, theta, power, lower=1e-9, upper=1e9):
    # ∫ qᵖ f(q) dq = ∫ e^{(p+1)u} f(eᵘ) du on a fine grid in u = log q
    support = getattr(mark, "q_min", lower)
    log_q = _log_grid(max(lower, support), upper)
    density = np.exp(np.asarray(mark.log_prob(jnp.exp(log_q), **theta)))
    return np.trapezoid(density * np.exp((power + 1.0) * log_q), log_q)


def test_registry_lists_every_family():
    assert set(MARKS) == set(CASES)
    for name, (mark, _) in CASES.items():
        assert isinstance(mark, Mark)
        assert isinstance(mark, MARKS[name])
    assert isinstance(make_mark("powerlaw", q_min=1.0), PowerLawMark)
    with pytest.raises(KeyError, match="unknown mark"):
        make_mark("gamma")


@pytest.mark.parametrize("name", sorted(CASES))
def test_normalised_all_families(name):
    mark, theta = CASES[name]
    np.testing.assert_allclose(_integrate(mark, theta, 0.0), 1.0, rtol=1e-6)


@pytest.mark.parametrize("name", sorted(CASES))
def test_truncated_mean_matches_quadrature(name):
    mark, theta = CASES[name]
    for lower, upper in ((0.0, np.inf), (0.8, 3.0), (2.5, 40.0)):
        expected = _integrate(mark, theta, 1.0, max(lower, 1e-9), min(upper, 1e9))
        value = float(mark.truncated_mean(lower, upper, **theta))
        np.testing.assert_allclose(value, expected, rtol=1e-5)


@pytest.mark.parametrize("name", sorted(CASES))
def test_sample_moments_match_closed_form(name):
    mark, theta = CASES[name]
    draws = np.asarray(mark.sample(jax.random.key(0), 400_000, **theta))
    mean = float(mark.truncated_mean(0.0, np.inf, **theta))
    second = _integrate(mark, theta, 2.0)
    # every case has finite moments up to the fourth (α > 4 for the power
    # laws), so the sample mean and second moment converge at rate 1/√S
    std = np.sqrt(second - mean**2)
    assert abs(draws.mean() - mean) < 5.0 * std / np.sqrt(draws.size)
    np.testing.assert_allclose(np.mean(draws**2), second, rtol=0.05)


def test_lognormal_closed_form_moments():
    mark = LognormalMark()
    mean = mark.truncated_mean(0.0, np.inf, mu=0.3, sigma=0.8)
    np.testing.assert_allclose(mean, np.exp(0.3 + 0.5 * 0.8**2))


def test_powerlaw_closed_form_mean():
    # E[Q] = q_min (α − 1)/(α − 2) for α > 2
    mean = PowerLawMark(q_min=0.5).truncated_mean(0.0, np.inf, alpha=6.5)
    np.testing.assert_allclose(mean, 0.5 * 5.5 / 4.5)


def test_lognormal_pareto_is_continuous_with_a_pareto_tail():
    mark, theta = CASES["lognormal_pareto"]
    q_break, alpha = theta["q_break"], theta["alpha"]
    below = mark.log_prob(q_break * (1.0 - 1e-10), **theta)
    above = mark.log_prob(q_break, **theta)
    np.testing.assert_allclose(below, above, rtol=1e-8)
    # above the break the survival is (1 − w)(Q_b/q)^α
    draws = np.asarray(mark.sample(jax.random.key(1), 400_000, **theta))
    tail_weight = np.mean(draws >= q_break)
    for level in (3.0, 4.0):
        expected = tail_weight * (q_break / level) ** alpha
        np.testing.assert_allclose(np.mean(draws >= level), expected, rtol=0.05)
    # the log-density falls with slope −(α + 1) in log q
    slope = (mark.log_prob(20.0, **theta) - mark.log_prob(10.0, **theta)) / np.log(2.0)
    np.testing.assert_allclose(slope, -(alpha + 1.0))


def test_mixture_of_one_component_is_the_lognormal():
    mixture = MixtureLognormalMark(n_components=1)
    q = jnp.array([0.2, 1.0, 5.0])
    np.testing.assert_allclose(
        mixture.log_prob(
            q, weights=jnp.ones(1), locs=jnp.array([0.3]), scales=jnp.array([0.8])
        ),
        LognormalMark().log_prob(q, mu=0.3, sigma=0.8),
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_numpyro_prior_gives_the_keyword_parameters(name):
    mark, theta = CASES[name]
    drawn = numpyro.handlers.seed(mark.numpyro_prior, rng_seed=0)("mark")
    assert set(drawn) == set(theta)
    for key, value in theta.items():
        assert jnp.shape(drawn[key]) == jnp.shape(value)
    assert bool(jnp.isfinite(mark.log_prob(1.0, **drawn)))


@pytest.mark.slow
def test_lognormal_pareto_recovery():
    from numpyro.infer import MCMC, NUTS

    mark = LognormalParetoMark(q_break_loc=2.0)
    truth = {"mu": 0.0, "sigma": 0.7, "alpha": 2.5, "q_break": 2.0}
    data = mark.sample(jax.random.key(2), 2000, **truth)

    def model(q):
        theta = mark.numpyro_prior("mark")
        numpyro.factor("marks", jnp.sum(mark.log_prob(q, **theta)))

    mcmc = MCMC(NUTS(model), num_warmup=400, num_samples=400, progress_bar=False)
    mcmc.run(jax.random.key(3), data)
    samples = mcmc.get_samples()
    for name in ("alpha", "q_break"):
        draws = np.asarray(samples[f"mark.{name}"])
        assert abs(draws.mean() - truth[name]) < 2.0 * draws.std(), name


def test_powerlaw_mean_diverges_or_is_logarithmic_for_small_alpha():
    mark = PowerLawMark(q_min=1.0)
    assert float(mark.truncated_mean(0.0, np.inf, alpha=1.5)) == np.inf
    assert float(mark.truncated_mean(0.0, np.inf, alpha=2.0)) == np.inf
    # α = 2: ∫₁² q · q⁻² dq = log 2
    np.testing.assert_allclose(mark.truncated_mean(1.0, 2.0, alpha=2.0), np.log(2.0))
    # α = 1.5 on a finite interval: ½ ∫₁⁴ q^(−1/2) dq = 2
    np.testing.assert_allclose(mark.truncated_mean(1.0, 4.0, alpha=1.5), 1.0)


def test_lognormal_pareto_mean_with_a_break_far_below_the_body():
    # log F_LN(Q_b) ≈ −930: the body's renormalisation stays in log space
    mark = LognormalParetoMark(q_break_loc=2.0)
    theta = {"mu": 5.0, "sigma": 0.1, "alpha": 2.5, "q_break": 2.0}
    mean = float(mark.truncated_mean(0.0, np.inf, **theta))
    assert np.isfinite(mean)
    np.testing.assert_allclose(mean, _integrate(mark, theta, 1.0), rtol=1e-5)
    # continuity puts almost all the mass on the Pareto, mean α Q_b/(α − 1)
    np.testing.assert_allclose(mean, 2.5 * 2.0 / 1.5, rtol=1e-2)


def test_lognormal_pareto_samples_a_body_far_below_its_centre():
    # Φ(z_b) underflows float64 here, yet the body keeps ~0.6 % of the mass
    from plumax.population._src.marks import _splice_log_weights

    mark = LognormalParetoMark(q_break_loc=2.0)
    theta = {"mu": 5.0, "sigma": 0.1, "alpha": 2.5, "q_break": 2.0}
    draws = np.asarray(mark.sample(jax.random.key(4), 400_000, **theta))
    assert draws.min() > 0.0
    params = (jnp.asarray(v) for v in (5.0, 0.1, 2.5, 2.0))
    body_weight = float(jnp.exp(_splice_log_weights(*params)[0]))
    below = draws < 2.0
    np.testing.assert_allclose(below.mean(), body_weight, rtol=0.05)
    assert np.all(np.isfinite(np.asarray(mark.log_prob(jnp.asarray(draws), **theta))))


@pytest.mark.parametrize("name", ["lognormal", "mixture_lognormal"])
def test_full_mean_gradient_is_finite_with_an_open_upper_bound(name):
    mark, theta = CASES[name]
    scale_key = "sigma" if name == "lognormal" else "scales"

    def mean(scale):
        return mark.truncated_mean(0.0, jnp.inf, **{**theta, scale_key: scale})

    gradient = jax.grad(lambda s: jnp.sum(mean(s)))(jnp.asarray(theta[scale_key]))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    if name == "lognormal":
        # d/dσ exp(μ + σ²/2) = σ exp(μ + σ²/2)
        expected = 0.8 * np.exp(0.3 + 0.5 * 0.8**2)
        np.testing.assert_allclose(gradient, expected, rtol=1e-10)


def _central_difference(fn, x, step=1e-6):
    return (fn(x + step) - fn(x - step)) / (2.0 * step)


def test_moment_gradients_in_the_tails_and_at_the_splice():
    lognormal = LognormalMark()

    def far_tail(mu):
        return lognormal.truncated_mean(np.exp(12.0), np.exp(13.0), mu=mu, sigma=1.0)

    gradient = jax.grad(far_tail)(0.0)
    assert bool(jnp.isfinite(gradient))
    np.testing.assert_allclose(gradient, _central_difference(far_tail, 0.0), rtol=1e-5)

    mark = LognormalParetoMark(q_break_loc=2.0)

    def full_mean(q_break):
        return mark.truncated_mean(
            0.0, jnp.inf, mu=0.0, sigma=1.0, alpha=2.5, q_break=q_break
        )

    gradient = jax.grad(full_mean)(2.0)
    assert bool(jnp.isfinite(gradient))
    np.testing.assert_allclose(gradient, _central_difference(full_mean, 2.0), rtol=1e-5)

    power_law = PowerLawMark(q_min=1.0)
    gradient = jax.grad(lambda a: power_law.truncated_mean(1.0, 4.0, alpha=a))(2.0)
    assert bool(jnp.isfinite(gradient))
