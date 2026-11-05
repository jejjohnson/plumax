"""Tests for the multi-satellite POD union and its calibration prior."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import lognorm

from plumax.population.importance import EventEvidence
from plumax.population.marks import LognormalMark
from plumax.population.pod import (
    TOD_BINS,
    InstrumentPOD,
    LogisticPOD,
    LogLogisticPOD,
    calibration_from_draw,
    calibration_prior,
    load_pod_calibrations,
    log_pod_union,
    pod_union,
    tod_bin_index,
)
from plumax.population.processes import (
    expected_events_by_tod_bin,
    padded_event_buffer,
)
from plumax.population.totals import total_emission


pytest.importorskip("numpyro")
pytest.importorskip("xtremax")

from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

from plumax.population.fitting import TMTPPPosterior, tmtpp_iw_model


SATELLITE_A = InstrumentPOD("a", "log_logistic", 3.0, 0.3, 2.0, 0.2)
SATELLITE_B = InstrumentPOD("b", "log_logistic", 10.0, 0.4, 1.5, 0.2)
PODS = [SATELLITE_A, SATELLITE_B]

# --- the coverage twin's design, fixed before its first run ---
# Truths are drawn from the fit's own prior, so 95 % coverage is exact up to MCMC error
# and any failure is a fault in the likelihood or the sampler.
TWIN_N_BASINS = 100
# Three binomial standard deviations about 0.95 at TWIN_N_BASINS.
TWIN_COVERAGE_BAND = 3.0 * math.sqrt(0.95 * 0.05 / TWIN_N_BASINS)
# Each instrument gets days of its own, which is what informs both: a union seen only
# as a whole barely tells a second instrument's threshold from the first's.
TWIN_WINDOW = (0.0, 200.0)
TWIN_INTERVALS = (
    (0.0, 80.0, jnp.array([True, False])),
    (80.0, 160.0, jnp.array([False, True])),
    (160.0, 200.0, jnp.array([True, True])),
)
# Far above the prior predictive counts: λ₀ ~ HalfNormal(3) over 200 days
TWIN_MAX_GROUND = 3000
TWIN_MAX_DETECTED = 2000
# Thresholds inside the mark prior's bulk, and steep: information on log Q₅₀ per event
# grows as k².
TWIN_PODS = [
    InstrumentPOD("a", "log_logistic", 1.0, 0.5, 4.0, 0.2),
    InstrumentPOD("b", "log_logistic", 1.5, 0.5, 4.0, 0.2),
]
# "Informed" means the catalog says at least as much about log Q₅₀ as the prior does:
# posterior precision ≥ 2 × prior precision, so spread ≤ 1/√2 of the prior's, for each
# instrument. test_coverage_twin_is_informed_by_design computes the expectation.
TWIN_MAX_POSTERIOR_SPREAD = 1.0 / math.sqrt(2.0)
TWIN_MARK = LognormalMark(mu_prior_scale=1.0, sigma_prior_scale=1.0)
TWIN_SUPPORT = (1e-6, 1e6)


def _hill(q, q50, slope):
    return expit(slope * np.log(q / q50))


@pytest.mark.parametrize(
    ("model", "curve"),
    [
        ("log_logistic", LogLogisticPOD(beta_0=-2.0 * math.log(3.0), beta_1=2.0)),
        ("logistic", LogisticPOD(Q_50=3.0, k=2.0)),
    ],
)
def test_single_instrument_identity(model, curve):
    pod = InstrumentPOD("only", model, 3.0, 0.3, 2.0)
    q = jnp.logspace(-2.0, 2.0, 41)
    np.testing.assert_allclose(pod_union([pod], q), curve(q), rtol=1e-12)
    np.testing.assert_allclose(
        log_pod_union([pod])(q), jnp.log(curve(q)), rtol=1e-10, atol=1e-13
    )


def test_hill_form_of_the_log_logistic_curve():
    # 06b's P_d = 1 / (1 + (Q₅₀ / Q)^k) is the registry's logistic in log Q
    q = np.logspace(-2.0, 2.0, 41)
    hill = 1.0 / (1.0 + (3.0 / q) ** 2.0)
    pod = InstrumentPOD("only", "log_logistic", 3.0, 0.3, 2.0)
    np.testing.assert_allclose(pod_union([pod], jnp.asarray(q)), hill, rtol=1e-12)


def test_union_is_the_independent_detection_closed_form():
    q = np.logspace(-1.0, 2.0, 31)
    expected = 1.0 - (1.0 - _hill(q, 3.0, 2.0)) * (1.0 - _hill(q, 10.0, 1.5))
    np.testing.assert_allclose(pod_union(PODS, jnp.asarray(q)), expected, rtol=1e-12)
    np.testing.assert_allclose(
        jnp.exp(log_pod_union(PODS)(jnp.asarray(q))), expected, rtol=1e-12
    )


def test_union_never_falls_below_a_member():
    q = jnp.logspace(-3.0, 3.0, 61)
    union = pod_union(PODS, q)
    assert jnp.all(union >= pod_union([SATELLITE_A], q))
    assert jnp.all(union >= pod_union([SATELLITE_B], q))


def test_coverage_masks_instruments_per_event():
    q = jnp.array([5.0, 5.0, 5.0, 5.0])
    coverage = jnp.array([[True, True], [True, False], [False, True], [False, False]])
    union = pod_union(PODS, q, coverage=coverage)
    np.testing.assert_allclose(union[0], pod_union(PODS, q[0]), rtol=1e-12)
    np.testing.assert_allclose(union[1], pod_union([SATELLITE_A], q[1]), rtol=1e-12)
    np.testing.assert_allclose(union[2], pod_union([SATELLITE_B], q[2]), rtol=1e-12)
    assert union[3] == 0.0


@pytest.mark.slow
def test_log_union_tail_is_accurate_with_a_finite_gradient():
    # far below every Q₅₀ a direct 1 − ∏(1 − P_d) rounds to zero
    q = jnp.array(1e-60)
    log_pod = log_pod_union(PODS)
    expected = np.logaddexp(
        2.0 * (math.log(1e-60) - math.log(3.0)),
        1.5 * (math.log(1e-60) - math.log(10.0)),
    )
    np.testing.assert_allclose(log_pod(q), expected, rtol=1e-12)
    assert jnp.isfinite(jax.grad(log_pod)(q))
    # and far above every Q₅₀
    assert jnp.isfinite(jax.grad(log_pod)(jnp.array(1e6)))


def test_log_union_gradient_matches_central_difference():
    def value(q50):
        calibration = {"a": {"q50": q50}, "b": {}}
        return log_pod_union(PODS, calibration=calibration)(jnp.array(5.0))

    step = 1e-5
    finite_difference = (value(3.0 + step) - value(3.0 - step)) / (2.0 * step)
    np.testing.assert_allclose(jax.grad(value)(3.0), finite_difference, rtol=1e-8)


def test_calibration_prior_has_the_published_lognormals():
    with handlers.seed(rng_seed=0), handlers.trace() as trace:
        draw = calibration_prior(PODS)
    site = trace["pod.b.q50"]["fn"]
    np.testing.assert_allclose(site.loc, math.log(10.0), rtol=1e-12)
    np.testing.assert_allclose(site.scale, 0.4, rtol=1e-12)
    assert trace["pod.a.slope"]["fn"].scale == 0.2
    assert set(draw) == {"a", "b"}


def test_zero_spread_stays_at_the_published_value():
    fixed = InstrumentPOD("fixed", "log_logistic", 3.0, 0.0, 2.0)
    with handlers.seed(rng_seed=0), handlers.trace() as trace:
        draw = calibration_prior([fixed])
    assert not trace
    assert float(draw["fixed"]["q50"]) == 3.0


def test_invalid_instruments_are_rejected():
    with pytest.raises(ValueError, match="registry curves in Q alone"):
        InstrumentPOD("x", "probit", 3.0, 0.3, 2.0)
    with pytest.raises(ValueError, match="must be > 0"):
        InstrumentPOD("x", "logistic", -3.0, 0.3, 2.0)
    with pytest.raises(ValueError, match="must be ≥ 0"):
        InstrumentPOD("x", "logistic", 3.0, -0.3, 2.0)
    with pytest.raises(ValueError, match="no instruments"):
        pod_union([], jnp.ones(2))
    with pytest.raises(ValueError, match="one column per instrument"):
        pod_union(PODS, jnp.ones(2), coverage=jnp.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="names repeat"):
        calibration_prior([SATELLITE_A, SATELLITE_A])


def test_load_calibrations_from_csv_and_yaml(tmp_path):
    table = tmp_path / "pods.csv"
    table.write_text(
        "name,model,q50,q50_std,slope,slope_std\n"
        "a,log_logistic,3.0,0.3,2.0,0.2\n"
        "b,log_logistic,10.0,0.4,1.5,0.2\n"
    )
    assert load_pod_calibrations(table) == PODS
    listing = tmp_path / "pods.yaml"
    listing.write_text(
        "- {name: a, model: log_logistic, q50: 3.0, q50_std: 0.3, slope: 2.0, "
        "slope_std: 0.2}\n"
        "- {name: b, model: log_logistic, q50: 10.0, q50_std: 0.4, slope: 1.5, "
        "slope_std: 0.2}\n"
    )
    assert load_pod_calibrations(listing) == PODS


def test_load_calibrations_rejects_bad_files(tmp_path):
    missing = tmp_path / "pods.csv"
    missing.write_text("name,model,q50\na,logistic,3.0\n")
    with pytest.raises(ValueError, match="no column"):
        load_pod_calibrations(missing)
    with pytest.raises(ValueError, match=r"expected \.csv"):
        load_pod_calibrations(tmp_path / "pods.json")


def _model_log_density(params, **arguments):
    density, _ = log_density(
        handlers.substitute(tmtpp_iw_model, data=params), (), arguments, {}
    )
    return float(density)


def _no_prior_params(mark_mu=0.5, mark_sigma=0.8, lambda_0=2.0):
    return {
        "lambda_0": lambda_0,
        "mark.mu": mark_mu,
        "mark.sigma": mark_sigma,
    }


def _prior_density(params) -> float:
    def prior():
        LognormalMark().numpyro_prior("mark")
        from plumax.population.intensity import INTENSITY_REGISTRY

        INTENSITY_REGISTRY["constant"].sample_priors()

    density, _ = log_density(handlers.substitute(prior, data=params), (), {}, {})
    return float(density)


def _thinned_rate(probability, mu=0.5, sigma=0.8):
    law = lognorm(s=sigma, scale=math.exp(mu))

    def integrand(log_q):
        q = math.exp(log_q)
        return probability(q) * law.pdf(q) * q

    return quad(integrand, -25.0, 25.0, limit=400, epsabs=1e-11)[0]


@pytest.mark.slow
def test_fixed_union_in_the_likelihood_is_the_closed_form():
    fixed = [
        InstrumentPOD("a", "log_logistic", 3.0, 0.0, 2.0),
        InstrumentPOD("b", "log_logistic", 10.0, 0.0, 1.5),
    ]
    window = (0.0, 10.0)
    marks = jnp.array([2.0, 7.0, 20.0])
    event_times, mask = padded_event_buffer([1.0, 4.0, 8.0], window)
    evidence = EventEvidence(log_weights=jnp.zeros((3, 1)), q_samples=marks[:, None])
    params = _no_prior_params()
    density = _model_log_density(
        params,
        event_times=event_times,
        mask=mask,
        evidence=evidence,
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=fixed,
        support=(1e-8, 1e8),
    )

    def probability(q):
        return 1.0 - (1.0 - _hill(q, 3.0, 2.0)) * (1.0 - _hill(q, 10.0, 1.5))

    law = lognorm(s=0.8, scale=math.exp(0.5))
    expected = sum(
        math.log(2.0) + law.logpdf(q) + math.log(probability(q))
        for q in np.asarray(marks)
    )
    expected -= 2.0 * 10.0 * _thinned_rate(probability)
    np.testing.assert_allclose(density - _prior_density(params), expected, rtol=1e-6)


@pytest.mark.slow
def test_coverage_follows_who_was_looking_when():
    # a looks for 6 days, b for 2 days, then 2 days with nobody
    fixed = [
        InstrumentPOD("a", "log_logistic", 3.0, 0.0, 2.0),
        InstrumentPOD("b", "log_logistic", 10.0, 0.0, 1.5),
    ]
    window = (0.0, 10.0)
    marks = jnp.array([2.0, 20.0])
    event_times, mask = padded_event_buffer([1.0, 7.0], window)
    evidence = EventEvidence(log_weights=jnp.zeros((2, 1)), q_samples=marks[:, None])
    params = _no_prior_params()
    density = _model_log_density(
        params,
        event_times=event_times,
        mask=mask,
        evidence=evidence,
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=fixed,
        coverage=jnp.array([[True, False], [False, True]]),
        coverage_intervals=[
            (0.0, 6.0, jnp.array([True, False])),
            (6.0, 8.0, jnp.array([False, True])),
            (8.0, 10.0, jnp.array([False, False])),
        ],
        support=(1e-8, 1e8),
    )
    law = lognorm(s=0.8, scale=math.exp(0.5))
    expected = (
        2.0 * math.log(2.0)
        + law.logpdf(2.0)
        + math.log(_hill(2.0, 3.0, 2.0))
        + law.logpdf(20.0)
        + math.log(_hill(20.0, 10.0, 1.5))
    )
    expected -= 2.0 * 6.0 * _thinned_rate(lambda q: _hill(q, 3.0, 2.0))
    expected -= 2.0 * 2.0 * _thinned_rate(lambda q: _hill(q, 10.0, 1.5))
    np.testing.assert_allclose(density - _prior_density(params), expected, rtol=1e-6)


@pytest.mark.slow
def test_coverage_intervals_run_under_nuts():
    # the compiled potential, where an interval's emptiness must be known on the host
    window = (0.0, 10.0)
    marks = jnp.array([2.0, 20.0, 5.0])
    event_times, mask = padded_event_buffer([1.0, 7.0, 2.0], window)
    mcmc = MCMC(NUTS(tmtpp_iw_model), num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(0),
        event_times,
        mask,
        EventEvidence(jnp.zeros((3, 1)), marks[:, None]),
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=PODS,
        coverage=jnp.array([[True, False], [False, True], [True, False]]),
        coverage_intervals=[
            (0.0, 6.0, jnp.array([True, False])),
            (6.0, 8.0, jnp.array([False, True])),
            (8.0, 10.0, jnp.array([False, False])),
        ],
        support=(1e-6, 1e6),
        n_nodes=201,
    )
    assert jnp.all(jnp.isfinite(mcmc.get_samples()["pod.a.q50"]))


def _tiny_model_arguments():
    window = (0.0, 10.0)
    event_times, mask = padded_event_buffer([1.0], window)
    evidence = EventEvidence(
        log_weights=jnp.zeros((1, 1)), q_samples=jnp.array([[5.0]])
    )
    return {
        "event_times": event_times,
        "mask": mask,
        "evidence": evidence,
        "mark": LognormalMark(),
        "intensity": "constant",
        "window": window,
        "support": (1e-6, 1e6),
    }


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"pods": PODS, "log_pod": jnp.log}, "not both"),
        ({"pods": PODS, "coverage": jnp.ones((1, 2), dtype=bool)}, "come together"),
        (
            {
                "pods": PODS,
                "coverage": jnp.ones((1, 2), dtype=bool),
                "coverage_intervals": [
                    (0.0, 6.0, jnp.ones(2, dtype=bool)),
                    (5.0, 8.0, jnp.ones(2, dtype=bool)),
                ],
            },
            "disjoint",
        ),
        (
            {
                "pods": PODS,
                "coverage": jnp.ones((1, 2), dtype=bool),
                "coverage_intervals": [(0.0, 11.0, jnp.ones(2, dtype=bool))],
            },
            "after the window",
        ),
    ],
)
def test_invalid_coverage_arguments_are_rejected(extra, message):
    with pytest.raises(ValueError, match=message), handlers.seed(rng_seed=0):
        tmtpp_iw_model(**_tiny_model_arguments(), **extra)


@pytest.mark.slow
def test_padded_evidence_matches_the_unpadded_catalog():
    window = (0.0, 10.0)
    marks = jnp.array([2.0, 7.0])
    params = _no_prior_params()
    arguments = {
        "mark": LognormalMark(),
        "intensity": "constant",
        "window": window,
        "pods": [
            InstrumentPOD("a", "log_logistic", 3.0, 0.0, 2.0),
            InstrumentPOD("b", "log_logistic", 10.0, 0.0, 1.5),
        ],
        "support": (1e-6, 1e6),
    }
    times, mask = padded_event_buffer([1.0, 4.0], window)
    plain = _model_log_density(
        params,
        event_times=times,
        mask=mask,
        evidence=EventEvidence(jnp.zeros((2, 1)), marks[:, None]),
        **arguments,
    )
    times, mask = padded_event_buffer([1.0, 4.0], window, max_events=5)
    padded_marks = jnp.array([2.0, 7.0, 1.0, 1.0, 1.0])
    padded = _model_log_density(
        params,
        event_times=times,
        mask=mask,
        evidence=EventEvidence(jnp.zeros((5, 1)), padded_marks[:, None]),
        **arguments,
    )
    np.testing.assert_allclose(padded, plain, rtol=1e-12)


def _union_posterior(q50_a, q50_b) -> TMTPPPosterior:
    samples = {
        "lambda_0": jnp.array([1.0, 2.0]),
        "mark.mu": jnp.array([0.0, 0.5]),
        "mark.sigma": jnp.array([1.0, 0.8]),
        "pod.a.q50": jnp.asarray(q50_a),
        "pod.b.q50": jnp.asarray(q50_b),
    }
    return TMTPPPosterior("constant", LognormalMark(), samples)


def _missing_quadrature(lambda_0, mu, sigma, q50_a, q50_b, window, duration):
    law = lognorm(s=sigma, scale=math.exp(mu))

    def integrand(log_q):
        q = math.exp(log_q)
        missed = (1.0 - _hill(q, q50_a, 2.0)) * (1.0 - _hill(q, q50_b, 1.5))
        return q * missed * law.pdf(q) * q

    moment = quad(integrand, math.log(1e-4), math.log(1e4), limit=400)[0]
    return lambda_0 * (window[1] - window[0]) * moment * duration


@pytest.mark.slow
def test_total_emission_composes_with_a_fixed_union():
    window = (0.0, 10.0)
    posterior = _union_posterior([3.0, 3.0], [10.0, 10.0])
    total = total_emission(
        posterior,
        horizon=window,
        event_duration=2.0,
        log_pod=log_pod_union(PODS),
        q_min=1e-4,
        q_max=1e4,
    )
    for index, (rate, mu, sigma) in enumerate([(1.0, 0.0, 1.0), (2.0, 0.5, 0.8)]):
        expected = _missing_quadrature(rate, mu, sigma, 3.0, 10.0, window, 2.0)
        np.testing.assert_allclose(total.missing_kg[index], expected, rtol=1e-6)


@pytest.mark.slow
def test_total_emission_uses_each_draws_calibration():
    window = (0.0, 10.0)
    posterior = _union_posterior([3.0, 6.0], [10.0, 30.0])
    total = total_emission(
        posterior,
        horizon=window,
        event_duration=2.0,
        log_pod=None,
        pods=PODS,
        q_min=1e-4,
        q_max=1e4,
    )
    draws = [(1.0, 0.0, 1.0, 3.0, 10.0), (2.0, 0.5, 0.8, 6.0, 30.0)]
    for index, draw in enumerate(draws):
        expected = _missing_quadrature(*draw, window, 2.0)
        np.testing.assert_allclose(total.missing_kg[index], expected, rtol=1e-6)


def test_total_emission_rejects_a_pod_and_instruments_together():
    with pytest.raises(ValueError, match="not both"):
        total_emission(
            _union_posterior([3.0, 3.0], [10.0, 10.0]),
            horizon=(0.0, 1.0),
            event_duration=1.0,
            log_pod=log_pod_union(PODS),
            pods=PODS,
            q_min=1e-4,
            q_max=1e4,
        )


def test_calibration_from_draw_falls_back_to_published_values():
    draw = {"pod.a.q50": jnp.array(4.0)}
    calibration = calibration_from_draw(PODS, draw)
    assert float(calibration["a"]["q50"]) == 4.0
    assert float(calibration["b"]["q50"]) == 10.0
    assert float(calibration["a"]["slope"]) == 2.0


TOD_SHIFTS = tuple(0.5 * math.sin(2.0 * math.pi * hour / 24.0) for hour in range(24))


def _binned(shifts=TOD_SHIFTS, model="log_logistic"):
    return InstrumentPOD("binned", model, 3.0, 0.0, 2.0, tod_bins=shifts)


def test_tod_bin_index_counts_hours_from_the_origin():
    times = jnp.array([0.0, 1.0 / 24.0, 0.5, 1.25, 2.999])
    assert tod_bin_index(times).tolist() == [0, 1, 12, 6, 23]
    assert tod_bin_index(jnp.array([0.0, 0.25]), tod_origin=20.0).tolist() == [20, 2]


@pytest.mark.parametrize("model", ["log_logistic", "logistic"])
def test_zero_shifts_are_the_unbinned_curve(model):
    binned = _binned((0.0,) * 24, model)
    plain = InstrumentPOD("plain", model, 3.0, 0.0, 2.0)
    q = jnp.logspace(-2.0, 2.0, 41)
    for tod_bin in (0, 7, 23):
        selected = jnp.array(tod_bin)
        np.testing.assert_allclose(
            pod_union([binned], q, tod_bin=selected), pod_union([plain], q), rtol=1e-14
        )
        np.testing.assert_allclose(
            log_pod_union([binned], tod_bin=selected)(q),
            log_pod_union([plain])(q),
            rtol=1e-14,
        )


@pytest.mark.parametrize("model", ["log_logistic", "logistic"])
def test_every_bin_reaches_certain_detection_at_large_rates(model):
    # the reason for a shift: a factor on P_d would stop short of 1
    binned = _binned(model=model)
    bins = jnp.arange(TOD_BINS)
    detected = pod_union([binned], jnp.full(TOD_BINS, 1e6), tod_bin=bins)
    np.testing.assert_allclose(detected, 1.0, atol=1e-10)


def test_each_bin_is_half_detected_at_its_shifted_threshold():
    binned = _binned()
    bins = jnp.arange(TOD_BINS)
    thresholds = 3.0 * jnp.exp(jnp.asarray(TOD_SHIFTS))
    np.testing.assert_allclose(
        pod_union([binned], thresholds, tod_bin=bins), 0.5, rtol=1e-12
    )


def test_an_unbinned_instrument_ignores_the_bin_in_a_union():
    plain = InstrumentPOD("plain", "log_logistic", 10.0, 0.0, 1.5)
    q = jnp.array(4.0)
    tod_bin = jnp.array(6)
    missed = (1.0 - _hill(4.0, 3.0 * math.exp(TOD_SHIFTS[6]), 2.0)) * (
        1.0 - _hill(4.0, 10.0, 1.5)
    )
    np.testing.assert_allclose(
        pod_union([_binned(), plain], q, tod_bin=tod_bin), 1.0 - missed, rtol=1e-12
    )


def test_one_rate_over_many_bins_broadcasts_a_mixed_union():
    plain = InstrumentPOD("plain", "log_logistic", 10.0, 0.0, 1.5)
    bins = jnp.arange(TOD_BINS)
    union = pod_union([_binned(), plain], jnp.array(4.0), tod_bin=bins)
    assert union.shape == (TOD_BINS,)
    np.testing.assert_allclose(
        union[6], pod_union([_binned(), plain], jnp.array(4.0), tod_bin=bins[6])
    )


def test_a_binned_instrument_needs_a_bin():
    with pytest.raises(ValueError, match="needs a tod_bin"):
        pod_union([_binned()], jnp.array(1.0))
    with pytest.raises(ValueError, match="needs a tod_bin"):
        log_pod_union([_binned()])


@pytest.mark.parametrize("shifts", [(0.0,) * 23, (0.0,) * 23 + (math.inf,)])
def test_a_bad_table_is_rejected(shifts):
    with pytest.raises(ValueError, match="24 finite shifts"):
        _binned(shifts)


def test_load_calibrations_reads_the_table(tmp_path):
    cells = " ".join(str(shift) for shift in TOD_SHIFTS)
    table = tmp_path / "pods.csv"
    table.write_text(
        f"name,model,q50,q50_std,slope,tod_bins\nbinned,log_logistic,3,0,2,{cells}\n"
    )
    assert load_pod_calibrations(table) == [_binned()]
    listing = tmp_path / "pods.yaml"
    listing.write_text(
        "- {name: binned, model: log_logistic, q50: 3, q50_std: 0, slope: 2, "
        f"tod_bins: {list(TOD_SHIFTS)}}}\n"
    )
    assert load_pod_calibrations(listing) == [_binned()]


def _binned_model_density(pods, **kwargs):
    window = (0.0, 2.0)
    marks = jnp.array([2.0, 20.0])
    event_times, mask = padded_event_buffer([0.25, 1.5], window)
    evidence = EventEvidence(jnp.zeros((2, 1)), marks[:, None])
    return _model_log_density(
        _no_prior_params(),
        event_times=event_times,
        mask=mask,
        evidence=evidence,
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=pods,
        support=(1e-8, 1e8),
        **kwargs,
    )


@pytest.mark.slow
def test_binned_likelihood_with_zero_shifts_is_the_unbinned_one():
    plain = InstrumentPOD("binned", "log_logistic", 3.0, 0.0, 2.0)
    np.testing.assert_allclose(
        _binned_model_density([_binned((0.0,) * 24)]),
        _binned_model_density([plain]),
        rtol=1e-10,
    )


@pytest.mark.slow
def test_binned_likelihood_with_no_intervals_has_no_compensator():
    # an empty coverage_intervals means nobody looked, with or without bins
    window = (0.0, 2.0)
    event_times, mask = padded_event_buffer([], window)
    density = _model_log_density(
        _no_prior_params(),
        event_times=event_times,
        mask=mask,
        evidence=EventEvidence(jnp.zeros((0, 1)), jnp.zeros((0, 1))),
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=[_binned()],
        coverage=jnp.zeros((0, 1), dtype=bool),
        coverage_intervals=[],
        support=(1e-8, 1e8),
    )
    np.testing.assert_allclose(density, _prior_density(_no_prior_params()), rtol=1e-12)


@pytest.mark.slow
def test_binned_likelihood_is_the_closed_form():
    # two days from 06:00: events at 12:00 and 18:00, hours 12 and 18, with every bin
    # looked at for two hours in total
    params = _no_prior_params()
    density = _binned_model_density([_binned()], tod_origin=6.0)
    law = lognorm(s=0.8, scale=math.exp(0.5))
    expected = (
        2.0 * math.log(2.0)
        + law.logpdf(2.0)
        + math.log(_hill(2.0, 3.0 * math.exp(TOD_SHIFTS[12]), 2.0))
        + law.logpdf(20.0)
        + math.log(_hill(20.0, 3.0 * math.exp(TOD_SHIFTS[18]), 2.0))
    )
    for shift in TOD_SHIFTS:
        threshold = 3.0 * math.exp(shift)
        rate = _thinned_rate(lambda q, q50=threshold: _hill(q, q50, 2.0))
        expected -= 2.0 * (2.0 / 24.0) * rate
    np.testing.assert_allclose(density - _prior_density(params), expected, rtol=1e-6)


@pytest.mark.slow
def test_total_emission_sums_the_missing_mass_over_the_hours():
    posterior = TMTPPPosterior(
        intensity="constant",
        mark=LognormalMark(),
        samples={
            "lambda_0": jnp.array([2.0]),
            "mark.mu": jnp.array([0.5]),
            "mark.sigma": jnp.array([0.8]),
        },
    )
    result = total_emission(
        posterior,
        horizon=(0.0, 2.0),
        event_duration=1.0,
        log_pod=None,
        q_min=1e-6,
        q_max=1e6,
        pods=[_binned()],
    )
    law = lognorm(s=0.8, scale=math.exp(0.5))
    expected = 0.0
    for shift in TOD_SHIFTS:
        threshold = 3.0 * math.exp(shift)

        def integrand(log_q, q50=threshold):
            q = math.exp(log_q)
            return q * (1.0 - _hill(q, q50, 2.0)) * law.pdf(q) * q

        rate = quad(integrand, math.log(1e-6), math.log(1e6), limit=200)[0]
        expected += 2.0 * (2.0 / 24.0) * rate
    np.testing.assert_allclose(result.missing_kg, [expected], rtol=1e-5)


def test_expected_events_by_bin_add_up_and_follow_the_clock():
    # a 1.5-day window from 06:00 over λ(t) = 2 + sin(2πt)
    def intensity(time):
        return 2.0 + jnp.sin(2.0 * jnp.pi * time)

    by_bin = expected_events_by_tod_bin(
        intensity, 0.25, 1.75, n_bins=TOD_BINS, tod_origin=0.0
    )

    # ∫ (2 + sin 2πt) dt on [a, b]
    def exact(start, stop):
        return 2.0 * (stop - start) + (
            math.cos(2.0 * math.pi * start) - math.cos(2.0 * math.pi * stop)
        ) / (2.0 * math.pi)

    np.testing.assert_allclose(jnp.sum(by_bin), exact(0.25, 1.75), rtol=1e-10)
    # hour 6 is seen twice in the window, hour 2 once
    twice = exact(0.25, 0.25 + 1.0 / 24.0) + exact(1.25, 1.25 + 1.0 / 24.0)
    once = exact(1.0 + 2.0 / 24.0, 1.0 + 3.0 / 24.0)
    np.testing.assert_allclose(by_bin[6], twice, rtol=1e-10)
    np.testing.assert_allclose(by_bin[2], once, rtol=1e-10)


@pytest.mark.slow
def test_tod_bins_with_coverage_run_under_nuts():
    # the compiled potential, with the shifts and the intervals together
    window = (0.0, 10.0)
    marks = jnp.array([2.0, 20.0, 5.0])
    event_times, mask = padded_event_buffer([1.0, 7.0, 2.0], window)
    other = InstrumentPOD("other", "log_logistic", 10.0, 0.2, 1.5, 0.2)
    mcmc = MCMC(NUTS(tmtpp_iw_model), num_warmup=20, num_samples=20, progress_bar=False)
    mcmc.run(
        jax.random.PRNGKey(0),
        event_times,
        mask,
        EventEvidence(jnp.zeros((3, 1)), marks[:, None]),
        mark=LognormalMark(),
        intensity="constant",
        window=window,
        pods=[_binned(), other],
        coverage=jnp.array([[True, False], [False, True], [True, False]]),
        coverage_intervals=[
            (0.0, 6.0, jnp.array([True, False])),
            (6.0, 8.0, jnp.array([False, True])),
        ],
        support=(1e-6, 1e6),
        n_nodes=201,
    )
    assert jnp.all(jnp.isfinite(mcmc.get_samples()["lambda_0"]))


def _twin_log_intensity(
    params: jax.Array, log_q: jax.Array, covering: jax.Array
) -> jax.Array:
    log_rate, mu, log_sigma = params[:3]
    log_q50 = params[jnp.array([3, 5])]
    log_slope = params[jnp.array([4, 6])]
    # log f(log Q | μ, σ)
    sigma = jnp.exp(log_sigma)
    log_mark = -0.5 * ((log_q - mu) / sigma) ** 2 - log_sigma - 0.9189385332046727
    # a = Σ_k c_k log(1 + e^{k_k (log Q − log Q₅₀,k)})
    miss_scale = jnp.sum(
        covering * jax.nn.softplus(jnp.exp(log_slope) * (log_q - log_q50))
    )
    # log r = log λ + log f + log(1 − e^{−a})
    return log_rate + log_mark + jnp.log(-jnp.expm1(-miss_scale))


def _expected_spread_ratios(n_truths: int, *, key: jax.Array) -> jax.Array:
    """Median expected posterior spread of log Q₅₀ per instrument, over its prior's.

    A Poisson process of intensity r(t, Q; ψ) has Fisher information
    F = Σ_w T_w ∫ r (∂_ψ log r)(∂_ψ log r)ᵀ d log Q, and Laplace puts the posterior
    covariance of ψ = (log λ, μ, log σ, log Q₅₀,k, log k_k) at (F + Λ_ψ)⁻¹. The half-normal
    priors of λ and σ count as no information, which errs towards a larger spread.
    """
    q50_std = jnp.array([pod.q50_std for pod in TWIN_PODS])
    slope_std = jnp.array([pod.slope_std for pod in TWIN_PODS])
    log_q50 = jnp.log(jnp.array([pod.q50 for pod in TWIN_PODS]))
    log_slope = jnp.log(jnp.array([pod.slope for pod in TWIN_PODS]))
    keys = jax.random.split(key, 5)
    rate = 3.0 * jnp.abs(jax.random.normal(keys[0], (n_truths,)))
    mu = jax.random.normal(keys[1], (n_truths,))
    sigma = jnp.abs(jax.random.normal(keys[2], (n_truths,)))
    q50_truth = log_q50 + q50_std * jax.random.normal(keys[3], (n_truths, 2))
    slope_truth = log_slope + slope_std * jax.random.normal(keys[4], (n_truths, 2))
    truths = jnp.column_stack(
        [
            jnp.log(rate),
            mu,
            jnp.log(sigma),
            q50_truth[:, 0],
            slope_truth[:, 0],
            q50_truth[:, 1],
            slope_truth[:, 1],
        ]
    )
    log_q = jnp.linspace(math.log(TWIN_SUPPORT[0]), math.log(TWIN_SUPPORT[1]), 1201)
    step = log_q[1] - log_q[0]
    # Λ_ψ, in the order of ψ
    prior_precision = jnp.diag(
        jnp.array(
            [
                0.0,
                1.0,
                0.0,
                1.0 / q50_std[0] ** 2,
                1.0 / slope_std[0] ** 2,
                1.0 / q50_std[1] ** 2,
                1.0 / slope_std[1] ** 2,
            ]
        )
    )

    def ratios(params: jax.Array) -> jax.Array:
        fisher = prior_precision
        for start, stop, covering in TWIN_INTERVALS:
            per_node = jax.vmap(_twin_log_intensity, (None, 0, None))
            gradients = jax.vmap(jax.grad(_twin_log_intensity), (None, 0, None))(
                params, log_q, jnp.asarray(covering, dtype=float)
            )
            intensity = jnp.exp(per_node(params, log_q, jnp.asarray(covering, float)))
            # F_w = T_w ∫ r (∂_ψ log r)(∂_ψ log r)ᵀ d log Q
            fisher = fisher + (stop - start) * step * jnp.einsum(
                "n,ni,nj->ij", intensity, gradients, gradients
            )
        posterior = jnp.linalg.inv(fisher)
        return jnp.sqrt(jnp.diag(posterior)[jnp.array([3, 5])]) / q50_std

    return jnp.nanmedian(jax.lax.map(ratios, truths), axis=0)


def test_coverage_twin_is_informed_by_design():
    # the calculation behind TWIN_PODS and TWIN_INTERVALS, which needs no MCMC
    ratios = _expected_spread_ratios(100, key=jax.random.PRNGKey(0))
    assert jnp.all(ratios < TWIN_MAX_POSTERIOR_SPREAD), ratios


def _simulate_basin(key: jax.Array):
    keys = jax.random.split(key, 11)
    # truths from the fit's prior: Q₅₀,k and k_k under the calibration
    truth = {}
    for index, pod in enumerate(TWIN_PODS):
        q50_noise = jax.random.normal(keys[2 * index])
        slope_noise = jax.random.normal(keys[2 * index + 1])
        truth[f"pod.{pod.name}.q50"] = pod.q50 * jnp.exp(pod.q50_std * q50_noise)
        truth[f"pod.{pod.name}.slope"] = pod.slope * jnp.exp(
            pod.slope_std * slope_noise
        )
    # λ₀ ~ HalfNormal(3), μ ~ N(0, 1), σ ~ HalfNormal(1)
    rate = 3.0 * jnp.abs(jax.random.normal(keys[4]))
    mu = jax.random.normal(keys[5])
    sigma = jnp.abs(jax.random.normal(keys[6]))

    # --- the ground process, then the union's thinning ---
    span = TWIN_WINDOW[1] - TWIN_WINDOW[0]
    n_ground = jax.random.poisson(keys[7], rate * span)
    marks = jnp.exp(mu + sigma * jax.random.normal(keys[8], (TWIN_MAX_GROUND,)))
    calibration = calibration_from_draw(TWIN_PODS, truth)
    # who was looking when each event happened
    event_times = span * jax.random.uniform(keys[10], (TWIN_MAX_GROUND,))
    coverage = jnp.zeros((TWIN_MAX_GROUND, 2), dtype=bool)
    for start, stop, covering in TWIN_INTERVALS:
        inside = (start <= event_times) & (event_times < stop)
        coverage = jnp.where(inside[:, None], jnp.asarray(covering), coverage)
    log_pod = log_pod_union(TWIN_PODS, coverage=coverage, calibration=calibration)
    uniforms = jax.random.uniform(keys[9], (TWIN_MAX_GROUND,))
    detected = (jnp.arange(TWIN_MAX_GROUND) < n_ground) & (
        jnp.log(uniforms) < log_pod(marks)
    )
    order = jnp.argsort(~detected)[:TWIN_MAX_DETECTED]
    n_detected = jnp.sum(detected)
    mask = detected[order]
    times = jnp.where(mask, event_times[order], span)
    q_samples = jnp.where(mask, marks[order], 1.0)[:, None]
    coverage = jnp.where(mask[:, None], coverage[order], True)
    return times, mask, q_samples, coverage, truth, n_ground, n_detected


def _fit_basin(key: jax.Array, times, mask, q_samples, coverage):
    # one compile for every basin, through lax.map rather than vmap: under vmap every
    # basin pays for the deepest NUTS tree of any basin at each step
    mcmc = MCMC(
        NUTS(tmtpp_iw_model), num_warmup=300, num_samples=300, progress_bar=False
    )
    mcmc.run(
        key,
        times,
        mask,
        EventEvidence(jnp.zeros_like(q_samples), q_samples),
        mark=TWIN_MARK,
        intensity="constant",
        window=TWIN_WINDOW,
        pods=TWIN_PODS,
        coverage=coverage,
        coverage_intervals=TWIN_INTERVALS,
        support=TWIN_SUPPORT,
        n_nodes=801,
    )
    samples = mcmc.get_samples()
    return {name: samples[f"pod.{name}.q50"] for name in ("a", "b")}


@pytest.mark.validation
def test_hierarchical_calibration_coverage(record_property):
    keys = jax.random.split(jax.random.PRNGKey(110), 2 * TWIN_N_BASINS)
    simulated = jax.vmap(_simulate_basin)(keys[:TWIN_N_BASINS])
    times, mask, q_samples, coverage, truth, n_ground, n_detected = simulated
    assert int(jnp.max(n_ground)) < TWIN_MAX_GROUND
    assert int(jnp.max(n_detected)) <= TWIN_MAX_DETECTED
    draws = jax.lax.map(
        lambda inputs: _fit_basin(*inputs),
        (keys[TWIN_N_BASINS:], times, mask, q_samples, coverage),
    )
    measured = {"median_detected": float(jnp.median(n_detected))}
    for pod in TWIN_PODS:
        name = pod.name
        low, high = jnp.quantile(draws[name], jnp.array([0.025, 0.975]), axis=1)
        true_value = truth[f"pod.{name}.q50"]
        inside = (low < true_value) & (true_value < high)
        measured[f"coverage_{name}"] = float(jnp.mean(inside))
        # the catalog informs this instrument's Q₅₀: not just its prior's coverage
        spread = jnp.median(jnp.std(jnp.log(draws[name]), axis=1))
        measured[f"spread_ratio_{name}"] = float(spread) / pod.q50_std
    for measure, value in measured.items():
        record_property(measure, value)
    for pod in TWIN_PODS:
        coverage_error = abs(measured[f"coverage_{pod.name}"] - 0.95)
        assert coverage_error <= TWIN_COVERAGE_BAND, measured
        assert measured[f"spread_ratio_{pod.name}"] < TWIN_MAX_POSTERIOR_SPREAD, (
            measured
        )
