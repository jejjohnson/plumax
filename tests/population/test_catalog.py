"""Tests for the Tier V cross-tier posterior catalog."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled.fusion import FusionPosterior
from plumax.lagrangian.inversion import GaussianPosterior, LognormalPosterior
from plumax.population.catalog import (
    EmissionCatalog,
    EmissionEvent,
    event_from_posterior,
)


def _event(rate: float, std: float, **kw: object) -> EmissionEvent:
    base = {"x": 0.0, "y": 0.0, "time": 0.0, "tier": "II"}
    base.update(kw)
    return EmissionEvent(emission_rate=rate, emission_std=std, **base)  # type: ignore[arg-type]


def test_event_rejects_negative_std() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _event(1.0, -0.5)


def test_catalog_vectorised_accessors() -> None:
    events = [
        _event(1.0, 0.1, x=10.0, y=20.0, time=100.0),
        _event(2.0, 0.2, x=30.0, y=40.0, time=200.0),
        _event(3.0, 0.3, x=50.0, y=60.0, time=300.0),
    ]
    cat = EmissionCatalog.from_events(events)
    assert cat.n_events == 3
    np.testing.assert_allclose(cat.emission_rate, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cat.emission_std, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(cat.x, [10.0, 30.0, 50.0])
    np.testing.assert_allclose(cat.y, [20.0, 40.0, 60.0])
    np.testing.assert_allclose(cat.time, [100.0, 200.0, 300.0])


def test_empty_catalog() -> None:
    cat = EmissionCatalog()
    assert cat.n_events == 0
    assert cat.emission_rate.shape == (0,)


def test_log_moments_math() -> None:
    # For Q ~ Normal(m, s^2) matched to a lognormal:
    # log_std^2 = ln(1 + (s/m)^2); log_mean = ln(m) - 0.5 log_std^2
    cat = EmissionCatalog.from_events([_event(10.0, 2.0), _event(5.0, 0.0)])
    log_mean, log_std = cat.log_moments()
    cv2 = (2.0 / 10.0) ** 2
    expected_log_var = np.log1p(cv2)
    np.testing.assert_allclose(log_std[0] ** 2, expected_log_var)
    np.testing.assert_allclose(log_mean[0], np.log(10.0) - 0.5 * expected_log_var)
    # Zero std -> exact lognormal at log(m), zero log-std.
    np.testing.assert_allclose(log_std[1], 0.0)
    np.testing.assert_allclose(log_mean[1], np.log(5.0))


def test_log_moments_recovers_mean() -> None:
    # The matched lognormal must reproduce the original arithmetic mean:
    # E[Q] = exp(log_mean + 0.5 log_std^2) == m.
    cat = EmissionCatalog.from_events([_event(7.5, 3.0)])
    log_mean, log_std = cat.log_moments()
    recovered_mean = np.exp(log_mean + 0.5 * log_std**2)
    np.testing.assert_allclose(recovered_mean, [7.5])


def test_log_moments_rejects_nonpositive_mean() -> None:
    cat = EmissionCatalog.from_events([_event(0.0, 1.0)])
    with pytest.raises(ValueError, match="positive"):
        cat.log_moments()


def test_event_from_gaussian_posterior() -> None:
    # GaussianPosterior is a grid-vector source-field posterior; the adapter
    # reduces it to the field total Q = 1ᵀmean with std = sqrt(1ᵀ Σ 1).
    mean = jnp.asarray([2.0, 1.5])
    cov = jnp.asarray([[0.04, 0.0], [0.0, 0.09]])
    post = GaussianPosterior(mean=mean, covariance=cov)
    ev = event_from_posterior(
        post, x=1.0, y=2.0, time=3.0, tier="II", instrument="GHGSat"
    )
    assert ev.emission_rate == pytest.approx(3.5)
    assert ev.emission_std == pytest.approx(np.sqrt(0.04 + 0.09))
    assert ev.instrument == "GHGSat"
    assert ev.tier == "II"


def test_event_from_lognormal_posterior() -> None:
    # LognormalPosterior carries mean (grid) + log_covariance; the adapter
    # totals the field and propagates std via the delta method.
    mean = jnp.asarray([3.0, 1.0])
    log_cov = jnp.asarray([[0.04, 0.0], [0.0, 0.09]])
    post = LognormalPosterior(
        mean=mean,
        log_increment=jnp.zeros(2),
        log_covariance=log_cov,
    )
    ev = event_from_posterior(post, x=0.0, y=0.0, time=0.0, tier="II")
    assert ev.emission_rate == pytest.approx(4.0)
    # Var = q_i^2 (exp(σ_ii) - 1) summed (off-diagonals zero here).
    expected_var = 9.0 * np.expm1(0.04) + 1.0 * np.expm1(0.09)
    assert ev.emission_std == pytest.approx(np.sqrt(expected_var))


def test_event_from_unknown_posterior_raises() -> None:
    class _Bogus:
        pass

    with pytest.raises(TypeError, match="must expose"):
        event_from_posterior(_Bogus(), x=0.0, y=0.0, time=0.0, tier="II")


def _build_fusion_posterior(rate: float, std: float) -> FusionPosterior:
    """Construct a real ``FusionPosterior`` robustly across field layouts.

    The adapter reads ``emission_rate`` / ``emission_std`` directly; this
    helper fills any other required dataclass fields with placeholders so
    the test exercises the genuine type.
    """
    kwargs: dict[str, object] = {}
    for f in dataclasses.fields(FusionPosterior):
        if f.name == "emission_rate":
            kwargs[f.name] = jnp.asarray(rate)
        elif f.name == "emission_std":
            kwargs[f.name] = jnp.asarray(std)
        elif (
            f.default is not dataclasses.MISSING
            or f.default_factory is not dataclasses.MISSING
        ):
            continue
        elif f.name == "instrument_names":
            kwargs[f.name] = ()
        else:
            kwargs[f.name] = jnp.asarray(0.0)
    return FusionPosterior(**kwargs)  # type: ignore[arg-type]


def test_event_from_fusion_posterior() -> None:
    post = _build_fusion_posterior(6.0, 1.5)
    ev = event_from_posterior(
        post, x=0.0, y=0.0, time=0.0, tier="IV", instrument="EMIT"
    )
    assert ev.emission_rate == pytest.approx(6.0)
    assert ev.emission_std == pytest.approx(1.5)
    assert ev.tier == "IV"
