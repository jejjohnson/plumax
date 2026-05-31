"""Tests for Tier V.B — spatio-temporal point process."""

from __future__ import annotations

import numpy as np
import pytest

from plumax.population.catalog import EmissionCatalog, EmissionEvent
from plumax.population.point_process import (
    PoissonRatePosterior,
    fit_inhomogeneous_intensity,
    fit_poisson_rate,
)


def _catalog(n: int) -> EmissionCatalog:
    events = [
        EmissionEvent(
            emission_rate=1.0,
            emission_std=0.1,
            x=0.0,
            y=0.0,
            time=float(i),
            tier="II",
        )
        for i in range(n)
    ]
    return EmissionCatalog.from_events(events)


def test_poisson_rate_recovery() -> None:
    # Draw a Poisson count for a known rate * exposure, then recover lambda.
    rng = np.random.default_rng(0)
    lam_true = 2.5
    area, duration = 4.0, 50.0
    exposure = area * duration
    n = int(rng.poisson(lam_true * exposure))
    post = fit_poisson_rate(_catalog(n), area=area, duration=duration)
    assert isinstance(post, PoissonRatePosterior)
    assert post.n_events == n
    assert post.rate_mean == pytest.approx(lam_true, rel=0.15)
    lo, hi = post.credible_interval(0.95)
    assert lo <= lam_true <= hi


def test_poisson_rate_conjugate_update() -> None:
    post = fit_poisson_rate(
        _catalog(10), area=2.0, duration=5.0, prior_shape=1.0, prior_rate=1.0
    )
    # Gamma(1 + 10, 1 + 10) posterior.
    assert post.alpha == pytest.approx(11.0)
    assert post.beta == pytest.approx(11.0)
    assert post.rate_mean == pytest.approx(1.0)


def test_poisson_rate_rejects_bad_exposure() -> None:
    with pytest.raises(ValueError, match="area"):
        fit_poisson_rate(_catalog(3), area=0.0, duration=1.0)
    with pytest.raises(ValueError, match="duration"):
        fit_poisson_rate(_catalog(3), area=1.0, duration=-2.0)


def test_inhomogeneous_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="covariates"):
        fit_inhomogeneous_intensity(np.zeros((0, 2)))
    with pytest.raises(ValueError, match="covariates"):
        fit_inhomogeneous_intensity(np.zeros((5,)))


@pytest.mark.slow
def test_inhomogeneous_intensity_smoke() -> None:
    rng = np.random.default_rng(3)
    cov = rng.normal(size=(40, 2))
    post = fit_inhomogeneous_intensity(cov, num_warmup=200, num_samples=200, seed=5)
    assert post.beta_mean.shape == (2,)
    assert np.isfinite(post.beta0_mean)
