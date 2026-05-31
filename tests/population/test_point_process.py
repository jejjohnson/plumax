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
    good_quad = np.zeros((4, 2))
    good_w = np.ones(4)
    with pytest.raises(ValueError, match="covariates"):
        fit_inhomogeneous_intensity(
            np.zeros((0, 2)), quadrature_covariates=good_quad, quadrature_weights=good_w
        )
    with pytest.raises(ValueError, match="covariates"):
        fit_inhomogeneous_intensity(
            np.zeros((5,)), quadrature_covariates=good_quad, quadrature_weights=good_w
        )


def test_inhomogeneous_rejects_bad_quadrature() -> None:
    cov = np.zeros((5, 2))
    # Covariate-dimension mismatch between events and quadrature points.
    with pytest.raises(ValueError, match="covariate"):
        fit_inhomogeneous_intensity(
            cov,
            quadrature_covariates=np.zeros((4, 3)),
            quadrature_weights=np.ones(4),
        )
    # Weight shape mismatch.
    with pytest.raises(ValueError, match="quadrature_weights"):
        fit_inhomogeneous_intensity(
            cov,
            quadrature_covariates=np.zeros((4, 2)),
            quadrature_weights=np.ones(3),
        )
    # Non-positive weights.
    with pytest.raises(ValueError, match="strictly positive"):
        fit_inhomogeneous_intensity(
            cov,
            quadrature_covariates=np.zeros((4, 2)),
            quadrature_weights=np.zeros(4),
        )


@pytest.mark.slow
def test_inhomogeneous_intensity_recovers_slope() -> None:
    # Simulate a 1-D inhomogeneous Poisson process with log λ(x) = β0 + β·x on a
    # fine grid by thinning, then check the exposure-aware fit recovers β. The
    # quadrature grid IS the simulation grid (weights = cell width), so the
    # integrated-intensity term is exact up to discretisation.
    rng = np.random.default_rng(0)
    # β0 = 1.0 gives ~70 events on this grid — enough to recover the slope.
    beta0_true, beta_true = 1.0, 1.5
    grid = np.linspace(-2.0, 2.0, 400)
    dx = float(grid[1] - grid[0])
    lam = np.exp(beta0_true + beta_true * grid)  # intensity per unit x
    # Thin: each cell yields an event with prob λ·dx (small).
    counts = rng.poisson(lam * dx)
    events = np.repeat(grid, counts)
    assert events.size > 30  # enough events for a meaningful fit
    post = fit_inhomogeneous_intensity(
        events[:, None],
        quadrature_covariates=grid[:, None],
        quadrature_weights=np.full(grid.size, dx),
        num_warmup=400,
        num_samples=400,
        seed=1,
    )
    assert post.beta_mean.shape == (1,)
    assert np.isfinite(post.beta0_mean)
    # Slope AND intercept recovered within a generous tolerance (finite
    # catalog + MC) — the intercept check confirms the exposure term makes the
    # rate density identifiable, not just the slope.
    assert abs(float(post.beta_mean[0]) - beta_true) < 0.5
    assert abs(float(post.beta0_mean) - beta0_true) < 0.5
