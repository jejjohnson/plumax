"""Tests for Tier V.A — instantaneous emission size distribution."""

from __future__ import annotations

import numpy as np
import pytest

from plumax.population.catalog import EmissionCatalog, EmissionEvent
from plumax.population.size_distribution import (
    SizeDistributionPosterior,
    fit_lognormal_size_distribution,
)


def _catalog_from_logQ(log_q: np.ndarray, rel_std: np.ndarray) -> EmissionCatalog:
    """Build a catalog of events with mean exp(log_q) and given CV per event."""
    means = np.exp(log_q)
    stds = rel_std * means
    events = [
        EmissionEvent(
            emission_rate=float(m),
            emission_std=float(s),
            x=0.0,
            y=0.0,
            time=float(i),
            tier="II",
        )
        for i, (m, s) in enumerate(zip(means, stds, strict=True))
    ]
    return EmissionCatalog.from_events(events)


def test_posterior_helpers() -> None:
    rng = np.random.default_rng(0)
    post = SizeDistributionPosterior(
        mu_samples=rng.normal(1.0, 0.1, size=500),
        sigma_samples=np.abs(rng.normal(0.5, 0.05, size=500)),
    )
    assert post.mu_mean == pytest.approx(1.0, abs=0.05)
    summary = post.summary()
    assert summary["mu"]["q2.5"] < summary["mu"]["q97.5"]
    pdf = post.fitted_pdf(np.array([1.0, 2.0, 5.0]))
    assert np.all(pdf > 0.0)
    med = post.quantile(0.5)
    assert med == pytest.approx(np.exp(post.mu_mean), rel=0.01)


def test_empty_catalog_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_lognormal_size_distribution(EmissionCatalog())


@pytest.mark.slow
def test_synthetic_recovery() -> None:
    rng = np.random.default_rng(42)
    mu_true, sigma_true = 1.5, 0.8
    n = 120
    log_q = rng.normal(mu_true, sigma_true, size=n)
    cat = _catalog_from_logQ(log_q, rel_std=np.full(n, 0.2))
    post = fit_lognormal_size_distribution(cat, num_warmup=300, num_samples=300, seed=1)
    s = post.summary()
    assert s["mu"]["q2.5"] <= mu_true <= s["mu"]["q97.5"]
    assert s["sigma"]["q2.5"] <= sigma_true <= s["sigma"]["q97.5"]


@pytest.mark.slow
def test_uncertainty_propagation_widens_posterior() -> None:
    # Core correctness check: a catalog of poorly-constrained events must
    # yield a WIDER theta posterior than the same point estimates fit with
    # negligible per-event uncertainty.
    rng = np.random.default_rng(7)
    mu_true, sigma_true = 1.0, 0.6
    n = 80
    log_q = rng.normal(mu_true, sigma_true, size=n)
    tight = _catalog_from_logQ(log_q, rel_std=np.full(n, 1e-3))
    wide = _catalog_from_logQ(log_q, rel_std=np.full(n, 0.8))
    post_tight = fit_lognormal_size_distribution(
        tight, num_warmup=300, num_samples=300, seed=2
    )
    post_wide = fit_lognormal_size_distribution(
        wide, num_warmup=300, num_samples=300, seed=2
    )
    width_tight = post_tight.summary()["mu"]["std"]
    width_wide = post_wide.summary()["mu"]["std"]
    assert width_wide > width_tight
