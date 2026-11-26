"""Shape, finiteness, and prior-sampling smoke tests for intensity modules."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import pytest

from plumax.population import intensity


T_GRID = jnp.linspace(0.0, 365.0, 257)


CONSTRUCTOR_ARGS = {
    "ConstantIntensity": dict(lambda_0=2.0),
    "DiurnalSinusoidalIntensity": dict(lambda_0=5.0, amplitude=2.0),
    "SeasonalSinusoidalIntensity": dict(lambda_0=2.0, amplitude=1.0),
    "DiurnalSeasonalIntensity": dict(lambda_0=3.0, amp_diurnal=1.5, amp_seasonal=1.0),
    "SeasonallyModulatedDiurnalIntensity": dict(
        lambda_0=4.0, amp_summer=3.0, amp_winter=0.5
    ),
    "OperationalScheduleIntensity": dict(lambda_active=8.0, lambda_idle=1.0),
    "WeibullRenewalIntensity": dict(scale=0.5, shape=3.0),
    "PeriodicBatchIntensity": dict(
        lambda_background=0.5, lambda_peak=20.0, period_days=0.5
    ),
    "CoalMineVentilationIntensity": dict(
        lambda_extraction=10.0, lambda_maintenance=1.0, lambda_idle=3.0
    ),
    "LandfillIntensity": dict(
        lambda_0=4.0, amp_seasonal=2.0, amp_diurnal=1.5, amp_barometric=1.0
    ),
    "OffshorePlatformIntensity": dict(lambda_0=5.0, amp_tidal=1.5, amp_operational=2.0),
    "WetlandPermafrostIntensity": dict(lambda_peak=5.0, lambda_frozen=0.1),
    "LivestockFeedlotIntensity": dict(lambda_0=3.0, amp_feeding=2.0, amp_seasonal=1.0),
}


def _time_grid(cls_name: str) -> jnp.ndarray:
    # The Weibull renewal hazard takes time-since-last-event, not calendar time.
    if cls_name == "WeibullRenewalIntensity":
        return jnp.linspace(0.05, 5.0, 64)
    return T_GRID


@pytest.mark.parametrize(
    "cls_name,kwargs", list(CONSTRUCTOR_ARGS.items()), ids=list(CONSTRUCTOR_ARGS)
)
def test_intensity_call_shape_finite_and_nonneg(cls_name, kwargs):
    cls = getattr(intensity, cls_name)
    model = cls(**kwargs)
    t = _time_grid(cls_name)
    lam = model(t)
    assert lam.shape == t.shape
    assert jnp.all(jnp.isfinite(lam)), f"{cls_name} produced non-finite λ"
    assert jnp.all(lam >= -1e-6), f"{cls_name} produced negative intensity"


@pytest.mark.parametrize(
    "cls_name",
    [n for n in CONSTRUCTOR_ARGS if n != "WeibullRenewalIntensity"],
)
def test_priors_sample_and_evaluate(cls_name):
    """Prior-factory returns a runnable module; λ(t) is finite on the grid."""
    cls = getattr(intensity, cls_name)

    def model():
        m = cls.sample_priors()
        lam = m(T_GRID)
        numpyro.deterministic("lam_finite", jnp.all(jnp.isfinite(lam)))

    with numpyro.handlers.seed(rng_seed=jr.PRNGKey(0)):
        trace = numpyro.handlers.trace(model).get_trace()
    assert bool(trace["lam_finite"]["value"])


@pytest.mark.parametrize(
    "kernel",
    [
        intensity.DiurnalSinusoidalIntensity(
            lambda_0=4.0, amplitude=3.0, phase_hours=14.0
        ),
        intensity.DiurnalSeasonalIntensity(
            lambda_0=4.0, amp_diurnal=3.0, amp_seasonal=0.0, phase_hours=14.0
        ),
        intensity.SeasonallyModulatedDiurnalIntensity(
            lambda_0=4.0, amp_summer=3.0, amp_winter=3.0, phase_hours=14.0
        ),
        intensity.LandfillIntensity(
            lambda_0=4.0,
            amp_seasonal=0.0,
            amp_diurnal=3.0,
            amp_barometric=0.0,
            phase_hours=14.0,
        ),
    ],
    ids=lambda kernel: type(kernel).__name__,
)
def test_the_daily_peak_is_at_phase_hours(kernel):
    # A sin(2πt − φ) peaked six hours after phase_hours; A cos(2πt − φ) peaks on it
    minutes = jnp.arange(24 * 60) / (24 * 60)
    peak_hour = float(minutes[jnp.argmax(kernel(minutes))] * 24)
    assert abs(peak_hour - 14.0) <= 1.0 / 60.0


@pytest.mark.parametrize(
    "kernel",
    [
        intensity.SeasonalSinusoidalIntensity(
            lambda_0=4.0, amplitude=3.0, peak_day=196.0
        ),
        intensity.DiurnalSeasonalIntensity(
            lambda_0=4.0, amp_diurnal=0.0, amp_seasonal=3.0, peak_day=196.0
        ),
        intensity.LandfillIntensity(
            lambda_0=4.0,
            amp_seasonal=3.0,
            amp_diurnal=0.0,
            amp_barometric=0.0,
            peak_day=196.0,
        ),
        intensity.LivestockFeedlotIntensity(
            lambda_0=4.0, amp_feeding=0.0, amp_seasonal=3.0, peak_day=196.0
        ),
    ],
    ids=lambda kernel: type(kernel).__name__,
)
def test_the_seasonal_peak_is_on_peak_day(kernel):
    # A sin(ω_s (t − t_peak)) peaked a quarter year (91 days) after peak_day
    days = jnp.arange(0.0, 365.25, 0.01)
    assert abs(float(days[jnp.argmax(kernel(days))]) - 196.0) <= 0.01
