"""Tier V.B — methane source intensity functions as Equinox modules.

Physically motivated intensity functions ``λ(t)`` [events day⁻¹] for CH₄
emission sources observed by satellite remote sensing, packaged as
``equinox.Module`` dataclasses. Each module bundles:

* deterministic parameters as frozen fields;
* ``__call__(t)`` — ``λ(t)`` evaluation, compatible with jit / vmap / grad;
* ``sample_priors(cls)`` — a NumPyro prior factory returning a configured
  instance for MCMC inference.

Conventions:

* Time ``t`` is always in **days**; fractional days encode sub-daily
  resolution.
* Intensity ``λ(t)`` is always in **events day⁻¹**.
* All modules are immutable pytrees (equinox).

NumPyro is only imported inside ``sample_priors``, so evaluating an
intensity never pulls it in.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float

from plumax.population._src._priors import PriorTable, PriorTableMixin


__all__ = [
    "INTENSITY_REGISTRY",
    "CoalMineVentilationIntensity",
    "ConstantIntensity",
    "DiurnalSeasonalIntensity",
    "DiurnalSinusoidalIntensity",
    "LandfillIntensity",
    "LivestockFeedlotIntensity",
    "OffshorePlatformIntensity",
    "OperationalScheduleIntensity",
    "PeriodicBatchIntensity",
    "SeasonalSinusoidalIntensity",
    "SeasonallyModulatedDiurnalIntensity",
    "WeibullRenewalIntensity",
    "WetlandPermafrostIntensity",
]


def _hour_of_day(t: jnp.ndarray) -> jnp.ndarray:
    """h = (t mod 1) · 24 [hours]."""
    return (t % 1.0) * 24.0


def _diurnal_phase(phase_hours: ArrayLike) -> ArrayLike:
    """φ_d = 2π · (φ_h / 24) [rad]."""
    return 2.0 * jnp.pi * (phase_hours / 24.0)


def _seasonal_phase(t: jnp.ndarray, peak_day: ArrayLike) -> jnp.ndarray:
    """ω_s (t − t_peak) with ω_s = 2π / 365.25 [rad]."""
    return 2.0 * jnp.pi / 365.25 * (t - peak_day)


def _shift_gate(
    t: jnp.ndarray, start: ArrayLike, end: ArrayLike, steepness: float = 50.0
) -> jnp.ndarray:
    """σ(s (h − h_start)) − σ(s (h − h_end)) at the hour of day h of ``t``."""
    hour = _hour_of_day(t)
    return jax.nn.sigmoid(steepness * (hour - start)) - jax.nn.sigmoid(
        steepness * (hour - end)
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. CONSTANT (Homogeneous Poisson)
# ═════════════════════════════════════════════════════════════════════════════


class ConstantIntensity(PriorTableMixin, eqx.Module):
    """Constant intensity for passive, continuously leaking infrastructure.

    Physical Motivation:
        Abandoned oil/gas wells, corroded subsurface pipelines, orphaned wellheads.
        No external forcing — the infrastructure is under constant, unchanging
        stress producing uncoordinated micro-leaks at a flat background rate.

        Typical sources: abandoned wells (~1–5 events day⁻¹), legacy distribution
        networks, chronic flange leaks.

    Equation:
            λ(t) = λ₀
            Λ(T) = λ₀ · T

    Attributes:
        lambda_0: Constant emission rate [events day⁻¹]. Typical: 0.1–10.0.

    Priors (sample_priors):
        λ₀ ~ HalfNormal(3.0)
    """

    lambda_0: Float[ArrayLike, ""]

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹]. Shape ``(N,)``.
        """
        return jnp.full_like(t, self.lambda_0)

    def integrate(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Closed-form compensator Λ(b) − Λ(a) = λ₀ · (b − a).

        Args:
            a: Lower limit [days].
            b: Upper limit [days].

        Returns:
            Expected event count on ``[a, b]``.
        """
        return self.lambda_0 * (jnp.asarray(b) - jnp.asarray(a))

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(self.lambda_0)

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(3.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. DIURNAL SINUSOIDAL (Inhomogeneous Poisson — Sub-Daily)
# ═════════════════════════════════════════════════════════════════════════════


class DiurnalSinusoidalIntensity(PriorTableMixin, eqx.Module):
    """Diurnal sinusoidal intensity for thermally driven infrastructure.

    Physical Motivation:
        Solar heating during afternoon hours (≈14:00) expands gas volumes inside
        storage tanks and pipeline networks, increasing PRV venting probability.
        Overnight cooling contracts the gas and suppresses emissions.

        Dominant sub-daily pattern for geostationary platforms (MTG-IRS) or
        high-revisit LEO constellations (GHGSat daily, Tanager daily targeted).

        Typical sources: tank farms, active compressor stations, LNG terminals,
        midstream gathering systems.

    Equation:
            λ(t) = max(0, λ₀ + A · cos(2πt − φ))
            φ = 2π · (φ_h / 24)

        Period = 1 day. Over one full cycle: Λ(1) = λ₀.

    Attributes:
        lambda_0: Baseline mean rate [events day⁻¹]. Must satisfy λ₀ > A. Typical: 2–8.
        amplitude: Diurnal half-amplitude [events day⁻¹]. Typical: 1–4.
        phase_hours: Hour-of-day of peak intensity [hours]. Typical: 14.0 (2 PM).

    Priors (sample_priors):
        λ₀  ~ HalfNormal(5.0)
        A   ~ HalfNormal(3.0)
        φ_h ~ Normal(14.0, 2.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amplitude: Float[ArrayLike, ""]
    phase_hours: Float[ArrayLike, ""] = 14.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        # φ = 2π (φ_h / 24)
        phase = _diurnal_phase(self.phase_hours)
        # λ = λ₀ + A cos(2π t − φ), largest at t = φ_h / 24
        raw = self.lambda_0 + self.amplitude * jnp.cos(2.0 * jnp.pi * t - phase)
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + |A|.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(self.lambda_0 + jnp.abs(self.amplitude))

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(5.0)),
        "amplitude": ("amplitude", lambda dist: dist.HalfNormal(3.0)),
        "phase_hours": ("phase_hours", lambda dist: dist.Normal(14.0, 2.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. SEASONAL SINUSOIDAL (Inhomogeneous Poisson — Annual)
# ═════════════════════════════════════════════════════════════════════════════


class SeasonalSinusoidalIntensity(PriorTableMixin, eqx.Module):
    """Annual sinusoidal intensity for biogenic / climate-driven sources.

    Physical Motivation:
        Microbial methanogenesis in saturated soils peaks during warmest, wettest
        months. Dominates at regional/area-flux scales observed by wide-swath
        mappers: TROPOMI (7×5.5 km, daily), MethaneSAT (200 m, ~weekly).

        Typical sources: tropical/boreal wetlands, rice paddies (monsoon Asia),
        permafrost thermokarst lakes, CAFOs.

    Equation:
            λ(t) = max(0, λ₀ + A · cos(2π/365.25 · (t − t_peak)))

        Over one full year: Λ(365.25) = λ₀ · 365.25.

    Attributes:
        lambda_0: Annual mean rate [events day⁻¹]. Typical: 0.5–3.0.
        amplitude: Seasonal half-amplitude [events day⁻¹]. Typical: 0.3–2.0.
        peak_day: Day-of-year of emission peak [days from Jan 1]. Typical: 196 (Jul 15).

    Priors (sample_priors):
        λ₀      ~ HalfNormal(2.0)
        A       ~ HalfNormal(1.5)
        t_peak  ~ Normal(196.0, 30.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amplitude: Float[ArrayLike, ""]
    peak_day: Float[ArrayLike, ""] = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        raw = self.lambda_0 + self.amplitude * jnp.cos(
            _seasonal_phase(t, self.peak_day)
        )
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + |A|.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(self.lambda_0 + jnp.abs(self.amplitude))

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(2.0)),
        "amplitude": ("amplitude", lambda dist: dist.HalfNormal(1.5)),
        "peak_day": ("peak_day", lambda dist: dist.Normal(196.0, 30.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. DIURNAL + SEASONAL COMPOUND (Additive Multi-Scale)
# ═════════════════════════════════════════════════════════════════════════════


class DiurnalSeasonalIntensity(PriorTableMixin, eqx.Module):
    """Additive diurnal + seasonal intensity for thermally active biogenic sources.

    Physical Motivation:
        Many sources exhibit *both* sub-daily thermal cycling and annual
        modulation. Landfills decompose faster on hot summer afternoons than cold
        winter nights. The aliasing between a satellite's fixed overpass time
        (Sentinel-2 at 10:30 AM, Landsat at 10:00 AM) and the diurnal peak
        systematically biases detection statistics.

        Typical sources: municipal landfills, wastewater treatment plants,
        agricultural biodigesters, tropical rice paddies.

    Equation:
            λ(t) = max(0, λ₀ + A_d·cos(2πt − φ_d) + A_s·cos(ω_s·(t − t_peak)))
            ω_s = 2π/365.25,  φ_d = 2π·(φ_h/24)

    Attributes:
        lambda_0: Grand mean intensity [events day⁻¹]. Typical: 1–5.
        amp_diurnal: Diurnal half-amplitude [events day⁻¹].
        amp_seasonal: Seasonal half-amplitude [events day⁻¹].
        phase_hours: Hour of diurnal peak [hours]. Default: 14.0.
        peak_day: Day of seasonal peak [days from Jan 1]. Default: 196.0.

    Priors (sample_priors):
        λ₀   ~ HalfNormal(3.0)
        A_d  ~ HalfNormal(2.0)
        A_s  ~ HalfNormal(1.5)
        φ_h  ~ Normal(14.0, 2.0)
        t_pk ~ Normal(196.0, 30.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amp_diurnal: Float[ArrayLike, ""]
    amp_seasonal: Float[ArrayLike, ""]
    phase_hours: Float[ArrayLike, ""] = 14.0
    peak_day: Float[ArrayLike, ""] = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        φ_d = _diurnal_phase(self.phase_hours)
        raw = (
            self.lambda_0
            + self.amp_diurnal * jnp.cos(2.0 * jnp.pi * t - φ_d)
            + self.amp_seasonal * jnp.cos(_seasonal_phase(t, self.peak_day))
        )
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + |A_d| + |A_s|.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_0 + jnp.abs(self.amp_diurnal) + jnp.abs(self.amp_seasonal)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(3.0)),
        "amp_diurnal": ("amp_diurnal", lambda dist: dist.HalfNormal(2.0)),
        "amp_seasonal": ("amp_seasonal", lambda dist: dist.HalfNormal(1.5)),
        "phase_hours": ("phase_hours", lambda dist: dist.Normal(14.0, 2.0)),
        "peak_day": ("peak_day", lambda dist: dist.Normal(196.0, 30.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. SEASONALLY MODULATED DIURNAL (Multiplicative Coupling)
# ═════════════════════════════════════════════════════════════════════════════


class SeasonallyModulatedDiurnalIntensity(PriorTableMixin, eqx.Module):
    """Diurnal intensity whose amplitude is seasonally modulated.

    Physical Motivation:
        Summer days produce much larger thermal swings (ΔT ≈ 20 K) than winter
        (ΔT ≈ 5 K). The diurnal emission peak is 3–4× larger in July than
        January. This *multiplicative* coupling is the dominant pattern at large
        landfills and open-pit coal mines observed from orbiting hyperspectral
        imagers (GHGSat 25 m, Tanager 30 m).

    Equation:
            A_d(t) = (A_sum + A_win)/2 + (A_sum − A_win)/2 · cos(ω_s·(t − t_pk))
            λ(t) = max(0, λ₀ + A_d(t) · cos(2πt − φ_d))

    Attributes:
        lambda_0: Grand mean intensity [events day⁻¹].
        amp_summer: Diurnal half-amplitude at seasonal peak [events day⁻¹].
        amp_winter: Diurnal half-amplitude at seasonal trough [events day⁻¹].
        phase_hours: Hour of diurnal peak [hours]. Default: 14.0.
        peak_day: Day of seasonal maximum [days from Jan 1]. Default: 196.0.

    Priors (sample_priors):
        λ₀      ~ HalfNormal(4.0)
        A_sum   ~ HalfNormal(3.0)
        A_win   ~ HalfNormal(1.0)
        φ_h     ~ Normal(14.0, 2.0)
        t_peak  ~ Normal(196.0, 30.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amp_summer: Float[ArrayLike, ""]
    amp_winter: Float[ArrayLike, ""]
    phase_hours: Float[ArrayLike, ""] = 14.0
    peak_day: Float[ArrayLike, ""] = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        φ_d = _diurnal_phase(self.phase_hours)
        amp_mean = (self.amp_summer + self.amp_winter) / 2.0
        amp_diff = (self.amp_summer - self.amp_winter) / 2.0
        # A(t) = A_mean + A_diff cos(ω_s (t − t_peak))
        seasonal_amplitude = amp_mean + amp_diff * jnp.cos(
            _seasonal_phase(t, self.peak_day)
        )
        # λ = λ₀ + A(t) cos(2π t − φ_d)
        raw = self.lambda_0 + seasonal_amplitude * jnp.cos(2.0 * jnp.pi * t - φ_d)
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + max(|A_sum|, |A_win|), since A_d(t) lies between the two.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_0
            + jnp.maximum(jnp.abs(self.amp_summer), jnp.abs(self.amp_winter))
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(4.0)),
        "amp_summer": ("amp_summer", lambda dist: dist.HalfNormal(3.0)),
        "amp_winter": ("amp_winter", lambda dist: dist.HalfNormal(1.0)),
        "phase_hours": ("phase_hours", lambda dist: dist.Normal(14.0, 2.0)),
        "peak_day": ("peak_day", lambda dist: dist.Normal(196.0, 30.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. OPERATIONAL SCHEDULE (Smooth Rectangular On/Off)
# ═════════════════════════════════════════════════════════════════════════════


class OperationalScheduleIntensity(PriorTableMixin, eqx.Module):
    """Step-function intensity for facilities with fixed shift schedules.

    Physical Motivation:
        Compressor stations, pump-jacks, and mine ventilation fans operate on
        strict crew shifts. Machinery ON → pressure cycling and mechanical stress
        generate leaks at λ_active. Machinery OFF → passive background at λ_idle.

        Critical for interpreting polar-orbiting satellites with fixed equatorial
        crossing times: Sentinel-2 (10:30), Landsat (10:00), TROPOMI (13:30).
        A satellite that always observes during the active window systematically
        overestimates the 24-hour average.

    Equation:
            λ(t) = λ_idle + (λ_active − λ_idle) · [σ(h − h_start) − σ(h − h_end)]
            h = (t mod 1) · 24,  σ(x) = 1/(1 + e^(−50x))

        Uses smooth sigmoid transitions (k=50) for differentiability in NUTS.

    Attributes:
        lambda_active: Intensity during operational hours [events day⁻¹]. Typical: 3–15.
        lambda_idle: Intensity during shutdown [events day⁻¹]. Typical: 0.1–1.0.
        start_hour: Start of active window [hours 0–24]. Default: 6.0.
        end_hour: End of active window [hours 0–24]. Default: 18.0.

    Priors (sample_priors):
        λ_active ~ HalfNormal(8.0)
        λ_idle   ~ HalfNormal(1.0)
        h_start  ~ Normal(6.0, 1.0)
        h_end    ~ Normal(18.0, 1.0)
    """

    lambda_active: Float[ArrayLike, ""]
    lambda_idle: Float[ArrayLike, ""]
    start_hour: Float[ArrayLike, ""] = 6.0
    end_hour: Float[ArrayLike, ""] = 18.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹]. Shape ``(N,)``.
        """
        # duty = σ(h − h_start) − σ(h − h_end)
        duty = _shift_gate(t, self.start_hour, self.end_hour)
        return self.lambda_idle + (self.lambda_active - self.lambda_idle) * duty

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ_idle + |λ_active − λ_idle|, since the duty cycle lies in (−1, 1).

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_idle + jnp.abs(self.lambda_active - self.lambda_idle)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_active": ("lambda_active", lambda dist: dist.HalfNormal(8.0)),
        "lambda_idle": ("lambda_idle", lambda dist: dist.HalfNormal(1.0)),
        "start_hour": ("start_hour", lambda dist: dist.Normal(6.0, 1.0)),
        "end_hour": ("end_hour", lambda dist: dist.Normal(18.0, 1.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. WEIBULL RENEWAL (Pressure-Relief Valve Recharge)
# ═════════════════════════════════════════════════════════════════════════════


class WeibullRenewalIntensity(PriorTableMixin, eqx.Module):
    """Weibull hazard function for pressure-driven, recharging sources.

    Physical Motivation:
        PRVs on LNG tanks and high-pressure wellheads don't fail randomly. After
        venting, pressure drops to zero and must physically *recharge*. The hazard
        starts at zero and climbs — a textbook increasing failure rate (IFR).

        This is a **Renewal Process**: intensity depends on elapsed time since the
        most recent event (Δt), not calendar time. Generalises the memoryless
        Exponential (k=1) to IFR (k>1) or DFR (k<1).

        Observable from any high-revisit platform (GHGSat, Tanager, geostationary).

    Equation:
            h(Δt) = (k/η) · (Δt/η)^(k−1)

        k > 1 → hazard increases (pressure recharge).
        k = 1 → constant hazard (memoryless Exponential).

    Attributes:
        scale: Weibull scale η [days]. Typical: 0.05–2.0.
        shape: Weibull shape k [dimensionless]. Typical: 2.0–5.0.

    Note:
        ``__call__`` takes ``t_since_last`` (Δt), not absolute time.

    Priors (sample_priors):
        η ~ HalfNormal(0.5)     [days]
        k ~ LogNormal(1.0, 0.5) [dimensionless]
    """

    scale: Float[ArrayLike, ""]
    shape: Float[ArrayLike, ""]

    def __call__(self, t_since_last: jnp.ndarray) -> jnp.ndarray:
        """Evaluate hazard rate.

        Args:
            t_since_last: Elapsed time since last event [days]. Must be > 0. Shape ``(N,)``.

        Returns:
            Instantaneous hazard [events day⁻¹]. Shape ``(N,)``.
        """
        Δt = jnp.clip(t_since_last, min=1e-8)
        shape, scale = self.shape, self.scale
        # h(Δt) = (k / η) (Δt / η)^(k − 1)
        return (shape / scale) * (Δt / scale) ** (shape - 1.0)

    _PRIORS: ClassVar[PriorTable] = {
        "scale": ("weibull_scale", lambda dist: dist.HalfNormal(0.5)),
        "shape": ("weibull_shape", lambda dist: dist.LogNormal(1.0, 0.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. PERIODIC BATCH VENTING (Impulse Train)
# ═════════════════════════════════════════════════════════════════════════════


class PeriodicBatchIntensity(PriorTableMixin, eqx.Module):
    """Periodic narrow-pulse intensity for scheduled batch venting.

    Physical Motivation:
        Glycol dehydrators flash accumulated CH₄ every 4–8 hours. Tank batteries
        undergo liquid unloading every 12–48 hours producing sharp spikes lasting
        minutes. Coal mine longwall shearers expose fresh seam every 6–12 hours.

        Between batches, only minor fugitive leaks persist. The result is a
        comb-like Gaussian pulse train on a low background.

        Observable primarily by high-revisit tasked satellites (GHGSat, Tanager)
        or continuous ground-based sensors.

    Equation:
            λ(t) = λ_bg + (λ_pk − λ_bg) · exp(−((t mod P) − P/2)² / (2·(f·P)²))

    Attributes:
        lambda_background: Inter-batch fugitive rate [events day⁻¹]. Typical: 0.1–1.0.
        lambda_peak: Peak rate during a batch vent [events day⁻¹]. Typical: 10–50.
        period_days: Time between successive batches [days]. Typical: 0.2–2.0.
        duty_fraction: Pulse width as fraction of period [dimensionless]. Default: 0.05.

    Priors (sample_priors):
        λ_bg  ~ HalfNormal(1.0)
        λ_pk  ~ HalfNormal(20.0)
        P     ~ LogNormal(ln(0.5), 0.5)   [days]
        f     ~ Beta(2, 20)               [dimensionless]
    """

    lambda_background: Float[ArrayLike, ""]
    lambda_peak: Float[ArrayLike, ""]
    period_days: Float[ArrayLike, ""]
    duty_fraction: Float[ArrayLike, ""] = 0.05

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹]. Shape ``(N,)``.
        """
        σ = self.duty_fraction * self.period_days
        phase = (t % self.period_days) - (self.period_days / 2.0)
        pulse = jnp.exp(-0.5 * (phase / σ) ** 2)
        return (
            self.lambda_background + (self.lambda_peak - self.lambda_background) * pulse
        )

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = max(λ_bg, λ_pk), since the pulse lies in (0, 1].

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(jnp.maximum(self.lambda_background, self.lambda_peak))

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_background": ("lambda_bg", lambda dist: dist.HalfNormal(1.0)),
        "lambda_peak": ("lambda_peak", lambda dist: dist.HalfNormal(20.0)),
        "period_days": ("period_days", lambda dist: dist.LogNormal(jnp.log(0.5), 0.5)),
        "duty_fraction": ("duty_fraction", lambda dist: dist.Beta(2.0, 20.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. COAL MINE VENTILATION (Shift + Weekly Maintenance)
# ═════════════════════════════════════════════════════════════════════════════


class CoalMineVentilationIntensity(PriorTableMixin, eqx.Module):
    """Multi-scale operational intensity for underground coal mines.

    Physical Motivation:
        Underground coal mines (>40 Mt CH₄ yr⁻¹ globally) exhibit three regimes:
          1. Extraction shifts (06–22, Mon–Sat): longwall shearer advance exposes
             fresh seam, releasing adsorbed CH₄. Fans at full capacity.
          2. Night idle (22–06): reduced crew, fans at minimum, slower desorption.
          3. Weekly maintenance (Sunday): extraction halted, CH₄ builds in panels.

        Observable by TROPOMI (daily, 7 km — integrates mine complex), MethaneSAT
        (weekly, 200 m — resolves vent shafts), GHGSat/Tanager (shaft-level).

    Equation:
            maintenance_gate = exp(−(d_week − d_maint)² / (2·0.3²))
            shift_gate = σ(h − h_start) − σ(h − h_end)
            λ(t) = gate_m · λ_maint + (1 − gate_m) · [gate_s · λ_ext + (1 − gate_s) · λ_idle]

    Attributes:
        lambda_extraction: Active extraction rate [events day⁻¹]. Typical: 5–20.
        lambda_maintenance: Maintenance shutdown rate [events day⁻¹]. Typical: 0.5–2.
        lambda_idle: Nightly idle rate [events day⁻¹]. Typical: 1–4.
        shift_start: Shift start [hours]. Default: 6.0.
        shift_end: Shift end [hours]. Default: 22.0.
        maintenance_day: Day of 7-day week for maintenance [0=Mon, 6=Sun]. Default: 6.0.

    Priors (sample_priors):
        λ_ext   ~ HalfNormal(10.0)
        λ_maint ~ HalfNormal(1.5)
        λ_idle  ~ HalfNormal(3.0)
    """

    lambda_extraction: Float[ArrayLike, ""]
    lambda_maintenance: Float[ArrayLike, ""]
    lambda_idle: Float[ArrayLike, ""]
    shift_start: Float[ArrayLike, ""] = 6.0
    shift_end: Float[ArrayLike, ""] = 22.0
    maintenance_day: Float[ArrayLike, ""] = 6.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹]. Shape ``(N,)``.
        """
        # d = t mod 7
        day_of_week = t % 7.0
        # shift = σ(h − h_start) − σ(h − h_end)
        shift = _shift_gate(t, self.shift_start, self.shift_end)
        # m = exp(−½ ((d − d_maint) / 0.3)²)
        maint = jnp.exp(-0.5 * ((day_of_week - self.maintenance_day) / 0.3) ** 2)
        return maint * self.lambda_maintenance + (1.0 - maint) * (
            shift * self.lambda_extraction + (1.0 - shift) * self.lambda_idle
        )

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = |λ_maint| + |λ_ext| + 2|λ_idle|, since gate_m ∈ [0, 1] and gate_s ∈ (−1, 1).

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            jnp.abs(self.lambda_maintenance)
            + jnp.abs(self.lambda_extraction)
            + 2.0 * jnp.abs(self.lambda_idle)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_extraction": ("lambda_extraction", lambda dist: dist.HalfNormal(10.0)),
        "lambda_maintenance": ("lambda_maintenance", lambda dist: dist.HalfNormal(1.5)),
        "lambda_idle": ("lambda_idle", lambda dist: dist.HalfNormal(3.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 10. LANDFILL (Seasonal + Diurnal + Barometric Pump)
# ═════════════════════════════════════════════════════════════════════════════


class LandfillIntensity(PriorTableMixin, eqx.Module):
    """Multi-component intensity for municipal solid waste landfills.

    Physical Motivation:
        Three superimposed physical mechanisms:
          1. Seasonal temperature (annual): soil T governs microbial activity.
             Summer peak → 2–3× winter rates.
          2. Diurnal thermal (daily): solar heating warms cover soil, enhancing
             gas diffusion. Peak ≈ 15:00 local.
          3. Barometric pumping (synoptic, ~3–7 days): passing low-pressure
             systems literally suck CH₄ out through the cover soil.

        Observable by all satellite tiers: TROPOMI (regional), MethaneSAT
        (complex-scale), GHGSat/Tanager (cell-level).

    Equation:
            λ(t) = max(0, λ₀ + A_s·cos(ω_s(t−t_pk)) + A_d·cos(ω_d·t−φ_d) + A_b·sin(ω_b·t))
            ω_s=2π/365.25, ω_d=2π, ω_b=2π/P_baro

    Attributes:
        lambda_0: Annual mean [events day⁻¹]. Typical: 2–6.
        amp_seasonal: Seasonal half-amplitude [events day⁻¹]. Typical: 1–3.
        amp_diurnal: Diurnal half-amplitude [events day⁻¹]. Typical: 0.5–2.
        amp_barometric: Barometric pumping half-amplitude [events day⁻¹]. Typical: 0.5–1.5.
        baro_period_days: Synoptic weather period [days]. Default: 5.0.
        peak_day: Seasonal peak [days from Jan 1]. Default: 196.0.
        phase_hours: Diurnal peak hour [hours]. Default: 15.0.

    Priors (sample_priors):
        λ₀      ~ HalfNormal(4.0)
        A_s     ~ HalfNormal(2.0)
        A_d     ~ HalfNormal(1.5)
        A_b     ~ HalfNormal(1.0)
        P_baro  ~ LogNormal(ln5, 0.3)
        t_peak  ~ Normal(196, 20)
        φ_h     ~ Normal(15, 1.5)
    """

    lambda_0: Float[ArrayLike, ""]
    amp_seasonal: Float[ArrayLike, ""]
    amp_diurnal: Float[ArrayLike, ""]
    amp_barometric: Float[ArrayLike, ""]
    baro_period_days: Float[ArrayLike, ""] = 5.0
    peak_day: Float[ArrayLike, ""] = 196.0
    phase_hours: Float[ArrayLike, ""] = 15.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        ω_d = 2.0 * jnp.pi
        ω_b = 2.0 * jnp.pi / self.baro_period_days
        φ_d = _diurnal_phase(self.phase_hours)
        raw = (
            self.lambda_0
            + self.amp_seasonal * jnp.cos(_seasonal_phase(t, self.peak_day))
            + self.amp_diurnal * jnp.cos(ω_d * t - φ_d)
            + self.amp_barometric * jnp.sin(ω_b * t)
        )
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + |A_s| + |A_d| + |A_b|.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_0
            + jnp.abs(self.amp_seasonal)
            + jnp.abs(self.amp_diurnal)
            + jnp.abs(self.amp_barometric)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(4.0)),
        "amp_seasonal": ("amp_seasonal", lambda dist: dist.HalfNormal(2.0)),
        "amp_diurnal": ("amp_diurnal", lambda dist: dist.HalfNormal(1.5)),
        "amp_barometric": ("amp_barometric", lambda dist: dist.HalfNormal(1.0)),
        "baro_period_days": (
            "baro_period_days",
            lambda dist: dist.LogNormal(jnp.log(5.0), 0.3),
        ),
        "peak_day": ("peak_day", lambda dist: dist.Normal(196.0, 20.0)),
        "phase_hours": ("phase_hours", lambda dist: dist.Normal(15.0, 1.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 11. OFFSHORE PLATFORM (Tidal + Operational)
# ═════════════════════════════════════════════════════════════════════════════


class OffshorePlatformIntensity(PriorTableMixin, eqx.Module):
    """Combined tidal and operational intensity for offshore oil & gas platforms.

    Physical Motivation:
        Semi-diurnal tides (M2: ~12.42 hr) modulate subsurface hydrostatic
        pressure, altering gas migration through sediments. Crew shift schedules
        simultaneously drive compressor cycling and flaring. These two independent
        forcings superimpose.

        Observable from Sentinel-2 (2–5 day, flare detection) and GHGSat/Tanager
        (hyperspectral, fugitive plume quantification).

    Equation:
            λ(t) = max(0, λ₀ + A_tide·sin(2π/P_tide · t) + A_ops·[σ(h−h_on)−σ(h−h_off)])

    Attributes:
        lambda_0: Baseline platform rate [events day⁻¹]. Typical: 3–8.
        amp_tidal: Tidal half-amplitude [events day⁻¹]. Typical: 0.5–2.
        amp_operational: Crew-shift half-amplitude [events day⁻¹]. Typical: 1–4.
        tidal_period_days: Dominant tidal period [days]. Default: 0.5175 (M2 ≈ 12.42 hr).
        shift_start: Day crew start [hours]. Default: 7.0.
        shift_end: Day crew end [hours]. Default: 19.0.

    Priors (sample_priors):
        λ₀     ~ HalfNormal(5.0)
        A_tide ~ HalfNormal(1.5)
        A_ops  ~ HalfNormal(2.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amp_tidal: Float[ArrayLike, ""]
    amp_operational: Float[ArrayLike, ""]
    tidal_period_days: Float[ArrayLike, ""] = 0.5175
    shift_start: Float[ArrayLike, ""] = 7.0
    shift_end: Float[ArrayLike, ""] = 19.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        # crew = σ(h − h_on) − σ(h − h_off)
        crew = _shift_gate(t, self.shift_start, self.shift_end)
        raw = (
            self.lambda_0
            + self.amp_tidal * jnp.sin(2.0 * jnp.pi / self.tidal_period_days * t)
            + self.amp_operational * crew
        )
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + |A_tide| + |A_ops|, since the crew gate lies in (−1, 1).

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_0 + jnp.abs(self.amp_tidal) + jnp.abs(self.amp_operational)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(5.0)),
        "amp_tidal": ("amp_tidal", lambda dist: dist.HalfNormal(1.5)),
        "amp_operational": ("amp_operational", lambda dist: dist.HalfNormal(2.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 12. WETLAND / PERMAFROST (Sigmoid-Gated Seasonal)
# ═════════════════════════════════════════════════════════════════════════════


class WetlandPermafrostIntensity(PriorTableMixin, eqx.Module):
    """Sigmoid-gated seasonal intensity for high-latitude wetland/permafrost.

    Physical Motivation:
        Arctic/boreal wetlands (~200 Mt CH₄ yr⁻¹) are gated by a binary physical
        constraint: soil must be thawed and saturated for methanogenesis. When the
        active layer freezes (Oct–Apr), emissions collapse to near zero. Thaw
        onset (May) ramps emissions over ~2–4 weeks. This sharp on/off is
        fundamentally different from a smooth sinusoid.

        Observable by TROPOMI (daily, 7 km — diffuse wetland complexes),
        MethaneSAT (weekly, 200 m — thermokarst lake clusters), GOSAT (~3 day).

    Equation:
            G_thaw(t)  = σ((t_yr − t_thaw) / w)
            G_freeze(t) = 1 − σ((t_yr − t_freeze) / w)
            λ(t) = λ_frozen + (λ_peak − λ_frozen) · G_thaw · G_freeze
            t_yr = t mod 365.25

    Attributes:
        lambda_peak: Peak summer rate [events day⁻¹]. Typical: 2–8.
        lambda_frozen: Winter baseline (residual ebullition) [events day⁻¹]. Typical: 0.01–0.3.
        thaw_onset_day: Day-of-year of thaw onset [days]. Default: 120 (≈ May 1).
        freeze_onset_day: Day-of-year of freeze onset [days]. Default: 280 (≈ Oct 7).
        transition_width: Sigmoid transition width [days]. Default: 15.0.

    Priors (sample_priors):
        λ_peak   ~ HalfNormal(5.0)
        λ_frozen ~ HalfNormal(0.2)
        t_thaw   ~ Normal(120, 15)
        t_freeze ~ Normal(280, 15)
        w        ~ HalfNormal(15.0)
    """

    lambda_peak: Float[ArrayLike, ""]
    lambda_frozen: Float[ArrayLike, ""]
    thaw_onset_day: Float[ArrayLike, ""] = 120.0
    freeze_onset_day: Float[ArrayLike, ""] = 280.0
    transition_width: Float[ArrayLike, ""] = 15.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹]. Shape ``(N,)``.
        """
        # t_yr = t mod 365.25
        day_of_year = t % 365.25
        width = self.transition_width
        # G_thaw = σ((t_yr − t_thaw) / w)
        g_thaw = jax.nn.sigmoid((day_of_year - self.thaw_onset_day) / width)
        # G_freeze = 1 − σ((t_yr − t_freeze) / w)
        g_freeze = 1.0 - jax.nn.sigmoid((day_of_year - self.freeze_onset_day) / width)
        return (
            self.lambda_frozen
            + (self.lambda_peak - self.lambda_frozen) * g_thaw * g_freeze
        )

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = max(λ_frozen, λ_peak), since G_thaw · G_freeze ∈ (0, 1).

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(jnp.maximum(self.lambda_frozen, self.lambda_peak))

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_peak": ("lambda_peak", lambda dist: dist.HalfNormal(5.0)),
        "lambda_frozen": ("lambda_frozen", lambda dist: dist.HalfNormal(0.2)),
        "thaw_onset_day": ("thaw_onset_day", lambda dist: dist.Normal(120.0, 15.0)),
        "freeze_onset_day": ("freeze_onset_day", lambda dist: dist.Normal(280.0, 15.0)),
        "transition_width": ("transition_width", lambda dist: dist.HalfNormal(15.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 13. LIVESTOCK FEEDLOT (Bimodal Feeding + Seasonal)
# ═════════════════════════════════════════════════════════════════════════════


class LivestockFeedlotIntensity(PriorTableMixin, eqx.Module):
    """Bimodal diurnal + seasonal intensity for concentrated livestock operations.

    Physical Motivation:
        Enteric fermentation in cattle (~100–120 kg CH₄ cow⁻¹ yr⁻¹) peaks 1–3
        hours after feeding as rumen microbial activity surges. Large CAFOs have
        fixed AM/PM feeding schedules → characteristic bimodal diurnal pattern.
        Seasonal modulation from temperature-dependent manure lagoon emissions
        and feed composition changes.

        Observable at facility scale by GHGSat (25 m), Tanager (30 m), and
        regionally by TROPOMI and MethaneSAT.

    Equation:
            G_am(t) = exp(−(h − h_am)² / (2w²))
            G_pm(t) = exp(−(h − h_pm)² / (2w²))
            λ(t) = max(0, λ₀ + A_f·(G_am + G_pm) + A_s·cos(ω_s(t − t_peak)))
            h = (t mod 1)·24

    Attributes:
        lambda_0: Baseline enteric rate [events day⁻¹]. Typical: 1–4.
        amp_feeding: Post-feeding surge amplitude [events day⁻¹]. Typical: 1–3.
        amp_seasonal: Seasonal half-amplitude [events day⁻¹]. Typical: 0.5–1.5.
        morning_feed_hour: AM feed time [hours]. Default: 7.0.
        evening_feed_hour: PM feed time [hours]. Default: 17.0.
        feed_response_width: Gaussian width of feeding pulse [hours]. Default: 2.0.
        peak_day: Seasonal peak [days from Jan 1]. Default: 196.0.

    Priors (sample_priors):
        λ₀   ~ HalfNormal(3.0)
        A_f  ~ HalfNormal(2.0)
        A_s  ~ HalfNormal(1.0)
        h_am ~ Normal(7.0, 1.0)
        h_pm ~ Normal(17.0, 1.0)
        w    ~ HalfNormal(2.0)
    """

    lambda_0: Float[ArrayLike, ""]
    amp_feeding: Float[ArrayLike, ""]
    amp_seasonal: Float[ArrayLike, ""]
    morning_feed_hour: Float[ArrayLike, ""] = 7.0
    evening_feed_hour: Float[ArrayLike, ""] = 17.0
    feed_response_width: Float[ArrayLike, ""] = 2.0
    peak_day: Float[ArrayLike, ""] = 196.0

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate intensity.

        Args:
            t: Time coordinates [days]. Shape ``(N,)``.

        Returns:
            Intensity [events day⁻¹], clamped ≥ 0. Shape ``(N,)``.
        """
        # h = (t mod 1) · 24
        hour_of_day = _hour_of_day(t)
        width = self.feed_response_width
        # G_am = exp(−(h − h_am)² / (2w²))
        morning_gate = jnp.exp(
            -0.5 * ((hour_of_day - self.morning_feed_hour) / width) ** 2
        )
        # G_pm = exp(−(h − h_pm)² / (2w²))
        evening_gate = jnp.exp(
            -0.5 * ((hour_of_day - self.evening_feed_hour) / width) ** 2
        )
        seasonal = self.amp_seasonal * jnp.cos(_seasonal_phase(t, self.peak_day))
        raw = (
            self.lambda_0 + self.amp_feeding * (morning_gate + evening_gate) + seasonal
        )
        return jnp.clip(raw, min=0.0)

    def max_intensity(self) -> jnp.ndarray:
        """Upper bound on λ(t) over all t, the envelope of a thinning sampler.

        λ_max = λ₀ + 2|A_f| + |A_s|, since each feeding gate is ≤ 1.

        Returns:
            The bound [events day⁻¹].
        """
        return jnp.asarray(
            self.lambda_0 + 2.0 * jnp.abs(self.amp_feeding) + jnp.abs(self.amp_seasonal)
        )

    _PRIORS: ClassVar[PriorTable] = {
        "lambda_0": ("lambda_0", lambda dist: dist.HalfNormal(3.0)),
        "amp_feeding": ("amp_feeding", lambda dist: dist.HalfNormal(2.0)),
        "amp_seasonal": ("amp_seasonal", lambda dist: dist.HalfNormal(1.0)),
        "morning_feed_hour": ("morning_feed_hour", lambda dist: dist.Normal(7.0, 1.0)),
        "evening_feed_hour": ("evening_feed_hour", lambda dist: dist.Normal(17.0, 1.0)),
        "feed_response_width": (
            "feed_response_width",
            lambda dist: dist.HalfNormal(2.0),
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

INTENSITY_REGISTRY = {
    "constant": ConstantIntensity,
    "diurnal": DiurnalSinusoidalIntensity,
    "seasonal": SeasonalSinusoidalIntensity,
    "diurnal_seasonal": DiurnalSeasonalIntensity,
    "modulated_diurnal": SeasonallyModulatedDiurnalIntensity,
    "operational": OperationalScheduleIntensity,
    "weibull_renewal": WeibullRenewalIntensity,
    "periodic_batch": PeriodicBatchIntensity,
    "coal_mine": CoalMineVentilationIntensity,
    "landfill": LandfillIntensity,
    "offshore": OffshorePlatformIntensity,
    "wetland_permafrost": WetlandPermafrostIntensity,
    "livestock_feedlot": LivestockFeedlotIntensity,
}
"""str → Module class mapping for dynamic dispatch and configuration files."""
