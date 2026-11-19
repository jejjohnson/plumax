"""Tier V.B — probability-of-detection (POD) models as Equinox modules.

Here "POD" means **probability of detection** — the chance that a satellite
detects a methane plume of a given size under given conditions. (It is
unrelated to the proper orthogonal decomposition used for reduced-order
modelling elsewhere in the docs.)

A family of GLM-style POD models ``P_d(x) ∈ [0, 1]`` for methane plume
detectability from satellite remote sensing, packaged as ``eqx.Module``.
Each module bundles:

* covariate transformation + link function in ``__call__``;
* ``sample_priors(cls)`` — a NumPyro prior factory for MCMC.

The progression mirrors :mod:`plumax.population.intensity`: simple →
complex, each model relaxing an assumption of the last.

Detection is not a deterministic threshold. It is a probabilistic event
governed by the interplay of source strength, atmospheric dilution,
instrument optics and radiometric conditions, modelled as a Bernoulli GLM
``Y ~ Bernoulli(P_d(x))`` where ``Y ∈ {0, 1}`` (detected / not detected)
and ``P_d`` is evaluated at covariates ``x``.

Conventions:

* Flux ``Q`` in [kg hr⁻¹]; wind speed ``U`` in [m s⁻¹]; pixel size ``p``
  in [m]; albedo ``A`` in [–], ``∈ [0, 1]``.
* Solar zenith angle ``θ`` in [°]; models use ``cos(θ) ∈ [0.26, 1.0]``.
* Spectral bands ``n_bands`` [dimensionless count]; noise ``σ_noise`` in
  [ppb] or instrument-specific units.
* All modules are immutable pytrees, jit / vmap / grad compatible.

NumPyro is only imported inside ``sample_priors``, so evaluating a POD
curve never pulls it in.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import erf as _jax_erf
from jax.typing import ArrayLike
from jaxtyping import Float

from plumax.population._src._priors import PriorTable, PriorTableMixin
from plumax.population._src.pod_union import (
    TOD_BINS,
    InstrumentPOD,
    calibration_from_draw,
    calibration_prior,
    load_pod_calibrations,
    log_pod_union,
    pod_union,
    tod_bin_index,
)


__all__ = [
    "POD_REGISTRY",
    "TOD_BINS",
    "AdditiveMultiCovariatePOD",
    "CloglogPOD",
    "ConcentrationProxyPOD",
    "FullVaryingCoefficientPOD",
    "InstrumentPOD",
    "LogLogisticPOD",
    "LogisticPOD",
    "ProbitPOD",
    "SNRBasedPOD",
    "SpectralAwarePOD",
    "VaryingCoefficientPOD",
    "calibration_from_draw",
    "calibration_prior",
    "cos_zenith",
    "load_pod_calibrations",
    "log_concentration_proxy",
    "log_flux",
    "log_flux_wind_ratio",
    "log_pod_union",
    "pod_union",
    "snr_proxy",
    "tod_bin_index",
]


# ═════════════════════════════════════════════════════════════════════════════
# COVARIATE TRANSFORMS
# ═════════════════════════════════════════════════════════════════════════════
# Standalone pure functions: each maps raw physical observables to the
# feature space consumed by the GLM linear predictor.
# All are jit/vmap/grad safe.
# ═════════════════════════════════════════════════════════════════════════════


def log_flux(Q: jnp.ndarray) -> jnp.ndarray:
    """Log-transform of methane mass flux.

    Physical Motivation:
        Flux spans 4+ orders of magnitude (1–10 000 kg hr⁻¹). The log
        compresses the super-emitter tail and linearises the relationship
        with detection probability in logit space.

    Equation:
            x = ln(Q)

    Args:
        Q: Methane mass flux [kg hr⁻¹]. Must be > 0. Shape ``(N,)``.

    Returns:
        Log-flux [ln(kg hr⁻¹)]. Shape ``(N,)``.
    """
    return jnp.log(jnp.clip(Q, min=1e-8))


def log_concentration_proxy(
    Q: jnp.ndarray,
    U: jnp.ndarray,
    p: jnp.ndarray,
) -> jnp.ndarray:
    """Log concentration proxy combining flux, wind, and pixel size.

    Physical Motivation:
        What a satellite actually "sees" is the column-integrated methane
        enhancement ΔΩ ∝ Q / (U · p). This proxy captures the three dominant
        first-order controls on detectability in a single dimensionless
        scalar:
          • Q ↑ → more CH₄ in the pixel → easier to detect.
          • U ↑ → plume diluted faster → harder to detect.
          • p ↑ → more background in the pixel → fill-factor dilution.

        Used by Varon et al. (2018), Cusworth et al. (2021).

    Equation:
            x = ln(Q / (U · p)) = ln(Q) − ln(U) − ln(p)

    Args:
        Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
        U: Wind speed [m s⁻¹]. Shape ``(N,)``.
        p: Pixel ground sampling distance [m]. Shape ``(N,)``.

    Returns:
        Log concentration proxy [dimensionless]. Shape ``(N,)``.
    """
    return (
        jnp.log(jnp.clip(Q, min=1e-8))
        - jnp.log(jnp.clip(U, min=1e-3))
        - jnp.log(jnp.clip(p, min=1.0))
    )


def log_flux_wind_ratio(
    Q: jnp.ndarray,
    U: jnp.ndarray,
) -> jnp.ndarray:
    """Log flux-to-wind ratio (pixel-free proxy).

    Physical Motivation:
        When pixel size is fixed for a single instrument, the Q/U ratio alone
        governs the column enhancement. This is the proxy used by the IMEO
        james_jax.py POD fits (Q/U in [kg hr⁻¹ / (m s⁻¹)]).

    Equation:
            x = ln(Q / U)

    Args:
        Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
        U: Wind speed [m s⁻¹]. Shape ``(N,)``.

    Returns:
        Log flux-wind ratio [ln(kg hr⁻¹ / (m s⁻¹))]. Shape ``(N,)``.
    """
    return jnp.log(jnp.clip(Q, min=1e-8)) - jnp.log(jnp.clip(U, min=1e-3))


def snr_proxy(
    Q: jnp.ndarray,
    U: jnp.ndarray,
    p: jnp.ndarray,
    albedo: jnp.ndarray,
    sigma_noise: jnp.ndarray,
) -> jnp.ndarray:
    """Signal-to-noise ratio proxy for radiometric detection.

    Physical Motivation:
        Detection fundamentally reduces to SNR. The signal is the column
        enhancement ΔΩ ∝ Q/(U·p). The noise floor depends on:
          • Albedo A: brighter surfaces → more reflected photons → lower
            shot noise → better SNR.
          • Instrument noise σ_n: dark current, read noise, spectral fitting
            residuals.

    Equation:
            SNR = (Q / (U · p)) · √A / σ_noise

        The model consumes ln(SNR) to linearise the logit relationship.

    Args:
        Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
        U: Wind speed [m s⁻¹]. Shape ``(N,)``.
        p: Pixel size [m]. Shape ``(N,)``.
        albedo: Surface albedo [–]. Shape ``(N,)``.
        sigma_noise: Instrument noise floor [ppb or instrument units]. Shape ``(N,)``.

    Returns:
        ln(SNR) [dimensionless]. Shape ``(N,)``.
    """
    signal = jnp.clip(Q, min=1e-8) / (jnp.clip(U, min=1e-3) * jnp.clip(p, min=1.0))
    noise_inv = jnp.sqrt(jnp.clip(albedo, min=1e-4)) / jnp.clip(sigma_noise, min=1e-6)
    return jnp.log(signal * noise_inv + 1e-12)


def cos_zenith(theta_deg: jnp.ndarray) -> jnp.ndarray:
    """Cosine of solar zenith angle.

    Physical Motivation:
        Solar zenith θ enters the radiative transfer equation through the
        air-mass factor 1/cos(θ). Higher sun (smaller θ, larger cos θ)
        means shorter path length → less atmospheric scattering → cleaner
        retrieval. Most instruments reject observations at θ > 70°.

    Equation:
            x = cos(θ · π / 180)

    Args:
        theta_deg: Solar zenith angle [degrees]. Shape ``(N,)``.

    Returns:
        cos(θ) [dimensionless], ∈ [0.26, 1.0] for θ ∈ [0°, 75°]. Shape ``(N,)``.
    """
    return jnp.cos(jnp.deg2rad(theta_deg))


# ═════════════════════════════════════════════════════════════════════════════
# 1. LOGISTIC (Simplest GLM — 1D Sigmoid)
# ═════════════════════════════════════════════════════════════════════════════


class LogisticPOD(PriorTableMixin, eqx.Module):
    """Single-covariate logistic POD: the classic S-curve.

    Physical Motivation:
        The simplest detection model: probability depends only on flux Q
        through a logistic sigmoid. This is the "Q₅₀" model universally
        used in first-pass satellite characterisation. The satellite has
        a detection midpoint Q₅₀ where P_d = 0.5, and a steepness k
        controlling how sharply the curve transitions from 0 → 1.

        Ignores wind, albedo, zenith — appropriate for controlled-release
        experiments under constant atmospheric conditions or as a baseline.

    Equation:
            P_d(Q) = σ(k · (Q − Q₅₀)) = 1 / (1 + exp(−k · (Q − Q₅₀)))

        Equivalent logit form:
            logit(P_d) = k · (Q − Q₅₀) = k·Q − k·Q₅₀ = β₁·Q + β₀

    Attributes:
        Q_50: Flux at 50% detection probability [kg hr⁻¹].
            Typical: GHGSat ≈ 100, TROPOMI ≈ 3000, MethaneSAT ≈ 200.
        k: Steepness of the sigmoid [hr kg⁻¹].
            Typical: 0.005–0.05.

    Priors (sample_priors):
        Q₅₀ ~ LogNormal(ln(500), 1.0)   [kg hr⁻¹]
        k   ~ HalfNormal(0.02)           [hr kg⁻¹]
    """

    Q_50: Float[ArrayLike, ""]
    k: Float[ArrayLike, ""]

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        return jax.nn.sigmoid(self.k * (Q - self.Q_50))

    _PRIORS: ClassVar[PriorTable] = {
        "Q_50": ("Q_50", lambda dist: dist.LogNormal(jnp.log(500.0), 1.0)),
        "k": ("k", lambda dist: dist.HalfNormal(0.02)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOG-LOGISTIC (Log-Flux GLM)
# ═════════════════════════════════════════════════════════════════════════════


class LogLogisticPOD(PriorTableMixin, eqx.Module):
    """Logistic POD on log-flux: the lognormal CDF approximation.

    Physical Motivation:
        Because flux spans orders of magnitude, the logistic curve on raw Q
        is poorly calibrated: it transitions too slowly across the super-
        emitter tail. Operating in log-space produces the lognormal CDF
        shape used by IMEO/Kamdar et al. for per-satellite POD fits.

        This is the model behind james_jax.py: P_d(Q) ≈ Φ_LN(Q; Q₅₀, σ).

    Equation:
            logit(P_d) = β₀ + β₁ · ln(Q)

        Equivalently, the transition midpoint is at Q = exp(−β₀/β₁).

    Attributes:
        beta_0: Intercept [logit units]. Controls the detection threshold.
        beta_1: Slope w.r.t. ln(Q) [logit per ln(kg hr⁻¹)].
            Positive → higher flux → more detectable.

    Priors (sample_priors):
        β₀ ~ Normal(−8.0, 3.0)    [logit]
        β₁ ~ HalfNormal(2.0)      [logit per ln(kg hr⁻¹)]
    """

    beta_0: Float[ArrayLike, ""]
    beta_1: Float[ArrayLike, ""]

    def __call__(self, Q: jnp.ndarray) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        logit = self.beta_0 + self.beta_1 * log_flux(Q)
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-8.0, 3.0)),
        "beta_1": ("beta_1", lambda dist: dist.HalfNormal(2.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3. CONCENTRATION-PROXY GLM (Flux + Wind + Pixel)
# ═════════════════════════════════════════════════════════════════════════════


class ConcentrationProxyPOD(PriorTableMixin, eqx.Module):
    """GLM on the log-concentration proxy ln(Q/(U·p)).

    Physical Motivation:
        The column-integrated methane enhancement ΔΩ ∝ Q/(U·p) is the
        primary observable. This model captures the three dominant controls
        in a single log-linear predictor. Assumes their effects are
        multiplicatively separable in concentration space (additive in
        log-space).

        Used in Varon et al. (2018), the canonical IME framework.

    Equation:
            logit(P_d) = β₀ + β₁ · ln(Q / (U · p))

    Attributes:
        beta_0: Intercept [logit units].
        beta_1: Sensitivity to log-concentration proxy
            [logit per dimensionless unit].

    Priors (sample_priors):
        β₀ ~ Normal(−3.0, 2.0)   [logit]
        β₁ ~ HalfNormal(2.0)     [logit per unit]
    """

    beta_0: Float[ArrayLike, ""]
    beta_1: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)
        # logit = β₀ + β₁ x
        logit = self.beta_0 + self.beta_1 * log_proxy
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-3.0, 2.0)),
        "beta_1": ("beta_1", lambda dist: dist.HalfNormal(2.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. ADDITIVE MULTI-COVARIATE GLM (Separable Effects)
# ═════════════════════════════════════════════════════════════════════════════


class AdditiveMultiCovariatePOD(PriorTableMixin, eqx.Module):
    """Additive GLM with separate coefficients for each covariate.

    Physical Motivation:
        Relaxes the single-proxy assumption: flux, wind, pixel size, albedo,
        and illumination each receive an independent coefficient. This allows
        the model to learn that, e.g., wind has a stronger dilution effect
        than pixel size, or that albedo matters more than zenith angle.

        The linear predictor is a standard multivariate logistic regression.

    Equation:
            logit(P_d) = β₀ + β_Q·ln(Q) + β_U·ln(U) + β_p·ln(p)
                            + β_A·A + β_θ·cos(θ)

        Expected sign constraints (encoded in priors):
          • β_Q > 0  (more flux → more detectable)
          • β_U < 0  (more wind → more dilution → less detectable)
          • β_p < 0  (larger pixel → more fill-factor dilution)
          • β_A > 0  (brighter surface → better SNR)
          • β_θ > 0  (higher sun → less atmospheric path → cleaner)

    Attributes:
        beta_0: Intercept [logit].
        beta_Q: Log-flux coefficient [logit per ln(kg hr⁻¹)].
        beta_U: Log-wind coefficient [logit per ln(m s⁻¹)].
        beta_p: Log-pixel coefficient [logit per ln(m)].
        beta_A: Albedo coefficient [logit per unit albedo].
        beta_theta: Cosine-zenith coefficient [logit per unit cos(θ)].

    Priors (sample_priors):
        Sign-constrained where physics dictates:
        β₀  ~ Normal(−5.0, 3.0)
        β_Q ~ HalfNormal(2.0)       (positive: more flux → detected)
        β_U ~ Normal(−1.0, 1.0)     (negative: dilution)
        β_p ~ Normal(−0.5, 0.5)     (negative: fill-factor)
        β_A ~ HalfNormal(1.5)       (positive: brighter → better)
        β_θ ~ HalfNormal(1.0)       (positive: higher sun → better)
    """

    beta_0: Float[ArrayLike, ""]
    beta_Q: Float[ArrayLike, ""]
    beta_U: Float[ArrayLike, ""]
    beta_p: Float[ArrayLike, ""]
    beta_A: Float[ArrayLike, ""]
    beta_theta: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.
            albedo: Surface albedo [–]. Shape ``(N,)``.
            cos_theta: Cosine of solar zenith angle [–]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        logit = (
            self.beta_0
            + self.beta_Q * log_flux(Q)
            + self.beta_U * jnp.log(jnp.clip(U, min=1e-3))
            + self.beta_p * jnp.log(jnp.clip(p, min=1.0))
            + self.beta_A * albedo
            + self.beta_theta * cos_theta
        )
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-5.0, 3.0)),
        "beta_Q": ("beta_Q", lambda dist: dist.HalfNormal(2.0)),
        "beta_U": ("beta_U", lambda dist: dist.Normal(-1.0, 1.0)),
        "beta_p": ("beta_p", lambda dist: dist.Normal(-0.5, 0.5)),
        "beta_A": ("beta_A", lambda dist: dist.HalfNormal(1.5)),
        "beta_theta": ("beta_theta", lambda dist: dist.HalfNormal(1.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 5. VARYING COEFFICIENT GLM (Environment-Dependent Sensitivity)
# ═════════════════════════════════════════════════════════════════════════════


class VaryingCoefficientPOD(PriorTableMixin, eqx.Module):
    """Varying coefficient GLM: sensitivity β₁ depends on environment.

    Physical Motivation:
        The slope of the detection curve is NOT constant — it changes with
        observing conditions. Under high albedo and overhead sun, the
        instrument's spectral retrieval is cleaner, so the sigmoid is
        *steeper* (the satellite becomes a better detector). Under low
        albedo or high zenith, the slope flattens.

        This is the "Refined Varying Coefficient Formulation" from the
        pod_numpyro.py module:

            β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ)

        The key insight: albedo and zenith don't shift the curve left/right
        (that's the additive model) — they change *how sharply* the
        instrument discriminates between detectable and undetectable plumes.

    Equation:
            β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ)
            logit(P_d) = β₀ + β₁(Z) · ln(Q / (U · p))

    Attributes:
        beta_0: Baseline threshold [logit]. Typical: [−6, −2].
        gamma_base: Intrinsic instrument sensitivity [logit per log-proxy-unit].
            Typical: [0.5, 3.0].
        gamma_albedo: Albedo enhancement [logit per log-unit per unit albedo].
            Typical: [0.5, 2.0].
        gamma_cos_theta: Illumination enhancement [logit per log-unit per unit cos(θ)].
            Typical: [0.3, 1.5].

    Priors (sample_priors):
        β₀  ~ Normal(−3.5, 2.0)
        γ_b ~ HalfNormal(1.5)
        γ_A ~ HalfNormal(1.0)
        γ_θ ~ Normal(0.5, 0.5)
    """

    beta_0: Float[ArrayLike, ""]
    gamma_base: Float[ArrayLike, ""]
    gamma_albedo: Float[ArrayLike, ""]
    gamma_cos_theta: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.
            albedo: Surface albedo [–]. Shape ``(N,)``.
            cos_theta: Cosine of solar zenith angle [–]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)
        # β₁ = γ₀ + γ_a · albedo + γ_θ · cos θ
        slope = (
            self.gamma_base
            + self.gamma_albedo * albedo
            + self.gamma_cos_theta * cos_theta
        )
        # logit = β₀ + β₁ x
        logit = self.beta_0 + slope * log_proxy
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-3.5, 2.0)),
        "gamma_base": ("gamma_base", lambda dist: dist.HalfNormal(1.5)),
        "gamma_albedo": ("gamma_albedo", lambda dist: dist.HalfNormal(1.0)),
        "gamma_cos_theta": ("gamma_cos_theta", lambda dist: dist.Normal(0.5, 0.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 6. SNR-BASED GLM (Radiometric First-Principles)
# ═════════════════════════════════════════════════════════════════════════════


class SNRBasedPOD(PriorTableMixin, eqx.Module):
    """GLM on log-SNR: radiometric first-principles detection model.

    Physical Motivation:
        At the hardware level, detection reduces to a hypothesis test on the
        retrieved methane column: "is the enhancement above the noise floor?"
        The test statistic is the signal-to-noise ratio:

            SNR = ΔΩ / σ_retrieval ∝ (Q / (U·p)) · √A / σ_noise

        where σ_noise encapsulates dark current, read noise, spectral fitting
        residuals, and atmospheric interference. Detection occurs when
        SNR > τ, which in the probabilistic GLM becomes:

            P_d = σ(β₀ + β₁ · ln(SNR))

        This model is appropriate when instrument noise characteristics are
        known or co-estimated, e.g., from calibration campaigns.

    Equation:
            SNR = (Q / (U · p)) · √A / σ_noise
            logit(P_d) = β₀ + β₁ · ln(SNR)

    Attributes:
        beta_0: Intercept [logit]. Encodes the detection threshold in SNR space.
        beta_1: SNR sensitivity [logit per ln(SNR)].

    Priors (sample_priors):
        β₀ ~ Normal(−2.0, 2.0)   [logit]
        β₁ ~ HalfNormal(2.0)     [logit per ln(SNR)]
    """

    beta_0: Float[ArrayLike, ""]
    beta_1: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        sigma_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.
            albedo: Surface albedo [–]. Shape ``(N,)``.
            sigma_noise: Instrument noise [ppb or retrieval units]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_snr = snr_proxy(Q, U, p, albedo, sigma_noise)
        # logit = β₀ + β₁ x
        logit = self.beta_0 + self.beta_1 * log_snr
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-2.0, 2.0)),
        "beta_1": ("beta_1", lambda dist: dist.HalfNormal(2.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. SPECTRAL-AWARE GLM (Hyperspectral vs. Multispectral)
# ═════════════════════════════════════════════════════════════════════════════


class SpectralAwarePOD(PriorTableMixin, eqx.Module):
    """GLM with spectral band count and resolution as covariates.

    Physical Motivation:
        Hyperspectral imagers (GHGSat: ~200 bands in SWIR, PRISMA: 250,
        EnMAP: 230) resolve individual CH₄ absorption lines, enabling
        matched-filter retrievals with far lower detection limits than
        multispectral instruments (Sentinel-2: ~13 bands, Landsat: ~11).
        The number of spectral channels in the CH₄-sensitive window
        (1600–1700 nm or 2300–2400 nm) directly governs retrieval precision.

        Spectral resolution Δλ [nm] determines how well absorption lines
        are resolved. Finer resolution → better spectral discrimination →
        lower false alarm rate.

    Equation:
            logit(P_d) = β₀ + β₁ · ln(Q/(U·p)) + β_n · ln(n_bands)
                            + β_Δλ · ln(Δλ) + β_A · A

    Attributes:
        beta_0: Intercept [logit].
        beta_proxy: Concentration proxy sensitivity [logit per unit].
        beta_n_bands: Spectral channel count effect [logit per ln(count)].
        beta_spectral_res: Spectral resolution effect [logit per ln(nm)].
        beta_albedo: Albedo effect [logit per unit albedo].

    Priors (sample_priors):
        β₀   ~ Normal(−5.0, 3.0)
        β_x  ~ HalfNormal(2.0)
        β_n  ~ HalfNormal(1.0)       (more bands → better)
        β_Δλ ~ Normal(−0.5, 0.5)     (finer resolution → better, negative)
        β_A  ~ HalfNormal(1.0)
    """

    beta_0: Float[ArrayLike, ""]
    beta_proxy: Float[ArrayLike, ""]
    beta_n_bands: Float[ArrayLike, ""]
    beta_spectral_res: Float[ArrayLike, ""]
    beta_albedo: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        n_bands: jnp.ndarray,
        spectral_res_nm: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.
            albedo: Surface albedo [–]. Shape ``(N,)``.
            n_bands: Number of spectral channels in CH₄ window [count]. Shape ``(N,)``.
            spectral_res_nm: Spectral resolution Δλ [nm]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)
        # logit = β₀ + β_proxy x + β_bands ln n_bands + β_res ln res + β_a albedo
        logit = (
            self.beta_0
            + self.beta_proxy * log_proxy
            + self.beta_n_bands * jnp.log(jnp.clip(n_bands, min=1.0))
            + self.beta_spectral_res * jnp.log(jnp.clip(spectral_res_nm, min=0.01))
            + self.beta_albedo * albedo
        )
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-5.0, 3.0)),
        "beta_proxy": ("beta_proxy", lambda dist: dist.HalfNormal(2.0)),
        "beta_n_bands": ("beta_n_bands", lambda dist: dist.HalfNormal(1.0)),
        "beta_spectral_res": ("beta_spectral_res", lambda dist: dist.Normal(-0.5, 0.5)),
        "beta_albedo": ("beta_albedo", lambda dist: dist.HalfNormal(1.0)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8. FULL VARYING-COEFFICIENT GLM (The Kitchen Sink)
# ═════════════════════════════════════════════════════════════════════════════


class FullVaryingCoefficientPOD(PriorTableMixin, eqx.Module):
    """Full varying-coefficient GLM with spectral and radiometric controls.

    Physical Motivation:
        The most expressive model in the family. Both the *intercept* and
        the *slope* of the detection curve vary with environmental and
        instrumental state. This captures:

          • **Intercept modifiers** (β₀ terms): direct additive effects on
            the logit baseline — e.g., more spectral bands directly lower
            the false-alarm threshold regardless of source strength.
          • **Slope modifiers** (γ terms): environment-dependent sensitivity
            — e.g., under high albedo, the instrument extracts more signal
            per unit concentration, steepening the POD curve.

    Equation:
            β₁(Z) = γ_base + γ_A · A + γ_θ · cos(θ) + γ_n · ln(n_bands)
            β₀(Z) = β₀_base + β₀_noise · ln(σ_noise)
            logit(P_d) = β₀(Z) + β₁(Z) · ln(Q / (U · p))

    Attributes:
        beta_0_base: Baseline intercept [logit].
        beta_0_noise: Noise intercept modifier [logit per ln(noise unit)].
        gamma_base: Intrinsic slope [logit per log-proxy-unit].
        gamma_albedo: Albedo slope modifier [logit per log-unit per albedo].
        gamma_cos_theta: Illumination slope modifier [logit per log-unit per cos(θ)].
        gamma_n_bands: Spectral slope modifier [logit per log-unit per ln(bands)].

    Priors (sample_priors):
        β₀_base  ~ Normal(−3.5, 2.0)
        β₀_noise ~ Normal(−0.5, 0.5)    (more noise → lower baseline)
        γ_base   ~ HalfNormal(1.5)
        γ_A      ~ HalfNormal(1.0)
        γ_θ      ~ Normal(0.5, 0.5)
        γ_n      ~ HalfNormal(0.5)
    """

    beta_0_base: Float[ArrayLike, ""]
    beta_0_noise: Float[ArrayLike, ""]
    gamma_base: Float[ArrayLike, ""]
    gamma_albedo: Float[ArrayLike, ""]
    gamma_cos_theta: Float[ArrayLike, ""]
    gamma_n_bands: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
        albedo: jnp.ndarray,
        cos_theta: jnp.ndarray,
        n_bands: jnp.ndarray,
        sigma_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.
            albedo: Surface albedo [–]. Shape ``(N,)``.
            cos_theta: Cosine of solar zenith angle [–]. Shape ``(N,)``.
            n_bands: Spectral channels in CH₄ window [count]. Shape ``(N,)``.
            sigma_noise: Instrument noise [ppb or retrieval units]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)

        # β₀ = β₀_base + β₀_noise ln σ_noise
        intercept = self.beta_0_base + self.beta_0_noise * jnp.log(
            jnp.clip(sigma_noise, min=1e-6)
        )
        # β₁ = γ₀ + γ_a · albedo + γ_θ · cos θ + γ_bands ln n_bands
        slope = (
            self.gamma_base
            + self.gamma_albedo * albedo
            + self.gamma_cos_theta * cos_theta
            + self.gamma_n_bands * jnp.log(jnp.clip(n_bands, min=1.0))
        )
        # logit = β₀ + β₁ x
        logit = intercept + slope * log_proxy
        return jax.nn.sigmoid(logit)

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0_base": ("beta_0_base", lambda dist: dist.Normal(-3.5, 2.0)),
        "beta_0_noise": ("beta_0_noise", lambda dist: dist.Normal(-0.5, 0.5)),
        "gamma_base": ("gamma_base", lambda dist: dist.HalfNormal(1.5)),
        "gamma_albedo": ("gamma_albedo", lambda dist: dist.HalfNormal(1.0)),
        "gamma_cos_theta": ("gamma_cos_theta", lambda dist: dist.Normal(0.5, 0.5)),
        "gamma_n_bands": ("gamma_n_bands", lambda dist: dist.HalfNormal(0.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. PROBIT GLM (Gaussian Link Alternative)
# ═════════════════════════════════════════════════════════════════════════════


class ProbitPOD(PriorTableMixin, eqx.Module):
    """Probit GLM on the concentration proxy.

    Physical Motivation:
        The probit link Φ(·) (standard normal CDF) arises naturally when the
        latent signal strength Y* = β₀ + β₁·x + ε, with ε ~ N(0,1). A
        detection occurs when Y* > 0. This is the correct model if the
        retrieval noise is Gaussian — which is a reasonable first
        approximation for well-calibrated optimal-estimation retrievals
        (e.g., IMAP-DOAS on TROPOMI, or the CO₂M matched filter).

        The probit and logit links produce nearly identical curves; the
        probit has slightly thinner tails, producing a sharper transition.

    Equation:
            P_d = Φ(β₀ + β₁ · ln(Q / (U · p)))

        where Φ is the standard normal CDF.

    Attributes:
        beta_0: Intercept [probit units].
        beta_1: Slope [probit per log-proxy unit].

    Priors (sample_priors):
        β₀ ~ Normal(−2.0, 2.0)   [probit]
        β₁ ~ HalfNormal(1.5)     [probit per unit]
    """

    beta_0: Float[ArrayLike, ""]
    beta_1: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)
        # z = β₀ + β₁ x
        probit_score = self.beta_0 + self.beta_1 * log_proxy
        # Φ(z) via the error function: Φ(z) = 0.5 · (1 + erf(z / √2))
        return 0.5 * (1.0 + _jax_erf(probit_score / jnp.sqrt(2.0)))

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-2.0, 2.0)),
        "beta_1": ("beta_1", lambda dist: dist.HalfNormal(1.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 10. CLOGLOG GLM (Complementary Log-Log — Asymmetric Extremes)
# ═════════════════════════════════════════════════════════════════════════════


class CloglogPOD(PriorTableMixin, eqx.Module):
    """Complementary log-log GLM for asymmetric detection tails.

    Physical Motivation:
        The logistic and probit links are symmetric: the curve approaches 0
        and 1 at the same rate. In practice, P_d often approaches 1 much
        faster than it departs from 0 — once a plume is large enough,
        detection is near-certain, but very small plumes can linger near
        P_d ≈ 0 for a wide range of Q. The cloglog link captures this
        asymmetry naturally.

        Arises from an extreme-value / Gumbel latent variable, appropriate
        when detection is a "first exceedance" event: the retrieval
        algorithm fires when the enhancement exceeds a noise threshold in
        at least one pixel.

    Equation:
            P_d = 1 − exp(−exp(β₀ + β₁ · ln(Q / (U · p))))

    Attributes:
        beta_0: Intercept [cloglog units].
        beta_1: Slope [cloglog per log-proxy unit].

    Priors (sample_priors):
        β₀ ~ Normal(−3.0, 2.0)   [cloglog]
        β₁ ~ HalfNormal(1.5)     [cloglog per unit]
    """

    beta_0: Float[ArrayLike, ""]
    beta_1: Float[ArrayLike, ""]

    def __call__(
        self,
        Q: jnp.ndarray,
        U: jnp.ndarray,
        p: jnp.ndarray,
    ) -> jnp.ndarray:
        """Evaluate P_d.

        Args:
            Q: Methane mass flux [kg hr⁻¹]. Shape ``(N,)``.
            U: Wind speed [m s⁻¹]. Shape ``(N,)``.
            p: Pixel size [m]. Shape ``(N,)``.

        Returns:
            Probability of detection [0, 1]. Shape ``(N,)``.
        """
        log_proxy = log_concentration_proxy(Q, U, p)
        # η = β₀ + β₁ x
        linear_predictor = self.beta_0 + self.beta_1 * log_proxy
        # P_d = 1 − exp(−e^η)
        return 1.0 - jnp.exp(-jnp.exp(linear_predictor))

    _PRIORS: ClassVar[PriorTable] = {
        "beta_0": ("beta_0", lambda dist: dist.Normal(-3.0, 2.0)),
        "beta_1": ("beta_1", lambda dist: dist.HalfNormal(1.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

POD_REGISTRY = {
    "logistic": LogisticPOD,
    "log_logistic": LogLogisticPOD,
    "concentration_proxy": ConcentrationProxyPOD,
    "additive_multi": AdditiveMultiCovariatePOD,
    "varying_coefficient": VaryingCoefficientPOD,
    "snr_based": SNRBasedPOD,
    "spectral_aware": SpectralAwarePOD,
    "full_varying": FullVaryingCoefficientPOD,
    "probit": ProbitPOD,
    "cloglog": CloglogPOD,
}
"""str → POD Module class mapping for dynamic dispatch."""
