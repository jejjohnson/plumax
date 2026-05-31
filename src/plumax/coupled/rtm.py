"""RTM-based observation operator for the Tier IV coupled forward.

The Tier IV v1 forward (:mod:`plumax.coupled.forward`) maps a plume column
enhancement to an observation through a **scalar averaging kernel**
(``y_inst = A_inst · col_z(...)``). The design's build-order step toward
L1-radiance fusion ([Tier IV §per-instrument-forward](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-per-instrument-forward),
[RTM stack](https://github.com/jejjohnson/plumax/blob/main/docs/design/04_rtm_stack.md))
replaces that scalar AK with a *physically real* radiative-transfer operator:
the same column enhancement is mapped to a gas column ``ΔVMR`` and then, via the
Beer–Lambert + spectral-response-function stack in :mod:`plumax.radtran`, to the
band-integrated normalised radiance each instrument would actually observe.

This module provides :class:`RadianceObservationOperator` — the RTM analogue of
:class:`plumax.adapters.RadtranObservationOperator`, but mapping a *per-receptor
column-enhancement field* (what Tier I gives) to per-receptor band-integrated
radiances, and :func:`radiance_response`, the drop-in parallel to
:func:`plumax.coupled.forward.column_response` for the radiance forward. The
existing scalar-AK path is untouched — the change is purely additive.

Differentiability
-----------------
The :mod:`plumax.radtran` Beer–Lambert forward and the
:class:`~plumax.radtran.srf.SpectralResponseFunction` are **NumPy-side** (the
LUT interpolation goes through ``xarray``), so :class:`RadianceObservationOperator`
is *not* JAX-traceable end-to-end: it consumes concrete arrays and returns
NumPy arrays, exactly like :class:`plumax.adapters.RadtranObservationOperator`.
The analytic Jacobian ``d(L_norm)/d(ΔVMR)`` returned by the radtran forward is
the tangent-linear used in :meth:`RadianceObservationOperator.linearize`. The
*Tier I plume* column response (:func:`plumax.coupled.forward.column_response`)
that feeds it remains differentiable; the column→ΔVMR conversion
(:func:`column_mass_to_delta_vmr`) is a pure JAX function and is differentiable.

Covariance / gaussx
-------------------
This operator deliberately returns a *band-integrated mean radiance* response
and does **not** roll its own spectral covariance / whitening: when a spectral
error covariance over bands is needed (matched-filter target construction,
whitened radiance residuals), route it through the gaussx-backed
:func:`plumax.radtran.build_lowrank_covariance_operator` +
:func:`plumax.radtran.matched_filter_pixel_op` (``gx.LowRankUpdate`` + Woodbury
``gx.solve``) rather than a hand-rolled dense inverse — see
:mod:`plumax.radtran.gaussx_solve`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from plumax.coupled.forward import PlumeSource, column_response
from plumax.lagrangian.inversion import _is_traced
from plumax.radtran.config import number_density_cm3
from plumax.radtran.forward import forward_nonlinear_normalized


if TYPE_CHECKING:
    import jax
    import xarray as xr

    from plumax.coupled.instrument import Instrument
    from plumax.radtran.srf import SpectralResponseFunction


# Methane molar mass and Avogadro's number for the mass→molecule conversion.
CH4_MOLAR_MASS_KG_PER_MOL: float = 0.0160425  # [kg/mol]
AVOGADRO_PER_MOL: float = 6.02214076e23  # [molecules/mol]
CM2_PER_M2: float = 1e4  # [cm²/m²]


def column_mass_to_delta_vmr(
    column_mass_kg_m2: jax.Array | float,
    *,
    p_atm: float,
    T_K: float,
    path_length_cm: float,
) -> jax.Array:
    """Convert a plume column **mass** enhancement [kg/m²] to a gas ``ΔVMR``.

    The Tier I plume column integral gives a column **mass enhancement**
    ``m`` [kg/m²] of methane sitting above the pixel. The radtran Beer–Lambert
    forward instead parameterises absorption by the *enhancement in volume
    mixing ratio* ``ΔVMR`` over the absorbing path. The two are related by the
    column number densities of the gas and of dry air:

    1. **Gas column number density** [molecules/cm²]::

           N_gas = m · (N_A / M_CH₄) / CM2_PER_M2

       (``N_A`` = Avogadro, ``M_CH₄`` = methane molar mass; dividing by
       ``CM2_PER_M2 = 1e4`` converts the per-m² areal density to per-cm².)

    2. **Air column number density** [molecules/cm²]::

           N_air = n(p, T) · L

       where ``n(p, T)`` is the ideal-gas number density [molecules/cm³]
       (:func:`plumax.radtran.config.number_density_cm3`) and ``L`` is the
       vertical path length [cm] over which the enhancement is distributed.

    3. **Volume mixing ratio enhancement** (dimensionless)::

           ΔVMR = N_gas / N_air

    This is the standard "slant/vertical column density divided by the air
    column" relation used in trace-gas retrievals (cf. the partial-column-to-VMR
    conversion in e.g. TROPOMI / matched-filter retrievals); keeping ``L`` and
    ``(p, T)`` explicit makes the per-instrument geometry auditable.

    Args:
        column_mass_kg_m2: Column mass enhancement [kg/m²] (scalar or array; the
            Tier I plume column response is linear in the emission rate ``Q``).
        p_atm: Layer pressure [atm], ``> 0``.
        T_K: Layer temperature [K], ``> 0``.
        path_length_cm: Vertical absorbing-path length [cm], ``> 0`` — the depth
            over which the column enhancement is spread (use the same value
            passed to the radtran forward for consistency).

    Returns:
        The dimensionless ``ΔVMR`` enhancement, same shape as
        ``column_mass_kg_m2``.

    Raises:
        ValueError: if ``p_atm``, ``T_K`` or ``path_length_cm`` is non-positive
            (eager check, skipped under JAX tracing via ``_is_traced``).
    """
    # Eager geometry validation, skipped under jit/grad/vmap (the scalars could
    # be tracers in a differentiable pipeline) — mirrors the inversion modules.
    if not _is_traced(p_atm, T_K, path_length_cm) and (
        p_atm <= 0.0 or T_K <= 0.0 or path_length_cm <= 0.0
    ):
        raise ValueError(
            "column_mass_to_delta_vmr: `p_atm`, `T_K` and `path_length_cm` "
            f"must be > 0 (got p_atm={p_atm!r}, T_K={T_K!r}, "
            f"path_length_cm={path_length_cm!r})."
        )
    m = jnp.asarray(column_mass_kg_m2, dtype=float)
    # Gas column number density [molecules/cm²].
    n_gas_cm2 = m * (AVOGADRO_PER_MOL / CH4_MOLAR_MASS_KG_PER_MOL) / CM2_PER_M2
    # Air column number density [molecules/cm²]; number_density_cm3 is a NumPy
    # scalar helper so wrap it back into the JAX graph.
    n_air_cm2 = number_density_cm3(p_atm, T_K) * path_length_cm
    return n_gas_cm2 / n_air_cm2


@dataclass(frozen=True)
class RadianceObservationOperator:
    """Map a per-receptor column-enhancement field to band radiances via the RTM.

    The RTM analogue of the scalar averaging kernel: instead of
    ``y = A · column``, this composes

        column mass [kg/m²]  →  ΔVMR  →  L_norm(ν) = exp(-Δτ)  →  band radiance

    using :func:`column_mass_to_delta_vmr` and the normalised Beer–Lambert
    forward :func:`plumax.radtran.forward_nonlinear_normalized`. With ``srf``
    supplied, the high-resolution normalised radiance ``exp(-Δτ(ν))`` is
    band-integrated by the instrument :class:`~plumax.radtran.srf.SpectralResponseFunction`;
    otherwise the full ``nu_obs`` spectrum is returned per receptor.

    The forward is **NumPy-side** (the LUT interpolation is not JAX-traceable),
    matching :class:`plumax.adapters.RadtranObservationOperator`; the analytic
    ``d(L_norm)/d(ΔVMR)`` Jacobian is exposed through :meth:`linearize`.

    Attributes:
        ds: Absorption-cross-section LUT dataset (HAPI schema; see
            :func:`plumax.hapi_lut.build_lut_dataset`).
        nu_obs: Observation wavenumbers [cm⁻¹], shape ``(n_nu,)``.
        T_K: Layer temperature [K].
        p_atm: Layer pressure [atm].
        vmr_background: Background gas volume mixing ratio.
        path_length_cm: Optical / absorbing-path length [cm].
        amf: Air-mass factor.
        srf: Optional band-integration operator. When ``None`` the operator
            returns the per-receptor high-resolution normalised radiance.
        var: LUT cross-section variable name.
    """

    ds: xr.Dataset
    nu_obs: np.ndarray
    T_K: float
    p_atm: float
    vmr_background: float
    path_length_cm: float
    amf: float
    srf: SpectralResponseFunction | None = None
    var: str = "absorption_cross_section"

    @property
    def n_channels(self) -> int:
        """Number of output channels per receptor (bands if SRF, else n_nu)."""
        if self.srf is not None:
            return int(self.srf.n_bands)
        return int(np.asarray(self.nu_obs).size)

    def delta_vmr(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Column mass [kg/m²] → ΔVMR via :func:`column_mass_to_delta_vmr`."""
        return column_mass_to_delta_vmr(
            column_mass_kg_m2,
            p_atm=self.p_atm,
            T_K=self.T_K,
            path_length_cm=self.path_length_cm,
        )

    def _forward_hr(self, delta_vmr: float):
        """Single-receptor normalised Beer–Lambert forward (NumPy ForwardResult)."""
        return forward_nonlinear_normalized(
            self.ds,
            np.asarray(self.nu_obs, dtype=float),
            T_K=self.T_K,
            p_atm=self.p_atm,
            vmr_background=self.vmr_background,
            delta_vmr=float(delta_vmr),
            path_length_cm=self.path_length_cm,
            amf=self.amf,
            var=self.var,
        )

    def _band_integrate(self, spectrum_hr: np.ndarray) -> np.ndarray:
        """Apply the SRF if present, else return the HR spectrum unchanged."""
        if self.srf is None:
            return np.asarray(spectrum_hr, dtype=float)
        return self.srf.apply(np.asarray(spectrum_hr, dtype=float))

    def predict_single(self, column_mass_kg_m2: float) -> np.ndarray:
        """Band-integrated normalised radiance for one receptor.

        Args:
            column_mass_kg_m2: Scalar column mass enhancement [kg/m²].

        Returns:
            Normalised radiance, shape ``(n_channels,)`` (``exp(-Δτ)``, band-
            integrated when an SRF is configured).
        """
        dvmr = float(self.delta_vmr(column_mass_kg_m2))
        return self._band_integrate(self._forward_hr(dvmr).radiance)

    def predict(self, column_enhancement: jax.Array | np.ndarray) -> np.ndarray:
        """Per-receptor band-integrated normalised radiance (vectorised).

        Args:
            column_enhancement: Per-receptor column mass enhancement [kg/m²],
                shape ``(n_obs,)`` (e.g. ``Q · column_response``).

        Returns:
            Per-receptor normalised radiance, shape ``(n_obs, n_channels)``.
        """
        cols = np.asarray(column_enhancement, dtype=float)
        if cols.ndim != 1:
            raise ValueError(
                "RadianceObservationOperator.predict: `column_enhancement` must "
                f"be 1-D (n_obs,); got shape {cols.shape}."
            )
        rows = [self.predict_single(float(c)) for c in cols]
        return np.stack(rows, axis=0)

    def linearize_single(self, column_mass_kg_m2: float) -> np.ndarray:
        """Band-integrated ``d(L_norm)/d(ΔVMR)`` at one receptor.

        Returns the radtran analytic Jacobian, band-integrated through the SRF
        when configured. This is the tangent-linear w.r.t. ``ΔVMR``; chain with
        ``d(ΔVMR)/d(column mass)`` (the constant slope of
        :func:`column_mass_to_delta_vmr`) for the column-space Jacobian.

        Args:
            column_mass_kg_m2: Scalar column mass enhancement [kg/m²].

        Returns:
            Jacobian, shape ``(n_channels,)`` (negative for an absorbing band).
        """
        dvmr = float(self.delta_vmr(column_mass_kg_m2))
        return self._band_integrate(self._forward_hr(dvmr).jacobian)


def radiance_response(
    source: PlumeSource,
    instrument: Instrument,
    operator: RadianceObservationOperator,
    emission_rate: jax.Array | float,
) -> np.ndarray:
    """Per-receptor band radiances for one instrument under the RTM operator.

    The radiance-forward parallel to :func:`plumax.coupled.forward.predict_observation`:
    it evaluates the Tier I unit-``Q`` column response
    (:func:`plumax.coupled.forward.column_response`), scales by the emission rate
    to get the per-receptor column **mass** enhancement
    ``Q · column_response`` [kg/m²], then maps each receptor through the RTM
    observation operator to a band-integrated normalised radiance.

    Unlike the scalar-AK forward this is **not** linear in ``Q`` (Beer–Lambert
    is exponential in the column), so it has no closed-form fusion path; it is
    intended for radiance simulation and as the forward of a gradient-based /
    linearised L1 inversion (future build-order step), leaving the existing
    closed-form :func:`plumax.coupled.fuse_observations` AK path intact.

    Args:
        source: Static source + met configuration.
        instrument: Observing instrument (receptors + AK). The RTM operator is
            the physical replacement for the scalar AK, so for a pure-RTM forward
            leave the instrument's ``averaging_kernel`` at its identity default;
            a non-identity AK is still applied to the column before the RTM (it
            then acts as a per-receptor column-scaling factor).
        operator: The configured :class:`RadianceObservationOperator`.
        emission_rate: Emission rate ``Q`` [kg/s] (scalar).

    Returns:
        Per-receptor normalised radiance, shape ``(n_obs, n_channels)``.
    """
    # Unit-Q column response is differentiable JAX; scale to the column mass and
    # hand concrete values to the NumPy-side RTM forward.
    response = column_response(source, instrument)
    column_mass = np.asarray(emission_rate, dtype=float) * np.asarray(
        response, dtype=float
    )
    return operator.predict(column_mass)


@dataclass(frozen=True)
class LinearisedRadianceOperator:
    """Tangent-linear radiance operator — the linearised counterpart of the RTM.

    The full :class:`RadianceObservationOperator` is exponential in the column
    (``L_norm = exp(-Δτ)``), so it has no closed-form inversion. Linearising the
    normalised Beer–Lambert forward about the background (``ΔVMR = 0``, where
    ``L_norm = 1``) gives the affine response

        L_norm(ΔVMR)  ≈  1 + J₀ · ΔVMR,        J₀ = d(L_norm)/d(ΔVMR)|₀,

    and since ``ΔVMR = slope · column_mass`` is itself linear (the constant slope
    of :func:`column_mass_to_delta_vmr`) and the Tier I column is linear in the
    emission rate ``Q``, the whole radiance forward becomes **linear in ``Q``**.
    That makes this the radiance analogue of the scalar averaging kernel: a
    per-channel linear gain ``J₀ · slope`` on the column mass, usable as a
    drop-in linear observation response (and, with one channel, in the
    closed-form fusion path the scalar AK uses).

    This is the well-known *linearised / optimal-estimation* radiance model
    (first-order Beer–Lambert), complementary to the exact nonlinear operator;
    it is accurate for small enhancements and exact in the optically-thin limit.

    Build with :func:`linearise` from a :class:`RadianceObservationOperator` so
    the two share the LUT, geometry and SRF configuration.

    Attributes:
        baseline: Background normalised radiance ``L_norm(0)``, shape
            ``(n_channels,)`` (all ones for the normalised forward; kept explicit
            so the affine offset is auditable / overridable).
        gain: Per-channel linear response to column mass ``d(L_norm)/d(mass)`` =
            ``J₀ · slope``, shape ``(n_channels,)`` [per kg/m²].
    """

    baseline: jax.Array
    gain: jax.Array

    @property
    def n_channels(self) -> int:
        """Number of output channels per receptor."""
        return int(jnp.asarray(self.gain).shape[0])

    def predict_single(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Linearised normalised radiance ``baseline + gain · mass`` for one receptor."""
        m = jnp.asarray(column_mass_kg_m2, dtype=float)
        return self.baseline + self.gain * m

    def predict(self, column_enhancement: jax.Array) -> jax.Array:
        """Per-receptor linearised radiance, shape ``(n_obs, n_channels)``.

        Fully differentiable / vectorised JAX (unlike the nonlinear operator's
        NumPy LUT path): ``baseline[None, :] + mass[:, None] · gain[None, :]``.
        """
        m = jnp.asarray(column_enhancement, dtype=float)
        if m.ndim != 1:
            raise ValueError(
                "LinearisedRadianceOperator.predict: `column_enhancement` must "
                f"be 1-D (n_obs,); got shape {m.shape}."
            )
        return self.baseline[None, :] + m[:, None] * self.gain[None, :]


def linearise(operator: RadianceObservationOperator) -> LinearisedRadianceOperator:
    """Build the :class:`LinearisedRadianceOperator` tangent to ``operator`` at 0.

    Evaluates the background radiance ``L_norm(0)`` and the analytic Jacobian
    ``J₀ = d(L_norm)/d(ΔVMR)|₀`` (both band-integrated through the operator's
    SRF), and folds in the constant ``d(ΔVMR)/d(mass)`` slope of
    :func:`column_mass_to_delta_vmr` to give the per-channel gain in column-mass
    units. The result is exact to first order in the enhancement.

    Args:
        operator: The configured nonlinear RTM observation operator.

    Returns:
        The matching :class:`LinearisedRadianceOperator`.
    """
    baseline = jnp.asarray(operator.predict_single(0.0), dtype=float)
    jac_dvmr = jnp.asarray(operator.linearize_single(0.0), dtype=float)
    # d(ΔVMR)/d(mass): the conversion is linear, so its slope is ΔVMR at unit mass.
    slope = float(operator.delta_vmr(1.0))
    return LinearisedRadianceOperator(baseline=baseline, gain=jac_dvmr * slope)


def radiance_response_linear(
    source: PlumeSource,
    instrument: Instrument,
    operator: LinearisedRadianceOperator,
    emission_rate: jax.Array | float,
) -> jax.Array:
    """Per-receptor linearised radiances for one instrument (differentiable).

    The tangent-linear parallel of :func:`radiance_response`: it scales the
    Tier I unit-``Q`` column response by ``Q`` and maps each receptor through the
    :class:`LinearisedRadianceOperator`. Unlike :func:`radiance_response` this is
    **linear in ``Q``** and stays in pure JAX end-to-end, so it is
    ``jax.grad`` / ``jax.jit``-friendly and (single-channel) feeds the
    closed-form :func:`plumax.coupled.fuse_observations` path.

    Args:
        source: Static source + met configuration.
        instrument: Observing instrument (receptors + AK).
        operator: The linearised radiance operator (from :func:`linearise`).
        emission_rate: Emission rate ``Q`` [kg/s].

    Returns:
        Per-receptor linearised normalised radiance, shape ``(n_obs, n_channels)``.
    """
    response = column_response(source, instrument)
    column_mass = jnp.asarray(emission_rate, dtype=float) * response
    return operator.predict(column_mass)


__all__ = [
    "AVOGADRO_PER_MOL",
    "CH4_MOLAR_MASS_KG_PER_MOL",
    "CM2_PER_M2",
    "LinearisedRadianceOperator",
    "RadianceObservationOperator",
    "column_mass_to_delta_vmr",
    "linearise",
    "radiance_response",
    "radiance_response_linear",
]
