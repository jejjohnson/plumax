"""RTM-based observation operator for the Tier IV coupled forward.

The Tier IV v1 forward (:mod:`plumax.coupled.forward`) maps a plume column
enhancement to an observation through a **scalar averaging kernel**
(``y_inst = A_inst · col_z(...)``). The design's build-order step toward
L1-radiance fusion ([Tier IV §per-instrument-forward](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-per-instrument-forward),
[RTM stack](https://github.com/jejjohnson/plumax/blob/main/docs/design/04_rtm_stack.md))
replaces that scalar AK with a *physically real* radiative-transfer operator:
the same column enhancement is mapped to a gas column ``ΔVMR`` and then, via
radtrax's single-slab Beer–Lambert model and its band instrument, to the
band-integrated normalised radiance each instrument would actually observe.

This module provides :class:`RadianceObservationOperator`, which maps a
*per-receptor column-enhancement field* (what Tier I gives) to per-receptor
band-integrated radiances (:class:`plumax.adapters.RadtranObservationOperator`
is its one-receptor case in ``ΔVMR``), and :func:`radiance_response`, the
drop-in parallel to :func:`plumax.coupled.forward.column_response` for the
radiance forward. The existing scalar-AK path is untouched.

Differentiability
-----------------
The table lookup ``a(ν) = σ(ν, T, p) n L AMF`` goes through ``xarray`` and
happens once, when the operator is built
(:meth:`radtrax.retrieval.var3d.RadianceObservationModel.from_lut`). Every
step after it is JAX: the column→ΔVMR conversion
(:func:`column_mass_to_delta_vmr`), ``exp(−a ΔVMR)`` and the band
integration, so :class:`RadianceObservationOperator`,
:class:`LayeredRadianceOperator` and :class:`LinearisedRadianceOperator` all
run under ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Covariance / gaussx
-------------------
This operator deliberately returns a *band-integrated mean radiance* response
and does **not** roll its own spectral covariance / whitening: when a spectral
error covariance over bands is needed (matched-filter target construction,
whitened radiance residuals), route it through the gaussx-backed
background estimators in :mod:`radtrax.retrieval.matched_filter.background` and
:func:`radtrax.retrieval.matched_filter.core.apply_pixel` (``gx.LowRankUpdate`` + Woodbury
``gx.solve``) rather than a hand-rolled dense inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from geophysix.columns import convert
from geophysix.constants import CH4
from jax.typing import ArrayLike
from jaxtyping import Float
from radtrax.retrieval.var3d import RadianceObservationModel
from radtrax.slab import (
    ATM_TO_PA,
    forward_nonlinear_normalized,
)

from plumax._src.constants import M_CH4
from plumax._src.errors import DomainError
from plumax._src.tracing import require_positive
from plumax.coupled.forward import PlumeSource, column_response


if TYPE_CHECKING:
    import xarray as xr
    from jaxtyping import Array, Float
    from radtrax import ForwardModel
    from radtrax.atmosphere import AtmosphericState, ViewingGeometry
    from radtrax.instrument import SpectralResponseFunction
    from radtrax.surface import Surface

    from plumax.coupled.instrument import Instrument


# Methane molar mass for the mass→mole conversion, kept under its rtm name (a
# plumax._src.constants value).
CH4_MOLAR_MASS_KG_PER_MOL: float = M_CH4  # [kg/mol]


def column_mass_to_delta_vmr(
    column_mass_kg_m2: jax.Array | float,
    *,
    p_atm: Float[ArrayLike, ""],
    T_K: Float[ArrayLike, ""],
    path_length_cm: Float[ArrayLike, ""],
) -> jax.Array:
    """Convert a plume column **mass** enhancement [kg/m²] to a gas ``ΔVMR``.

    The Tier I plume column integral gives a column **mass enhancement**
    ``m`` [kg/m²] of methane sitting above the pixel. The radtrax slab Beer–Lambert
    forward instead parameterises absorption by the *enhancement in volume
    mixing ratio* ``ΔVMR`` over the absorbing path. The column in ppm·m at the
    layer's ``(p, T)``, :func:`geophysix.columns.convert` from kg/m², spread over
    the path length ``L`` gives it::

        X = 10⁶ (m / M_CH₄) R T / p,   ΔVMR = 10⁻⁶ X / L

    which is the gas column number density divided by the air column,
    ``N_gas / (n(p, T) L)``: the standard "slant/vertical column density divided
    by the air column" relation used in trace-gas retrievals (cf. the
    partial-column-to-VMR conversion in e.g. TROPOMI / matched-filter
    retrievals); keeping ``L`` and ``(p, T)`` explicit makes the per-instrument
    geometry auditable.

    Args:
        column_mass_kg_m2: Column mass enhancement [kg/m²] (scalar or array; the
            Tier I plume column response is linear in the emission rate ``Q``).
        p_atm: Layer pressure [atm], ``> 0``.
        T_K: Layer temperature [K], ``> 0``.
        path_length_cm: Vertical absorbing-path length [cm], ``> 0`` — the depth
            over which the column enhancement is spread (use the same value
            passed to the radtrax slab forward for consistency).

    Returns:
        The dimensionless ``ΔVMR`` enhancement, same shape as
        ``column_mass_kg_m2``.

    Raises:
        ValueError: if ``p_atm``, ``T_K`` or ``path_length_cm`` is non-positive
            (eager check, skipped under JAX tracing via ``_is_traced``).
    """
    # Eager geometry validation, skipped under jit/grad/vmap (the scalars could
    # be tracers in a differentiable pipeline) — mirrors the inversion modules.
    require_positive("p_atm", p_atm, caller="column_mass_to_delta_vmr")
    require_positive("T_K", T_K, caller="column_mass_to_delta_vmr")
    require_positive(
        "path_length_cm", path_length_cm, caller="column_mass_to_delta_vmr"
    )
    column_mass = jnp.asarray(column_mass_kg_m2, dtype=float)
    pressure = jnp.asarray(p_atm, dtype=float) * ATM_TO_PA
    # X = 10⁶ (m / M) R T / p  [ppm·m]
    column_ppm_m = convert(
        column_mass,
        "kg_m2",
        "ppm_m",
        gas=CH4,
        pressure=pressure,
        temperature=jnp.asarray(T_K, dtype=float),
    )
    # L [m]
    path_length_m = jnp.asarray(path_length_cm, dtype=float) / 100.0
    # ΔVMR = 10⁻⁶ X / L
    return column_ppm_m * 1e-6 / path_length_m


def _require_inside_table(
    ds: xr.Dataset, nu_cm: np.ndarray, T_K: float, p_atm: float
) -> None:
    """Fail where the table lookup would return NaN: outside its ν, T or p range."""
    queries = (
        ("wavenumber", np.asarray(nu_cm, dtype=float), "the spectral grid", "cm⁻¹"),
        ("temperature", np.asarray(T_K, dtype=float), "`T_K`", "K"),
        ("pressure", np.asarray(p_atm, dtype=float), "`p_atm`", "atm"),
    )
    for coord, values, name, unit in queries:
        grid = np.asarray(ds[coord], dtype=float)
        low, high = float(grid.min()), float(grid.max())
        if float(values.min()) < low or float(values.max()) > high:
            raise DomainError(
                f"RadianceObservationOperator: {name} "
                f"[{float(values.min()):.6g}, {float(values.max()):.6g}] {unit} "
                f"leaves the table's {coord} range [{low:.6g}, {high:.6g}] {unit}, "
                "where the lookup returns NaN; extend the table or narrow the grid."
            )


class RadianceObservationOperator(eqx.Module):
    """Map a per-receptor column-enhancement field to normalised radiance via the slab.

    The RTM analogue of the scalar averaging kernel: instead of
    ``y = A · column``, this composes

        column mass m [kg/m²]  →  ΔVMR  →  L_norm(ν) = exp(−a(ν) ΔVMR)  →  band radiance

    with :func:`column_mass_to_delta_vmr` and the single-slab Beer–Lambert
    absorption ``a(ν) = σ(ν, T, p) n L AMF``, which the constructor looks up
    from the table once. With ``srf`` the operator holds radtrax's
    :class:`~radtrax.retrieval.var3d.RadianceObservationModel`, whose
    spectral grid is the SRF's own (``ν = 10⁷ / λ``), and returns one value
    per band, ``y_b = Σ_g f_bg exp(−a_g ΔVMR)``. Without it, it holds ``a(ν)``
    on ``nu_obs`` and returns the high-resolution spectrum.

    Every step after the lookup is JAX: ``predict`` is a ``jax.vmap`` over
    receptors, :meth:`linearize_single` is ``jax.jacfwd``, and
    :class:`~plumax.coupled.CoupledProblem` takes the operator as it takes
    :class:`LayeredRadianceOperator`.

    Args:
        ds: Absorption-cross-section table (HAPI schema; see
            :func:`radtrax.spectroscopy.hitran.build_lut_dataset`).
        T_K: Layer temperature [K].
        p_atm: Layer pressure [atm].
        path_length_cm: Optical / absorbing-path length [cm].
        amf: Air-mass factor.
        srf: Band integration; its wavelength grid is the spectral grid.
        nu_obs: Wavenumbers [cm⁻¹] of the spectrum returned without ``srf``,
            ``(G,)``.
        var: Table cross-section variable name.

    Attributes:
        model: The radtrax slab model with its band instrument, or ``None``
            without an SRF.
        absorption_coeff_hr: ``a(ν)`` on ``nu_obs``, ``(G,)``, or ``None``
            with an SRF (the model holds it on the SRF's grid).
        T_K: Layer temperature [K], for the mass → ΔVMR conversion.
        p_atm: Layer pressure [atm], for the mass → ΔVMR conversion.
        path_length_cm: Path length [cm], for the mass → ΔVMR conversion.

    Raises:
        ValueError: If not exactly one of ``srf`` and ``nu_obs`` is given.
        DomainError: If the spectral grid, ``T_K`` or ``p_atm`` leaves the
            table, where the lookup would return NaN.
    """

    model: RadianceObservationModel | None
    absorption_coeff_hr: Float[Array, " G"] | None
    T_K: float
    p_atm: float
    path_length_cm: float

    def __init__(
        self,
        ds: xr.Dataset,
        *,
        T_K: float,
        p_atm: float,
        path_length_cm: float,
        amf: float,
        srf: SpectralResponseFunction | None = None,
        nu_obs: Float[ArrayLike, " G"] | None = None,
        var: str = "absorption_cross_section",
    ) -> None:
        if (srf is None) == (nu_obs is None):
            raise ValueError(
                "RadianceObservationOperator: give exactly one of `srf` (its "
                "wavelength grid is the spectral grid) and `nu_obs` (the "
                "spectrum without band integration)."
            )
        self.T_K, self.p_atm = float(T_K), float(p_atm)
        self.path_length_cm = float(path_length_cm)
        if srf is not None:
            # ν = 10⁷ / λ on the SRF's grid
            nu_hr = 1e7 / np.asarray(srf.wavelengths_hr_nm, dtype=float)
            _require_inside_table(ds, nu_hr, self.T_K, self.p_atm)
            # a(ν) = σ(ν, T, p) n L AMF, on the SRF's grid in ascending ν
            self.model = RadianceObservationModel.from_lut(
                ds,
                srf=srf,
                T_K=self.T_K,
                p_atm=self.p_atm,
                path_length_cm=self.path_length_cm,
                amf=float(amf),
                var=var,
            )
            self.absorption_coeff_hr = None
            return
        nu = np.asarray(nu_obs, dtype=float)
        _require_inside_table(ds, nu, self.T_K, self.p_atm)
        # a(ν) = −∂L_norm/∂ΔVMR at ΔVMR = 0, from radtrax's slab lookup
        tangent = forward_nonlinear_normalized(
            ds,
            nu,
            T_K=self.T_K,
            p_atm=self.p_atm,
            vmr_background=0.0,
            delta_vmr=0.0,
            path_length_cm=self.path_length_cm,
            amf=float(amf),
            var=var,
        )
        self.model = None
        self.absorption_coeff_hr = -jnp.asarray(tangent.jacobian, dtype=float)

    @property
    def n_channels(self) -> int:
        """Number of output channels per receptor (bands if SRF, else n_nu)."""
        if self.model is not None:
            return self.model.n_bands
        return int(jnp.asarray(self.absorption_coeff_hr).shape[0])

    def delta_vmr(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Column mass [kg/m²] → ΔVMR via :func:`column_mass_to_delta_vmr`."""
        return column_mass_to_delta_vmr(
            column_mass_kg_m2,
            p_atm=self.p_atm,
            T_K=self.T_K,
            path_length_cm=self.path_length_cm,
        )

    def predict_delta_vmr(self, delta_vmr: jax.Array | float) -> jax.Array:
        """Normalised radiance for one receptor's ``ΔVMR``, shape ``(n_channels,)``.

        Args:
            delta_vmr: Scalar ``ΔVMR`` enhancement.

        Returns:
            ``exp(−a ΔVMR)``, band-integrated when an SRF is configured.
        """
        delta_vmr = jnp.asarray(delta_vmr, dtype=float)
        if self.model is not None:
            # one pixel: () → (1, 1)
            field = einx.id(" -> h w", delta_vmr, h=1, w=1)
            # y_b = Σ_g f_bg exp(−a_g ΔVMR): (1, 1) → (1, 1, B) → (B,)
            return self.model.forward(field)[0, 0]
        # L_norm(ν) = exp(−a(ν) ΔVMR)                                        (G,)
        return jnp.exp(-jnp.asarray(self.absorption_coeff_hr) * delta_vmr)

    def predict_single(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Normalised radiance for one receptor, shape ``(n_channels,)``.

        Args:
            column_mass_kg_m2: Scalar column mass enhancement [kg/m²].

        Returns:
            ``exp(−a ΔVMR)``, band-integrated when an SRF is configured.
        """
        return self.predict_delta_vmr(self.delta_vmr(column_mass_kg_m2))

    def predict(self, column_enhancement: jax.Array | np.ndarray) -> jax.Array:
        """Per-receptor normalised radiance, shape ``(n_obs, n_channels)``.

        Args:
            column_enhancement: Per-receptor column mass enhancement [kg/m²],
                shape ``(n_obs,)`` (e.g. ``Q · column_response``).
        """
        column_mass = jnp.asarray(column_enhancement, dtype=float)
        if column_mass.ndim != 1:
            raise ValueError(
                "RadianceObservationOperator.predict: `column_enhancement` must "
                f"be 1-D (n_obs,); got shape {column_mass.shape}."
            )
        return jax.vmap(self.predict_single)(column_mass)

    def linearize_single(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """``d(L_norm)/d(ΔVMR)`` at one receptor, by ``jax.jacfwd``.

        The tangent-linear with respect to ``ΔVMR``, band-integrated when an
        SRF is configured; chain with ``d(ΔVMR)/d(column mass)`` (the
        constant slope of :func:`column_mass_to_delta_vmr`) for the
        column-space Jacobian, which :func:`linearise` takes directly.

        Args:
            column_mass_kg_m2: Scalar column mass enhancement [kg/m²].

        Returns:
            Jacobian, shape ``(n_channels,)`` (negative for an absorbing band).
        """
        return jax.jacfwd(self.predict_delta_vmr)(self.delta_vmr(column_mass_kg_m2))


class LayeredRadianceOperator(eqx.Module):
    """Map a column-mass enhancement to band radiance through radtrax's layered model.

    The layered counterpart of :class:`RadianceObservationOperator`, on
    :class:`radtrax.ForwardModel` instead of the single slab::

        column mass m [kg/m²]  →  ΔX = m / M_CH₄ [mol/m²]
            →  state.add_column(gas, ΔX, profile)  →  y = F(state, surface, geometry)

    and, when ``normalise`` (the default), ``y(ΔX) / y(0)``, so zero
    enhancement gives ones as the slab operator's normalised radiance does.
    Every step is JAX, so ``jax.grad`` of a misfit flows through the plume
    model and the radiative transfer together, and ``predict`` is a
    ``jax.vmap`` over receptors. The layers remove the slab's error of
    broadening every line at one pressure; with one layer on a table node,
    an identity instrument and no shift it equals the slab operator.

    Band values are divided after band integration, ``y_b(ΔX) / y_b(0)``,
    which is what a sensor's ratio is. The slab operator instead integrates
    ``exp(-Δτ)`` over each band, so with an SRF the two differ by the
    background spectrum's weighting within the band.

    Attributes:
        model: The layered forward model (opacity, solver, instrument).
        state: The background atmosphere; it must carry ``gas``.
        surface: The surface reflectance.
        geometry: Sun and sensor directions.
        profile: The plume's relative weight per layer, ``(L,)``; ``None``
            puts the whole enhancement in the bottom layer.
        gas: The gas the plume adds.
        normalise: Divide by the background band values ``y(0)``.

    Examples:
        >>> import jax.numpy as jnp
        >>> from radtrax import ForwardModel
        >>> from radtrax.atmosphere import ViewingGeometry, layers_from_levels
        >>> from radtrax.instrument import BandInstrument
        >>> from radtrax.solvers import BeerLambert
        >>> from radtrax.spectroscopy import CrossSectionTable, TableOpacity
        >>> from radtrax.surface import Lambertian
        >>> nu = jnp.array([4300.0, 4300.1, 4300.2])
        >>> table = CrossSectionTable(
        ...     jnp.log(jnp.array([1e4, 1e5])),
        ...     jnp.array([200.0, 300.0]),
        ...     jnp.full((2, 2, 3), 1e-21),
        ... )
        >>> model = ForwardModel(
        ...     TableOpacity(nu, {"CH4": table}),
        ...     BeerLambert(),
        ...     BandInstrument(nu, jnp.full((1, 3), 1 / 3)),
        ... )
        >>> atm = layers_from_levels(
        ...     jnp.array([1e5, 5e4]),
        ...     jnp.array([288.0, 255.0]),
        ...     {"CH4": jnp.full(2, 1.9e-6)},
        ... )
        >>> op = LayeredRadianceOperator(
        ...     model, atm, Lambertian(0.3), ViewingGeometry(1.0, 1.0)
        ... )
        >>> float(op.predict_single(0.0)[0])
        1.0
    """

    model: ForwardModel
    state: AtmosphericState
    surface: Surface
    geometry: ViewingGeometry
    profile: Float[Array, " L"] | None = None
    gas: str = eqx.field(static=True, default="CH4")
    normalise: bool = eqx.field(static=True, default=True)

    @property
    def n_channels(self) -> int:
        """Number of output channels per receptor (the instrument's bands)."""
        return int(jax.eval_shape(self.predict_single, 0.0).shape[0])

    def delta_column(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Column mass [kg/m²] → column enhancement ``ΔX`` [mol/m²]."""
        return jnp.asarray(column_mass_kg_m2, dtype=float) / CH4_MOLAR_MASS_KG_PER_MOL

    def predict_single(self, column_mass_kg_m2: jax.Array | float) -> jax.Array:
        """Band values for one receptor, shape ``(n_channels,)``.

        Args:
            column_mass_kg_m2: Scalar column mass enhancement [kg/m²].

        Returns:
            ``y(ΔX) / y(0)`` when ``normalise``, else ``y(ΔX)``.
        """
        plume = self.state.add_column(
            self.gas, self.delta_column(column_mass_kg_m2), self.profile
        )
        radiance = self.model(plume, self.surface, self.geometry)
        if self.normalise:
            # y(ΔX) / y(0)
            radiance = radiance / self.model(self.state, self.surface, self.geometry)
        return radiance

    def predict(self, column_enhancement: jax.Array | np.ndarray) -> jax.Array:
        """Per-receptor band values, shape ``(n_obs, n_channels)``.

        Args:
            column_enhancement: Per-receptor column mass enhancement [kg/m²],
                shape ``(n_obs,)`` (e.g. ``Q · column_response``).
        """
        column_mass = jnp.asarray(column_enhancement, dtype=float)
        if column_mass.ndim != 1:
            raise ValueError(
                "LayeredRadianceOperator.predict: `column_enhancement` must be "
                f"1-D (n_obs,); got shape {column_mass.shape}."
            )
        return jax.vmap(self.predict_single)(column_mass)


def radiance_response(
    source: PlumeSource,
    instrument: Instrument,
    operator: RadianceObservationOperator | LayeredRadianceOperator,
    emission_rate: jax.Array | float,
) -> jax.Array:
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
        operator: The configured :class:`RadianceObservationOperator` or
            :class:`LayeredRadianceOperator`; with either the whole map from
            ``Q`` to radiance is JAX and differentiable in ``Q``.
        emission_rate: Emission rate ``Q`` [kg/s] (scalar).

    Returns:
        Per-receptor normalised radiance, shape ``(n_obs, n_channels)``.
    """
    response = column_response(source, instrument)
    return operator.predict(jnp.asarray(emission_rate, dtype=float) * response)


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
        column_mass = jnp.asarray(column_mass_kg_m2, dtype=float)
        # L = L₀ + g m
        return self.baseline + self.gain * column_mass

    def predict(self, column_enhancement: jax.Array) -> jax.Array:
        """Per-receptor linearised radiance, shape ``(n_obs, n_channels)``.

        Fully differentiable / vectorised JAX:
        ``baseline[None, :] + mass[:, None] · gain[None, :]``.
        """
        column_mass = jnp.asarray(column_enhancement, dtype=float)
        if column_mass.ndim != 1:
            raise ValueError(
                "LinearisedRadianceOperator.predict: `column_enhancement` must "
                f"be 1-D (n_obs,); got shape {column_mass.shape}."
            )
        if column_mass.shape[0] == 0:
            # einx rejects a zero-size axis; no receptors, no radiances
            dtype = jnp.result_type(self.baseline, column_mass, self.gain)
            return jnp.zeros((0, np.shape(self.gain)[0]), dtype=dtype)
        # g m
        scaled = einx.multiply("obs, channel -> obs channel", column_mass, self.gain)
        # L = L₀ + g m
        return einx.add("channel, obs channel -> obs channel", self.baseline, scaled)


def linearise(
    operator: RadianceObservationOperator | LayeredRadianceOperator,
) -> LinearisedRadianceOperator:
    """Build the :class:`LinearisedRadianceOperator` tangent to ``operator`` at 0.

    Evaluates the background radiance ``L_norm(0)`` and the gain
    ``g = d(L_norm)/d(mass)|₀``, ``jax.jacfwd`` of the operator's
    ``predict_single`` at zero, so the slope of
    :func:`column_mass_to_delta_vmr` is already folded in and the gain is in
    column-mass units. The result is exact to first order in the enhancement.

    Args:
        operator: The configured nonlinear RTM observation operator.

    Returns:
        The matching :class:`LinearisedRadianceOperator`.
    """
    zero = jnp.asarray(0.0)
    return LinearisedRadianceOperator(
        baseline=operator.predict_single(zero),
        # g = ∂L_norm/∂m at m = 0
        gain=jax.jacfwd(operator.predict_single)(zero),
    )


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
    "CH4_MOLAR_MASS_KG_PER_MOL",
    "LayeredRadianceOperator",
    "LinearisedRadianceOperator",
    "RadianceObservationOperator",
    "column_mass_to_delta_vmr",
    "linearise",
    "radiance_response",
    "radiance_response_linear",
]
