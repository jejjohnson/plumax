"""Coupled Tier I → column → averaging-kernel forward, per instrument.

The Tier IV v1 forward (design build-order step 1): compose the steady-state
Gaussian plume (Tier I) with a column integral and a per-instrument averaging
kernel to predict each satellite's observed column-enhancement vector,

    y_inst = A_inst · col_z( plume(Q, x₀, wind, stability) )  +  bias_inst  + ε,

fused across a list of instruments that each keep their native receptor
geometry (design §multi-instrument-fusion — no pre-regridding).

Because the Gaussian plume is **linear in the emission rate Q**, the per-receptor
column enhancement factors as ``Q · response``, where ``response`` is the column
the model predicts at unit emission. :func:`column_response` returns that
unit-Q response so the inversion in :mod:`plumax.coupled.fusion` can treat the
forward as the linear map ``y = Q · response + bias`` and solve in closed form
(the design's "linear-conditional-Gaussian limit"). :func:`predict_observation`
/ :func:`predict_multi` give the full (Q-scaled, bias-added) prediction for
simulation and diagnostics.

The transport is pluggable: :func:`build_coupled_forward` takes any
:class:`~plumax.coupled.backends.TransportBackend`, an object with a
``background`` and a ``column_response(instrument)``. ``PlumeSource`` is the
Tier I one, and an
:class:`~plumax.coupled.backends.AbstractControlledSource`: it declares its
position and wind as control blocks, so the nonlinear inversion can fit them.
:class:`~plumax.coupled.backends.LagrangianSource` (Tier II) and
:class:`~plumax.coupled.backends.EulerianSource` (Tier III) are the others. A
backend may return a design matrix ``(n_obs, n_param)`` instead of a vector,
one column per emission parameter, and the forward is then ``y = R q + bias``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax._src.controlled import AbstractControlledSource
from plumax._src.wind import wind_components
from plumax.coupled.instrument import Instrument
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.observation import ColumnObservation
from plumax.gauss_plume.plume import plume_concentration_vmap


if TYPE_CHECKING:
    from plumax.coupled.backends import TransportBackend


class PlumeSource(AbstractControlledSource):
    """Static source + meteorology for the Tier I coupled forward.

    The emission rate ``Q`` is deliberately *not* stored here — it is the scalar
    the inversion solves for, so the forward is parameterised by everything
    *except* ``Q`` (location, wind, stability, background) and is linear in it.

    As a controlled source it declares ``location`` ``(x, y)`` and ``wind``
    ``(speed, direction)``, scaled by ``emission_rate`` or
    ``log_emission_rate``; the height, stability class and levels stay fixed.

    Attributes:
        location: ``(x, y, z)`` source coordinates [m].
        wind_speed: Wind speed [m/s], ``> 0``.
        wind_direction: Meteorological direction [deg from North] (``270`` = wind
            from the west, flowing east).
        stability_class: Pasquill class ``'A'``–``'F'``.
        column_z: 1-D vertical grid [m] for the column integral, shape
            ``(n_z,)``, strictly increasing.
        background: Additive background column (same units as the AK output),
            applied per receptor. Default ``0``.
    """

    location: tuple[float, float, float]
    wind_speed: float
    wind_direction: float
    stability_class: str
    column_z: jax.Array
    background: float = 0.0

    def wind_uv(self) -> tuple[float, float]:
        """Wind ``(u, v)`` [m/s] from speed + meteorological direction."""
        wind_east, wind_north = wind_components(
            self.wind_speed, self.wind_direction, xp=np
        )
        return float(wind_east), float(wind_north)

    @property
    def control_blocks(self) -> Mapping[str, tuple[int, ...]]:
        """``location: (2,)`` and ``wind: (2,)``."""
        return {"location": (2,), "wind": (2,)}

    @property
    def emission_blocks(self) -> tuple[str, ...]:
        """``emission_rate`` and ``log_emission_rate``."""
        return ("emission_rate", "log_emission_rate")

    def column_response(self, instrument: Instrument) -> jax.Array:
        """Unit-``Q`` column response, the Tier I :class:`TransportBackend` method.

        Delegates to the module-level :func:`column_response`.
        """
        return column_response(self, instrument)

    def response_at(
        self,
        instrument: Instrument,
        control: Mapping[str, Float[Array, "*shape"]],
    ) -> Float[Array, " obs"]:
        """Unit-``Q`` column with the position and wind taken from ``control``.

        Blocks missing from ``control`` keep this source's values; see
        :func:`plume_response`.

        Args:
            instrument: The observing instrument.
            control: The control, by block.

        Returns:
            The unit-emission column through the averaging kernel.
        """
        location = control.get("location", jnp.asarray(self.location[:2]))
        wind = control.get("wind", jnp.asarray([self.wind_speed, self.wind_direction]))
        return plume_response(self, instrument, location, wind)


def column_response(source: PlumeSource, instrument: Instrument) -> jax.Array:
    """Per-receptor column enhancement at **unit emission** for one instrument.

    Evaluates the Gaussian plume at unit ``Q`` on each receptor column, then
    integrates over ``source.column_z`` (trapezoidal) and applies the instrument
    averaging kernel with the shared
    :class:`~plumax.gauss_plume.observation.ColumnObservation` (whose module
    docstring states the units). The full observation at emission ``Q`` is then
    ``Q · column_response + bias`` (plus the background; see
    :func:`predict_observation`). Linearity in ``Q`` is what makes the coupled
    inversion closed-form. It is :func:`plume_response` at the source's own
    position and wind.

    Args:
        source: The static source + met configuration.
        instrument: The observing instrument (receptors + AK).

    Returns:
        Unit-``Q`` AK-weighted column response, shape ``(n_obs,)``.
    """
    location = jnp.asarray(source.location[:2], dtype=float)
    wind = jnp.asarray([source.wind_speed, source.wind_direction], dtype=float)
    return plume_response(source, instrument, location, wind)


def plume_response(
    source: PlumeSource,
    instrument: Instrument,
    location: Float[Array, " 2"],
    wind: Float[Array, " 2"],
) -> Float[Array, " obs"]:
    """Tier I unit-emission column at a traced source position and wind.

    ``coupled.column_response`` with the horizontal position and the wind
    (speed, meteorological direction) as JAX arrays, so the coupled cost can
    differentiate through them. The source height, stability class and levels
    stay those of ``source``.

    Args:
        source: The Tier I source (height, stability, levels).
        instrument: The observing instrument.
        location: Source ``(x, y)`` [m].
        wind: ``(speed [m/s], direction [deg from North])``.

    Returns:
        The unit-emission column through the averaging kernel.
    """
    levels = jnp.asarray(source.column_z, dtype=float)
    receptors = jnp.asarray(instrument.receptors, dtype=float)
    num_levels = levels.shape[0]
    # every receptor at every level, level fastest
    easting = einx.id("obs -> (obs level)", receptors[:, 0], level=num_levels)
    northing = einx.id("obs -> (obs level)", receptors[:, 1], level=num_levels)
    height = einx.id("level -> (obs level)", levels, obs=receptors.shape[0])
    wind_u, wind_v = wind_components(wind[0], wind[1])
    concentration = plume_concentration_vmap(
        easting,
        northing,
        height,
        location[0],
        location[1],
        source.location[2],
        wind_u,
        wind_v,
        1.0,
        get_dispersion_params(source.stability_class),
    )
    profiles = einx.id("(obs level) -> obs level", concentration, level=num_levels)
    observation = ColumnObservation.trapezoid(levels, averaging_kernel=instrument.ak)
    return observation.column(profiles)


def predict_observation(
    source: PlumeSource,
    instrument: Instrument,
    emission_rate: jax.Array | float,
    bias: jax.Array | float = 0.0,
) -> jax.Array:
    """Predict one instrument's observation vector ``y_inst``.

    ``y = Q · column_response + background + bias`` — the affine forward the
    coupled cost compares against the observed enhancement.

    Args:
        source: Static source + met configuration.
        instrument: The observing instrument.
        emission_rate: Emission rate ``Q`` [kg/s] (scalar).
        bias: Per-instrument additive offset (scalar). Default ``0``.

    Returns:
        Predicted observation, shape ``(n_obs,)``.
    """
    response = column_response(source, instrument)
    return emission_rate * response + source.background + bias


@dataclass(frozen=True)
class CoupledForward:
    """Multi-instrument coupled forward: a list of per-instrument observations.

    Holds the shared source/met configuration and the instrument list, and
    pre-computes each instrument's unit-``Q`` :func:`column_response` (the static,
    ``Q``-independent part of the forward). Use :func:`build_coupled_forward`.

    Attributes:
        source: Shared transport backend (``PlumeSource``, ``LagrangianSource``
            or ``EulerianSource``).
        instruments: The observing instruments (fusion list).
        responses: Per-instrument unit-``Q`` column responses (one ``(n_obs,)``
            array each, or an ``(n_obs, n_param)`` design matrix for a
            multi-parameter emission), aligned with ``instruments``.
    """

    source: PlumeSource | TransportBackend
    instruments: tuple[Instrument, ...]
    responses: tuple[jax.Array, ...]

    def predict(
        self,
        emission_rate: jax.Array | float,
        biases: Sequence[jax.Array | float] | None = None,
    ) -> list[jax.Array]:
        """Predicted observation vector per instrument (list, native resolution).

        Args:
            emission_rate: Emission rate ``Q`` [kg/s], or the ``(n_param,)``
                emission vector when the responses are design matrices.
            biases: Per-instrument additive biases, one per instrument. Defaults
                to zero for every instrument.

        Returns:
            A list of ``(n_obs_i,)`` predictions, aligned with ``instruments``.
        """
        if biases is None:
            biases = [0.0] * len(self.instruments)
        if len(biases) != len(self.instruments):
            raise ValueError(
                f"CoupledForward.predict: got {len(biases)} biases for "
                f"{len(self.instruments)} instruments."
            )
        return [
            _scale_response(resp, emission_rate) + self.source.background + bias
            for resp, bias in zip(self.responses, biases, strict=True)
        ]


def _scale_response(response: jax.Array, emission_rate: jax.Array | float) -> jax.Array:
    """``Q · r`` for a response vector, ``R q`` for a design matrix."""
    if response.ndim == 1:
        return emission_rate * response
    return response @ jnp.asarray(emission_rate, dtype=response.dtype)


def build_coupled_forward(
    source: PlumeSource | TransportBackend, instruments: Sequence[Instrument]
) -> CoupledForward:
    """Assemble a :class:`CoupledForward`, precomputing unit-``Q`` responses.

    Args:
        source: Shared transport backend: the Tier I ``PlumeSource``, or a
            :class:`~plumax.coupled.backends.LagrangianSource` /
            :class:`~plumax.coupled.backends.EulerianSource`.
        instruments: One or more observing instruments to fuse.

    Returns:
        The configured :class:`CoupledForward`.

    Raises:
        ValueError: if ``instruments`` is empty.
    """
    insts = tuple(instruments)
    if not insts:
        raise ValueError("build_coupled_forward: need at least one instrument.")
    responses = tuple(source.column_response(inst) for inst in insts)
    return CoupledForward(source=source, instruments=insts, responses=responses)


__all__ = [
    "CoupledForward",
    "PlumeSource",
    "build_coupled_forward",
    "column_response",
    "plume_response",
    "predict_observation",
]
