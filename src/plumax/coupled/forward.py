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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from plumax.coupled.instrument import Instrument
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.plume import plume_concentration_vmap


@dataclass(frozen=True)
class PlumeSource:
    """Static source + meteorology for the Tier I coupled forward.

    The emission rate ``Q`` is deliberately *not* stored here — it is the scalar
    the inversion solves for, so the forward is parameterised by everything
    *except* ``Q`` (location, wind, stability, background) and is linear in it.

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
        theta = np.deg2rad(self.wind_direction)
        u = -self.wind_speed * np.sin(theta)
        v = -self.wind_speed * np.cos(theta)
        return float(u), float(v)


def column_response(source: PlumeSource, instrument: Instrument) -> jax.Array:
    """Per-receptor column enhancement at **unit emission** for one instrument.

    Evaluates the Gaussian plume at unit ``Q`` on each receptor column, integrates
    over ``source.column_z`` (trapezoidal), and applies the instrument averaging
    kernel. The full observation at emission ``Q`` is then
    ``Q · column_response + bias`` (plus the background; see
    :func:`predict_observation`). Linearity in ``Q`` is what makes the coupled
    inversion closed-form.

    Args:
        source: The static source + met configuration.
        instrument: The observing instrument (receptors + AK).

    Returns:
        Unit-``Q`` AK-weighted column response, shape ``(n_obs,)``.
    """
    z = jnp.asarray(source.column_z, dtype=float)
    receptors = jnp.asarray(instrument.receptors, dtype=float)
    n_obs = receptors.shape[0]
    n_z = z.shape[0]

    # Build (n_obs * n_z) receptor points: every (x, y) crossed with every z.
    x = jnp.repeat(receptors[:, 0], n_z)
    y = jnp.repeat(receptors[:, 1], n_z)
    z_tiled = jnp.tile(z, n_obs)

    sx, sy, sz = source.location
    u, v = source.wind_uv()
    dispersion = get_dispersion_params(source.stability_class)

    conc = plume_concentration_vmap(
        x, y, z_tiled, sx, sy, sz, u, v, 1.0, dispersion
    )  # unit Q
    conc = conc.reshape(n_obs, n_z)
    column = jnp.trapezoid(conc, x=z, axis=1)  # (n_obs,)
    return instrument.ak * column


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
        source: Shared source + met configuration.
        instruments: The observing instruments (fusion list).
        responses: Per-instrument unit-``Q`` column responses (one ``(n_obs,)``
            array each), aligned with ``instruments``.
    """

    source: PlumeSource
    instruments: tuple[Instrument, ...]
    responses: tuple[jax.Array, ...]

    def predict(
        self,
        emission_rate: jax.Array | float,
        biases: Sequence[jax.Array | float] | None = None,
    ) -> list[jax.Array]:
        """Predicted observation vector per instrument (list, native resolution).

        Args:
            emission_rate: Emission rate ``Q`` [kg/s].
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
            emission_rate * resp + self.source.background + b
            for resp, b in zip(self.responses, biases, strict=True)
        ]


def build_coupled_forward(
    source: PlumeSource, instruments: Sequence[Instrument]
) -> CoupledForward:
    """Assemble a :class:`CoupledForward`, precomputing unit-``Q`` responses.

    Args:
        source: Shared source + met configuration.
        instruments: One or more observing instruments to fuse.

    Returns:
        The configured :class:`CoupledForward`.

    Raises:
        ValueError: if ``instruments`` is empty.
    """
    insts = tuple(instruments)
    if not insts:
        raise ValueError("build_coupled_forward: need at least one instrument.")
    responses = tuple(column_response(source, inst) for inst in insts)
    return CoupledForward(source=source, instruments=insts, responses=responses)


__all__ = [
    "CoupledForward",
    "PlumeSource",
    "build_coupled_forward",
    "column_response",
    "predict_observation",
]
