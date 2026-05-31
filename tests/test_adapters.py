"""Tests for the pipekit_cycle protocol adapters in plumax.adapters.

These assert *structural* conformance: the adapters expose exactly the methods
and properties the runtime-checkable protocols require, so ``isinstance``
against ``pipekit_cycle.protocols`` succeeds even though plumax never imports
that package at runtime.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import xarray as xr
from pipekit_cycle.protocols import ForwardModel, ObservationOperator

from plumax.adapters import EulerianForwardModel, RadtranObservationOperator


class _ConstRHS:
    """Minimal RHS stub: a unit tendency everywhere."""

    def __call__(self, t, concentration, args=None):
        return jnp.ones_like(concentration)


def _toy_lut() -> xr.Dataset:
    nu = np.linspace(4000.0, 4500.0, 51)
    T_grid = np.array([260.0, 300.0])
    P_grid = np.array([0.5, 1.0])
    sigma = np.zeros((nu.size, T_grid.size, P_grid.size))
    for i_T in range(T_grid.size):
        for i_P in range(P_grid.size):
            sigma[:, i_T, i_P] = 3e-21 * np.exp(-0.5 * ((nu - 4300.0) / 5.0) ** 2)
    return xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"],
                sigma,
            )
        },
        coords={"wavenumber": nu, "temperature": T_grid, "pressure": P_grid},
    )


def test_eulerian_forward_model_satisfies_protocol():
    fm = EulerianForwardModel(_ConstRHS(), dt=0.5)
    assert isinstance(fm, ForwardModel)
    assert fm.dt == 0.5
    assert fm.state_signature is None


def test_eulerian_forward_model_step():
    fm = EulerianForwardModel(_ConstRHS(), dt=0.5)
    state = jnp.zeros((3, 3, 3))
    advanced = fm.step(state, 0.5)
    # c <- c + dt * 1 = 0.5 everywhere.
    assert jnp.allclose(advanced, 0.5)


def test_radtran_obs_operator_satisfies_protocol():
    H = RadtranObservationOperator(
        _toy_lut(),
        np.linspace(4100.0, 4400.0, 25),
        T_K=300.0,
        p_atm=1.0,
        vmr_background=1.9e-6,
        path_length_cm=1.0e5,
        amf=2.0,
    )
    assert isinstance(H, ObservationOperator)


def test_radtran_obs_operator_zero_enhancement_is_unit_radiance():
    nu_obs = np.linspace(4100.0, 4400.0, 25)
    H = RadtranObservationOperator(
        _toy_lut(),
        nu_obs,
        T_K=300.0,
        p_atm=1.0,
        vmr_background=1.9e-6,
        path_length_cm=1.0e5,
        amf=2.0,
    )
    # Normalised radiance at ΔVMR = 0 is exp(0) = 1 across the band.
    radiance = H(0.0)
    assert radiance.shape == nu_obs.shape
    assert np.allclose(radiance, 1.0)
    # A positive enhancement absorbs -> normalised radiance drops below 1.
    assert np.all(H(1.0e-6) <= 1.0)
    # Jacobian has the spectral shape and is non-positive (absorption).
    jac = H.linearize(1.0e-6)
    assert jac.shape == nu_obs.shape
    assert np.all(jac <= 0.0)
