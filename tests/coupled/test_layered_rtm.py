"""The layered radiance operator on radtrax's ForwardModel (plumax#135).

Two checks from the issue: with one layer on a table node, an identity
instrument and no shift it equals the slab operator to 1e-12 (radtrax gate
V1's setting); and ``jax.grad`` of a radiance misfit with respect to the
emission rate, through the plume model and a three-layer atmosphere, matches
central differences.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from radtrax import ForwardModel
from radtrax.atmosphere import (
    ATM_TO_PA,
    AtmosphericState,
    ViewingGeometry,
    air_mass_factor,
    layers_from_levels,
)
from radtrax.instrument import BandInstrument
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.surface import Lambertian

from plumax.coupled import (
    Instrument,
    LayeredRadianceOperator,
    PlumeSource,
    RadianceObservationOperator,
    linearise,
    radiance_response,
)


SZA, VZA = 35.0, 10.0
T0, P0, X0, DZ_CM = 260.0, 1.0, 1.8e-6, 8.4e5  # a node of the table below


@pytest.fixture
def synthetic_lut() -> xr.Dataset:
    """σ(ν, T, P): two Gaussian lines on a coarse ν/T/P grid (as test_rtm.py)."""
    nu = np.linspace(4000.0, 4500.0, 201)
    T_grid = np.array([220.0, 260.0, 300.0])
    P_grid = np.array([0.5, 1.0])
    sigma = np.zeros((nu.size, T_grid.size, P_grid.size))
    for i_T, T in enumerate(T_grid):
        for i_P, p in enumerate(P_grid):
            for c, s, w in ((4200.0, 1e-21, 3.0), (4300.0, 3e-21, 5.0)):
                sigma[:, i_T, i_P] += (
                    s
                    * np.exp(-(300.0 - T) / 150.0)
                    * np.exp(-0.5 * ((nu - c) / (w * p)) ** 2)
                )
    return xr.Dataset(
        {
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"],
                sigma,
            )
        },
        coords={
            "wavenumber": nu,
            "wavelength": ("wavenumber", 1e7 / nu),
            "temperature": T_grid,
            "pressure": P_grid,
        },
    )


def _identity_model(ds: xr.Dataset) -> ForwardModel:
    nu = ds["wavenumber"].values
    return ForwardModel(
        TableOpacity.from_datasets({"CH4": ds}),
        BeerLambert(),
        BandInstrument(nu, np.eye(nu.size)),
    )


def _one_layer() -> AtmosphericState:
    return AtmosphericState(
        p_pa=jnp.array([P0 * ATM_TO_PA]),
        t_k=jnp.array([T0]),
        z_m=jnp.array([DZ_CM / 200]),
        dz_m=jnp.array([DZ_CM / 100]),
        vmr={"CH4": jnp.array([X0])},
    )


def _three_layers() -> AtmosphericState:
    """Inside the table: 1.0 to 0.5 atm, 290 to 240 K."""
    return layers_from_levels(
        jnp.array([1.0, 0.85, 0.7, 0.55]) * ATM_TO_PA,
        jnp.array([290.0, 275.0, 260.0, 245.0]),
        {"CH4": jnp.full(4, X0)},
    )


def _two_band_model(ds: xr.Dataset) -> ForwardModel:
    """Two boxcar bands, one on each line."""
    nu = ds["wavenumber"].values
    response = np.stack(
        [np.where(np.abs(nu - c) <= 20.0, 1.0, 0.0) for c in (4200.0, 4300.0)]
    )
    response /= response.sum(axis=1, keepdims=True)
    return ForwardModel(
        TableOpacity.from_datasets({"CH4": ds}),
        BeerLambert(),
        BandInstrument(nu, response),
    )


def _source() -> PlumeSource:
    return PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 21),
    )


def _receptors() -> np.ndarray:
    gx, gy = np.meshgrid(
        np.linspace(100.0, 1500.0, 5), np.linspace(-200.0, 200.0, 4), indexing="ij"
    )
    return np.column_stack([gx.reshape(-1), gy.reshape(-1)])


def test_one_layer_on_a_node_equals_the_slab(synthetic_lut):
    nu = synthetic_lut["wavenumber"].values
    slab = RadianceObservationOperator(
        ds=synthetic_lut,
        nu_obs=nu,
        T_K=T0,
        p_atm=P0,
        vmr_background=X0,
        path_length_cm=DZ_CM,
        amf=float(air_mass_factor(SZA, VZA)),
    )
    layered = LayeredRadianceOperator(
        _identity_model(synthetic_lut),
        _one_layer(),
        Lambertian(0.3),
        ViewingGeometry.from_angles(SZA, VZA),
    )
    masses = np.array([0.0, 0.05, 0.4, 2.0])  # kg/m²
    np.testing.assert_allclose(
        np.asarray(layered.predict(masses)), slab.predict(masses), rtol=1e-12
    )
    assert float(slab.predict(masses).min()) < 0.5  # the lines absorb


def test_zero_enhancement_is_ones_and_a_plume_absorbs(synthetic_lut):
    op = LayeredRadianceOperator(
        _two_band_model(synthetic_lut),
        _three_layers(),
        Lambertian(0.3),
        ViewingGeometry.from_angles(SZA, VZA),
        profile=jnp.array([0.6, 0.3, 0.1]),
    )
    assert op.n_channels == 2
    np.testing.assert_array_equal(np.asarray(op.predict_single(0.0)), 1.0)
    assert np.all(np.asarray(op.predict_single(0.5)) < 1.0)
    with pytest.raises(ValueError, match="must be 1-D"):
        op.predict(jnp.ones((2, 2)))


def test_linearise_is_the_tangent_at_zero(synthetic_lut):
    op = LayeredRadianceOperator(
        _two_band_model(synthetic_lut),
        _three_layers(),
        Lambertian(0.3),
        ViewingGeometry.from_angles(SZA, VZA),
    )
    lin = linearise(op)
    np.testing.assert_array_equal(np.asarray(lin.baseline), 1.0)
    h = 1e-6  # kg/m²
    central = (op.predict_single(h) - op.predict_single(-h)) / (2 * h)
    np.testing.assert_allclose(np.asarray(lin.gain), np.asarray(central), rtol=1e-7)


def test_grad_of_misfit_in_emission_rate_matches_central_differences(synthetic_lut):
    op = LayeredRadianceOperator(
        _two_band_model(synthetic_lut),
        _three_layers(),
        Lambertian(0.3),
        ViewingGeometry.from_angles(SZA, VZA),
        profile=jnp.array([0.6, 0.3, 0.1]),
    )
    source, inst = _source(), Instrument("EMIT", _receptors())
    y_obs = radiance_response(source, inst, op, 40.0)

    def misfit(q: jax.Array) -> jax.Array:
        return jnp.sum((radiance_response(source, inst, op, q) - y_obs) ** 2)

    q, h = 25.0, 1e-3
    grad = float(jax.grad(misfit)(jnp.asarray(q)))
    central = float((misfit(q + h) - misfit(q - h)) / (2 * h))
    assert grad < 0.0  # below the truth, the misfit falls as Q rises
    np.testing.assert_allclose(grad, central, rtol=1e-6)
