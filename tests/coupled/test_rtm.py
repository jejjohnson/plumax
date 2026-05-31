"""Tests for the Tier IV RTM-based coupled observation operator.

Uses a HAPI-free synthetic absorption-cross-section LUT (the same fixture
pattern as ``tests/radtran/conftest.py``) so nothing here needs HAPI or the
network.
"""

from __future__ import annotations

from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from plumax.coupled import (
    Instrument,
    PlumeSource,
    RadianceObservationOperator,
    column_mass_to_delta_vmr,
    radiance_response,
)
from plumax.radtran.forward import forward_nonlinear_normalized
from plumax.radtran.srf import SpectralResponseFunction


# ── fixtures (mirror tests/radtran/conftest.py — no HAPI / network) ────────────


@pytest.fixture
def synthetic_lut() -> xr.Dataset:
    """Tiny σ(ν, T, P) LUT: 2 Gaussian lines on a coarse ν/T/P grid."""
    nu = np.linspace(4000.0, 4500.0, 201)  # cm^-1
    wl = 1e7 / nu  # nm
    T_grid = np.array([220.0, 260.0, 300.0])
    P_grid = np.array([0.5, 1.0])

    centres = np.array([4200.0, 4300.0])
    strengths_T = np.array([1e-21, 3e-21])
    widths = np.array([3.0, 5.0])

    sigma = np.zeros((nu.size, T_grid.size, P_grid.size), dtype=float)
    for i_T, T in enumerate(T_grid):
        temp_fac = np.exp(-(300.0 - T) / 150.0)
        for i_P, p in enumerate(P_grid):
            for c, s, w in zip(centres, strengths_T, widths, strict=True):
                sigma[:, i_T, i_P] += (
                    s * temp_fac * np.exp(-0.5 * ((nu - c) / (w * p)) ** 2)
                )

    return xr.Dataset(
        data_vars={
            "absorption_cross_section": (
                ["wavenumber", "temperature", "pressure"],
                sigma,
                {"units": "cm^2 / molecule"},
            ),
        },
        coords={
            "wavenumber": (["wavenumber"], nu, {"units": "cm^-1"}),
            "wavelength": (["wavenumber"], wl, {"units": "nm"}),
            "temperature": (["temperature"], T_grid, {"units": "K"}),
            "pressure": (["pressure"], P_grid, {"units": "atm"}),
        },
        attrs={"molecule": "TOY_CH4"},
    )


@pytest.fixture
def nu_obs() -> np.ndarray:
    return np.linspace(4150.0, 4350.0, 64)


def _operator(ds, nu, *, srf=None) -> RadianceObservationOperator:
    return RadianceObservationOperator(
        ds=ds,
        nu_obs=nu,
        T_K=290.0,
        p_atm=1.0,
        vmr_background=1.8e-6,
        path_length_cm=8.4e5,
        amf=2.0,
        srf=srf,
    )


def _band_srf(nu: np.ndarray) -> SpectralResponseFunction:
    """A single-band SRF over the wavelengths corresponding to ``nu_obs``."""
    wl = np.sort(1e7 / nu)  # nm, ascending (SRF requires increasing grid)
    centre = float(np.mean(wl))
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([centre]),
        band_widths_nm=np.array([float(wl[-1] - wl[0])]),
        band_names=("B0",),
        srf_type="gaussian",
    )


# ── column mass → ΔVMR conversion ──────────────────────────────────────────────


def test_column_mass_to_delta_vmr_zero_and_linear():
    z = column_mass_to_delta_vmr(0.0, p_atm=1.0, T_K=290.0, path_length_cm=8.4e5)
    assert float(z) == 0.0
    # Linear in column mass.
    a = float(
        column_mass_to_delta_vmr(1e-3, p_atm=1.0, T_K=290.0, path_length_cm=8.4e5)
    )
    b = float(
        column_mass_to_delta_vmr(2e-3, p_atm=1.0, T_K=290.0, path_length_cm=8.4e5)
    )
    assert a > 0.0
    assert b == pytest.approx(2.0 * a, rel=1e-10)


def test_column_mass_to_delta_vmr_validates_geometry():
    with pytest.raises(ValueError, match="must be > 0"):
        column_mass_to_delta_vmr(1e-3, p_atm=-1.0, T_K=290.0, path_length_cm=8.4e5)


# ── (1) zero enhancement → normalised radiance ≈ 1 ─────────────────────────────


def test_zero_enhancement_unit_radiance(synthetic_lut, nu_obs):
    op = _operator(synthetic_lut, nu_obs)
    out = op.predict_single(0.0)
    assert out.shape == (nu_obs.size,)
    np.testing.assert_allclose(out, 1.0, rtol=0, atol=1e-12)


# ── (2) positive enhancement reduces radiance, monotonically ───────────────────


def test_positive_enhancement_monotonic_absorption(synthetic_lut, nu_obs):
    op = _operator(synthetic_lut, nu_obs)
    masses = [0.0, 1e-3, 2e-3, 4e-3]
    # Minimum over band = deepest absorption; must decrease with column mass.
    mins = [float(op.predict_single(m).min()) for m in masses]
    assert all(mn <= 1.0 + 1e-12 for mn in mins)
    assert all(later < earlier for earlier, later in pairwise(mins))
    # Mean normalised radiance also decreases (more absorption overall).
    means = [float(op.predict_single(m).mean()) for m in masses]
    assert all(later < earlier for earlier, later in pairwise(means))


# ── (3) Jacobian sign / shape ──────────────────────────────────────────────────


def test_jacobian_sign_and_shape(synthetic_lut, nu_obs):
    op = _operator(synthetic_lut, nu_obs)
    jac = op.linearize_single(2e-3)
    assert jac.shape == (nu_obs.size,)
    # d(L_norm)/d(ΔVMR) <= 0 (more gas absorbs more); strictly < 0 on the lines.
    assert np.all(jac <= 1e-15)
    assert jac.min() < 0.0
    # Finite-difference check against predict_single in ΔVMR space. The atol is
    # set relative to the Jacobian magnitude: off the absorption lines the
    # cross-section is ~0 and both sides are FD roundoff, which a tight absolute
    # tolerance would flag spuriously.
    dvmr0 = float(op.delta_vmr(2e-3))
    eps = dvmr0 * 1e-4
    f0 = op._forward_hr(dvmr0).radiance
    f1 = op._forward_hr(dvmr0 + eps).radiance
    fd = (f1 - f0) / eps
    np.testing.assert_allclose(jac, fd, rtol=1e-3, atol=1e-3 * np.abs(jac).max())


def test_srf_band_integration_shape(synthetic_lut, nu_obs):
    srf = _band_srf(nu_obs)
    op = _operator(synthetic_lut, nu_obs, srf=srf)
    assert op.n_channels == 1
    assert op.predict_single(0.0).shape == (1,)
    np.testing.assert_allclose(op.predict_single(0.0), 1.0, atol=1e-12)
    # Band-integrated radiance still drops below 1 under absorption.
    assert float(op.predict_single(2e-3)[0]) < 1.0
    assert op.linearize_single(2e-3).shape == (1,)


# ── (4) composition with the coupled forward → per-receptor radiances ──────────


def _grid(nx, ny, x0, x1, y0, y1):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.reshape(-1), gy.reshape(-1)])


def _source():
    return PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 21),
    )


def test_radiance_response_shape_and_absorption(synthetic_lut, nu_obs):
    source = _source()
    inst = Instrument("EMIT", _grid(5, 4, 100.0, 1500.0, -200.0, 200.0))
    op = _operator(synthetic_lut, nu_obs)

    out = radiance_response(source, inst, op, emission_rate=1e3)
    assert out.shape == (inst.n_obs, nu_obs.size)
    # Every normalised radiance is in (0, 1].
    assert np.all(out > 0.0)
    assert np.all(out <= 1.0 + 1e-9)
    # At least some receptors (near the plume) show absorption < 1.
    assert float(out.min()) < 1.0
    # Zero emission → no absorption anywhere.
    out0 = radiance_response(source, inst, op, emission_rate=0.0)
    np.testing.assert_allclose(out0, 1.0, atol=1e-12)


def test_predict_requires_1d(synthetic_lut, nu_obs):
    op = _operator(synthetic_lut, nu_obs)
    with pytest.raises(ValueError, match="must be 1-D"):
        op.predict(np.zeros((3, 2)))


# ── (5) twin check: operator == radtran.forward_nonlinear_normalized ───────────


def test_twin_consistency_single_receptor(synthetic_lut, nu_obs):
    op = _operator(synthetic_lut, nu_obs)
    mass = 3e-3
    dvmr = float(op.delta_vmr(mass))
    ref = forward_nonlinear_normalized(
        synthetic_lut,
        nu_obs,
        T_K=290.0,
        p_atm=1.0,
        vmr_background=1.8e-6,
        delta_vmr=dvmr,
        path_length_cm=8.4e5,
        amf=2.0,
    )
    np.testing.assert_allclose(op.predict_single(mass), ref.radiance, rtol=1e-12)
    np.testing.assert_allclose(op.linearize_single(mass), ref.jacobian, rtol=1e-12)
