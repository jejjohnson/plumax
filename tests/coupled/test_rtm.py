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
    LinearisedRadianceOperator,
    PlumeSource,
    RadianceObservationOperator,
    column_mass_to_delta_vmr,
    linearise,
    radiance_response,
    radiance_response_linear,
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


def test_column_mass_to_delta_vmr_differentiable_in_geometry():
    # The conversion is advertised as pure-JAX: jit/grad over the (traced)
    # geometry must work — the ideal-gas density is inlined with jnp rather than
    # routed through the NumPy radtran.config helper.
    import jax

    @jax.jit
    def dvmr_of_p(p):
        return column_mass_to_delta_vmr(1e-3, p_atm=p, T_K=290.0, path_length_cm=8.4e5)

    val = dvmr_of_p(1.0)
    assert np.isfinite(float(val))
    # ΔVMR ∝ 1/p_air, so the pressure derivative is finite and negative.
    g = jax.grad(dvmr_of_p)(1.0)
    assert np.isfinite(float(g))
    assert float(g) < 0.0
    # Matches the concrete value.
    ref = column_mass_to_delta_vmr(1e-3, p_atm=1.0, T_K=290.0, path_length_cm=8.4e5)
    np.testing.assert_allclose(float(val), float(ref), rtol=1e-12)


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


def test_srf_band_integration_aligns_spectrum_to_wavelength_grid(synthetic_lut, nu_obs):
    # Regression: the spectrum is on nu_obs (decreasing wavelength) while the
    # SRF grid is increasing wavelength. The band radiance must match a manual
    # wavelength-aligned integral; a reversed contraction (the bug) would differ
    # for an ASYMMETRIC SRF, so use an edge-weighted (non-symmetric) band.
    wl = np.sort(1e7 / nu_obs)  # ascending nm
    # Triangular SRF centred near the short-wavelength edge → asymmetric over the
    # observed window, so sample order matters.
    centre = float(wl[0] + 0.25 * (wl[-1] - wl[0]))
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([centre]),
        band_widths_nm=np.array([0.5 * (wl[-1] - wl[0])]),
        band_names=("edge",),
        srf_type="triangular",
    )
    op = _operator(synthetic_lut, nu_obs, srf=srf)
    band = op.predict_single(3e-3)

    # Manual reference: HR spectrum interpolated onto the SRF wavelength grid,
    # then SRF-applied — i.e. the correct (forward) ordering.
    dvmr = float(op.delta_vmr(3e-3))
    spec = np.asarray(op._forward_hr(dvmr).radiance)
    wl_obs = 1e7 / np.asarray(nu_obs, dtype=float)
    order = np.argsort(wl_obs)
    spec_on_grid = np.interp(wl, wl_obs[order], spec[order])
    ref = srf.apply(spec_on_grid)
    np.testing.assert_allclose(band, ref, rtol=1e-12)


def test_srf_grid_wider_than_nu_obs_is_rejected(synthetic_lut, nu_obs):
    # An SRF grid reaching outside the modelled nu_obs window would make
    # np.interp repeat the edge spectrum (extrapolation), biasing the band —
    # so it must be rejected rather than silently extrapolated.
    wl = np.sort(1e7 / nu_obs)
    pad = 0.2 * (wl[-1] - wl[0])
    wide = np.linspace(wl[0] - pad, wl[-1] + pad, wl.size)  # extends both edges
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wide,
        band_centers_nm=np.array([float(np.mean(wide))]),
        band_widths_nm=np.array([0.5 * (wide[-1] - wide[0])]),
        band_names=("wide",),
        srf_type="gaussian",
    )
    op = _operator(synthetic_lut, nu_obs, srf=srf)
    with pytest.raises(ValueError, match="extends beyond the modelled nu_obs"):
        op.predict_single(1e-3)


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


# ── linearised (tangent-linear) radiance operator ─────────────────────────────


def test_linearise_baseline_and_gain_shapes(synthetic_lut, nu_obs):
    lin = linearise(_operator(synthetic_lut, nu_obs))
    assert isinstance(lin, LinearisedRadianceOperator)
    assert lin.n_channels == nu_obs.size
    # Background radiance is the normalised unit (exp(0) = 1) on every channel.
    np.testing.assert_allclose(np.asarray(lin.baseline), 1.0, atol=1e-12)
    # Absorbing channels have negative gain (more mass → less radiance).
    assert float(jnp.min(lin.gain)) < 0.0


def test_linearised_matches_nonlinear_to_first_order(synthetic_lut, nu_obs):
    # For a small enhancement the linearised radiance ≈ the exact Beer–Lambert
    # forward; the tangent is exact at 0 and first-order accurate near it.
    op = _operator(synthetic_lut, nu_obs)
    lin = linearise(op)
    small = 1e-4  # small column mass [kg/m²]
    exact = np.asarray(op.predict_single(small))
    approx = np.asarray(lin.predict_single(small))
    np.testing.assert_allclose(approx, exact, rtol=0, atol=5e-4)
    # Exact at zero enhancement.
    np.testing.assert_allclose(
        np.asarray(lin.predict_single(0.0)),
        np.asarray(op.predict_single(0.0)),
        atol=1e-12,
    )


def test_linearised_is_linear_in_column_mass(synthetic_lut, nu_obs):
    lin = linearise(_operator(synthetic_lut, nu_obs))
    # predict(m) = baseline + gain·m exactly — assert against the analytic gain
    # rather than differencing the (≈1) baseline, which would lose all precision
    # to catastrophic cancellation on the tiny absorption signal.
    m = 2e-3
    np.testing.assert_allclose(
        np.asarray(lin.predict_single(m)),
        np.asarray(lin.baseline) + m * np.asarray(lin.gain),
        rtol=1e-12,
    )


def test_linearised_predict_vectorised(synthetic_lut, nu_obs):
    lin = linearise(_operator(synthetic_lut, nu_obs))
    cols = jnp.array([0.0, 1e-3, 2e-3])
    out = lin.predict(cols)
    assert out.shape == (3, nu_obs.size)
    # Row 0 (zero mass) is the baseline.
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(lin.baseline), atol=1e-12)


def test_linearised_is_jittable_and_differentiable(synthetic_lut, nu_obs):
    # The linearised operator is pure JAX (no NumPy LUT path), so it composes
    # under jit / grad — the property that lets it feed gradient-based and
    # closed-form inversions.
    import jax

    lin = linearise(_operator(synthetic_lut, nu_obs))

    @jax.jit
    def total(mass):
        return jnp.sum(lin.predict(mass))

    cols = jnp.array([1e-3, 2e-3])
    val = total(cols)
    assert np.isfinite(float(val))
    g = jax.grad(total)(cols)
    assert g.shape == (2,)
    assert np.all(np.isfinite(np.asarray(g)))


def test_radiance_response_linear_is_linear_in_Q(synthetic_lut, nu_obs):
    # radiance_response_linear is linear in the emission rate Q (unlike the
    # nonlinear radiance_response), enabling closed-form / gradient inversion.
    src = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 21),
    )
    inst = Instrument(
        "RTMLIN",
        np.column_stack([np.linspace(200.0, 1500.0, 6), np.zeros(6)]),
    )
    lin = linearise(_operator(synthetic_lut, nu_obs))
    base = radiance_response_linear(src, inst, lin, emission_rate=0.0)
    r1 = radiance_response_linear(src, inst, lin, emission_rate=1.0)
    r2 = radiance_response_linear(src, inst, lin, emission_rate=2.0)
    assert r1.shape == (6, nu_obs.size)
    # (response(Q) − response(0)) scales linearly with Q. Off-line channels are
    # ~0 ± roundoff, so floor the absolute tolerance to the signal magnitude
    # rather than demanding pure relative agreement on numerical dust.
    d1 = np.asarray(r1) - np.asarray(base)
    d2 = np.asarray(r2) - np.asarray(base)
    np.testing.assert_allclose(d2, 2.0 * d1, rtol=1e-6, atol=1e-9 * np.abs(d1).max())
