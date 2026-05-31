"""Spectral Response Function primitive."""

from __future__ import annotations

import numpy as np
import pytest

from plumax.radtran.srf import SpectralResponseFunction, _bin_widths


def _make_srf(srf_type: str = "gaussian") -> SpectralResponseFunction:
    wl = np.linspace(1400.0, 2500.0, 1101)
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([1610.0, 2190.0]),
        band_widths_nm=np.array([90.0, 180.0]),
        band_names=("B11", "B12"),
        srf_type=srf_type,
    )


@pytest.mark.parametrize("srf_type", ["gaussian", "rectangular", "triangular"])
def test_srf_matrix_is_L1_normalised(srf_type):
    srf = _make_srf(srf_type)
    np.testing.assert_allclose(srf.matrix.sum(axis=1), 1.0, atol=1e-12)


def test_flat_input_maps_to_flat_output():
    srf = _make_srf("gaussian")
    flat = np.full(srf.n_lambda, 0.3)
    out = srf.apply(flat)
    np.testing.assert_allclose(out, 0.3, atol=1e-12)


def test_apply_preserves_batched_leading_dims():
    srf = _make_srf("gaussian")
    cube = np.random.default_rng(0).random((5, 7, srf.n_lambda))
    out = srf.apply(cube)
    assert out.shape == (5, 7, srf.n_bands)
    # Consistent with per-pixel apply on a single spectrum.
    np.testing.assert_allclose(srf.apply(cube[0, 0]), out[0, 0])


def test_gaussian_peaks_at_band_centre():
    srf = _make_srf("gaussian")
    peak_idx_0 = int(np.argmax(srf.matrix[0]))
    peak_idx_1 = int(np.argmax(srf.matrix[1]))
    assert abs(srf.wavelengths_hr_nm[peak_idx_0] - 1610.0) < 2.0
    assert abs(srf.wavelengths_hr_nm[peak_idx_1] - 2190.0) < 2.0


def test_rectangular_is_top_hat():
    srf = _make_srf("rectangular")
    # Width 90 nm at 1610 nm → nonzero inside [1565, 1655].
    lam = srf.wavelengths_hr_nm
    inside = (lam >= 1565.0) & (lam <= 1655.0)
    assert (srf.matrix[0][inside] > 0).all()
    assert (srf.matrix[0][~inside] == 0).all()


def test_adjoint_is_transpose_of_forward():
    srf = _make_srf("gaussian")
    # < S x, y > = < x, S^T y >  — random vectors.
    rng = np.random.default_rng(1)
    x = rng.normal(size=srf.n_lambda)
    y = rng.normal(size=srf.n_bands)
    lhs = float(srf.apply(x) @ y)
    rhs = float(x @ srf.adjoint(y))
    assert lhs == pytest.approx(rhs, rel=1e-10, abs=1e-10)


def test_jacobian_equals_apply_for_linear_operator():
    srf = _make_srf("gaussian")
    v = np.random.default_rng(2).normal(size=srf.n_lambda)
    np.testing.assert_allclose(srf.apply(v), srf.jacobian(v), atol=1e-12)


def _make_srf_on_grid(wl: np.ndarray, srf_type: str = "gaussian"):
    return SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([1610.0, 2190.0]),
        band_widths_nm=np.array([90.0, 180.0]),
        band_names=("B11", "B12"),
        srf_type=srf_type,
    )


def _trapz_band_reference(srf, spectrum):
    """Independent reference: L_b = ∫ f_b(λ) L(λ) dλ / ∫ f_b(λ) dλ (trapezoid)."""
    raw = srf._build_parametric_matrix()
    wl = np.asarray(srf.wavelengths_hr_nm, dtype=float)
    num = np.array([np.trapezoid(raw[b] * spectrum, x=wl) for b in range(raw.shape[0])])
    den = np.array([np.trapezoid(raw[b], x=wl) for b in range(raw.shape[0])])
    return num / den


def test_uniform_grid_matches_trapezoidal_quadrature():
    # On a uniform grid, band integration must equal a true trapezoidal
    # quadrature (∫f·L / ∫f) — including the half-weighted endpoints.
    wl = np.linspace(1400.0, 2500.0, 1101)
    srf = _make_srf_on_grid(wl, "gaussian")
    spectrum = np.random.default_rng(7).random(srf.n_lambda)
    np.testing.assert_allclose(
        srf.apply(spectrum), _trapz_band_reference(srf, spectrum), rtol=1e-10
    )


def _nonuniform_wavelength_grid() -> np.ndarray:
    # A wavelength grid derived from a uniform wavenumber grid (1e7 / nu) is
    # non-uniform; the constructor requires strictly-increasing samples, so we
    # sort it ascending (the spacing stays non-uniform — finer at long λ).
    nu = np.linspace(4000.0, 7150.0, 1400)  # cm^-1, uniform
    return np.sort(1e7 / nu)  # nm, ascending, non-uniform spacing


def test_nonuniform_grid_flat_input_still_flat():
    wl = _nonuniform_wavelength_grid()
    srf = _make_srf_on_grid(wl, "gaussian")
    np.testing.assert_allclose(srf.matrix.sum(axis=1), 1.0, atol=1e-12)
    out = srf.apply(np.full(srf.n_lambda, 0.3))
    np.testing.assert_allclose(out, 0.3, atol=1e-12)


def test_nonuniform_grid_matches_trapezoidal_quadrature():
    # Band integral must equal a reference width-weighted (trapezoidal) average,
    # not a plain sum that over-weights the densely-sampled side.
    wl = _nonuniform_wavelength_grid()
    srf = _make_srf_on_grid(wl, "gaussian")
    spectrum = np.random.default_rng(3).random(srf.n_lambda)
    np.testing.assert_allclose(
        srf.apply(spectrum), _trapz_band_reference(srf, spectrum), rtol=1e-10
    )


def test_bin_widths_are_trapezoidal_with_half_endpoints():
    # Uniform grid: interior weights = spacing d, endpoints = d/2 (half), and
    # the total equals the full integration span (n-1)*d, not n*d.
    d = 2.5
    s = np.arange(6) * d
    w = _bin_widths(s)
    assert w[0] == pytest.approx(d / 2.0)
    assert w[-1] == pytest.approx(d / 2.0)
    np.testing.assert_allclose(w[1:-1], d)
    assert w.sum() == pytest.approx((s.size - 1) * d)


def test_bin_widths_handle_descending_and_nonuniform():
    s = np.array([10.0, 7.0, 1.0])  # descending, non-uniform
    w = _bin_widths(s)
    # Endpoints: half of their single adjacent interval (|3|/2, |6|/2).
    assert w[0] == pytest.approx(1.5)
    assert w[-1] == pytest.approx(3.0)
    assert w[1] == pytest.approx(1.5 + 3.0)
    assert np.all(w > 0)


def test_bin_widths_endpoint_response_not_overweighted():
    # A custom SRF whose only support is the first sample must integrate to that
    # sample's value — the endpoint must not be double-weighted (the np.gradient
    # bug would inflate it). Row normalisation makes the band a pure passthrough
    # of the edge sample regardless of weight, so we assert via apply on a ramp.
    wl = np.linspace(1500.0, 2500.0, 11)
    mat = np.zeros((1, wl.size))
    mat[0, 0] = 1.0  # response only at the grid edge
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([1500.0]),
        band_widths_nm=np.array([50.0]),
        band_names=("edge",),
        srf_type="custom",
        custom_srfs=mat,
    )
    spectrum = np.linspace(1.0, 2.0, wl.size)
    np.testing.assert_allclose(srf.apply(spectrum), spectrum[0], rtol=1e-12)


def test_apply_rejects_wrong_last_axis():
    srf = _make_srf("gaussian")
    with pytest.raises(ValueError, match="last axis size"):
        srf.apply(np.zeros(srf.n_lambda - 1))


def test_custom_srf_requires_matrix():
    wl = np.linspace(1400.0, 2500.0, 101)
    with pytest.raises(ValueError, match="custom_srfs"):
        SpectralResponseFunction(
            wavelengths_hr_nm=wl,
            band_centers_nm=np.array([1610.0]),
            band_widths_nm=np.array([90.0]),
            band_names=("B11",),
            srf_type="custom",
        )


def test_custom_srf_accepted_and_normalised():
    wl = np.linspace(1400.0, 2500.0, 101)
    mat = np.zeros((1, 101))
    mat[0, 50:55] = 1.0  # unnormalised
    srf = SpectralResponseFunction(
        wavelengths_hr_nm=wl,
        band_centers_nm=np.array([wl[52]]),
        band_widths_nm=np.array([10.0]),
        band_names=("B0",),
        srf_type="custom",
        custom_srfs=mat,
    )
    assert srf.matrix.sum() == pytest.approx(1.0, rel=1e-12)


def test_zero_total_response_raises():
    wl = np.linspace(1400.0, 2500.0, 101)
    with pytest.raises(ValueError, match="zero total response"):
        SpectralResponseFunction(
            wavelengths_hr_nm=wl,
            band_centers_nm=np.array([800.0]),  # well outside the grid
            band_widths_nm=np.array([10.0]),
            band_names=("B0",),
            srf_type="rectangular",
        )


def test_rejects_nonmonotonic_wavelengths():
    with pytest.raises(ValueError, match="strictly increasing"):
        SpectralResponseFunction(
            wavelengths_hr_nm=np.array([1500.0, 1400.0, 2000.0]),
            band_centers_nm=np.array([1610.0]),
            band_widths_nm=np.array([90.0]),
            band_names=("B11",),
        )


def test_rejects_unknown_srf_type():
    wl = np.linspace(1400.0, 2500.0, 101)
    with pytest.raises(ValueError, match="unknown srf_type"):
        SpectralResponseFunction(
            wavelengths_hr_nm=wl,
            band_centers_nm=np.array([1610.0]),
            band_widths_nm=np.array([90.0]),
            band_names=("B11",),
            srf_type="sinc",  # type: ignore[arg-type]
        )
