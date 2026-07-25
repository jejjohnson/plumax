"""Spectral Response Function — band integration of hyperspectral radiance.

The HAPI LUTs in :mod:`plumax.hapi_lut` tabulate σ(ν, T, P) on a
fine line-by-line wavenumber grid. Real instruments observe *band-integrated*
radiance

    L_b = ∫ f_b(λ) · L(λ) dλ          / ∫ f_b(λ) dλ

where ``f_b(λ)`` is the instrument's Spectral Response Function for band ``b``.
This module builds the ``(n_bands, n_λ)`` SRF matrix once — each row is a
normalised response profile — and applies it as a plain matrix multiplication
against a high-resolution radiance spectrum (1-D) or radiance cube (3-D).

The class mirrors the API in the Orbio Project Eucalyptus tutorial
(``SpectralFilter`` + ``get_sensor_spectral_response_function``) but exposes
a linear-operator view: ``apply`` (forward), ``jacobian`` (tangent action —
identical to forward since SRF is linear), and ``adjoint`` (``Sᵀ`` for
reverse-mode / VJP use in 3DVar). These three methods match the operator
interface used by the observation-operator classes in the original
``jej_vc_snippets/methane_retrieval/lut_obs_op.py``.

Three parametric SRF shapes are supported:

- ``'gaussian'``   : ``exp[-(λ-λ_c)² / (2 σ²)]`` with ``FWHM = 2.355 σ``.
- ``'rectangular'`` : 1 inside ``[λ_c − W/2, λ_c + W/2]``, 0 outside.
- ``'triangular'``  : linear ramp peaking at ``λ_c``, zero at the edges.

Plus ``'custom'`` which accepts an arbitrary ``(n_bands, n_λ)`` matrix
(each row gets normalised to sum to 1).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np


SRFType = Literal["gaussian", "rectangular", "triangular", "custom"]


def _bin_widths(samples: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for (possibly non-uniform) 1-D samples.

    Each interior sample gets half of each neighbouring interval and the
    endpoints get a single half-interval — the standard trapezoidal-rule
    weights. Works for ascending or descending grids (e.g. wavelengths from
    ``1e7 / nu``); a single-sample grid falls back to unit weight.
    """
    s = np.asarray(samples, dtype=float)
    if s.size < 2:
        return np.ones_like(s)
    # Each interval contributes half its width to the sample on either side, so
    # interior samples get half of both neighbouring intervals and the endpoints
    # get a single half-interval — the exact trapezoidal-rule weights. (Unlike
    # ``np.gradient``, which gives the endpoints a full interval and would
    # over-weight an SRF with response at the grid edge by 2×.) ``abs`` makes
    # the result independent of ascending/descending sample order.
    half = np.abs(np.diff(s)) / 2.0
    widths = np.zeros_like(s)
    widths[:-1] += half
    widths[1:] += half
    return widths


@dataclass
class SpectralResponseFunction:
    """Band-integration operator for a multispectral / hyperspectral instrument.

    Attributes
    ----------
    wavelengths_hr_nm : np.ndarray
        High-resolution wavelength grid [nm], shape ``(n_lambda,)``. Must be
        monotone increasing.
    band_centers_nm : np.ndarray
        Band centre wavelengths [nm], shape ``(n_bands,)``.
    band_widths_nm : np.ndarray
        Band FWHM (for 'gaussian'), full-width (for 'rectangular'/'triangular')
        [nm], shape ``(n_bands,)``.
    band_names : tuple of str
        Band labels, length ``n_bands``.
    srf_type : str
        'gaussian' | 'rectangular' | 'triangular' | 'custom'.
    custom_srfs : np.ndarray or None
        Used only when ``srf_type == 'custom'``. Shape ``(n_bands, n_lambda)``.

    Notes
    -----
    Rows of :attr:`matrix` are L1-normalised so that a flat input spectrum
    ``L(λ) = c`` maps to ``L_b = c`` for every band; this matches the
    "dot-product-in-frequency" convention in the Eucalyptus tutorial.
    """

    wavelengths_hr_nm: np.ndarray
    band_centers_nm: np.ndarray
    band_widths_nm: np.ndarray
    band_names: tuple[str, ...] = ("B0",)
    srf_type: SRFType = "gaussian"
    custom_srfs: np.ndarray | None = None
    matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.wavelengths_hr_nm = np.asarray(self.wavelengths_hr_nm, dtype=float)
        self.band_centers_nm = np.asarray(self.band_centers_nm, dtype=float)
        self.band_widths_nm = np.asarray(self.band_widths_nm, dtype=float)
        self.band_names = tuple(self.band_names)

        if self.wavelengths_hr_nm.ndim != 1:
            raise ValueError(
                "SpectralResponseFunction: `wavelengths_hr_nm` must be 1-D."
            )
        if self.band_centers_nm.shape != self.band_widths_nm.shape:
            raise ValueError(
                "SpectralResponseFunction: band_centers and band_widths shapes must match."
            )
        if len(self.band_names) != self.band_centers_nm.size:
            raise ValueError(
                "SpectralResponseFunction: `band_names` length must match number of bands."
            )
        if np.any(np.diff(self.wavelengths_hr_nm) <= 0.0):
            raise ValueError(
                "SpectralResponseFunction: `wavelengths_hr_nm` must be strictly increasing."
            )

        if self.srf_type == "custom":
            if self.custom_srfs is None:
                raise ValueError("srf_type='custom' requires `custom_srfs`.")
            mat = np.asarray(self.custom_srfs, dtype=float)
            expected = (self.n_bands, self.n_lambda)
            if mat.shape != expected:
                raise ValueError(
                    f"`custom_srfs` shape {mat.shape} must equal {expected}"
                )
        elif self.srf_type in {"gaussian", "rectangular", "triangular"}:
            mat = self._build_parametric_matrix()
        else:
            raise ValueError(
                f"SpectralResponseFunction: unknown srf_type {self.srf_type!r}. "
                "Use 'gaussian', 'rectangular', 'triangular', or 'custom'."
            )

        # Weight each high-res sample by its wavelength-bin width before
        # normalising, so band integration is a proper (trapezoidal) quadrature
        # rather than a plain sum. This matters when ``wavelengths_hr_nm`` is
        # non-uniform — e.g. derived from a uniform wavenumber grid via
        # ``1e7 / nu`` — otherwise the densely-sampled side of each band is
        # over-weighted. On a uniform grid the widths are constant and cancel in
        # the row normalisation below, leaving the matrix unchanged.
        widths = _bin_widths(np.asarray(self.wavelengths_hr_nm, dtype=float))
        mat = mat * widths[None, :]

        # L1-normalise each band so a flat input gives a flat output.
        row_sums = mat.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise ValueError(
                "SpectralResponseFunction: at least one band has zero total response "
                "over the supplied wavelength grid — extend the grid or check band "
                "centres/widths."
            )
        self.matrix = mat / row_sums

    # ── shape helpers ────────────────────────────────────────────────────────

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return int(self.band_centers_nm.size)

    @property
    def n_lambda(self) -> int:
        """Number of high-resolution wavelength samples."""
        return int(self.wavelengths_hr_nm.size)

    # ── SRF builders ─────────────────────────────────────────────────────────

    def _build_parametric_matrix(self) -> np.ndarray:
        """Build the ``(n_bands, n_lambda)`` SRF matrix for a parametric shape."""
        wl = self.wavelengths_hr_nm[None, :]  # (1, n_lambda)
        centers = self.band_centers_nm[:, None]  # (n_bands, 1)
        widths = self.band_widths_nm[:, None]  # (n_bands, 1)
        dwl = wl - centers

        if self.srf_type == "gaussian":
            sigma = widths / 2.355  # FWHM → σ
            return np.exp(-(dwl**2) / (2.0 * sigma**2))
        if self.srf_type == "rectangular":
            return (np.abs(dwl) <= widths / 2.0).astype(float)
        if self.srf_type == "triangular":
            return np.maximum(0.0, 1.0 - np.abs(dwl) / (widths / 2.0))
        raise AssertionError(f"unreachable srf_type {self.srf_type!r}")

    # ── linear operator ──────────────────────────────────────────────────────

    def apply(self, radiance_hr: np.ndarray) -> np.ndarray:
        """Forward operator: map HR spectrum/cube to band-integrated values.

        Parameters
        ----------
        radiance_hr : np.ndarray
            Input radiance. Shape ``(n_lambda,)`` for a single spectrum, or
            ``(..., n_lambda)`` for a cube (spatial + spectral).

        Returns
        -------
        radiance_bands : np.ndarray
            Band-integrated radiance with the final axis replaced by
            ``n_bands``.
        """
        x = np.asarray(radiance_hr, dtype=float)
        if x.shape[-1] != self.n_lambda:
            raise ValueError(
                f"SpectralResponseFunction.apply: last axis size {x.shape[-1]} "
                f"must equal n_lambda={self.n_lambda}"
            )
        # ``S @ xᵀ``: contract last axis of `x` (wavelength) with second axis
        # of `matrix`, so `...l, bl -> ...b`.
        return np.einsum("bl, ...l -> ...b", self.matrix, x)

    def jacobian(self, tangent_hr: np.ndarray) -> np.ndarray:
        """Jacobian-vector product ``S · v``. Linear operator → same as ``apply``."""
        return self.apply(tangent_hr)

    def adjoint(self, cotangent_bands: np.ndarray) -> np.ndarray:
        """Adjoint operator ``Sᵀ``: map band-space cotangents back to HR.

        Used for reverse-mode / VJP gradient calculations.

        Parameters
        ----------
        cotangent_bands : np.ndarray
            Band-space cotangent, shape ``(..., n_bands)``.

        Returns
        -------
        cotangent_hr : np.ndarray
            HR-space cotangent, shape ``(..., n_lambda)``.
        """
        u = np.asarray(cotangent_bands, dtype=float)
        if u.shape[-1] != self.n_bands:
            raise ValueError(
                f"SpectralResponseFunction.adjoint: last axis size {u.shape[-1]} "
                f"must equal n_bands={self.n_bands}"
            )
        return np.einsum("bl, ...b -> ...l", self.matrix, u)


def check_srf_lut_coverage(
    srf: SpectralResponseFunction,
    lut_wavelengths_nm: np.ndarray,
    *,
    context: str,
    tol: float = 1e-2,
) -> np.ndarray:
    """Warn when SRF bands respond outside the LUT's wavelength coverage.

    Resampling a cross-section / target spectrum onto the SRF grid zero-fills
    wavelengths beyond the LUT range (``np.interp(..., left=0, right=0)``),
    silently attenuating any band whose response reaches there — a band whose
    support lies wholly outside gets an exactly-zero (and misleading) value.
    This computes each band's fraction of SRF mass *outside* the LUT coverage
    and warns for bands exceeding ``tol``.

    Parameters
    ----------
    srf
        Instrument SRF.
    lut_wavelengths_nm
        Wavelengths [nm] covered by the LUT (any order); coverage is taken as
        ``[min, max]``.
    context
        Caller name, prepended to the warning message.
    tol
        Out-of-coverage SRF-mass fraction above which a band is flagged.

    Returns
    -------
    frac_out : np.ndarray
        Per-band fraction of SRF response outside the LUT coverage.
    """
    lam = np.asarray(lut_wavelengths_nm, dtype=float)
    lam_min, lam_max = float(lam.min()), float(lam.max())
    hr = srf.wavelengths_hr_nm
    in_range = (hr >= lam_min) & (hr <= lam_max)
    total = srf.matrix.sum(axis=1)
    covered = srf.matrix[:, in_range].sum(axis=1)
    frac_out = np.where(total > 0.0, 1.0 - covered / total, 0.0)
    for b in np.nonzero(frac_out > tol)[0]:
        warnings.warn(
            f"{context}: band {srf.band_names[b]!r} has {frac_out[b] * 100:.1f}% "
            f"of its SRF response outside the LUT wavelength coverage "
            f"[{lam_min:.1f}, {lam_max:.1f}] nm; its band-integrated value is "
            "silently zero-filled there.",
            stacklevel=3,
        )
    return frac_out
