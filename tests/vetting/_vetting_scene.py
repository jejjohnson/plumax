"""The plume-vetting notebook's 64 × 64 EMIT-like scene, for the vetting gates.

The plume-vetting notebook (section "One scene") builds it: 30
bands outside methane's band and 9 inside it (2296–2356 nm), three surface
classes, noise at SNR 300, candidate A a methane plume and candidate B a
plume-shaped patch of a surface whose reflectance dips by 6 % at 2335 nm.
The same seeds and steps, with radtrax's `transmittance_table` and
`AbsorptionWindow`, so the gates test what the notebook shows. Built once per
session (``notebook_scene`` is cached); ``vet_d_norm`` is the notebook's
``vet_candidate`` chain up to ``D_norm``: plumax's seeds and pool, then
radtrax's absorption-shape test on them.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import radtrax
from radtrax.atmosphere import ViewingGeometry, afgl_us_standard
from radtrax.instrument import GaussianInstrument
from radtrax.retrieval.absorption import (
    AbsorptionWindow,
    d_norm,
    fit_transmittance,
    pair_pixels,
    ratio_spectrum,
    transmittance_table,
)
from radtrax.retrieval.matched_filter import apply_image, estimate_cov_shrunk
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.spectroscopy.synthetic import NU_CM, synthetic_table
from radtrax.surface import Lambertian
from scipy.ndimage import gaussian_filter

from plumax.vetting import background_pool, select_seeds


@dataclasses.dataclass(frozen=True)
class Scene:
    """The cube, its matched filter, the candidates and what vetting needs."""

    radiance: np.ndarray  # L (H, W, B)
    mf: np.ndarray  # ΔX_MF [mol/m²] (H, W)
    masks: np.ndarray  # (C, H, W): A, then B
    wavelength_nm: np.ndarray  # (B,)
    table: np.ndarray  # E (D, B_fit)
    delta_grid: np.ndarray  # ΔX [mol/m²] (D,)
    window: AbsorptionWindow


@functools.cache
def notebook_scene() -> Scene:
    """The notebook's scene, built as its cells 6, 13, 15 and 19 build it."""
    centers_nm = np.arange(2296.0, 2357.0, 7.5)  # (9,)
    opacity = TableOpacity.from_datasets(
        {"CH4": synthetic_table(seed=0), "H2O": synthetic_table(seed=1)}
    )
    instrument = GaussianInstrument(NU_CM, centers_nm, np.full(centers_nm.size, 8.5))
    model = radtrax.ForwardModel(opacity, BeerLambert(), instrument)
    state = afgl_us_standard(z_top_km=20.0, gases=["CH4", "H2O"])
    geometry = ViewingGeometry.from_angles(35.0, 10.0)
    delta_grid = np.linspace(0.0, 0.4, 801)  # (D,)
    # In 8 chunks of about 100 columns (each computes its own y(0)): one
    # vmap over all 801 holds several GB of (D, L, G) intermediates on the
    # 14,000-wavenumber grid.
    table = np.concatenate(
        [
            np.asarray(transmittance_table(model, state, geometry, c))
            for c in np.array_split(delta_grid, 8)
        ]
    )  # (D, 9)

    rng = np.random.default_rng(0)
    height = width = 64
    wl_out = np.linspace(1000.0, 2050.0, 30)
    wl = np.concatenate([wl_out, centers_nm])  # (39,)
    in_band = np.arange(wl_out.size, wl.size)

    field = gaussian_filter(rng.standard_normal((height, width)), 6.0)
    cls = np.digitize(field, np.quantile(field, [1 / 3, 2 / 3]))  # (H, W)
    coef = np.array([[0.28, 0.06, -0.04], [0.18, 0.08, -0.04], [0.40, -0.03, -0.02]])
    x_wl = (wl - 1700.0) / 700.0
    rho_cls = coef @ np.stack([np.ones_like(x_wl), x_wl, x_wl**2])  # (3, B)
    bright = 1 + 0.5 * gaussian_filter(rng.standard_normal((height, width)), 2.0)
    rho = rho_cls[cls] * bright[..., None]  # (H, W, B)

    ii, jj = np.mgrid[:height, :width]
    along = jj - 14.0
    sigma_y = 1.5 + 0.18 * np.clip(along, 0, None)
    plume = 0.20 * np.exp(-((ii - 40.0) ** 2) / (2 * sigma_y**2)) * 1.5 / sigma_y
    delta_x = np.where(along > 0, plume, 0.0)
    delta_x += 0.20 * np.exp(-((ii - 40.0) ** 2 + (jj - 14.0) ** 2) / 2.0)
    mask_a = delta_x > 0.02

    u, v = jj - 44.0, ii - 16.0
    ang = np.deg2rad(25.0)
    ellipse = ((u * np.cos(ang) + v * np.sin(ang)) / 11.0) ** 2
    ellipse += ((-u * np.sin(ang) + v * np.cos(ang)) / 3.2) ** 2
    mask_b = ellipse < 1
    dip = 1 - 0.06 * np.exp(-((wl - 2335.0) ** 2) / (2 * 15.0**2))  # (B,)

    y0 = np.asarray(model(state, Lambertian(1.0), geometry))  # (9,)
    e_pix = np.stack(
        [np.interp(delta_x.ravel(), delta_grid, table[:, b]) for b in range(9)], -1
    ).reshape(height, width, 9)
    l_true = np.empty((height, width, wl.size))
    l_true[..., : wl_out.size] = (
        rho[..., : wl_out.size] * y0.mean() * (2300 / wl_out) ** 2
    )
    l_true[..., in_band] = rho[..., in_band] * y0 * e_pix
    l_true[mask_b] *= dip
    radiance = l_true * (1 + rng.standard_normal(l_true.shape) / 300.0)
    masks = np.stack([mask_a, mask_b])  # (C, H, W)

    ln_l = np.log(radiance)
    t = radtrax.unit_absorption_target(model, state, Lambertian(1.0), geometry)
    target = np.concatenate([np.zeros(wl_out.size), np.asarray(t)])
    mu = ln_l.reshape(-1, wl.size).mean(0)
    mf = np.asarray(apply_image(ln_l, mu, estimate_cov_shrunk(ln_l, mu), target))
    mf_sigma = float(np.std(mf[~masks.any(0)]))
    window = AbsorptionWindow(
        "CH4",
        fit_nm=((2290.0, 2360.0),),
        absorbing_nm=((2290.0, 2360.0),),
        mf_background_mol_m2=2 * mf_sigma,
    )
    return Scene(radiance, mf, masks, wl, table, delta_grid, window)


def vet_d_norm(scene: Scene, mask: np.ndarray, *, box: int = 16) -> float:
    """``D_norm`` of one mask: seeds, pool, pairs, ratio, fit (the notebook's chain)."""
    seeds = select_seeds(scene.mf, mask)
    pool = background_pool(
        scene.mf,
        seeds,
        threshold=scene.window.mf_background_mol_m2,
        exclude=scene.masks,
        box=box,
    )
    i_in, i_out = pair_pixels(
        scene.radiance,
        np.flatnonzero(seeds),
        np.flatnonzero(pool),
        scene.window.pairing_bands(scene.wavelength_nm),
    )
    fit_b = scene.window.fit_bands(scene.wavelength_nm)
    ratio = ratio_spectrum(scene.radiance, i_in, i_out)[fit_b]
    fit = fit_transmittance(
        ratio, scene.table, scene.delta_grid, scene.wavelength_nm[fit_b]
    )
    return float(d_norm(ratio, fit))
