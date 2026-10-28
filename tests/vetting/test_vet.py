"""VettingResult and vet_candidate, one candidate vetted end to end (#156).

``vet_candidate`` is checked as the composition of its steps (plumax's seeds
and pool, radtrax's absorption-shape test; each argument reaches its step),
on a noise-free scene whose answer is known (uniform ``ΔX`` over identical
surfaces), on its shape checks, and, gate VT2 (part), on the plume-vetting
notebook's scene, where it must print the theory page's numbers. The band
grid is EMIT's 285 centres spaced evenly (no EMIT wavelength table yet,
radtrax#114), as in radtrax's window tests, and the table is Beer–Lambert
with radtrax's fit tests' band-structured ``ε``.

The assertions on radtrax's numbers (``ΔX̂``, ``D_norm``) are tolerances, not
grid-node equalities: the fit may refine ``ΔX̂`` between the table's nodes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from radtrax import ShapeError
from radtrax.retrieval.absorption import (
    AbsorptionWindow,
    d_norm,
    fit_transmittance,
    pair_pixels,
    ratio_spectrum,
)

from plumax.vetting import (
    VettingResult,
    background_pool,
    select_seeds,
    vet_candidate,
)
from tests.vetting._vetting_scene import notebook_scene, vet_scene


EMIT_NM = np.linspace(381.00558, 2493.2576, 285)
WINDOW = AbsorptionWindow.ch4_emit()
# mol/m² per ppm·m: the CH₄ preset's 30 ppm·m background cut, converted.
MOL_M2_PER_PPM_M = WINDOW.mf_background_mol_m2 / 30.0
FIT = WINDOW.fit_bands(EMIT_NM)  # (B,)
WL_FIT = EMIT_NM[FIT]  # (Bf,)
GRID = np.linspace(0.0, 0.4, 81)  # ΔX [mol/m²], step 0.005           (D,)
STEP = GRID[1]
H, W = 20, 24


def _eps(wl: np.ndarray) -> np.ndarray:
    """radtrax's fit tests' band-structured absorption coefficient [m²/mol]."""
    centres, widths = (2200, 2265, 2320, 2360, 2390), (8, 14, 10, 6, 12)
    depths = (0.2, 0.5, 0.9, 0.6, 0.3)
    return 0.05 + sum(
        d * np.exp(-0.5 * ((wl - c) / w) ** 2)
        for c, w, d in zip(centres, widths, depths, strict=True)
    )


EPS = _eps(WL_FIT)  # (Bf,)
TABLE = np.exp(-GRID[:, None] * EPS[None])  # E (D, Bf)


def _mask() -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[7:13, 9:15] = True  # 36 pixels
    return m


def _scene(
    delta: float, brightness: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(cube, mf, mask)``: one surface, ``ΔX = delta`` uniform over the mask.

    ``L(λ) = b · ρ(λ) · E(λ; ΔX)`` on the fitting bands and ``b · ρ(λ)``
    elsewhere, with ``b = 1`` unless ``brightness`` (``(H, W)``) is given;
    the matched filter reads ``ΔX`` exactly.
    """
    mask = _mask()
    rho = 0.3 + 0.1 * np.sin(EMIT_NM / 300.0)  # (B,)
    b = np.ones((H, W)) if brightness is None else brightness
    cube = b[..., None] * np.broadcast_to(rho, (H, W, EMIT_NM.size)).copy()
    e = np.exp(-delta * EPS)  # (Bf,)
    cube[mask[:, :, None] & FIT[None, None, :]] *= np.tile(e, int(mask.sum()))
    mf = np.where(mask, delta, 0.0)  # (H, W)
    return cube, mf, mask


# ── the composition of the steps ──────────────────────────────────────────


def test_the_result_is_the_steps_composed() -> None:
    """Each argument reaches its step, and each field is that step's output."""
    rng = np.random.default_rng(0)
    cube, mf, mask = _scene(0.1, 1.0 + 0.2 * rng.standard_normal((H, W)))
    exclude = np.stack([mask, np.roll(mask, 8, axis=1)])
    kw = {"k": 20, "keep": 0.8, "degree": 3, "box": 5}
    r = vet_candidate(
        cube, mf, mask, EMIT_NM, TABLE, GRID, WINDOW, exclude=exclude, **kw
    )

    seeds = select_seeds(mf, mask, k=20)
    pool = background_pool(
        mf, seeds, threshold=WINDOW.mf_background_mol_m2, exclude=exclude, box=5
    )
    i_in, i_out = pair_pixels(
        cube,
        np.flatnonzero(seeds),
        np.flatnonzero(pool),
        WINDOW.pairing_bands(EMIT_NM),
        keep=0.8,
    )
    ratio = ratio_spectrum(cube, i_in, i_out)[FIT]
    fit = fit_transmittance(ratio, TABLE, GRID, WL_FIT, degree=3)
    assert isinstance(r, VettingResult)
    assert r.n_pairs == i_in.size == int(0.8 * seeds.sum())
    np.testing.assert_array_equal(r.ratio, ratio)
    assert r.ratio.shape == (int(FIT.sum()),)
    assert float(r.delta) == float(fit.delta) == float(r.fit.delta)
    np.testing.assert_array_equal(r.fit.coefficients, fit.coefficients)
    assert r.fit.coefficients.shape == (4,)
    assert float(r.d_norm) == float(d_norm(ratio, fit))
    no_gas = fit_transmittance(
        ratio, np.ones((1, FIT.sum())), np.zeros(1), WL_FIT, degree=3
    )
    assert float(r.residual_ss_no_gas) == float(no_gas.residual_ss)


def test_the_no_gas_residual_is_the_fit_at_zero() -> None:
    """``SSE(0)`` is the grid's first row (``E = 1``); ``SSE(ΔX̂)`` is the least."""
    rng = np.random.default_rng(1)
    cube, mf, mask = _scene(0.2, 1.0 + 0.2 * rng.standard_normal((H, W)))
    r = vet_candidate(cube, mf, mask, EMIT_NM, TABLE, GRID, WINDOW)
    at_zero = fit_transmittance(r.ratio, TABLE[:1], GRID[:1], WL_FIT)
    np.testing.assert_allclose(r.residual_ss_no_gas, at_zero.residual_ss, rtol=1e-12)
    assert float(r.residual_ss_no_gas) > 10 * float(r.fit.residual_ss)


def test_n_pairs_is_static() -> None:
    """``n_pairs`` is a Python int and not a leaf of the result's pytree."""
    cube, mf, mask = _scene(0.1)
    r = vet_candidate(cube, mf, mask, EMIT_NM, TABLE, GRID, WINDOW)
    assert type(r.n_pairs) is int
    leaves = jax.tree_util.tree_leaves(r)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    # delta, d_norm, ratio, residual_ss_no_gas, and the fit's own leaves.
    assert len(leaves) == 4 + len(jax.tree_util.tree_leaves(r.fit))


# ── a noise-free scene: uniform ΔX over identical surfaces ────────────────


def test_a_noise_free_uniform_column_is_returned_within_one_grid_step() -> None:
    """Off the grid, ``ΔX̂`` is within one step of the truth (#156)."""
    for delta in (0.0123, 0.1234, 0.3071):
        cube, mf, mask = _scene(delta)
        r = vet_candidate(cube, mf, mask, EMIT_NM, TABLE, GRID, WINDOW)
        assert abs(float(r.delta) - delta) <= STEP


def test_a_noise_free_column_on_a_grid_node_is_recovered() -> None:
    """On a node the ratio is ``E`` itself, so the fit finds it.

    The ratio is plumax's pixel sets through radtrax's pairing, exact to
    float64 rounding; ``ΔX̂`` and ``D_norm`` are radtrax's fit, held to a
    tolerance (within a grid step, a near-flat continuum, a near-zero score).
    """
    delta = float(GRID[30])
    cube, mf, mask = _scene(delta)
    r = vet_candidate(cube, mf, mask, EMIT_NM, TABLE, GRID, WINDOW)
    # Identical surfaces: R = E(ΔX) up to float64 rounding of the means.
    np.testing.assert_allclose(r.ratio, TABLE[30], rtol=1e-12)
    assert abs(float(r.delta) - delta) <= STEP
    np.testing.assert_allclose(r.fit.continuum, 1.0, atol=1e-2)
    assert float(r.d_norm) < 0.05


# ── shapes ────────────────────────────────────────────────────────────────


def test_a_table_off_the_windows_fit_bands_raises() -> None:
    cube, mf, mask = _scene(0.1)
    with pytest.raises(ShapeError, match=rf"table must be \(D, {FIT.sum()}\)"):
        vet_candidate(cube, mf, mask, EMIT_NM, TABLE[:, :-1], GRID, WINDOW)
    with pytest.raises(ShapeError, match="table must be"):
        vet_candidate(cube, mf, mask, EMIT_NM, TABLE[0], GRID, WINDOW)


def test_wavelengths_off_the_cubes_bands_raise() -> None:
    cube, mf, mask = _scene(0.1)
    with pytest.raises(ShapeError, match="wavelength_nm has shape"):
        vet_candidate(cube, mf, mask, EMIT_NM[:-1], TABLE, GRID, WINDOW)
    with pytest.raises(ShapeError, match="wavelength_nm has shape"):
        vet_candidate(cube[0], mf, mask, EMIT_NM, TABLE, GRID, WINDOW)


# ── gate VT2 (part): the notebook scene ───────────────────────────────────


@pytest.mark.slow  # compile-bound: the scene's forward model and matched filter
def test_vt2_vetting_reverses_the_matched_filters_order() -> None:
    """Gate VT2 (part): the theory page's table, rows A and B.

    A, the plume: ``ΔX̂ = 1,741 ± 5`` ppm·m (the epic's tolerance; radtrax
    refines ``ΔX̂`` between the table's nodes, 12.4 ppm·m apart) and
    ``D_norm = 0.14 ± 0.01``. B, the carbonate: ``ΔX̂ = 0 + 5`` ppm·m and
    ``D_norm > 0.9``. The matched filter scores B above A; vetting reverses
    the order.
    """
    scene = notebook_scene()
    tol = 5.0 * MOL_M2_PER_PPM_M  # mol/m²
    mask_a, mask_b = scene.masks
    assert scene.mf[mask_b].mean() > scene.mf[mask_a].mean()
    a, b = vet_scene(scene, mask_a), vet_scene(scene, mask_b)
    assert float(a.delta) == pytest.approx(1741.0 * MOL_M2_PER_PPM_M, abs=tol)
    assert float(a.d_norm) == pytest.approx(0.14, abs=0.01)
    assert 0.0 <= float(b.delta) <= tol
    assert float(b.d_norm) > 0.9
    # The spectra an analyst plots: R / P against E on the window's 9 bands.
    q = a.ratio / a.fit.continuum
    assert q.shape == a.fit.transmittance.shape == (9,)
    assert bool(jnp.all(jnp.isfinite(q)))
