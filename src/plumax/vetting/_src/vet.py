"""One candidate, vetted: seeds, pool, pairs, ratio, fit and score in one call.

`vet_candidate` runs Xiang et al.'s steps in their fixed order with the
window's band selection: plumax's `select_seeds` and `background_pool`, then
radtrax's absorption-shape test on the two sets (``pair_pixels`` on the
window's pairing bands, ``ratio_spectrum`` on its fitting bands,
``fit_transmittance`` and ``d_norm``, from ``radtrax.retrieval.absorption``).
It returns a `VettingResult`, the scores and the spectra an analyst plots
(``R / P`` against ``E``). Theory: the plume-vetting page.

The result carries scores, not a verdict: the decision rule (a fixed
``D_norm`` cut, the scene null's ``p̂``, or a likelihood ratio between the
fits with and without the gas) is #158's. For the last,
``residual_ss_no_gas`` is the fit's residual at ``ΔX = 0`` beside the
``fit.residual_ss`` at ``ΔX̂``; for ``p̂``, vet each of
`null_masks`' copies the same way and pass the scores to `null_pvalue`.

``ΔX̂`` is the column of the *mean* transmittance over the pairs, not the
mean column (``E`` is convex in ``ΔX``), so it sits below the mean of the
truth over the seeds: a score, not a retrieval.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Bool, Float
from radtrax import ShapeError
from radtrax.retrieval.absorption import (
    AbsorptionWindow,
    TransmittanceFit,
    d_norm,
    fit_transmittance,
    pair_pixels,
    ratio_spectrum,
)

from plumax.vetting.seeds import background_pool, select_seeds


__all__ = ["VettingResult", "vet_candidate"]


class VettingResult(eqx.Module):
    """One candidate's scores and the spectra behind them.

    Attributes:
        delta: ``ΔX̂``, the fitted column enhancement [mol/m²], ``()``.
        d_norm: ``D_norm`` of the fit (dimensionless), ``()``.
        n_pairs: The number of pairs the ratio averages (static).
        ratio: ``R``, the ratio spectrum on the window's fitting bands,
            ``(Bf,)``; NaN in a band with no usable pair.
        fit: radtrax's ``TransmittanceFit`` of ``ratio``
            (``radtrax.retrieval.absorption``): its continuum ``P``,
            transmittance ``E`` and residual ``fit.residual_ss`` at ``ΔX̂``.
        residual_ss_no_gas: The residual sum of squares of the continuum
            alone (``E = 1``, ``ΔX = 0``) on the same bands and degree,
            ``()``; with ``fit.residual_ss``, the likelihood ratio
            ``SSE(0) / SSE(ΔX̂)`` of #158's third option.
    """

    delta: Float[Array, ""]
    d_norm: Float[Array, ""]
    n_pairs: int = eqx.field(static=True)
    ratio: Float[Array, " Bf"]
    fit: TransmittanceFit
    residual_ss_no_gas: Float[Array, ""]


def vet_candidate(
    cube: Float[ArrayLike, "H W B"],
    mf: Float[ArrayLike, "H W"],
    mask: Bool[ArrayLike, "H W"],
    wavelength_nm: Float[ArrayLike, " B"],
    table: Float[ArrayLike, "D Bf"],
    delta_grid: Float[ArrayLike, " D"],
    window: AbsorptionWindow,
    *,
    exclude: Bool[ArrayLike, " C H W"] | None = None,
    k: int = 40,
    keep: float = 0.5,
    degree: int = 2,
    box: int = 200,
) -> VettingResult:
    """Vet one candidate mask: seeds, pool, pairs, ratio, fit and ``D_norm``.

    With ``F`` the window's fitting bands and ``O`` its pairing bands,

    ``I = select_seeds(MF, M)``, ``P = background_pool(MF, I)`` below the
    window's ``mf_background_mol_m2``;

    ``(i_n, o_n) = pair_pixels(L, I, P, O)``, the best ``keep`` of a
    one-to-one assignment of the two sets as flat pixel indices;

    ``R(λ) = ∑_n L_in,n(λ) / ∑_n L_out,n(λ)`` for ``λ ∈ F``;

    ``(ΔX̂, ĉ) = argmin ∑_λ (R(λ) − P(λ; c) E(λ; ΔX))²`` over the grid, and
    ``D_norm`` of that fit.

    Runs eagerly, not under ``jit`` (pairing is NumPy and SciPy), and is not
    differentiable as a whole (seeds, pairs and ``ΔX̂`` are discrete); the
    pieces that are differentiable say so.

    Args:
        cube: Radiance, ``(H, W, B)``.
        mf: Matched-filter column ``ΔX`` [mol/m²], ``(H, W)``, in the units
            of ``window.mf_background_mol_m2``.
        mask: The candidate ``M``, ``(H, W)``.
        wavelength_nm: The cube's band centres [nm], ``(B,)``.
        table: ``E(λ; ΔX_d)`` on the window's fitting bands, ``(D, Bf)``
            with ``Bf = window.fit_bands(wavelength_nm).sum()``, from
            radtrax's ``transmittance_table``.
        delta_grid: The columns ``ΔX_d`` of the table's rows [mol/m²],
            ``(D,)``, starting at 0.
        window: The gas's ``AbsorptionWindow``
            (``radtrax.retrieval.absorption``).
        exclude: The scene's ``C`` candidate masks, ``(C, H, W)``, kept out
            of the background pool; pass every candidate of the scene.
        k: Number of seeds before growth; the reference code's 40.
        keep: Fraction of the pairs kept, in (0, 1].
        degree: The continuum's Legendre degree.
        box: Margin, in pixels, around the seeds' bounding box that the pool
            is drawn from.

    Returns:
        The candidate's [`VettingResult`][plumax.vetting.VettingResult].

    Raises:
        ShapeError: If ``wavelength_nm`` is not ``(B,)`` with the cube's
            ``B``, or ``table`` does not have ``Bf`` columns; and whatever
            the steps raise (`select_seeds`, `background_pool`,
            `pair_pixels`, `fit_transmittance`) on their own inputs.

    Examples:
        >>> import numpy as np
        >>> from radtrax.retrieval.absorption import AbsorptionWindow
        >>> from plumax.vetting import vet_candidate
        >>> wl = np.array([1000.0, 1100.0, 1200.0, 2300.0, 2310.0, 2320.0])
        >>> eps = np.array([2.0, 5.0, 3.0])  # m²/mol on the fitting bands
        >>> grid = np.linspace(0.0, 0.2, 21)  # mol/m²
        >>> table = np.exp(-grid[:, None] * eps)  # (21, 3)
        >>> mask = np.zeros((6, 6), bool)
        >>> mask[2:4, 2:4] = True
        >>> mf = np.where(mask, 0.1, 0.0)
        >>> e = np.where(mask[..., None], np.exp(-0.1 * eps), 1.0)  # (6, 6, 3)
        >>> cube = np.concatenate([np.ones((6, 6, 3)), e], axis=-1)
        >>> window = AbsorptionWindow(
        ...     "CH4", ((2290.0, 2330.0),), ((2290.0, 2330.0),), 0.01
        ... )
        >>> r = vet_candidate(cube, mf, mask, wl, table, grid, window, degree=0)
        >>> print(f"{r.delta:.2f} {r.d_norm:.2f} {r.n_pairs}")
        0.10 0.00 2
    """
    wavelength_nm = np.asarray(wavelength_nm)
    cube = jnp.asarray(cube)
    table = jnp.asarray(table)
    if cube.ndim != 3 or wavelength_nm.shape != cube.shape[2:]:
        raise ShapeError(
            f"vet_candidate: wavelength_nm has shape {wavelength_nm.shape}, the "
            f"cube {cube.shape}; need (B,) with the cube's (H, W, B)"
        )
    fit_b = window.fit_bands(wavelength_nm)  # (B,)
    n_fit = int(fit_b.sum())
    if table.ndim != 2 or table.shape[1] != n_fit:
        raise ShapeError(
            f"vet_candidate: table must be (D, {n_fit}), one column per band of "
            f"the window's {n_fit} fitting bands, got {table.shape}"
        )

    seeds = select_seeds(mf, mask, k=k)  # (H, W)
    pool = background_pool(
        mf, seeds, threshold=window.mf_background_mol_m2, exclude=exclude, box=box
    )  # (H, W)
    i_in, i_out = pair_pixels(
        cube,
        np.flatnonzero(seeds),
        np.flatnonzero(pool),
        window.pairing_bands(wavelength_nm),
        keep=keep,
    )  # (N,), (N,)
    ratio = ratio_spectrum(cube, i_in, i_out)[fit_b]  # (Bf,)
    wl_fit = wavelength_nm[fit_b]  # (Bf,)
    fit = fit_transmittance(ratio, table, delta_grid, wl_fit, degree=degree)
    no_gas = fit_transmittance(
        ratio,
        jnp.ones_like(table[:1]),
        jnp.zeros(1, table.dtype),
        wl_fit,
        degree=degree,
    )
    return VettingResult(
        delta=fit.delta,
        d_norm=d_norm(ratio, fit),
        n_pairs=int(i_in.size),
        ratio=ratio,
        fit=fit,
        residual_ss_no_gas=no_gas.residual_ss,
    )
