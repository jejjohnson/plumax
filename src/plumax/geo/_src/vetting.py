"""Plume vetting on GeoArrays (behind the ``[geo]`` extra): one Dataset per scene.

`PlumeVetting` is the L2 operator over the array-level vetting in
:mod:`plumax.vetting` (#157): it reads a radiance GeoArray, its
matched-filter GeoArray and a stack of candidate masks, runs
`vet_candidate` on each candidate and the scene null (`null_masks`,
`null_pvalue`) beside it, and returns the scores as one ``xarray.Dataset``.
It returns scores, not a verdict: the decision rule is #158's.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

import einx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, ArrayLike, Bool, Float
from radtrax import DegenerateInputError, ParameterError, ShapeError
from radtrax.retrieval.absorption import AbsorptionWindow

from plumax._src.tracing import require_positive
from plumax.errors import MissingExtraError
from plumax.vetting import VettingResult, null_masks, null_pvalue, vet_candidate
from plumax.vetting._src.vet import (
    DEFAULT_BOX,
    DEFAULT_DEGREE,
    DEFAULT_K,
    DEFAULT_KEEP,
    _check_vet_inputs,
)


try:
    from geoops.carrier import (
        invalid_values,
        require_geoarray,
        require_grid_match,
        single_band,
    )
    from pipekit_ops import Operator
except ModuleNotFoundError as err:  # pragma: no cover - the base-install gate
    raise MissingExtraError(
        "plumax.geo.vetting needs the [geo] extra: pip install 'plumax[geo]'"
    ) from err


if TYPE_CHECKING:
    from geoarray import GeoArray


__all__ = ["PlumeVetting"]


class PlumeVetting(Operator):
    """Vet every candidate of a scene and its scene null; scores as a Dataset.

    Called as ``PlumeVetting(window, table, delta_grid)(cube, mf, masks)``.
    For each candidate ``M_c`` of ``masks``, with every candidate kept out of
    the background pool,

    ``(ΔX̂_c, D_norm,c) = vet_candidate(L, MF, M_c)``, and

    ``p̂_c = null_pvalue(D_norm,c, {D_norm(M_s)})`` over ``n_null`` copies
    ``M_s`` of ``M_c`` from `null_masks`, each vetted as ``M_c`` is; a copy
    that cannot be scored (radtrax's ``DegenerateInputError``) scores
    ``NaN`` and `null_pvalue` leaves it out.

    The cube is ``(B, H, W)`` with its band centres in nm in
    ``attrs["wavelengths"]``; ``mf`` is ``(H, W)`` or ``(1, H, W)`` on the
    cube's pixel grid, in mol/m² (the units of
    ``window.mf_background_mol_m2``). Nodata pixels of either carrier
    (non-finite, or equal to its ``fill_value``) are dropped from
    every candidate and kept out of the pool and the null copies. The
    candidates' random draws split ``key`` in candidate order, so the same
    key gives the same ``p_null``.

    The Dataset has dims ``candidate`` (``C``) and ``band`` (the window's
    ``Bf`` fitting bands, coordinate ``wavelength_nm``) and holds
    ``delta_x_mol_m2`` (``ΔX̂``), ``d_norm``, ``p_null`` (``NaN`` when
    ``n_null = 0``) and ``n_pairs`` per candidate; ``ratio`` (``R``),
    ``continuum`` (``P(λ; ĉ)``) and ``fit`` (``P(λ; ĉ) E(λ; ΔX̂)``, the fitted
    ratio) per candidate and band; and attributes ``window`` (the window's
    fields as JSON) and ``key`` (the key's raw data).

    Args:
        window: The gas's ``AbsorptionWindow``
            (``radtrax.retrieval.absorption``).
        table: ``E(λ; ΔX_d)`` on the window's fitting bands, ``(D, Bf)``,
            from radtrax's ``transmittance_table``.
        delta_grid: The columns ``ΔX_d`` of the table's rows [mol/m²],
            ``(D,)``, starting at 0.
        n_null: Number of null copies per candidate; ``0`` skips the null.
        key: A ``jax.random`` key for the null copies; ``None`` uses
            ``jax.random.key(0)``.
        k: Number of seeds before growth (`vet_candidate`).
        keep: Fraction of the pairs kept, in (0, 1].
        degree: The continuum's Legendre degree.
        box: Margin, in pixels, around the seeds' bounding box that the pool
            is drawn from.

    Raises:
        ParameterError: If ``n_null`` is negative.
        ShapeError: If the cube is not ``(B, H, W)``, has no
            ``attrs["wavelengths"]`` with ``B`` entries, ``table`` does not
            have one column per fitting band, or ``masks`` is not
            ``(C, H, W)`` on the cube's grid; and whatever `vet_candidate`
            raises on a candidate.

    Examples:
        A pipeline sketch, after radtrax's matched filter::

            from plumax.geo import PlumeVetting

            scores = PlumeVetting(window, table, delta_grid, n_null=100)(
                cube, mf, masks
            )
            scores["d_norm"], scores["p_null"]
    """

    _terminal: ClassVar[bool] = True

    def __init__(
        self,
        window: AbsorptionWindow,
        table: Float[ArrayLike, "D Bf"],
        delta_grid: Float[ArrayLike, " D"],
        *,
        n_null: int = 100,
        key: Array | None = None,
        k: int = DEFAULT_K,
        keep: float = DEFAULT_KEEP,
        degree: int = DEFAULT_DEGREE,
        box: int = DEFAULT_BOX,
    ) -> None:
        require_positive(
            "n_null",
            n_null,
            caller="PlumeVetting",
            allow_zero=True,
            error=ParameterError,
        )
        self.window = window
        self.table = table
        self.delta_grid = delta_grid
        self.n_null = int(n_null)
        self.key = jax.random.key(0) if key is None else key
        self.k = k
        self.keep = keep
        self.degree = degree
        self.box = box

    def _apply(
        self,
        cube: GeoArray,
        enhancement: GeoArray,
        masks: Bool[np.ndarray, "C H W"],
    ) -> xr.Dataset:
        require_geoarray(cube, "PlumeVetting")
        require_grid_match(cube, enhancement, "PlumeVetting", names=("cube", "mf"))
        radiance = np.asarray(cube)
        if radiance.ndim != 3:
            raise ShapeError(
                f"PlumeVetting: the cube must be (B, H, W), got {radiance.shape}"
            )
        wavelength_nm = self._wavelengths(cube, radiance.shape[0])
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3 or masks.shape[1:] != radiance.shape[1:]:
            raise ShapeError(
                f"PlumeVetting: masks must be (C, H, W) with the cube's (H, W) "
                f"{radiance.shape[1:]}, got {masks.shape}"
            )
        # vet_candidate takes the cube as (H, W, B)
        radiance = einx.id("channel y x -> y x channel", radiance)
        score = np.asarray(single_band(np.asarray(enhancement), name="mf"))

        # nodata pixels never enter a candidate, the pool or a null copy
        # invalid = ∃ band: nodata
        invalid = einx.any("[band] y x", np.asarray(invalid_values(cube)))
        invalid |= np.asarray(single_band(invalid_values(enhancement), name="mf"))
        exclude = masks
        if invalid.any():
            # exclude = masks ∪ {invalid}; einx.id cannot size an empty stack
            if masks.shape[0] == 0:
                exclude = einx.id("y x -> 1 y x", invalid)
            else:
                exclude = einx.id("mask y x, y x -> (mask + 1) y x", masks, invalid)
            masks = masks & ~invalid

        fit_bands = self.window.fit_bands(wavelength_nm)
        num_fit_bands = int(fit_bands.sum())
        table = jnp.asarray(self.table)
        _check_vet_inputs(
            table,
            num_fit_bands,
            "PlumeVetting",
            "fitting band of the cube's wavelengths",
        )

        def vet(mask: np.ndarray) -> VettingResult:
            return vet_candidate(
                radiance,
                score,
                mask,
                wavelength_nm,
                table,
                self.delta_grid,
                self.window,
                exclude=exclude,
                k=self.k,
                keep=self.keep,
                degree=self.degree,
                box=self.box,
            )

        rows: list[VettingResult] = []
        p_null: list[float] = []
        key = self.key
        for mask in masks:
            result = vet(mask)
            rows.append(result)
            if self.n_null == 0:
                p_null.append(np.nan)
                continue
            key, subkey = jax.random.split(key)
            copies = null_masks(mask, self.n_null, key=subkey, exclude=exclude)
            null_scores = np.array([self._null_score(vet, copy) for copy in copies])
            p_null.append(null_pvalue(float(result.d_norm), null_scores))

        return self._dataset(rows, p_null, wavelength_nm[fit_bands])

    @staticmethod
    def _wavelengths(cube: GeoArray, num_bands: int) -> np.ndarray:
        wavelengths = (getattr(cube, "attrs", None) or {}).get("wavelengths")
        if wavelengths is None:
            raise ShapeError(
                "PlumeVetting: the cube needs its band centres in nm as "
                "attrs['wavelengths']"
            )
        wavelength_nm = np.asarray(wavelengths, dtype=float)
        if wavelength_nm.shape != (num_bands,):
            raise ShapeError(
                f"PlumeVetting: attrs['wavelengths'] has shape "
                f"{wavelength_nm.shape}, the cube {num_bands} bands"
            )
        return wavelength_nm

    @staticmethod
    def _null_score(
        vet: Callable[[np.ndarray], VettingResult], copy: np.ndarray
    ) -> float:
        try:
            return float(vet(copy).d_norm)
        except DegenerateInputError:
            # a copy that cannot be scored; null_pvalue leaves NaN out
            return float("nan")

    def _dataset(
        self,
        rows: list[VettingResult],
        p_null: list[float],
        wavelength_nm: np.ndarray,
    ) -> xr.Dataset:
        candidate = ("candidate",)
        per_band = ("candidate", "band")
        if rows:
            ratio = np.stack([np.asarray(row.ratio) for row in rows])
            continuum = np.stack([np.asarray(row.fit.continuum) for row in rows])
            transmittance = np.stack(
                [np.asarray(row.fit.transmittance) for row in rows]
            )
        else:
            ratio = continuum = transmittance = np.empty((0, wavelength_nm.size))
        column_enhancement = np.array([float(row.delta) for row in rows])
        d_norm = np.array([float(row.d_norm) for row in rows])
        num_pairs = np.array([row.n_pairs for row in rows], dtype=np.int64)
        return xr.Dataset(
            {
                "delta_x_mol_m2": (candidate, column_enhancement),
                "d_norm": (candidate, d_norm),
                "p_null": (candidate, np.array(p_null, dtype=float)),
                "n_pairs": (candidate, num_pairs),
                "ratio": (per_band, ratio),
                "continuum": (per_band, continuum),
                "fit": (per_band, continuum * transmittance),
            },
            coords={
                "candidate": np.arange(len(rows)),
                "wavelength_nm": ("band", wavelength_nm),
            },
            attrs={
                "window": json.dumps(dataclasses.asdict(self.window)),
                "key": _key_data(self.key),
            },
        )

    def get_config(self) -> dict[str, Any]:
        """The JSON-safe configuration, without the arrays (table, grid, key)."""
        return {
            "window": json.loads(json.dumps(dataclasses.asdict(self.window))),
            "n_null": self.n_null,
            "k": self.k,
            "keep": self.keep,
            "degree": self.degree,
            "box": self.box,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        table: Float[ArrayLike, "D Bf"] | None = None,
        delta_grid: Float[ArrayLike, " D"] | None = None,
        key: Array | None = None,
    ) -> PlumeVetting:
        """Rebuild from `get_config`'s output and the arrays it leaves out.

        Raises:
            TypeError: If ``table`` or ``delta_grid`` is not given; the
                config holds neither.
        """
        if table is None or delta_grid is None:
            raise TypeError(
                "PlumeVetting.from_config needs the table= and delta_grid= "
                "arrays, which get_config leaves out"
            )
        config = dict(config)
        window = dict(config.pop("window"))
        for name in ("fit_nm", "absorbing_nm"):
            window[name] = tuple(tuple(band) for band in window[name])
        return cls(
            AbsorptionWindow(**window),
            table,
            delta_grid,
            key=key,
            **config,
        )


def _key_data(key: Array) -> np.ndarray:
    """A key's raw ``uint32`` data, for a typed or a legacy ``uint32`` key."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return np.asarray(jax.random.key_data(key))
    return np.asarray(key)
