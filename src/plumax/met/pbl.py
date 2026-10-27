"""Boundary-layer diagnostics on a ``MetField``, and the Hanna factory.

- :func:`pbl_height_bulk_richardson` re-diagnoses the PBL height from the
  analysis grid's own profiles, through
  :func:`xrtoolz.atm.pbl_height_bulk_richardson` (load-time, NumPy; needs
  the ``io`` extra). A ``MetField`` already carries the source model's
  ``pbl_height``; this is for checking it, or for sources without one.
- :func:`hanna_turbulence_from` builds the Tier II
  :class:`~plumax.lagrangian.turbulence.HannaTurbulence` at one column from
  the met and a :class:`~plumax.met.surface_layer.SurfaceLayer`, instead of
  hand-typed constants.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from plumax.lagrangian.turbulence import HannaTurbulence
from plumax.met.field import MetField
from plumax.met.surface_layer import CP_DRY, P_REFERENCE, R_DRY, SurfaceLayer


def pbl_height_bulk_richardson(field: MetField, *, ri_crit: float = 0.25) -> np.ndarray:
    """PBL height from the bulk Richardson number of the analysis profiles.

    Builds a ``(time, level, y, x)`` Dataset of the dry potential
    temperature ``θ = T (p_ref/p)^{R_d/c_p}`` (the ``MetField`` has no
    humidity, so ``θ_v = θ``), the wind at T-points (``u`` and ``v``
    averaged from the neighbouring U- and V-points) and the analysis
    heights above the grid's surface, and hands it to
    :func:`xrtoolz.atm.pbl_height_bulk_richardson`.

    Args:
        field: Met on the analysis grid.
        ri_crit: Critical bulk Richardson number.

    Returns:
        PBL height [m], ``(n_time, ny, nx)``; NaN where no level reaches
        ``ri_crit`` inside the grid.

    Raises:
        ImportError: If the ``io`` extra (xrtoolz) is not installed.
    """
    try:
        from xrtoolz.atm import pbl_height_bulk_richardson as bulk_richardson
    except ImportError as err:
        raise ImportError(
            "pbl_height_bulk_richardson runs through xrtoolz; install it with "
            "`pip install 'plumax[io]'`"
        ) from err
    grid = field.plume_grid
    z = np.asarray(grid.z, dtype=np.float64)
    height = z - (z[0] - 0.5 * grid.dz)
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    u_t = 0.5 * (u + np.concatenate([u[..., :1], u[..., :-1]], axis=-1))
    v_t = 0.5 * (v + np.concatenate([v[..., :1, :], v[..., :-1, :]], axis=-2))
    temperature = np.asarray(field.temperature, dtype=np.float64)
    pressure = np.asarray(field.pressure, dtype=np.float64)
    theta = temperature * (P_REFERENCE / pressure) ** (R_DRY / CP_DRY)
    dims = ("time", "level", "y", "x")
    ds = xr.Dataset(
        {"theta_v": (dims, theta), "u": (dims, u_t), "v": (dims, v_t)},
        coords={"z_agl": ("level", height)},
    )
    out = bulk_richardson(ds, theta_v="theta_v", ri_crit=ri_crit, height="z_agl")
    return out["pbl_height"].transpose("time", "y", "x").values


def hanna_turbulence_from(
    field: MetField, layer: SurfaceLayer, *, it: int, iy: int, ix: int
) -> HannaTurbulence:
    """Hanna (1982) turbulence at one analysis column and met time.

    ``HannaTurbulence`` selects its regime by the sign of ``L`` and treats
    ``L = 0`` as neutral, so a neutral column (``1/L = 0``) is passed as
    ``L = 0`` rather than ``±inf``.

    Args:
        field: Met on the analysis grid (for ``pbl_height``).
        layer: Surface-layer scales from
            :func:`~plumax.met.surface_layer.surface_layer_from_met`.
        it: Met time index.
        iy: Analysis y index.
        ix: Analysis x index.

    Returns:
        The turbulence model, with plain-float parameters.
    """
    inv_l = float(layer.inverse_obukhov_length[it, iy, ix])
    return HannaTurbulence(
        u_star=float(layer.u_star[it, iy, ix]),
        pbl_height=float(field.pbl_height[it, iy, ix]),
        obukhov_length=0.0 if inv_l == 0.0 else 1.0 / inv_l,
        w_star=float(layer.w_star[it, iy, ix]),
    )
