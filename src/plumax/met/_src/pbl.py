"""Boundary-layer diagnostics on a ``MetField``, and the Hanna turbulence factory.

The bulk-Richardson PBL height runs through xrtoolz at load time (NumPy, the
``io`` extra); the Hanna factory turns a :class:`SurfaceLayer` into Tier II's
turbulence model at one column.
"""

from __future__ import annotations

import einx
import numpy as np
import xarray as xr
from jaxtyping import Float

from plumax._src.errors import MissingExtraError
from plumax.lagrangian.turbulence import HannaTurbulence
from plumax.met._src.surface_layer import CP_DRY, P_REFERENCE, R_DRY, SurfaceLayer
from plumax.met.field import MetField


def pbl_height_bulk_richardson(
    field: MetField, *, ri_crit: float = 0.25
) -> Float[np.ndarray, "time y x"]:
    """PBL height h from the bulk Richardson number of the analysis profiles.

    Ri_b(z) = g (θ(z) − θ_s)(z − z_s) / (θ_s |𝐮(z) − 𝐮_s|²),  h: first Ri_b = Ri_c,

    with θ = T (p_ref/p)^{R_d/c_p} (a ``MetField`` has no humidity, so θ_v = θ)
    and the surface values taken from the lowest level.

    Algorithm:
        1. Build θ, the T-point wind and the heights above the grid's surface.
        2. Hand them to :func:`xrtoolz.atm.pbl_height_bulk_richardson`, which
           finds the first crossing and interpolates between levels.

    Args:
        field: Met on the analysis grid.
        ri_crit: Critical bulk Richardson number Ri_c.

    Returns:
        h [m]; NaN where no level reaches Ri_c inside the grid.

    Raises:
        MissingExtraError: If the ``io`` extra (xrtoolz) is not installed.

    Examples:
        A neutral layer under an inversion at 400 m, with shear:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.met.field import MetField
        >>> grid = make_grid(
        ...     (0.0, 400.0, 4), (0.0, 400.0, 4), (0.0, 1000.0, 20), dtype=jnp.float64
        ... )
        >>> z = grid.z - (grid.z[0] - 0.5 * grid.dz)
        >>> theta = 300.0 + 0.01 * jnp.maximum(z - 400.0, 0.0)
        >>> column = lambda f: jnp.broadcast_to(f[None, :, None, None], (1, 20, 4, 4))
        >>> field = MetField(
        ...     plume_grid=grid, times=jnp.zeros(1), u=column(2.0 + 0.005 * z),
        ...     v=jnp.zeros((1, 20, 4, 4)), w=jnp.zeros((1, 20, 4, 4)),
        ...     temperature=column(theta), pressure=jnp.full((1, 20, 4, 4), 1.0e5),
        ...     pbl_height=jnp.zeros((1, 4, 4)),
        ... )
        >>> height = pbl_height_bulk_richardson(field)
        >>> height.shape
        (1, 4, 4)
        >>> bool(((height > 350.0) & (height < 1000.0)).all())
        True
    """
    try:
        import xrtoolz.atm
    except ImportError as err:
        raise MissingExtraError(
            "pbl_height_bulk_richardson runs through xrtoolz; install it with "
            "`pip install 'plumax[io]'`"
        ) from err
    grid = field.plume_grid
    grid_z = np.asarray(grid.z, dtype=np.float64)
    # z_surface = z₀ − Δz/2
    grid_surface = grid_z[0] - 0.5 * grid.dz
    # h_k = z_k − z_surface
    heights = grid_z - grid_surface
    u = np.asarray(field.u, dtype=np.float64)
    v = np.asarray(field.v, dtype=np.float64)
    # each U-point's western neighbour, the edge column repeating itself
    u_west = einx.id(
        "t z y first, t z y rest -> t z y (first + rest)", u[..., :1], u[..., :-1]
    )
    # each V-point's southern neighbour
    v_south = einx.id(
        "t z first x, t z rest x -> t z (first + rest) x",
        v[..., :1, :],
        v[..., :-1, :],
    )
    # u_T = (u_{i−1/2} + u_{i+1/2}) / 2
    u_centre = 0.5 * (u + u_west)
    # v_T = (v_{j−1/2} + v_{j+1/2}) / 2
    v_centre = 0.5 * (v + v_south)
    temperature = np.asarray(field.temperature, dtype=np.float64)
    pressure = np.asarray(field.pressure, dtype=np.float64)
    # θ = T (p_ref / p)^{R_d / c_p}
    theta = temperature * (P_REFERENCE / pressure) ** (R_DRY / CP_DRY)
    dims = ("time", "level", "y", "x")
    profiles = xr.Dataset(
        {"theta_v": (dims, theta), "u": (dims, u_centre), "v": (dims, v_centre)},
        coords={"z_agl": ("level", heights)},
    )
    result = xrtoolz.atm.pbl_height_bulk_richardson(
        profiles, theta_v="theta_v", ri_crit=ri_crit, height="z_agl"
    )
    return result["pbl_height"].transpose("time", "y", "x").values


def hanna_turbulence_from(
    field: MetField, layer: SurfaceLayer, *, it: int, iy: int, ix: int
) -> HannaTurbulence:
    """Hanna (1982) turbulence at one analysis column and met time.

    L = 1 / (1/L), passed as 0 at neutral.

    ``HannaTurbulence`` selects its regime by the sign of L and treats L = 0
    as neutral, so a neutral column (1/L = 0) is passed as L = 0 rather than
    ±inf.

    Args:
        field: Met on the analysis grid (for the PBL height h).
        layer: Surface-layer scales from :class:`MoninObukhov`.
        it: Met time index.
        iy: Analysis y index.
        ix: Analysis x index.

    Returns:
        The turbulence model, with plain-float parameters.

    Examples:
        A neutral column maps to Hanna's L = 0, an unstable one keeps its sign:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.met._src.surface_layer import MoninObukhov
        >>> from plumax.met.field import MetField
        >>> grid = make_grid(
        ...     (0.0, 400.0, 4), (0.0, 400.0, 4), (0.0, 200.0, 4), dtype=jnp.float64
        ... )
        >>> shape = (1, 4, 4, 4)
        >>> field = MetField(
        ...     plume_grid=grid, times=jnp.zeros(1), u=jnp.full(shape, 5.0),
        ...     v=jnp.zeros(shape), w=jnp.zeros(shape),
        ...     temperature=jnp.full(shape, 300.0), pressure=jnp.full(shape, 1.0e5),
        ...     pbl_height=jnp.full((1, 4, 4), 800.0),
        ... )
        >>> neutral = MoninObukhov()(field, z0=0.1, sensible_heat_flux=0.0)
        >>> turbulence = hanna_turbulence_from(field, neutral, it=0, iy=1, ix=0)
        >>> turbulence.obukhov_length, turbulence.pbl_height
        (0.0, 800.0)
        >>> sunny = MoninObukhov()(field, z0=0.1, sensible_heat_flux=200.0)
        >>> hanna_turbulence_from(field, sunny, it=0, iy=1, ix=0).obukhov_length < 0
        True
    """
    inverse_length = float(layer.inverse_obukhov_length[it, iy, ix])
    # L = 1 / (1/L), or Hanna's neutral sentinel 0
    obukhov_length = 0.0 if inverse_length == 0.0 else 1.0 / inverse_length
    return HannaTurbulence(
        u_star=float(layer.u_star[it, iy, ix]),
        pbl_height=float(field.pbl_height[it, iy, ix]),
        obukhov_length=obukhov_length,
        w_star=float(layer.w_star[it, iy, ix]),
    )
