"""Boundary-layer diagnostics on a ``MetField``, and the Hanna turbulence factory.

The bulk-Richardson PBL height is diagnostix's, on the arrays of the field, so
it traces; the Hanna factory turns a :class:`SurfaceLayer` into Tier II's
turbulence model at one column.
"""

from __future__ import annotations

import einx
import jax.numpy as jnp
from diagnostix.columns import boundary_layer_height
from geophysix.thermo import potential_temperature as theta_of
from jaxtyping import Array, Float

from plumax._src.constants import P_REFERENCE
from plumax.lagrangian.turbulence import DEFAULT_CORIOLIS_PARAMETER, HannaTurbulence
from plumax.met._src.surface_layer import SurfaceLayer
from plumax.met.field import MetField


def pbl_height_bulk_richardson(
    field: MetField, *, ri_crit: float = 0.25
) -> Float[Array, "time y x"]:
    """PBL height h from the bulk Richardson number of the analysis profiles.

    Ri_b(z) = g (θ(z) − θ_s)(z − z_s) / (θ_s |𝐮(z) − 𝐮_s|²),  h: first Ri_b = Ri_c,

    with θ = T (p_ref/p)^{R_d/c_p} (a ``MetField`` has no humidity, so θ_v = θ)
    and the surface values taken from the lowest level.

    Algorithm:
        1. Build θ, the T-point wind and the heights above the grid's surface.
        2. Hand the columns to :func:`diagnostix.columns.boundary_layer_height`,
           which finds the first crossing and interpolates between levels.

    Args:
        field: Met on the analysis grid.
        ri_crit: Critical bulk Richardson number Ri_c.

    Returns:
        h [m]; NaN where no level reaches Ri_c inside the grid, and where a
        level at or below h is NaN. Under autodiff the level of the crossing is
        piecewise constant, and the gradient flows through the interpolation.

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
    grid = field.plume_grid
    # z_surface = z₀ − Δz/2
    grid_surface = grid.z[0] - 0.5 * grid.dz
    # h_k = z_k − z_surface
    heights = grid.z - grid_surface
    wind_x, wind_y = jnp.asarray(field.u), jnp.asarray(field.v)
    # each U-point's western neighbour, the edge column repeating itself
    wind_x_west = einx.id(
        "t z y first, t z y rest -> t z y (first + rest)",
        wind_x[..., :1],
        wind_x[..., :-1],
    )
    # each V-point's southern neighbour
    wind_y_south = einx.id(
        "t z first x, t z rest x -> t z (first + rest) x",
        wind_y[..., :1, :],
        wind_y[..., :-1, :],
    )
    # u_T = (u_{i−1/2} + u_{i+1/2}) / 2
    wind_x_centre = 0.5 * (wind_x + wind_x_west)
    # v_T = (v_{j−1/2} + v_{j+1/2}) / 2
    wind_y_centre = 0.5 * (wind_y + wind_y_south)
    # θ = T (p_ref / p)^{R_d / c_p}
    potential_temperature = theta_of(
        jnp.asarray(field.temperature),
        jnp.asarray(field.pressure),
        reference_pressure=P_REFERENCE,
    )
    # the levels last, as the column kernel reads them
    theta, u, v = (
        einx.id("t z y x -> t y x z", profile)
        for profile in (potential_temperature, wind_x_centre, wind_y_centre)
    )
    return boundary_layer_height(theta, u, v, heights, critical=ri_crit)


def hanna_turbulence_from(
    field: MetField,
    layer: SurfaceLayer,
    *,
    it: int,
    iy: int,
    ix: int,
    coriolis_parameter: float = DEFAULT_CORIOLIS_PARAMETER,
) -> HannaTurbulence:
    """Hanna (1982) turbulence at one analysis column and met time.

    L = 1 / (1/L), passed as 0 at neutral.

    ``HannaTurbulence`` treats L = 0 as neutral (as well as any h/|L| < 1),
    so a neutral column (1/L = 0) is passed as L = 0 rather than ±inf.
    ``MetField`` carries no latitude, so the Coriolis parameter f is an
    argument with FLEXPART's default.

    Args:
        field: Met on the analysis grid (for the PBL height h).
        layer: Surface-layer scales from :class:`MoninObukhov`.
        it: Met time index.
        iy: Analysis y index.
        ix: Analysis x index.
        coriolis_parameter: Coriolis parameter f = 2Ω sin φ [1/s] for the
            neutral profiles; the default 1e-4 s⁻¹ (latitude ≈ 43°) is the
            value FLEXPART's ``hanna.f90`` hard-codes.

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
        coriolis_parameter=coriolis_parameter,
    )
