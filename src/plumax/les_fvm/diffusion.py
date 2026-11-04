"""K-theory eddy diffusion for the Eulerian dispersion solver.

Under the K-theory closure the turbulent flux is modelled as a gradient
term

    u' C'  ≈  -K · ∇C,

giving a diffusion contribution ``∂_t C ⊇ ∇·(K ∇C)`` that closes the
mean-flow tracer equation.  Here ``K = diag(K_h, K_h, K_z)`` is an
anisotropic diagonal diffusivity — horizontal eddies are taken isotropic
in (x, y) while the vertical component is smaller in a stable boundary
layer.

Horizontal diffusion is assembled via :class:`finitevolx.Diffusion3D`
(one 2-D Laplacian per z-level) and vertical diffusion by the local
:func:`_vertical_ops.vertical_diffusion_tendency` helper.

The :func:`pg_eddy_diffusivity` helper converts Pasquill-Gifford σ rates
into an effective ``(K_h, K_z)`` using the standard Taylor relation

    K = (1/2) d(σ²)/dt  ≈  σ² / (2 t),

evaluated at a user-supplied reference travel distance.  This is a
rough calibration and is exposed primarily as a convenience for
cross-checks against ``gauss_puff`` at the reference distance.

:func:`mo_eddy_diffusivity` follows the diagnosed boundary layer instead: a
K-profile ``K_z(z)`` from the :mod:`plumax.met.surface_layer` scales. And
:func:`smagorinsky_k_h` gives a field-valued ``K_h`` from the resolved horizontal
deformation of a spatially varying wind.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from finitevolx import (
    Diffusion3D,
    avg_xy_bwd,
    shear_strain,
    tensor_strain,
)
from jaxtyping import Array, Float

from plumax.gauss_puff.dispersion import (
    calculate_pg_dispersion,
    get_pg_params,
)
from plumax.les_fvm._vertical_ops import vertical_diffusion_tendency
from plumax.les_fvm.grid import PlumeGrid3D


if TYPE_CHECKING:
    from plumax.les_fvm.wind import PrescribedWindField
    from plumax.met.surface_layer import SurfaceLayer


class EddyDiffusivity(eqx.Module):
    """Scalar or field-valued ``(K_h, K_z)`` eddy diffusivity.

    Attributes
    ----------
    horizontal : float or Float[Array, "nz ny nx"]
        Horizontal eddy diffusivity [m²/s].  A scalar is broadcast over
        the grid.  A field is expected on interior T-points.
    vertical : float or Float[Array, "nz ny nx"]
        Vertical eddy diffusivity [m²/s].  Same broadcasting rules.
    """

    horizontal: float | Float[Array, "nz ny nx"]
    vertical: float | Float[Array, "nz ny nx"]

    def as_arrays(
        self,
    ) -> tuple[
        float | Float[Array, "Nz Ny Nx"],
        float | Float[Array, "Nz Ny Nx"],
    ]:
        """Return ``(K_h, K_z)`` in forms consumable by the tendency calls."""
        return _broadcast_or_scalar(self.horizontal), _broadcast_or_scalar(
            self.vertical
        )


def _broadcast_or_scalar(
    k: float | Float[Array, "nz ny nx"],
) -> float | Float[Array, "Nz Ny Nx"]:
    """Pass scalars through, pad interior fields to the full ghosted layout."""
    k_arr = jnp.asarray(k)
    if k_arr.ndim == 0:
        return k_arr
    return jnp.pad(k_arr, ((1, 1), (1, 1), (1, 1)), mode="edge")


@eqx.filter_jit
def diffusion_tendency(
    concentration: Float[Array, "Nz Ny Nx"],
    eddy_diffusivity: EddyDiffusivity,
    plume_grid: PlumeGrid3D,
    periodic: tuple[bool, bool] = (False, False),
) -> Float[Array, "Nz Ny Nx"]:
    """Full 3-D eddy diffusion tendency ``∇·(K ∇C)`` at interior T-points.

    Parameters
    ----------
    concentration : Float[Array, "Nz Ny Nx"]
        Tracer at T-points.
    eddy_diffusivity : EddyDiffusivity
        ``(K_h, K_z)`` diagonal diffusivity.
    plume_grid : PlumeGrid3D
    periodic : tuple[bool, bool], default ``(False, False)``
        ``(x_periodic, y_periodic)``.  For a **field** ``K_h`` on a periodic
        axis the coefficient ghost is *wrapped* (rather than left edge-padded
        by :meth:`EddyDiffusivity.as_arrays`) so the paired periodic seam
        faces share one diffusivity and their diffusive fluxes cancel.  A
        scalar ``K_h`` is uniform and already conserves, so it is untouched.

    Returns
    -------
    Float[Array, "Nz Ny Nx"]
        Diffusive tendency at T-points, zero on every ghost face.

    Notes
    -----
    The horizontal operator runs in finitevolX ``wall="open"`` mode, so the
    lateral wall diffusive fluxes ``K_h·(C_edge − C_ghost)/dx`` are assembled
    from the ghost ring.  The caller must fill the ghost cells with the
    desired lateral BC (Dirichlet / periodic) *before* calling — see
    :func:`~plumax.les_fvm.boundary.apply_boundary_conditions`.  This is what
    makes horizontal diffusion honor ``bc_x`` / ``bc_y``.
    """
    k_h, k_z = eddy_diffusivity.as_arrays()
    x_periodic, y_periodic = periodic
    if jnp.ndim(k_h) > 0:
        # Field K_h: wrap the coefficient ghost on periodic axes so a periodic
        # seam uses one shared face diffusivity (scalar K_h needs nothing).
        if x_periodic:
            k_h = k_h.at[:, :, 0].set(k_h[:, :, -2]).at[:, :, -1].set(k_h[:, :, 1])
        if y_periodic:
            k_h = k_h.at[:, 0, :].set(k_h[:, -2, :]).at[:, -1, :].set(k_h[:, 1, :])
    horizontal_op = Diffusion3D(grid=plume_grid.grid)
    horizontal = horizontal_op(concentration, k_h, wall="open")
    vertical = vertical_diffusion_tendency(concentration, k_z, plume_grid.dz)
    return horizontal + vertical


def pg_eddy_diffusivity(
    stability_class: str,
    wind_speed: float,
    reference_distance: float = 200.0,
) -> EddyDiffusivity:
    """Rough K-theory calibration from Pasquill-Gifford σ curves.

    For a neutral puff travelling at ``u`` under stability class ``C``,
    the cross-wind and vertical spreads at a reference distance
    ``x_ref`` are ``σ_y(x_ref)`` and ``σ_z(x_ref)``.  Taylor's identity
    ``K = σ² / (2 t)`` with ``t = x_ref / u`` gives a simple constant
    effective diffusivity.  The result is the *average* K over
    ``[0, x_ref]``; users wanting a sharper match across a wide range
    of travel distances should pass a spatially-varying field directly.

    Parameters
    ----------
    stability_class : str
        One of ``"A"``..``"F"``.
    wind_speed : float
        Representative wind speed ``u`` [m/s].  Must be > 0.
    reference_distance : float, default 200.0
        Reference downwind distance ``x_ref`` [m] at which the σ curves
        are sampled.

    Returns
    -------
    EddyDiffusivity
        Scalar ``(K_h, K_z)`` calibrated at ``x_ref``.

    Raises
    ------
    ValueError
        If ``wind_speed`` or ``reference_distance`` is non-positive.
    """
    if wind_speed <= 0.0:
        raise ValueError(
            f"pg_eddy_diffusivity: wind_speed must be > 0 (got {wind_speed!r})"
        )
    if reference_distance <= 0.0:
        raise ValueError(
            "pg_eddy_diffusivity: reference_distance must be > 0 "
            f"(got {reference_distance!r})"
        )
    params = get_pg_params(stability_class)
    distance = jnp.asarray(reference_distance, dtype=jnp.float32)
    _, sigma_y, sigma_z = calculate_pg_dispersion(distance, params)
    t_ref = reference_distance / wind_speed
    k_h = float(sigma_y**2 / (2.0 * t_ref))
    k_z = float(sigma_z**2 / (2.0 * t_ref))
    return EddyDiffusivity(horizontal=k_h, vertical=k_z)


def mo_eddy_diffusivity(
    grid: PlumeGrid3D,
    surface_layer: SurfaceLayer,
    pbl_height: float | Float[Array, "time y x"],
    *,
    it: int,
    k_h: float | Float[Array, "nz ny nx"],
) -> EddyDiffusivity:
    """Monin–Obukhov K-profile for ``K_z``, with a caller-supplied ``K_h``.

    K_z(z) = κ u* z (1 − min(z/z_i, 1))² / φ_h(z/L).

    The cubic-in-depth profile vanishes at the ground and at the boundary-layer
    top, and φ_h (:func:`plumax.met.surface_layer.phi_h`) carries the stability:
    below 1 in unstable air, above 1 in stable air. The ``min`` matters: without
    it (1 − z/z_i)² grows again above z_i, and the free troposphere would mix.

    Algorithm:
        1. Take the height above ground of each T-level, z = (k + ½) Δz, the
           convention :class:`~plumax.met.surface_layer.MoninObukhov` uses.
        2. Take u*, 1/L and z_i at time index ``it``.
        3. Evaluate the profile on every T-point of the interior.

    Args:
        grid: The analysis grid.
        surface_layer: u* and 1/L on ``(time, y, x)``, from
            :class:`~plumax.met.surface_layer.MoninObukhov`.
        pbl_height: Boundary-layer height z_i [m], a scalar or
            ``MetField.pbl_height``.
        it: Time index into ``surface_layer`` (and a field ``pbl_height``).
        k_h: Horizontal diffusivity [m²/s], passed through: a constant, or
            :func:`smagorinsky_k_h`.

    Returns:
        ``EddyDiffusivity`` with ``vertical`` on interior T-points.

    Raises:
        equinox.EquinoxRuntimeError: Under ``jit`` too, if ``pbl_height`` is
            negative.

    Examples:
        Neutral air at mid-PBL gives κ u* z_i / 8:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.met.surface_layer import SurfaceLayer
        >>> grid = make_grid((0.0, 40.0, 4), (0.0, 40.0, 4), (0.0, 160.0, 4))
        >>> layer = SurfaceLayer(
        ...     u_star=jnp.full((1, 4, 4), 0.5),
        ...     inverse_obukhov_length=jnp.zeros((1, 4, 4)),
        ...     w_star=jnp.zeros((1, 4, 4)),
        ... )
        >>> eddy = mo_eddy_diffusivity(grid, layer, 120.0, it=0, k_h=1.0)
        >>> eddy.vertical.shape
        (4, 4, 4)
        >>> bool(jnp.allclose(eddy.vertical[1], 0.4 * 0.5 * 120.0 / 8.0))
        True

        Above the boundary-layer top nothing mixes:

        >>> bool((eddy.vertical[3] == 0.0).all())
        True
    """
    # met.field imports les_fvm.grid, so importing met at module level is a cycle
    from plumax.met.surface_layer import KARMAN, phi_h

    # z = z_k − z_ground = (k + ½) Δz
    height = grid.z - grid.z[0] + 0.5 * grid.dz
    u_star = surface_layer.u_star[it]
    inverse_length = surface_layer.inverse_obukhov_length[it]
    depth = jnp.asarray(pbl_height, dtype=height.dtype)
    depth = eqx.error_if(depth, jnp.any(depth < 0.0), "pbl_height must be ≥ 0")
    if depth.ndim == 3:
        depth = depth[it]
    depth = jnp.broadcast_to(depth, u_star.shape)

    # ζ = z / L
    zeta = einx.multiply("level, y x -> level y x", height, inverse_length)
    # z / z_i, infinite for a collapsed layer so that K_z = 0 there
    relative_height = einx.divide("level, y x -> level y x", height, depth)
    # (1 − min(z/z_i, 1))²
    depth_shape = (1.0 - jnp.minimum(relative_height, 1.0)) ** 2
    # κ u* z
    surface_scale = KARMAN * einx.multiply("level, y x -> level y x", height, u_star)
    # K_z = κ u* z (1 − min(z/z_i, 1))² / φ_h(ζ)
    vertical = surface_scale * depth_shape / phi_h(zeta)
    return EddyDiffusivity(horizontal=k_h, vertical=vertical)


def smagorinsky_k_h(
    wind: PrescribedWindField,
    grid: PlumeGrid3D,
    *,
    c_s: float = 0.2,
    t: float | Float[Array, ""],
) -> Float[Array, "nz ny nx"]:
    """Smagorinsky horizontal diffusivity from the resolved deformation.

    K_h = (c_s Δ)² |D|,  Δ = √(Δx Δy),  |D|² = D_T² + D_S²,

    with the tension D_T = ∂u/∂x − ∂v/∂y and the shear D_S = ∂v/∂x + ∂u/∂y.
    These are the two independent components of the trace-free horizontal strain,
    so uniform translation, solid-body rotation and isotropic horizontal
    divergence all give K_h = 0. Vertical shear is left out on purpose: K_h only
    mixes along levels, and ∂u/∂z in the boundary layer would swamp it on a grid
    with Δz ≪ Δx.

    Algorithm:
        1. Evaluate the wind at ``t`` on its C-grid stagger: u on U-points, v on
           V-points, level by level.
        2. D_T lands on T-points; D_S lands on corners, and D_S² is averaged from
           the four corners to the T-point, so a checkerboard in D_S cannot
           cancel.
        3. K_h = (c_s Δ)² |D| on interior T-points.

    The wind's ghost ring is an edge pad, so its normal gradient across the
    domain edge reads zero and the outermost interior cells are biased low;
    cells one further in are exact for a linear wind.

    Args:
        wind: The wind field.
        grid: The grid it lives on.
        c_s: Smagorinsky constant c_s (> 0).
        t: Time [s].

    Returns:
        K_h [m²/s] on interior T-points, the layout ``EddyDiffusivity`` pads.

    Raises:
        ValueError: If ``c_s`` is not positive.

    Examples:
        A uniform wind has no deformation, and a shear flow has K_h = (c_s Δ)² S:

        >>> import jax.numpy as jnp
        >>> from plumax.les_fvm.grid import make_grid
        >>> from plumax.les_fvm.wind import uniform_wind_field, wind_field_from_callable
        >>> grid = make_grid((0.0, 80.0, 8), (0.0, 80.0, 8), (0.0, 40.0, 4))
        >>> uniform = uniform_wind_field(grid, u=5.0, v=1.0)
        >>> bool((smagorinsky_k_h(uniform, grid, t=0.0) == 0.0).all())
        True
        >>> shear = wind_field_from_callable(
        ...     grid, lambda t, x, y, z: (0.1 * y, jnp.zeros_like(y), jnp.zeros_like(y))
        ... )
        >>> k_h = smagorinsky_k_h(shear, grid, t=0.0)
        >>> bool(jnp.allclose(k_h[:, 1:-1, 1:-1], (0.2 * 10.0) ** 2 * 0.1, rtol=1e-4))
        True

    References:
        Smagorinsky (1963), Mon. Wea. Rev. 91, 99–164.
    """
    if c_s <= 0.0:
        raise ValueError(f"smagorinsky_k_h: c_s must be > 0 (got {c_s!r})")
    dx, dy = grid.dx, grid.dy
    u, v, _ = wind(jnp.asarray(t, dtype=grid.x.dtype))

    def squared_deformation(
        u_level: Float[Array, "Ny Nx"], v_level: Float[Array, "Ny Nx"]
    ) -> Float[Array, "ny nx"]:
        # D_T = ∂u/∂x − ∂v/∂y at T-points
        tension = tensor_strain(u_level, v_level, dx, dy)[1:-1, 1:-1]
        # D_S = ∂v/∂x + ∂u/∂y at corners
        shear = shear_strain(u_level, v_level, dx, dy)
        # ⟨D_S²⟩ at T-points
        shear_squared = avg_xy_bwd(shear**2)
        # |D|² = D_T² + ⟨D_S²⟩
        return tension**2 + shear_squared

    deformation_squared = jax.vmap(squared_deformation)(u[1:-1], v[1:-1])
    # |D| with a zero gradient at rest: √ has an infinite slope at 0, so take it of
    # 1 there (the double-where trick), keeping uniform-wind adjoints finite
    is_at_rest = deformation_squared == 0.0
    safe_squared = jnp.where(is_at_rest, 1.0, deformation_squared)
    deformation = jnp.where(is_at_rest, 0.0, jnp.sqrt(safe_squared))
    # K_h = (c_s Δ)² |D|, with Δ² = Δx Δy
    return c_s**2 * dx * dy * deformation


def make_eddy_diffusivity(
    spec: float | tuple[float, float] | str | EddyDiffusivity,
    stability_class: str | None = None,
    wind_speed: float | None = None,
    reference_distance: float = 200.0,
) -> EddyDiffusivity:
    """Coerce a user-facing diffusivity spec into an :class:`EddyDiffusivity`.

    Accepts:

    - :class:`EddyDiffusivity`                       (returned unchanged)
    - ``float``                                      (isotropic scalar K)
    - ``(K_h, K_z)`` tuple                           (anisotropic scalars)
    - ``"pg"``                                       (call :func:`pg_eddy_diffusivity`;
      requires ``stability_class`` + ``wind_speed``)
    """
    if isinstance(spec, EddyDiffusivity):
        return spec
    if isinstance(spec, str):
        if spec.lower() != "pg":
            raise ValueError(
                f"make_eddy_diffusivity: unknown string spec {spec!r}; "
                "only 'pg' is supported"
            )
        if stability_class is None or wind_speed is None:
            raise ValueError(
                "make_eddy_diffusivity: 'pg' spec requires both "
                "`stability_class` and `wind_speed`"
            )
        return pg_eddy_diffusivity(
            stability_class=stability_class,
            wind_speed=wind_speed,
            reference_distance=reference_distance,
        )
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise ValueError(
                "make_eddy_diffusivity: tuple spec must be (K_h, K_z), "
                f"got length {len(spec)}"
            )
        return EddyDiffusivity(horizontal=float(spec[0]), vertical=float(spec[1]))
    k = float(spec)
    return EddyDiffusivity(horizontal=k, vertical=k)


__all__ = [
    "EddyDiffusivity",
    "diffusion_tendency",
    "make_eddy_diffusivity",
    "mo_eddy_diffusivity",
    "pg_eddy_diffusivity",
    "smagorinsky_k_h",
]
