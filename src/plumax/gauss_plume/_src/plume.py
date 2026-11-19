"""Steady-state Gaussian plume forward model (JAX).

Implements the analytical solution to the steady-state advection-diffusion
equation with a ground-reflection boundary condition, in wind-aligned
coordinates (x', y', z):

    C(x', y', z) = (Q / (2π u σ_y σ_z))
                   · exp(-y'² / (2 σ_y²))
                   · [exp(-(z - H)² / (2 σ_z²)) + exp(-(z + H)² / (2 σ_z²))]

where Q is the emission rate [kg/s], u = √(u² + v²) the wind speed [m/s],
σ_y(x'), σ_z(x') the Briggs dispersion coefficients, and H the source height.

Public surface
--------------
- ``rotate_to_wind_frame``       : JIT-compiled (x, y) → (x', y') rotation
- ``plume_concentration``        : JIT-compiled forward model
- ``plume_concentration_vmap``   : vmapped over receptor locations
- ``simulate_plume``              : xarray wrapper building a 3-D field
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einx
import jax.numpy as jnp
import numpy as np
from functax import quad as functax_quad
from integrix import rule as integrix_rule
from jax import jit, vmap
from jax.typing import ArrayLike
from jaxtyping import Float

from plumax._src.dataset import label_concentrations
from plumax._src.domain import (
    check_domain,
    check_source,
    flat_mesh,
    node_axis,
)
from plumax._src.tracing import require_positive
from plumax._src.wind import wind_components
from plumax.gauss_plume._src.dispersion import check_stability
from plumax.gauss_plume.dispersion import (
    calculate_briggs_dispersion,
    get_dispersion_params,
)
from plumax.gauss_plume.mo_dispersion import GolderDispersion


if TYPE_CHECKING:  # xarray is imported lazily inside simulate_plume
    import xarray as xr

    from plumax.gauss_plume.plume_rise import BriggsPlumeRise


# ── Coordinate transform ─────────────────────────────────────────────────────


@jit
def rotate_to_wind_frame(
    x: jnp.ndarray,
    y: jnp.ndarray,
    source_x: Float[ArrayLike, ""],
    source_y: Float[ArrayLike, ""],
    wind_u: Float[ArrayLike, ""],
    wind_v: Float[ArrayLike, ""],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Rotate fixed-frame coordinates into the wind-aligned frame.

    The wind-aligned frame has its origin at the source, its x'-axis pointing
    downwind, and its y'-axis pointing crosswind (90° counter-clockwise from
    x', following the right-hand rule). The rotation angle is
    ``θ = atan2(wind_v, wind_u)`` [rad].

    Parameters
    ----------
    x, y : jnp.ndarray
        Receptor coordinates in the fixed frame [m]. Must have matching
        shapes (or be broadcast-compatible).
    source_x, source_y : float
        Source location in the fixed frame [m].
    wind_u, wind_v : float
        Wind velocity components [m/s].

    Returns
    -------
    x_wind : jnp.ndarray
        Downwind coordinate x' [m] (> 0 downwind, < 0 upwind of source).
    y_wind : jnp.ndarray
        Crosswind coordinate y' [m].
    """
    # Δx = x − x_s
    spacing_x = x - source_x
    # Δy = y − y_s
    spacing_y = y - source_y

    # θ = atan2(v, u)
    angle = jnp.arctan2(wind_v, wind_u)
    cos_theta = jnp.cos(angle)
    sin_theta = jnp.sin(angle)

    # x' = Δx cos θ + Δy sin θ
    x_wind = spacing_x * cos_theta + spacing_y * sin_theta
    # y' = −Δx sin θ + Δy cos θ
    y_wind = -spacing_x * sin_theta + spacing_y * cos_theta
    return x_wind, y_wind


# ── Forward model ────────────────────────────────────────────────────────────


# Minimum wind speed (below which the plume model is not physically valid — a
# puff model should be used instead). The model clamps to this value rather
# than raising so that upstream Bayesian inference remains differentiable
# when wind draws dip into a calm regime.
MIN_WIND_SPEED: float = 0.5  # m/s


@jit
def plume_concentration(
    x: jnp.ndarray,
    y: jnp.ndarray,
    z: jnp.ndarray,
    source_x: Float[ArrayLike, ""],
    source_y: Float[ArrayLike, ""],
    source_z: Float[ArrayLike, ""],
    wind_u: Float[ArrayLike, ""],
    wind_v: Float[ArrayLike, ""],
    emission_rate: Float[ArrayLike, ""],
    dispersion_params: jnp.ndarray | GolderDispersion,
    plume_rise: BriggsPlumeRise | None = None,
) -> jnp.ndarray:
    """Evaluate the steady-state Gaussian plume concentration.

    Computes the mass concentration [kg/m³] at receptor points ``(x, y, z)``
    in the fixed reference frame. Coordinates are rotated into the
    wind-aligned frame internally, upwind points are masked to zero, and
    ground reflection is applied.

    Parameters
    ----------
    x, y, z : jnp.ndarray
        Receptor coordinates in the fixed frame [m]. Broadcast-compatible.
    source_x, source_y, source_z : float
        Source location [m]. ``source_z`` is the emission height above ground.
    wind_u, wind_v : float
        Wind velocity components [m/s]. The total wind speed is clamped
        below to :data:`MIN_WIND_SPEED` for numerical stability.
    emission_rate : float
        Continuous emission rate Q [kg/s].
    dispersion_params : jnp.ndarray, shape (6,), or GolderDispersion
        ``[a_y, b_y, c_y, a_z, b_z, c_z]`` (Briggs parameters of one PG
        class), or a :class:`~plumax.gauss_plume.GolderDispersion` that picks
        the PG class from the Obukhov length and roughness length.
    plume_rise : BriggsPlumeRise, optional
        Plume rise Δh(x'). When given, the release height at each receptor is
        the effective height ``source_z + Δh(x')`` with Δh evaluated at the
        receptor's downwind distance and the (clamped) wind speed; it is
        differentiable in the stack parameters. Default ``None`` keeps the
        fixed height ``source_z``.

    Returns
    -------
    concentration : jnp.ndarray
        Mass concentration [kg/m³], same shape as the broadcast of the
        input coordinates. Zero at upwind locations (x' ≤ 0).
    """
    wind_speed = jnp.sqrt(wind_u**2 + wind_v**2)
    wind_speed = jnp.maximum(wind_speed, MIN_WIND_SPEED)

    x_downwind, y_crosswind = rotate_to_wind_frame(
        x, y, source_x, source_y, wind_u, wind_v
    )

    x_positive = jnp.maximum(x_downwind, 1.0)
    if isinstance(dispersion_params, GolderDispersion):
        crosswind_spread, vertical_spread = dispersion_params(x_positive)
    else:
        crosswind_spread, vertical_spread = calculate_briggs_dispersion(
            x_positive, dispersion_params
        )

    # 2π U σ_y σ_z
    normalization = 2.0 * jnp.pi * wind_speed * crosswind_spread * vertical_spread

    # exp(−y'² / 2σ_y²)
    exp_y = jnp.exp(-0.5 * jnp.square(y_crosswind / crosswind_spread))

    if plume_rise is None:
        release_height = source_z
    else:
        # H_eff = H + Δh(x')
        release_height = source_z + plume_rise(x_positive, wind_speed=wind_speed)

    # z − H_eff and z + H_eff, the direct and ground-reflected images
    z_diff = z - release_height
    height_sum = z + release_height
    # exp(−(z − H)² / 2σ_z²)
    exp_z_direct = jnp.exp(-0.5 * jnp.square(z_diff / vertical_spread))
    # exp(−(z + H)² / 2σ_z²)
    exp_z_reflected = jnp.exp(-0.5 * jnp.square(height_sum / vertical_spread))

    concentration = (
        (emission_rate / normalization) * exp_y * (exp_z_direct + exp_z_reflected)
    )
    return jnp.where(x_downwind > 0.0, concentration, 0.0)


plume_concentration_vmap = vmap(
    plume_concentration,
    in_axes=(0, 0, 0, None, None, None, None, None, None, None),
    out_axes=0,
)
"""Vectorised over receptor locations — x, y, z batched along axis 0."""


# ── Xarray wrapper ───────────────────────────────────────────────────────────


def simulate_plume(
    emission_rate: float,
    source_location: tuple[float, float, float],
    wind_speed: float,
    wind_direction: float,
    stability_class: str | GolderDispersion,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    background_conc: float = 0.0,
    plume_rise: BriggsPlumeRise | None = None,
) -> xr.Dataset:
    """Simulate a steady-state Gaussian plume on a 3-D grid.

    Wraps :func:`plume_concentration` with grid construction, wind-from-angle
    to (u, v) conversion, and column integration, returning an
    :class:`xarray.Dataset` suitable for plotting / I/O.

    Wind-direction convention follows the meteorological "from" convention:
    ``wind_direction = 270°`` is a wind *from* the west (flowing east);
    ``wind_direction = 0°`` is a wind *from* the north (flowing south).

        u = -wind_speed · sin(θ),     v = -wind_speed · cos(θ),
            with θ = wind_direction in radians.

    Parameters
    ----------
    emission_rate : float
        Continuous emission rate Q [kg/s]. Must be strictly positive.
    source_location : tuple of float, (3,)
        (x, y, z) source coordinates [m]. ``z ≥ 0``.
    wind_speed : float
        Wind speed magnitude [m/s]. Must be strictly positive.
    wind_direction : float
        Wind direction [degrees from North, meteorological "from" convention].
    stability_class : str or GolderDispersion
        One of ``'A'``, ..., ``'F'``, or a
        :class:`~plumax.gauss_plume.GolderDispersion` that picks the class from
        the Obukhov length and roughness length.
    domain_x, domain_y, domain_z : tuple of (float, float, int)
        ``(start, stop, n_points)`` for each axis. ``start < stop`` and
        ``n_points ≥ 2``.
    background_conc : float
        Additive background concentration [kg/m³]. Default ``0.0``.
    plume_rise : BriggsPlumeRise, optional
        Plume rise added to the source height ``z`` (see
        :func:`plume_concentration`). Default ``None``: no rise.

    Returns
    -------
    ds : xarray.Dataset
        Variables: ``concentration`` (x, y, z) [kg/m³],
        ``column_concentration`` (x, y) [kg/m²].

    Raises
    ------
    ValueError
        On invalid stability class, non-positive emission rate, non-positive
        wind speed, malformed domain tuples, or negative source height.
    """
    import xarray as xr  # lazy — only this entry point requires xarray

    require_positive("emission_rate", emission_rate, caller="simulate_plume")
    require_positive("wind_speed", wind_speed, caller="simulate_plume")
    is_golder = isinstance(stability_class, GolderDispersion)
    if not is_golder:
        check_stability(stability_class, caller="simulate_plume")
    check_source(source_location, caller="simulate_plume")
    check_domain("domain_x", domain_x, caller="simulate_plume")
    check_domain("domain_y", domain_y, caller="simulate_plume")
    check_domain("domain_z", domain_z, caller="simulate_plume")

    if isinstance(stability_class, GolderDispersion):
        dispersion_params = stability_class
        stability_attr = "golder"
    else:
        dispersion_params = get_dispersion_params(stability_class)
        stability_attr = stability_class

    easting = node_axis(domain_x)
    northing = node_axis(domain_y)
    height = node_axis(domain_z)
    num_x, num_y = len(easting), len(northing)

    wind_u, wind_v = wind_components(wind_speed, wind_direction, xp=np)

    src_x, src_y, src_z = source_location

    easting_flat, northing_flat, height_flat = flat_mesh(easting, northing, height)
    conc_flat = plume_concentration(
        jnp.asarray(easting_flat),
        jnp.asarray(northing_flat),
        jnp.asarray(height_flat),
        src_x,
        src_y,
        src_z,
        float(wind_u),
        float(wind_v),
        emission_rate,
        dispersion_params,
        plume_rise=plume_rise,
    )
    conc_flat = np.asarray(conc_flat)
    concentration = einx.id("(x y z) -> x y z", conc_flat, x=num_x, y=num_y)
    concentration = concentration + background_conc

    # Trapezoidal vertical integration: the z grid from ``np.linspace`` includes
    # both endpoints, so a plain ``sum * dz`` over-counts a vertically uniform
    # component (e.g. a background) by one grid interval, a factor
    # num_z / (num_z − 1).
    # h_k, the trapezoid weights on the levels z_k
    levels = functax_quad.trapezoid(height)
    # Ω(x, y) = Σ_k h_k C(x, y, z_k)
    integral = integrix_rule.integrate(levels, concentration, axis=2)
    column = np.asarray(integral.value)

    dataset = xr.Dataset(
        data_vars={
            "concentration": (["x", "y", "z"], concentration),
            "column_concentration": (["x", "y"], column),
        },
        coords={"x": easting, "y": northing, "z": height},
        attrs={
            "title": "Steady-state Gaussian plume (JAX)",
            "emission_rate": emission_rate,
            "emission_rate_units": "kg/s",
            "source_x": src_x,
            "source_y": src_y,
            "source_z": src_z,
            "wind_speed": wind_speed,
            "wind_speed_units": "m/s",
            "wind_direction": wind_direction,
            "wind_direction_units": "degrees from North (meteorological)",
            "wind_u": float(wind_u),
            "wind_v": float(wind_v),
            "stability_class": stability_attr,
            "background_concentration": background_conc,
        },
    )
    label_concentrations(dataset)
    return dataset
