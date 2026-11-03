"""Met-ensemble footprints and the inversion that marginalises over them.

The Tier II inversion treats the footprint F as exact; transport error only
enters through a tuned ``R_repr``. With an ensemble of meteorologies (WRF
members, ERA5 EDA) the transport error is explicit: one footprint F_e per
member, and the linearised transport error at the prior mean q_a,

    R_transport = cov_e(F_e q_a) = Σ_e (F_e − F̄) q_a q_aᵀ (F_e − F̄)ᵀ / (E − 1),

is added to the observation covariance of the inversion on the mean footprint
F̄ (v1 of the Tier II design, Step 6):

    y = F̄ q + ε,    ε ~ N(0, R + R_transport).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax.lagrangian.footprint import compute_footprint
from plumax.lagrangian.inversion import (
    GaussianPosterior,
    _blue_update,
    _check_inversion_shapes,
)


if TYPE_CHECKING:
    from plumax.lagrangian.particles import TurbulenceModel
    from plumax.lagrangian.wind_interp import GriddedWind


def ensemble_footprints(
    winds: Sequence[GriddedWind | Callable[[jax.Array], jax.Array]],
    receptors: Float[Array, "obs 3"],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    *,
    n_particles: int = 5000,
    t_back: float = 600.0,
    dt: float = 1.0,
    pbl_height: float = 1000.0,
    pbl_fraction: float = 0.5,
    air_density: float = 1.2,
    receptor_time: float = 0.0,
    seed: int = 0,
) -> Float[Array, "member obs cell"]:
    """Footprint matrix F_e of every ensemble member.

    F_e[i, s] = compute_footprint(receptor i, wind e)[s], cells flattened
    row-major over ``(x, y)``.

    Every member reuses the same particle draws for a receptor (common random
    numbers: receptor ``i`` uses seed ``seed + i`` in every member). The spread
    between members is then the effect of the winds alone; independent draws
    would add particle-sampling noise that :func:`ensemble_inversion` would count
    as transport error.

    Args:
        winds: One mean wind per member: a
            :class:`~plumax.lagrangian.wind_interp.GriddedWind` (e.g.
            ``gridded_wind_from_met`` of each member's ``MetField``) or a
            uniform ``t -> (u, v, w)`` callable.
        receptors: Receptor positions (x, y, z) [m].
        turbulence: Turbulence model, shared by the members.
        domain_x: ``(start, stop, n_cells)`` along x for the surface grid.
        domain_y: ``(start, stop, n_cells)`` along y for the surface grid.
        n_particles: Particles per receptor and member.
        t_back: Backward integration time [s].
        dt: Time step [s].
        pbl_height: Boundary-layer height [m].
        pbl_fraction: Fraction of the PBL that sees the surface flux.
        air_density: Air density [kg m⁻³].
        receptor_time: Observation time [s] on the winds' clock.
        seed: Base PRNG seed.

    Returns:
        The member footprints, units s m² kg⁻¹ (see ``compute_footprint``).

    Raises:
        ValueError: If ``winds`` is empty.
    """
    if len(winds) == 0:
        raise ValueError("ensemble_footprints: need at least one member wind.")
    receptor_points = np.asarray(receptors, dtype=float)
    members = []
    for wind in winds:
        rows = []
        for index, receptor in enumerate(receptor_points):
            footprint, _, _ = compute_footprint(
                tuple(receptor),
                turbulence,
                domain_x,
                domain_y,
                wind=wind,
                n_particles=n_particles,
                t_back=t_back,
                dt=dt,
                pbl_height=pbl_height,
                pbl_fraction=pbl_fraction,
                air_density=air_density,
                receptor_time=receptor_time,
                seed=seed + index,
            )
            rows.append(einx.rearrange("x y -> (x y)", jnp.asarray(footprint)))
        members.append(jnp.stack(rows))
    return jnp.stack(members)


def transport_covariance(
    footprints: Float[Array, "member obs cell"],
    prior_mean: Float[Array, " cell"],
) -> tuple[Float[Array, "obs cell"], Float[Array, "obs obs"]]:
    """Mean footprint F̄ and linearised transport-error covariance at q_a.

    R_transport = Σ_e d_e d_eᵀ / (E − 1),   d_e = (F_e − F̄) q_a,

    and zero for a single member.

    Args:
        footprints: Member footprints F_e.
        prior_mean: Prior mean q_a, the linearisation point.

    Returns:
        ``(F̄, R_transport)``.

    Examples:
        Two members that disagree on one receptor:

        >>> import jax.numpy as jnp
        >>> footprints = jnp.array([[[1.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 1.0]]])
        >>> mean_footprint, covariance = transport_covariance(
        ...     footprints, jnp.array([1.0, 1.0])
        ... )
        >>> mean_footprint.tolist(), covariance.tolist()
        ([[2.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 0.0]])

        Identical members carry no transport error:

        >>> same = jnp.stack([footprints[0], footprints[0]])
        >>> bool((transport_covariance(same, jnp.ones(2))[1] == 0.0).all())
        True
    """
    n_members = footprints.shape[0]
    # F̄ = F_0 + mean_e(F_e − F_0): identical members give F̄ = F_0 exactly
    offsets = footprints - footprints[0]
    mean_footprint = footprints[0] + einx.mean("member obs cell -> obs cell", offsets)
    # d_e = (F_e − F̄) q_a
    deviation = einx.subtract(
        "member obs cell, obs cell -> member obs cell", footprints, mean_footprint
    )
    response = einx.dot("member obs cell, cell -> member obs", deviation, prior_mean)
    # Σ_e d_e d_eᵀ / (E − 1)
    scatter = einx.dot("member a, member b -> a b", response, response)
    covariance = scatter / max(n_members - 1, 1)
    return mean_footprint, covariance


def ensemble_inversion(
    footprints: Float[Array, "member obs cell"],
    observation: Float[Array, " obs"],
    *,
    prior_mean: Float[Array, " cell"],
    prior_cov: Float[Array, "cell cell"],
    obs_variance: Float[Array, " obs"],
) -> GaussianPosterior:
    """Gaussian inversion that marginalises the footprint over a met ensemble.

    q* = q_a + B F̄ᵀ (F̄ B F̄ᵀ + R + R_transport)⁻¹ (y − F̄ q_a),
    P* = B − B F̄ᵀ (F̄ B F̄ᵀ + R + R_transport)⁻¹ F̄ B,

    with F̄ and R_transport from :func:`transport_covariance`. One member gives
    :func:`~plumax.lagrangian.linear_gaussian_inversion` on its footprint.

    Args:
        footprints: Member footprints F_e (e.g. :func:`ensemble_footprints`).
        observation: Observed values y.
        prior_mean: Prior mean q_a.
        prior_cov: Prior covariance B.
        obs_variance: Diagonal of the retrieval + representation covariance R.

    Returns:
        The :class:`~plumax.lagrangian.GaussianPosterior` over q.

    Raises:
        ImportError: Without the ``assim`` extra (``pip install
            'plumax[assim]'``), as ``linear_gaussian_inversion``.
    """
    member_footprints = jnp.asarray(footprints, dtype=float)
    if member_footprints.ndim != 3:
        raise ValueError(
            "ensemble_inversion: `footprints` must be 3-D (member, obs, cell)."
        )
    y = jnp.asarray(observation, dtype=float)
    qa = jnp.asarray(prior_mean, dtype=float)
    b = jnp.asarray(prior_cov, dtype=float)
    r = jnp.asarray(obs_variance, dtype=float)
    _check_inversion_shapes(member_footprints[0], y, qa, b, r)

    mean_footprint, transport = transport_covariance(member_footprints, qa)
    # R_total = R + R_transport
    total_covariance = jnp.diag(r) + transport
    # y − F̄ q_a
    innovation = y - einx.dot("obs cell, cell -> obs", mean_footprint, qa)
    increment, posterior_cov = _blue_update(
        mean_footprint,
        innovation,
        b,
        total_covariance,
        caller="ensemble_inversion",
    )
    return GaussianPosterior(
        mean=qa + increment,
        covariance=posterior_cov,
        prior_mean=qa,
        prior_covariance=b,
    )
