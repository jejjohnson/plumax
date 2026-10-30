"""Tier II and Tier III transport behind the coupled forward, through the shared column.

``coupled.build_coupled_forward`` needs one thing from a transport tier: the
column each instrument would see per unit emission, already through that
instrument's averaging kernel,

    y_inst = Q · r_inst + c_bg + bias_inst,    r_inst = A_inst col_z(c(·; Q = 1)).

:class:`TransportBackend` names that contract. The Tier I ``PlumeSource``
meets it, and this module adds the other two tiers:

- :class:`LagrangianSource` (Tier II) runs backward footprints from every
  receptor level (``lagrangian.ensemble_footprints``, one per met member),
  averages them over the members, and turns the surface flux of a unit source
  into a concentration profile at each receptor.
- :class:`EulerianSource` (Tier III) runs the ``les_fvm`` finite-volume
  transport for a unit constant emission and reads the field at the
  instrument's overpass time.

Both apply ``gauss_plume.ColumnObservation``, so every tier shares one column
and one set of units (kg m⁻² for the observation, s m⁻² per unit emission).
The emission rate stays a scalar for a point source. A Lagrangian source can
carry several emission patterns (one column each, for example one per cell),
and the response is then a design matrix with one column per pattern.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import einx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from plumax._src.errors import DomainError, GridMismatchError
from plumax.gauss_plume.observation import ColumnObservation
from plumax.lagrangian.met_ensemble import ensemble_footprints


if TYPE_CHECKING:
    from plumax.coupled.instrument import Instrument
    from plumax.lagrangian.particles import TurbulenceModel
    from plumax.lagrangian.wind_interp import GriddedWind
    from plumax.les_fvm.fourdvar import EulerianForward4DVar
    from plumax.les_fvm.grid import PlumeGrid3D


@runtime_checkable
class TransportBackend(Protocol):
    """A transport tier as the coupled forward sees it.

    r_inst = A_inst col_z(c(·; Q = 1)), the column per unit emission at each
    of the instrument's receptors.

    Attributes:
        background: Additive background column c_bg, in the units of the
            observation.
    """

    background: float

    def column_response(
        self, instrument: Instrument
    ) -> Float[Array, " obs"] | Float[Array, "obs param"]:
        """Column per unit emission at the instrument's receptors.

        Args:
            instrument: The observing instrument (receptors and kernel).

        Returns:
            The response r_inst: one value per receptor for a scalar emission
            rate, or one column per emission parameter.
        """
        ...


@dataclasses.dataclass(frozen=True)
class LagrangianSource:
    """Tier II transport: met-ensemble backward footprints through the shared column.

    c_k(x_i) = ρ_air F̄(x_i, z_k) q,   F̄ = mean_e F_e,
    r_i = A_i Σ_k h_k c_k(x_i),

    with F_e the footprint of member e at receptor (x_i, z_k), q the surface
    flux of a unit source and h_k the trapezoid weights over ``column_z``.

    Attributes:
        emission_pattern: Surface flux per unit emission q [m⁻²] on the
            footprint grid, cells flattened row-major over ``(x, y)``. A
            second axis holds one pattern per emission parameter.
        winds: One mean wind per met-ensemble member (a ``GriddedWind`` or a
            ``t -> (u, v, w)`` callable). One member is a deterministic run.
        turbulence: Turbulence model shared by the members.
        domain_x: ``(start, stop, n_cells)`` of the footprint grid along x.
        domain_y: ``(start, stop, n_cells)`` of the footprint grid along y.
        column_z: Receptor levels z_k [m] of the column, strictly increasing.
        n_particles: Particles per receptor level and member.
        t_back: Backward integration time [s].
        dt: Particle time step [s].
        pbl_height: Boundary-layer height [m].
        pbl_fraction: Fraction of the boundary layer that sees the surface flux.
        air_density: Air density ρ_air [kg m⁻³].
        seed: Base PRNG seed of the particle draws.
        background: Additive background column c_bg.
    """

    emission_pattern: Float[Array, " cell"] | Float[Array, "cell param"]
    winds: tuple[GriddedWind | Callable[[jax.Array], jax.Array], ...]
    turbulence: TurbulenceModel
    domain_x: tuple[float, float, int]
    domain_y: tuple[float, float, int]
    column_z: Float[Array, " level"]
    n_particles: int = 1000
    t_back: float = 600.0
    dt: float = 1.0
    pbl_height: float = 1000.0
    pbl_fraction: float = 0.5
    air_density: float = 1.2
    seed: int = 0
    background: float = 0.0

    def __post_init__(self) -> None:
        if len(self.winds) == 0:
            raise ValueError("LagrangianSource: need at least one member wind.")
        n_cells = int(self.domain_x[2]) * int(self.domain_y[2])
        pattern_cells = np.shape(self.emission_pattern)[0]
        if pattern_cells != n_cells:
            raise GridMismatchError(
                f"LagrangianSource: `emission_pattern` has {pattern_cells} cells "
                f"but the footprint grid has {n_cells}."
            )

    @classmethod
    def point(
        cls,
        location: tuple[float, float],
        *,
        winds: Sequence[GriddedWind | Callable[[jax.Array], jax.Array]],
        turbulence: TurbulenceModel,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        column_z: Float[Array, " level"],
        **settings: float | int,
    ) -> LagrangianSource:
        """A point source: unit emission spread over the cell that holds it.

        q_s = 1 / A_cell in the containing cell s, zero elsewhere.

        Args:
            location: Source ``(x, y)`` [m].
            winds: One mean wind per met member.
            turbulence: Turbulence model.
            domain_x: ``(start, stop, n_cells)`` along x.
            domain_y: ``(start, stop, n_cells)`` along y.
            column_z: Receptor levels z_k [m].
            **settings: Any other field of :class:`LagrangianSource`.

        Returns:
            The source, with a one-cell emission pattern.

        Raises:
            DomainError: If the location lies outside the footprint grid.
        """
        x_edges = np.linspace(domain_x[0], domain_x[1], int(domain_x[2]) + 1)
        y_edges = np.linspace(domain_y[0], domain_y[1], int(domain_y[2]) + 1)
        ix = int(np.searchsorted(x_edges, location[0], side="right")) - 1
        iy = int(np.searchsorted(y_edges, location[1], side="right")) - 1
        if not (0 <= ix < x_edges.size - 1 and 0 <= iy < y_edges.size - 1):
            raise DomainError(
                f"LagrangianSource.point: {location} lies outside the footprint grid."
            )
        # A_cell = Δx Δy
        cell_area = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        pattern = np.zeros((x_edges.size - 1, y_edges.size - 1))
        # q_s = 1 / A_cell in the containing cell
        pattern[ix, iy] = 1.0 / cell_area
        return cls(
            emission_pattern=jnp.asarray(pattern.reshape(-1)),
            winds=tuple(winds),
            turbulence=turbulence,
            domain_x=domain_x,
            domain_y=domain_y,
            column_z=jnp.asarray(column_z),
            **settings,
        )

    def footprints(self, instrument: Instrument) -> Float[Array, "member point cell"]:
        """Member footprints F_e at every receptor level, level fastest.

        Args:
            instrument: The observing instrument; its overpass time is the
                receptor time of the backward runs.

        Returns:
            One row per (receptor, level) pair, units s m² kg⁻¹.
        """
        levels = np.asarray(self.column_z, dtype=float)
        receptors = np.asarray(instrument.receptors, dtype=float)
        # every receptor (x_i, y_i) at every level z_k, level fastest
        horizontal = np.repeat(receptors, levels.size, axis=0)
        heights = np.tile(levels, receptors.shape[0])
        points = np.column_stack([horizontal, heights])
        return ensemble_footprints(
            list(self.winds),
            jnp.asarray(points),
            self.turbulence,
            self.domain_x,
            self.domain_y,
            n_particles=self.n_particles,
            t_back=self.t_back,
            dt=self.dt,
            pbl_height=self.pbl_height,
            pbl_fraction=self.pbl_fraction,
            air_density=self.air_density,
            receptor_time=instrument.observation_time,
            seed=self.seed,
        )

    def column_response(
        self, instrument: Instrument
    ) -> Float[Array, " obs"] | Float[Array, "obs param"]:
        """Column per unit emission at the instrument's receptors.

        Algorithm:
            1. Backward footprints from every receptor level, per member.
            2. Average over the members: F̄ = mean_e F_e.
            3. Concentration of a unit source: c = ρ_air F̄ q.
            4. Column and averaging kernel: r_i = A_i Σ_k h_k c_k(x_i).

        Args:
            instrument: The observing instrument.

        Returns:
            The response r_inst, one column per emission pattern when the
            pattern has a parameter axis.
        """
        n_levels = int(np.shape(self.column_z)[0])
        footprints = self.footprints(instrument)
        # F̄ = mean_e F_e
        mean_footprint = einx.mean("member point cell -> point cell", footprints)
        pattern = jnp.asarray(self.emission_pattern, dtype=mean_footprint.dtype)
        observation = ColumnObservation.trapezoid(
            jnp.asarray(self.column_z, dtype=mean_footprint.dtype),
            averaging_kernel=instrument.ak,
        )
        if pattern.ndim == 1:
            # F̄ q
            mixing_ratio = einx.dot(
                "point cell, cell -> point", mean_footprint, pattern
            )
            # c = ρ_air F̄ q
            concentration = self.air_density * mixing_ratio
            profiles = einx.rearrange(
                "(obs level) -> obs level", concentration, level=n_levels
            )
            return observation.column(profiles)
        # F̄ q_p, one per pattern
        mixing_ratio = einx.dot(
            "point cell, cell param -> point param", mean_footprint, pattern
        )
        # c_p = ρ_air F̄ q_p
        concentration = self.air_density * mixing_ratio
        profiles = einx.rearrange(
            "(obs level) param -> param obs level", concentration, level=n_levels
        )
        columns = jax.vmap(observation.column)(profiles)
        return einx.rearrange("param obs -> obs param", columns)


@dataclasses.dataclass(frozen=True)
class EulerianSource:
    """Tier III transport: the ``les_fvm`` finite-volume run through the shared column.

    r_i = A_i Σ_k Δz c_k(x_i, t_inst; Q = 1),

    the field of a unit constant emission at the overpass time t_inst, read in
    the T-cell that holds each receptor. The forward is linear in the emission
    with a zero initial field, so ``Q · r`` is the run at constant emission Q.

    Attributes:
        forward: The differentiable transport (``les_fvm.build_forward``);
            its save times must include each instrument's overpass time.
        background: Additive background column c_bg.
    """

    forward: EulerianForward4DVar
    background: float = 0.0

    @functools.cached_property
    def unit_history(self) -> Float[Array, "time level y x"]:
        """Interior field history of a unit constant emission, one solve."""
        n_times = self.forward.save_times.shape[0]
        unit = jnp.ones(n_times, dtype=self.forward.save_times.dtype)
        return self.forward.concentration_history(unit)

    @property
    def plume_grid(self) -> PlumeGrid3D:
        """The analysis grid the transport runs on."""
        n_times = self.forward.save_times.shape[0]
        unit = jnp.ones(n_times, dtype=self.forward.save_times.dtype)
        return self.forward.rhs_factory(unit).plume_grid

    def time_index(self, instrument: Instrument) -> int:
        """Index of the save time that equals the instrument's overpass time.

        Raises:
            GridMismatchError: If no save time matches it.
        """
        save_times = np.asarray(self.forward.save_times, dtype=float)
        matches = np.flatnonzero(np.isclose(save_times, instrument.observation_time))
        if matches.size == 0:
            raise GridMismatchError(
                f"EulerianSource: overpass time {instrument.observation_time} s of "
                f"{instrument.name!r} is not one of the forward's save times; "
                "build the forward with it in `save_times`."
            )
        return int(matches[0])

    def column_response(self, instrument: Instrument) -> Float[Array, " obs"]:
        """Column per unit emission at the instrument's receptors.

        Args:
            instrument: The observing instrument.

        Returns:
            The response r_inst.

        Raises:
            GridMismatchError: If the overpass time is not a save time.
            DomainError: If a receptor lies outside the grid.
        """
        index = self.time_index(instrument)
        observation = ColumnObservation.for_grid(self.plume_grid, instrument=instrument)
        return observation.from_field(self.unit_history[index])


__all__ = ["EulerianSource", "LagrangianSource", "TransportBackend"]
