"""pipekit ``Operator`` wrappers over the plumax forward models.

Each forward model in plumax is a pure function with a long, model-specific
signature. These wrappers split that signature the pipekit way: the *static*
grid / domain / scheme configuration is captured at construction time, and the
*dynamic* source / wind parameters flow through ``_apply`` as a single mapping
carrier. The result is an :class:`pipekit.Operator` that composes with ``|``,
supports the config round-trip via :class:`pipekit.ConfigMixin`, and can be
dropped into a :class:`pipekit.Sequential` pipeline.

Example:
    >>> from plumax.operators import GaussianPlume
    >>> plume = GaussianPlume(
    ...     stability_class="D",
    ...     domain_x=(-100.0, 1000.0, 56),
    ...     domain_y=(-300.0, 300.0, 31),
    ...     domain_z=(0.0, 100.0, 11),
    ... )
    >>> ds = plume({
    ...     "emission_rate": 1.0,
    ...     "source_location": (0.0, 0.0, 10.0),
    ...     "wind_speed": 5.0,
    ...     "wind_direction": 270.0,
    ... })
    >>> ds["concentration"].dims
    ('x', 'y', 'z')
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pipekit import Operator

from plumax.gauss_plume import simulate_plume
from plumax.gauss_puff import simulate_puff
from plumax.lagrangian import simulate_lagrangian
from plumax.les_fvm import simulate_eulerian_dispersion


if TYPE_CHECKING:
    from collections.abc import Mapping

    import xarray as xr


class GaussianPlume(Operator):
    """Steady-state Gaussian plume forward model as a pipekit operator.

    Construction captures the static grid and stability configuration. Calling
    the operator with a mapping of dynamic parameters (``emission_rate``,
    ``source_location``, ``wind_speed``, ``wind_direction``) returns the
    simulated :class:`xarray.Dataset`.

    Args:
        stability_class: Pasquill stability class ``'A'``–``'F'``.
        domain_x: ``(start, stop, n_points)`` for the x axis [m].
        domain_y: ``(start, stop, n_points)`` for the y axis [m].
        domain_z: ``(start, stop, n_points)`` for the z axis [m].
        background_conc: Additive background concentration [kg/m³].
    """

    def __init__(
        self,
        stability_class: str,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        domain_z: tuple[float, float, int],
        background_conc: float = 0.0,
    ) -> None:
        self.stability_class = stability_class
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.background_conc = background_conc

    def _apply(self, params: Mapping[str, Any]) -> xr.Dataset:
        return simulate_plume(
            emission_rate=params["emission_rate"],
            source_location=params["source_location"],
            wind_speed=params["wind_speed"],
            wind_direction=params["wind_direction"],
            stability_class=self.stability_class,
            domain_x=self.domain_x,
            domain_y=self.domain_y,
            domain_z=self.domain_z,
            background_conc=self.background_conc,
        )


class GaussianPuff(Operator):
    """Time-resolved Gaussian puff forward model as a pipekit operator.

    Static grid, time axis, and scheme configuration are captured at
    construction; the dynamic parameters (``emission_rate``,
    ``source_location``, ``wind_speed``, ``wind_direction``) flow through
    ``_apply``. ``wind_speed`` / ``wind_direction`` are per-timestep arrays.

    Note: because ``time_array`` (and an optional ``turbulence`` model) are not
    JSON primitives, this operator's :meth:`get_config` is a debug repr rather
    than a faithful YAML round-trip; hence ``forbid_in_yaml = True``.

    Args:
        stability_class: Pasquill stability class ``'A'``–``'F'``.
        domain_x / domain_y / domain_z: ``(start, stop, n_points)`` per axis.
        time_array: Output times [s].
        release_frequency: Puff release frequency [Hz].
        scheme: Dispersion scheme, ``'pg'`` (default) or ``'briggs'``.
        background_conc: Additive background concentration [kg/m³].
        turbulence: Optional :class:`~plumax.gauss_puff.OUTurbulence` model.
        turbulence_seed: Seed / generator for the turbulence draws.
    """

    forbid_in_yaml = True

    def __init__(
        self,
        stability_class: str,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        domain_z: tuple[float, float, int],
        time_array: Any,
        release_frequency: float = 1.0,
        scheme: str = "pg",
        background_conc: float = 0.0,
        turbulence: Any = None,
        turbulence_seed: Any = None,
    ) -> None:
        self.stability_class = stability_class
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.time_array = time_array
        self.release_frequency = release_frequency
        self.scheme = scheme
        self.background_conc = background_conc
        self.turbulence = turbulence
        self.turbulence_seed = turbulence_seed

    def _apply(self, params: Mapping[str, Any]) -> xr.Dataset:
        return simulate_puff(
            emission_rate=params["emission_rate"],
            source_location=params["source_location"],
            wind_speed=params["wind_speed"],
            wind_direction=params["wind_direction"],
            stability_class=self.stability_class,
            domain_x=self.domain_x,
            domain_y=self.domain_y,
            domain_z=self.domain_z,
            time_array=self.time_array,
            release_frequency=self.release_frequency,
            scheme=self.scheme,
            background_conc=self.background_conc,
            turbulence=self.turbulence,
            turbulence_seed=self.turbulence_seed,
        )


class EulerianDispersion(Operator):
    """Eulerian finite-volume dispersion runner as a pipekit operator.

    :func:`~plumax.les_fvm.simulate_eulerian_dispersion` is keyword-only with a
    large static configuration (domain, time window, flow, eddy diffusivity,
    boundary conditions). That configuration is captured at construction and
    merged with the per-call ``params`` mapping (typically ``emission_rate``
    and ``source_location``) inside ``_apply``.

    Because the stored configuration may hold non-serialisable objects
    (``WindSchedule``, boundary-condition atoms), :meth:`get_config` reports the
    configured keys only and ``forbid_in_yaml = True``.

    Args:
        **config: Keyword arguments forwarded verbatim to
            :func:`~plumax.les_fvm.simulate_eulerian_dispersion`.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml = True

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    def _apply(self, params: Mapping[str, Any]) -> xr.Dataset:
        return simulate_eulerian_dispersion(**{**self.config, **dict(params)})

    def get_config(self) -> dict[str, Any]:
        return {"config_keys": sorted(self.config)}


class LagrangianDispersion(Operator):
    """Markov-1 Lagrangian dispersion forward model as a pipekit operator.

    Static grid, turbulence and integration configuration are captured at
    construction; the dynamic parameters (``emission_rate``, ``source_location``
    and either ``wind`` or ``wind_speed`` / ``wind_direction``) flow through
    ``_apply``. The turbulence model is a non-primitive object, so
    ``forbid_in_yaml = True`` and :meth:`get_config` is a debug repr.

    Args:
        turbulence: A turbulence model (e.g.
            :class:`~plumax.lagrangian.HomogeneousTurbulence`).
        domain_x / domain_y / domain_z: ``(start, stop, n_cells)`` per axis.
        n_particles: Ensemble size.
        t_end: Integration horizon [s].
        dt: Time step [s].
        pbl_height: Optional reflecting PBL lid [m].
        background_conc: Additive background concentration [kg/m³].
        seed: PRNG seed.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml = True

    def __init__(
        self,
        turbulence: Any,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        domain_z: tuple[float, float, int],
        n_particles: int = 5000,
        t_end: float = 600.0,
        dt: float = 1.0,
        pbl_height: float | None = None,
        background_conc: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.turbulence = turbulence
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.domain_z = domain_z
        self.n_particles = n_particles
        self.t_end = t_end
        self.dt = dt
        self.pbl_height = pbl_height
        self.background_conc = background_conc
        self.seed = seed

    def _apply(self, params: Mapping[str, Any]) -> xr.Dataset:
        return simulate_lagrangian(
            emission_rate=params["emission_rate"],
            source_location=params["source_location"],
            turbulence=self.turbulence,
            domain_x=self.domain_x,
            domain_y=self.domain_y,
            domain_z=self.domain_z,
            n_particles=self.n_particles,
            t_end=self.t_end,
            dt=self.dt,
            wind=params.get("wind"),
            wind_speed=params.get("wind_speed"),
            wind_direction=params.get("wind_direction"),
            pbl_height=self.pbl_height,
            background_conc=self.background_conc,
            seed=self.seed,
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "stability_class": None,
            "domain_x": self.domain_x,
            "domain_y": self.domain_y,
            "domain_z": self.domain_z,
            "n_particles": self.n_particles,
            "t_end": self.t_end,
            "dt": self.dt,
        }


__all__ = [
    "EulerianDispersion",
    "GaussianPlume",
    "GaussianPuff",
    "LagrangianDispersion",
]
