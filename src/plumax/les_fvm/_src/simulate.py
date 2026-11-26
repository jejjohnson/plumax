"""High-level diffrax runner returning an xarray ``Dataset``.

This module wires every other ``les_fvm`` piece together behind a single
user-facing entry point, :func:`simulate_eulerian_dispersion`, modelled
after ``gauss_puff.simulate_puff`` for cross-model consistency.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import xarray as xr
from integrix.march import stable_step
from jaxtyping import Array, Float

from plumax._src.dataset import label_concentrations
from plumax._src.tracing import require_positive
from plumax.gauss_puff.wind import WindSchedule
from plumax.les_fvm._src.boundary import AxisBCSpec
from plumax.les_fvm.boundary import build_default_concentration_bc
from plumax.les_fvm.diffusion import EddyDiffusivity, make_eddy_diffusivity
from plumax.les_fvm.dynamics import EulerianDispersionRHS
from plumax.les_fvm.grid import PlumeGrid3D, coord_to_numpy, make_grid
from plumax.les_fvm.source import make_gaussian_source
from plumax.les_fvm.wind import (
    PrescribedWindField,
    uniform_wind_field,
    wind_field_from_schedule,
)


SolverName = Literal["tsit5", "dopri5", "ssprk3", "heun"]
AdvectionScheme = Literal[
    "upwind1",
    "weno3",
    "weno5",
    "weno7",
    "weno9",
    "minmod",
    "van_leer",
    "superbee",
    "mc",
    "naive",
]


def simulate_eulerian_dispersion(
    *,
    # Domain and resolution
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    # Time
    t_start: float,
    t_end: float,
    save_interval: float,
    # Source
    emission_rate: float | Callable[[Float[Array, ""]], Float[Array, ""]],
    source_location: tuple[float, float, float],
    source_radius: float | None = None,
    # Flow
    wind_schedule: WindSchedule | None = None,
    uniform_wind: tuple[float, float, float] | None = None,
    wind_field: PrescribedWindField | None = None,
    # SGS / eddy diffusivity
    eddy_diffusivity: (float | tuple[float, float] | str | EddyDiffusivity) = "pg",
    stability_class: str | None = None,
    pg_reference_distance: float = 200.0,
    # Numerics
    advection_scheme: AdvectionScheme = "weno5",
    bc_x: AxisBCSpec = (
        "dirichlet",
        "outflow",
    ),
    bc_y: AxisBCSpec = "periodic",
    bc_z: AxisBCSpec = (
        "neumann",
        "neumann",
    ),
    solver: SolverName = "tsit5",
    dt0: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-8,
    max_steps: int = 100_000,
    max_wind_speed: float | None = None,
    # Initial condition
    initial_concentration: Float[Array, "Nz Ny Nx"] | None = None,
    seed: int = 0,
) -> xr.Dataset:
    """Simulate 3-D Eulerian tracer transport under a prescribed wind.

    Parameters
    ----------
    domain_x, domain_y, domain_z : tuple (min, max, n_interior)
        Physical extent + number of interior cells per axis.
    t_start, t_end : float
        Simulation window [s].
    save_interval : float
        Save cadence [s].  Saved times are
        ``np.arange(t_start, t_end + save_interval, save_interval)``.
    emission_rate : float or callable
        Methane source rate [kg/s] (constant or ``t -> q(t)``).
    source_location : tuple (x, y, z)
        Release point [m].
    source_radius : float, optional
        Gaussian source radius [m].  Defaults to ``2 · max(dx, dy, dz)``.
    wind_schedule : WindSchedule, optional
    uniform_wind : (u, v, w), optional
    wind_field : PrescribedWindField, optional
        One of these three must be provided — they are mutually exclusive.
    eddy_diffusivity : float | (K_h, K_z) | "pg" | EddyDiffusivity
        K-theory eddy diffusivity spec.  The string ``"pg"`` pulls a rough
        calibration from the PG σ curves — requires ``stability_class`` and
        a non-zero mean wind speed.
    stability_class : str, optional
        Required when ``eddy_diffusivity="pg"``.
    pg_reference_distance : float, default 200.0
        Reference downwind distance for PG → K-theory conversion.
    advection_scheme : {"upwind1", "weno3", "weno5", ...}, default "weno5"
        Horizontal reconstruction scheme.
    bc_x, bc_y, bc_z : BC specs
        See :func:`boundary.build_default_concentration_bc`.
    solver : {"tsit5", "dopri5", "ssprk3", "heun"}, default "tsit5"
        diffrax time integrator.  ``"ssprk3"`` and ``"heun"`` are
        explicit-fixed-step and ignore ``rtol``/``atol``.  ``"tsit5"`` and
        ``"dopri5"`` are adaptive, with a step of at most
        :func:`stable_step_bound`.
    dt0 : float
        Initial / fixed step size [s].
    rtol, atol : float
        Adaptive-step controller tolerances (ignored for fixed-step solvers).
    max_steps : int
        Upper bound on diffrax internal steps.
    max_wind_speed : float, optional
        Conservative upper bound on each wind component ``|u|, |v|, |w|``
        [m/s], used only by the fixed-step stability guard. Supply it for a
        callable wind whose peak the guard cannot otherwise sample reliably;
        when ``None`` the peak is estimated from the field (dense time grid +
        any ``WindSchedule`` knots in the window).
    initial_concentration : ndarray, optional
        Interior-shaped ``(nz, ny, nx)`` initial tracer field.  Defaults
        to zero.
    seed : int
        Currently unused (reserved for future stochastic variants).

    Returns
    -------
    xr.Dataset
        ``concentration(time, z, y, x)`` [kg/m³] and the diagnostic
        ``column_concentration(time, y, x)`` [kg/m²].  The dataset
        carries model metadata in ``ds.attrs``.

    Raises
    ------
    ValueError
        For inconsistent wind/source/emission specs.
    """
    del seed  # Reserved for future stochastic branch.

    plume_grid = make_grid(domain_x, domain_y, domain_z)

    wind = _resolve_wind_field(
        plume_grid=plume_grid,
        wind_schedule=wind_schedule,
        uniform_wind=uniform_wind,
        wind_field=wind_field,
    )

    source = make_gaussian_source(
        plume_grid=plume_grid,
        emission_rate=emission_rate,
        source_location=source_location,
        source_radius=source_radius,
    )

    eddy = _resolve_eddy_diffusivity(
        eddy_diffusivity=eddy_diffusivity,
        stability_class=stability_class,
        wind_schedule=wind_schedule,
        uniform_wind=uniform_wind,
        pg_reference_distance=pg_reference_distance,
        t_start=t_start,
        end_time=t_end,
    )
    # Validate inputs independently of the solver: a negative / non-finite K is
    # unstable (or guard-defeating) on the adaptive path too, and a malformed
    # `max_wind_speed` would silently disable the fixed-step guard.
    _max_diffusivity(eddy)
    if max_wind_speed is not None and (
        not np.isfinite(max_wind_speed) or max_wind_speed < 0.0
    ):
        raise ValueError(
            f"max_wind_speed must be finite and non-negative (got {max_wind_speed!r})."
        )

    horizontal_bc, vertical_bc = build_default_concentration_bc(
        bc_x=bc_x, bc_y=bc_y, bc_z=bc_z
    )

    rhs = EulerianDispersionRHS(
        plume_grid=plume_grid,
        wind_field=wind,
        eddy_diffusivity=eddy,
        source=source,
        horizontal_bc=horizontal_bc,
        vertical_bc=vertical_bc,
        advection_scheme=advection_scheme,
    )

    concentration0 = _build_initial_concentration(
        plume_grid=plume_grid, initial_concentration=initial_concentration
    )
    save_times = _build_save_times(
        t_start=t_start, end_time=t_end, save_interval=save_interval
    )
    # In-window WindSchedule knots — a piecewise-linear schedule attains its
    # speed extrema at knots, so feed them to the fixed-step stability guard so
    # a fast knot between the dense samples is not missed.
    knot_times: tuple[float, ...] = ()
    if wind_schedule is not None:
        knots = np.asarray(wind_schedule.times, dtype=float)
        knot_times = tuple(float(knot) for knot in knots if t_start <= knot <= t_end)
    solution = _solve(
        rhs=rhs,
        t_start=t_start,
        end_time=t_end,
        initial_time_step=dt0,
        initial_state=concentration0,
        save_times=save_times,
        solver_name=solver,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        extra_sample_times=knot_times,
        max_wind_speed=max_wind_speed,
    )
    return _to_dataset(
        plume_grid=plume_grid,
        times=np.asarray(solution.ts),
        concentration_history=np.asarray(solution.ys),
        attrs={
            "title": "Eulerian 3-D methane dispersion (gridax)",
            "model": "L2: prescribed wind + K-theory diffusivity",
            "advection_scheme": advection_scheme,
            "solver": solver,
        },
    )


def _resolve_wind_field(
    *,
    plume_grid: PlumeGrid3D,
    wind_schedule: WindSchedule | None,
    uniform_wind: tuple[float, float, float] | None,
    wind_field: PrescribedWindField | None,
) -> PrescribedWindField:
    provided = sum(arg is not None for arg in (wind_schedule, uniform_wind, wind_field))
    if provided != 1:
        raise ValueError(
            "simulate_eulerian_dispersion: exactly one of "
            "`wind_schedule`, `uniform_wind`, `wind_field` must be provided"
        )
    if wind_field is not None:
        return wind_field
    if wind_schedule is not None:
        return wind_field_from_schedule(plume_grid=plume_grid, schedule=wind_schedule)
    u_mean, v_mean, w_mean = uniform_wind
    return uniform_wind_field(plume_grid=plume_grid, u=u_mean, v=v_mean, w=w_mean)


def _resolve_eddy_diffusivity(
    *,
    eddy_diffusivity,
    stability_class: str | None,
    wind_schedule: WindSchedule | None,
    uniform_wind: tuple[float, float, float] | None,
    pg_reference_distance: float,
    t_start: float,
    end_time: float,
) -> EddyDiffusivity:
    wind_speed = None
    if uniform_wind is not None:
        u_mean, v_mean, _ = uniform_wind
        wind_speed = float(np.hypot(u_mean, v_mean))
    elif wind_schedule is not None:
        # Time-mean speed over the *simulated* window [t_start, t_end]. We
        # sample the piecewise-linear schedule at densely-spaced points and
        # take the mean of |V(t)| — this respects the simulation window
        # and knot spacing, so adding knots outside the run interval or
        # changing knot density does not move the calibration.
        wind_speed = _mean_schedule_speed(wind_schedule, t_start, end_time)
    if (
        isinstance(eddy_diffusivity, str)
        and eddy_diffusivity.lower() == "pg"
        and wind_speed is not None
        and wind_speed == 0.0
    ):
        raise ValueError(
            "simulate_eulerian_dispersion: eddy_diffusivity='pg' requires a "
            "non-zero mean wind speed to compute the K-theory calibration"
        )
    return make_eddy_diffusivity(
        eddy_diffusivity,
        stability_class=stability_class,
        wind_speed=wind_speed,
        reference_distance=pg_reference_distance,
    )


def _mean_schedule_speed(
    schedule: WindSchedule, t_start: float, end_time: float
) -> float:
    """Time-mean wind speed of a piecewise-linear schedule over a window.

    Integrates ``|V(t)|`` with np.interp on a dense uniform sample grid
    and returns the window mean.  100 samples is enough — the result is
    only used to calibrate a scalar K-theory diffusivity.
    """
    if end_time <= t_start:
        # Degenerate window — fall back to the instantaneous speed at t_start.
        east_at_start = float(
            np.interp(t_start, np.asarray(schedule.times), np.asarray(schedule.u_wind))
        )
        north_at_start = float(
            np.interp(t_start, np.asarray(schedule.times), np.asarray(schedule.v_wind))
        )
        # |V| = √(u² + v²)
        return float(np.hypot(east_at_start, north_at_start))
    t_sample = np.linspace(t_start, end_time, 100)
    u_sample = np.interp(
        t_sample, np.asarray(schedule.times), np.asarray(schedule.u_wind)
    )
    v_sample = np.interp(
        t_sample, np.asarray(schedule.times), np.asarray(schedule.v_wind)
    )
    # ⟨|V|⟩ = mean √(u² + v²)
    return float(np.hypot(u_sample, v_sample).mean())


def _build_save_times(
    *, t_start: float, end_time: float, save_interval: float
) -> Float[Array, " M"]:
    """Return save-time knots clipped to ``[t_start, end_time]``.

    ``jnp.arange(t_start, end_time + 0.5 * save_interval, save_interval)`` can
    step past ``end_time`` when the interval does not evenly divide the run
    window (e.g. ``0..10`` with ``save_interval = 6`` gives ``[0, 6, 12]``).
    diffrax's ``SaveAt(ts=...)`` rejects save points outside the integration
    interval, so we clip explicitly and dedupe the trailing value.
    """
    require_positive("save_interval", save_interval)
    if end_time < t_start:
        raise ValueError(
            f"end_time must be >= t_start (got t_start={t_start}, end_time={end_time})"
        )
    raw = np.arange(t_start, end_time + 0.5 * save_interval, save_interval)
    raw = raw[raw <= end_time + 1e-9]  # guard against fp round-up
    # Snap the last knot to exactly end_time when it falls short, so the final
    # save point matches the user-requested simulation horizon.
    if raw.size > 0 and raw[-1] < end_time - 1e-9:
        raw = np.concatenate([raw, np.asarray([end_time])])
    return jnp.asarray(raw, dtype=jnp.float32)


def _build_initial_concentration(
    *,
    plume_grid: PlumeGrid3D,
    initial_concentration: Float[Array, "nz ny nx"] | None,
) -> Float[Array, "Nz Ny Nx"]:
    num_levels, num_rows, num_cols = plume_grid.shape
    if initial_concentration is None:
        return jnp.zeros((num_levels, num_rows, num_cols), dtype=plume_grid.x.dtype)
    interior = jnp.asarray(initial_concentration, dtype=plume_grid.x.dtype)
    expected = plume_grid.interior_shape
    if interior.shape != expected:
        raise ValueError(
            f"simulate_eulerian_dispersion: initial_concentration shape "
            f"{interior.shape!r} does not match interior shape {expected!r}"
        )
    return jnp.pad(interior, ((1, 1), (1, 1), (1, 1)), mode="constant")


def stable_step_bound(
    *,
    dx: float,
    dy: float,
    dz: float,
    max_u: float,
    max_v: float,
    max_w: float,
    k_h: float,
    k_z: float,
    cfl_safety: float = 1.0,
) -> float:
    """Maximum stable step for an explicit fixed-step transport solve.

    Advection and diffusion are advanced in the *same* explicit step, so
    their tendencies add and the conservative forward-Euler / Heun limit is
    set by the summed rate::

        adv_rate  = |u|/dx + |v|/dy + |w|/dz
        diff_rate = 2·(K_h/dx² + K_h/dy² + K_z/dz²)
        bound     = cfl_safety / (adv_rate + diff_rate)

    Using ``min(1/adv_rate, 1/diff_rate)`` instead would permit up to a 2×
    step when the two rates are comparable. The bound is ``inf`` only when
    the field is both motionless and diffusion-free.

    Parameters
    ----------
    dx, dy, dz : float
        Grid spacing [m].
    max_u, max_v, max_w : float
        Component-wise maximum wind speed |u|, |v|, |w| over the run [m/s];
        the sign is ignored.
    k_h, k_z : float
        Maximum horizontal / vertical eddy diffusivity [m²/s].
    cfl_safety : float, default 1.0
        Multiplier on the raw stability bound.  ``1.0`` is the forward-Euler
        sum-CFL / FTCS-diffusion limit; the RK schemes used here are at
        least as stable, so this is a conservative guard.

    Returns
    -------
    float
        The maximum stable ``dt0`` [s].

    Raises
    ------
    integrix.exceptions.ValidationError
        A ``ValueError``: a spacing or ``cfl_safety`` is not finite and
        positive, a speed or diffusivity is not finite, or a diffusivity is
        negative.

    Notes
    -----
    This is :func:`integrix.march.stable_step` on the three axes: a
    first-order (upwind / forward-Euler) sum-CFL estimate — a
    conservative safety net that catches gross step-size errors (the
    ``dt0``-orders-of-magnitude-too-large failure mode). It is **not** a
    rigorous stability certificate for the higher-order or low-dissipation
    reconstructions (e.g. the default ``weno5``, or ``naive``): those
    semidiscrete operators have different stability regions, and an explicit
    RK step can amplify their weakly-damped modes even below this bound. Keep
    ``dt0`` comfortably under the reported value when using such schemes.
    """
    # h_max = CFL / (Σᵢ |uᵢ|/Δxᵢ + 2 Σᵢ Kᵢ/Δxᵢ²), with K_h on both horizontal axes
    bound = stable_step(
        (dx, dy, dz),
        speed=(max_u, max_v, max_w),
        diffusivity=(k_h, k_h, k_z),
        cfl=cfl_safety,
    )
    return float(bound)


def _max_wind_components(
    wind: PrescribedWindField, sample_times: tuple[float, ...]
) -> tuple[float, float, float]:
    """Component-wise max |u|, |v|, |w| over the interior at ``sample_times``.

    Raises on a non-finite sample: ``max(prev, nan)`` keeps ``prev`` (a NaN wind
    would read as calm and let the guard approve a step the RHS then corrupts),
    so reject it explicitly — the same treatment as the diffusivity check.
    """
    max_u = max_v = max_w = 0.0
    for time in sample_times:
        wind_east, wind_north, wind_up = wind(jnp.asarray(time, dtype=jnp.float32))
        for name, comp in (("u", wind_east), ("v", wind_north), ("w", wind_up)):
            if not bool(jnp.all(jnp.isfinite(comp))):
                raise ValueError(
                    f"wind field has a non-finite {name} component at "
                    f"t={float(time):g}; the CFL guard cannot bound a NaN/inf wind."
                )
        max_u = max(max_u, float(jnp.max(jnp.abs(wind_east))))
        max_v = max(max_v, float(jnp.max(jnp.abs(wind_north))))
        max_w = max(max_w, float(jnp.max(jnp.abs(wind_up))))
    return max_u, max_v, max_w


def _max_diffusivity(eddy: EddyDiffusivity) -> tuple[float, float]:
    """Return ``(max K_h, max K_z)``, rejecting negative or non-finite ``K``.

    A negative ``K`` is anti-diffusion: its modes grow every explicit step, so
    no positive ``dt0`` is stable, and taking ``|K|`` would let the guard
    approve a step the real (negative-``K``) tendency then blows up on. A
    NaN/inf ``K`` makes every ``dt0 > bound`` comparison false, silently
    disabling the guard while the tendency corrupts the solve. Reject both.
    """
    horizontal_diffusivity = jnp.asarray(eddy.as_arrays()[0])
    vertical_diffusivity = jnp.asarray(eddy.as_arrays()[1])
    if not bool(
        jnp.all(jnp.isfinite(horizontal_diffusivity))
        & jnp.all(jnp.isfinite(vertical_diffusivity))
    ):
        raise ValueError("eddy diffusivity must be finite; got a NaN/inf K.")
    min_diffusivity = min(
        float(jnp.min(horizontal_diffusivity)), float(jnp.min(vertical_diffusivity))
    )
    if min_diffusivity < 0.0:
        raise ValueError(
            "eddy diffusivity must be non-negative; a negative K is "
            f"anti-diffusion and is unconditionally unstable (min K = {min_diffusivity:g})."
        )
    return float(jnp.max(horizontal_diffusivity)), float(jnp.max(vertical_diffusivity))


def _stable_step(
    *,
    plume_grid: PlumeGrid3D,
    wind: PrescribedWindField,
    eddy: EddyDiffusivity,
    t_start: float,
    end_time: float,
    extra_sample_times: tuple[float, ...] = (),
    max_wind_speed: float | None = None,
) -> float:
    """Return :func:`stable_step_bound` for a run, infinite for still air and K = 0.

    The field is always sampled on a dense time grid over ``[t_start, end_time]``
    unioned with ``extra_sample_times`` (e.g. the in-window ``WindSchedule``
    knots, where a piecewise-linear schedule attains its extrema) — both to
    estimate the peak speed that drives the advective CFL term and to reject a
    non-finite wind. When ``max_wind_speed`` is given it overrides the sampled
    maxima as a conservative per-component bound — the robust choice for a
    callable whose peak sits between samples — while the sampling still runs so
    the finiteness check is never bypassed.
    """
    base = tuple(float(time) for time in np.linspace(t_start, end_time, 33))
    # Union (dedup) the dense grid with the supplied extra times.
    sample_times = tuple(
        dict.fromkeys(base + tuple(float(time) for time in extra_sample_times))
    )
    # Always sample: this validates finiteness (raises on a NaN/inf wind) even
    # when an override supplies the maxima.
    sampled_u, sampled_v, sampled_w = _max_wind_components(wind, sample_times)
    if max_wind_speed is not None:
        peak = float(max_wind_speed)
        max_u = max_v = max_w = peak
    else:
        max_u, max_v, max_w = sampled_u, sampled_v, sampled_w
    horizontal_diffusivity, vertical_diffusivity = _max_diffusivity(eddy)
    return stable_step_bound(
        dx=plume_grid.dx,
        dy=plume_grid.dy,
        dz=plume_grid.dz,
        max_u=max_u,
        max_v=max_v,
        max_w=max_w,
        k_h=horizontal_diffusivity,
        k_z=vertical_diffusivity,
    )


def _check_fixed_step_stability(
    initial_time_step: float, bound: float, solver_name: str
) -> None:
    """Raise if ``initial_time_step`` exceeds the stable step ``bound``."""
    if initial_time_step > bound:
        raise ValueError(
            f"simulate_eulerian_dispersion: dt0={initial_time_step:g} s exceeds the stable "
            f"step ~{bound:g} s for fixed-step solver {solver_name!r} "
            f"(advective CFL / explicit-diffusion limit on this grid+wind+K). "
            f"Reduce dt0, soften K / coarsen the grid, or use an adaptive "
            f"solver ('tsit5'/'dopri5')."
        )


def _pick_solver(name: SolverName):
    if name == "tsit5":
        return diffrax.Tsit5(), True
    if name == "dopri5":
        return diffrax.Dopri5(), True
    if name == "ssprk3":
        from integrix.march import RK3SSP

        return RK3SSP(), False
    if name == "heun":
        return diffrax.Heun(), False
    raise ValueError(f"simulate_eulerian_dispersion: unknown solver {name!r}")


@eqx.filter_jit
def _solve_jit(
    rhs: EulerianDispersionRHS,
    t_start: float,
    end_time: float,
    initial_time_step: float,
    initial_state: Float[Array, "Nz Ny Nx"],
    save_times: Float[Array, " M"],
    solver,
    stepsize_controller,
    max_steps: int,
):
    term = diffrax.ODETerm(rhs)
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=t_start,
        t1=end_time,
        dt0=initial_time_step,
        y0=initial_state,
        saveat=diffrax.SaveAt(ts=save_times),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )


def _solve(
    *,
    rhs: EulerianDispersionRHS,
    t_start: float,
    end_time: float,
    initial_time_step: float,
    initial_state: Float[Array, "Nz Ny Nx"],
    save_times: Float[Array, " M"],
    solver_name: SolverName,
    rtol: float,
    atol: float,
    max_steps: int,
    extra_sample_times: tuple[float, ...] = (),
    max_wind_speed: float | None = None,
):
    solver, adaptive = _pick_solver(solver_name)
    bound = _stable_step(
        plume_grid=rhs.plume_grid,
        wind=rhs.wind_field,
        eddy=rhs.eddy_diffusivity,
        t_start=t_start,
        end_time=end_time,
        extra_sample_times=extra_sample_times,
        max_wind_speed=max_wind_speed,
    )
    if adaptive:
        # The error control alone does not keep the step stable: ``atol`` is in
        # kg/m³, and a plume below it has no error to control. Cap the step.
        initial_time_step = min(initial_time_step, bound)
        stepsize_controller = diffrax.PIDController(
            rtol=rtol, atol=atol, dtmax=bound if math.isfinite(bound) else None
        )
    else:
        # Fixed-step solvers integrate at exactly the initial step with no error control,
        # so guard the CFL / explicit-diffusion stability limit up front —
        # otherwise an over-large step silently blows up mid-solve.
        _check_fixed_step_stability(initial_time_step, bound, solver_name)
        stepsize_controller = diffrax.ConstantStepSize()
    return _solve_jit(
        rhs=rhs,
        t_start=float(t_start),
        end_time=float(end_time),
        initial_time_step=float(initial_time_step),
        initial_state=initial_state,
        save_times=save_times,
        solver=solver,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )


def _to_dataset(
    *,
    plume_grid: PlumeGrid3D,
    times: np.ndarray,
    concentration_history: np.ndarray,
    attrs: dict,
) -> xr.Dataset:
    """Package solver output as an xarray ``Dataset`` on the interior grid."""
    # concentration_history has shape (n_saves, Nz, Ny, Nx) — drop ghost rings.
    interior = concentration_history[:, 1:-1, 1:-1, 1:-1]
    x_coords, y_coords, z_coords = coord_to_numpy(plume_grid)
    dataset = xr.Dataset(
        {"concentration": (["time", "z", "y", "x"], interior)},
        coords={
            "time": (["time"], times),
            "z": (["z"], z_coords),
            "y": (["y"], y_coords),
            "x": (["x"], x_coords),
        },
        attrs=attrs,
    )
    dataset["column_concentration"] = (
        dataset["concentration"].sum(dim="z") * plume_grid.dz
    )
    label_concentrations(dataset)
    return dataset
