"""Tier III — strong-constraint 4D-Var inversion over the Eulerian transport model.

Recovers a time-resolved emission signal from a sequence of column observations
by directly minimising the strong-constraint 4D-Var cost
([design](https://github.com/jejjohnson/plumax/blob/main/docs/design/03_tier3_eulerian.md#tier3-cost)):

    J(S) = ½‖S − S_b‖²_B  +  ½ Σ_t ‖H_t c(S, t) − y_t‖²_{R_t}

where ``c(S, t)`` is the 3-D tracer field produced by the finite-volume solver
:func:`~plumax.les_fvm.simulate_eulerian_dispersion` for emission signal ``S``,
``H_t`` is the column observation operator (vertical integral → optional spatial
sampling), ``S_b`` / ``B`` the source prior, and ``R_t`` the observation-error
covariance at overpass ``t``. (The IC / background term ``½‖c_0 − c_b(0)‖²`` of
the full three-term design cost is future work — this v1 pins the initial field
and inverts only the source.)

The solver here is the **direct strong-constraint** path: a single L-BFGS
minimisation of the full nonlinear cost in whitened control space. The key
enabler (design §adjoint) is that **JAX reverse-mode autodiff through the diffrax
FV solve is the exact discrete adjoint** — there is no hand-written adjoint
model; ``jax.value_and_grad`` of the cost supplies the gradient the optimiser
consumes. The incremental / Gauss-Newton inner-loop variant (design
§incremental) is a separate future addition.

The control vector here is the **time-resolved scalar emission rate** ``S = q``
at a known source location — the v1 scope. A per-cell space–time source field is
a natural extension (the forward already accepts an arbitrary
``emission_fn(t)``); the cost / control-transform machinery below is written
against a flat control vector so that extension does not change the solver.

Control-variable transform (design §control-transform): optimisation runs in
**whitened space** ``χ = L⁻¹ (S − S_b)`` where ``B = L Lᵀ``. The prior term is
then the identity ``½‖χ‖²``, which is what makes the problem well-conditioned;
the Matérn-3/2 temporal prior from :mod:`plumax.lagrangian.inversion` supplies
``B`` and its Cholesky factor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from plumax.lagrangian.inversion import _is_traced
from plumax.les_fvm.boundary import build_default_concentration_bc
from plumax.les_fvm.diffusion import make_eddy_diffusivity
from plumax.les_fvm.dynamics import EulerianDispersionRHS
from plumax.les_fvm.grid import make_grid
from plumax.les_fvm.source import GaussianSource, make_gaussian_source
from plumax.les_fvm.wind import uniform_wind_field


if TYPE_CHECKING:
    import optimistix as optx


# ── differentiable forward: emission signal → column-observation time series ──


@dataclass(frozen=True)
class ColumnObservationOperator:
    """Sample the column-integrated field at fixed receptor cells.

    ``H_t`` in the 4D-Var cost: vertically integrate the 3-D concentration to a
    column (``∑_z c · dz``), then gather the values at a set of receptor pixels.
    With ``receptor_index = None`` every interior column is an observation.

    Attributes:
        dz: Vertical cell size [m] for the column integral.
        receptor_index: Optional ``(n_obs, 2)`` int array of ``(iy, ix)``
            interior-cell indices to sample. ``None`` flattens the full
            ``(ny, nx)`` column grid.
    """

    dz: float
    receptor_index: jax.Array | None = None

    def __call__(self, concentration: jax.Array) -> jax.Array:
        """Map an interior field ``(nz, ny, nx)`` to observations ``(n_obs,)``."""
        column = jnp.sum(concentration, axis=0) * self.dz  # (ny, nx)
        if self.receptor_index is None:
            return column.reshape(-1)
        iy = self.receptor_index[:, 0]
        ix = self.receptor_index[:, 1]
        return column[iy, ix]


@dataclass(frozen=True)
class EulerianForward4DVar:
    """Differentiable map from an emission time series to column observations.

    Bundles the fixed transport configuration (grid, wind, diffusivity, BCs,
    source geometry, save times) so that :meth:`predict` is a pure function of
    the control vector ``q`` — the time-resolved emission rate at the save
    times — and is therefore ``jax.grad`` / ``jax.linearize``-able for 4D-Var.

    Build with :func:`build_forward` rather than directly.

    Attributes:
        rhs_factory: ``q_series -> EulerianDispersionRHS`` closing over the
            static transport configuration; the emission rate is the only
            free input (linearly interpolated between save times).
        observation_op: The column ``H_t``.
        save_times: Observation/integration times [s], shape ``(n_t,)``.
        initial_concentration: Padded initial field ``(Nz, Ny, Nx)``.
        solver_kwargs: diffrax solve settings (solver, dt0, tolerances).
    """

    rhs_factory: Callable[[jax.Array], EulerianDispersionRHS]
    observation_op: ColumnObservationOperator
    save_times: jax.Array
    initial_concentration: jax.Array
    t0: float
    t1: float
    solver_kwargs: dict = field(default_factory=dict)

    def concentration_history(self, q_series: jax.Array) -> jax.Array:
        """Solve the FV transport for ``q_series``; return interior fields.

        Returns the ``(n_t, nz, ny, nx)`` interior concentration history (ghost
        rings stripped). Differentiable in ``q_series`` via the diffrax solve.
        """
        import diffrax

        rhs = self.rhs_factory(jnp.asarray(q_series))
        kw = {
            "solver": diffrax.Heun(),
            "dt0": 1.0,
            "stepsize_controller": diffrax.ConstantStepSize(),
            "max_steps": 100_000,
            **self.solver_kwargs,
        }
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            kw["solver"],
            t0=self.t0,
            t1=self.t1,
            dt0=kw["dt0"],
            y0=self.initial_concentration,
            saveat=diffrax.SaveAt(ts=self.save_times),
            stepsize_controller=kw["stepsize_controller"],
            max_steps=kw["max_steps"],
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )
        return sol.ys[:, 1:-1, 1:-1, 1:-1]

    def predict(self, q_series: jax.Array) -> jax.Array:
        """Column observations stacked over time, shape ``(n_t, n_obs)``."""
        history = self.concentration_history(q_series)
        return jax.vmap(self.observation_op)(history)


def _interpolated_emission(
    q_series: jax.Array, save_times: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Build ``t -> q(t)`` by linear interpolation of ``q_series`` at save times."""

    def emission_fn(t: jax.Array) -> jax.Array:
        return jnp.interp(t, save_times, q_series)

    return emission_fn


def build_forward(
    *,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    domain_z: tuple[float, float, int],
    save_times: jax.Array,
    source_location: tuple[float, float, float],
    uniform_wind: tuple[float, float, float],
    eddy_diffusivity: float | tuple[float, float] = 1.0,
    source_radius: float | None = None,
    advection_scheme: str = "upwind1",
    receptor_index: jax.Array | None = None,
    solver_kwargs: dict | None = None,
) -> EulerianForward4DVar:
    """Assemble a differentiable :class:`EulerianForward4DVar`.

    The transport configuration mirrors
    :func:`~plumax.les_fvm.simulate_eulerian_dispersion` but is frozen except
    for the emission time series, which becomes the 4D-Var control vector.
    A uniform wind and scalar / anisotropic constant diffusivity keep the v1
    inversion fast and the twin experiment exactly reproducible.

    Args:
        domain_x / domain_y / domain_z: ``(min, max, n_interior)`` per axis.
        save_times: Observation/integration times [s], shape ``(n_t,)``.
        source_location: ``(x, y, z)`` release point [m].
        uniform_wind: Constant ``(u, v, w)`` [m/s].
        eddy_diffusivity: Constant scalar ``K`` or ``(K_h, K_z)`` [m²/s].
        source_radius: Gaussian source radius [m] (defaults to ``2·max(dx,…)``).
        advection_scheme: Reconstruction scheme (``"upwind1"`` is the robust,
            cheap default for inversion; WENO variants are available).
        receptor_index: Optional ``(n_obs, 2)`` ``(iy, ix)`` receptor cells.
        solver_kwargs: Overrides for the diffrax solve.

    Returns:
        A configured :class:`EulerianForward4DVar`.
    """
    # Validate the time grid up front: t0/t1, SaveAt(ts=...) and jnp.interp all
    # assume ≥ 2 strictly-increasing save times. Catching it here gives a clear
    # error instead of an opaque diffrax / interpolation failure later.
    st = np.asarray(save_times, dtype=float)
    if st.ndim != 1 or st.size < 2:
        raise ValueError(
            "build_forward: `save_times` must be a 1-D array with ≥ 2 entries "
            f"(got shape {st.shape})."
        )
    if np.any(np.diff(st) <= 0.0):
        raise ValueError("build_forward: `save_times` must be strictly increasing.")

    # Build the grid (and hence the state, save times and control vector) in the
    # canonical float dtype so they all agree: float64 when JAX x64 is enabled,
    # float32 otherwise. A mismatch makes the diffrax checkpoint buffer reject
    # the (promoted) tendency.
    dtype = jnp.result_type(float)
    plume_grid = make_grid(domain_x, domain_y, domain_z, dtype=dtype)
    wind_field = uniform_wind_field(
        plume_grid=plume_grid,
        u=uniform_wind[0],
        v=uniform_wind[1],
        w=uniform_wind[2],
    )
    eddy = make_eddy_diffusivity(eddy_diffusivity)
    horizontal_bc, vertical_bc = build_default_concentration_bc(
        bc_x=("dirichlet", "outflow"), bc_y="periodic", bc_z=("neumann", "neumann")
    )
    save_times = jnp.asarray(save_times, dtype=plume_grid.x.dtype)

    # Build the source *geometry* once, with a placeholder unit rate. This runs
    # the (eager) source-location bounds check and fixes the static ``density``
    # profile outside the differentiated path; only the emission rate ``q``
    # flows through the trace, so the per-call factory just swaps ``emission_fn``
    # and never re-runs the host-side ``float(...)`` validation.
    base_source = make_gaussian_source(
        plume_grid=plume_grid,
        emission_rate=1.0,
        source_location=source_location,
        source_radius=source_radius,
    )

    def rhs_factory(q_series: jax.Array) -> EulerianDispersionRHS:
        source = GaussianSource(
            emission_fn=_interpolated_emission(q_series, save_times),
            density=base_source.density,
        )
        return EulerianDispersionRHS(
            plume_grid=plume_grid,
            wind_field=wind_field,
            eddy_diffusivity=eddy,
            source=source,
            horizontal_bc=horizontal_bc,
            vertical_bc=vertical_bc,
            advection_scheme=advection_scheme,
        )

    nz, ny, nx = plume_grid.shape
    observation_op = ColumnObservationOperator(
        dz=plume_grid.dz, receptor_index=receptor_index
    )
    return EulerianForward4DVar(
        rhs_factory=rhs_factory,
        observation_op=observation_op,
        save_times=save_times,
        initial_concentration=jnp.zeros((nz, ny, nx), dtype=plume_grid.x.dtype),
        t0=float(save_times[0]),
        t1=float(save_times[-1]),
        solver_kwargs=dict(solver_kwargs or {}),
    )


# ── 4D-Var cost in whitened control space ────────────────────────────────────


@dataclass(frozen=True)
class FourDVarProblem:
    """A strong-constraint 4D-Var problem over a scalar emission time series.

    Holds the differentiable forward, the prior ``(S_b, B = L Lᵀ)``, and the
    observations ``(y_t, R_t)``. The cost is evaluated in whitened space
    ``χ = L⁻¹ (S − S_b)`` so the prior term is the identity ``½‖χ‖²`` and the
    Hessian is well-conditioned (design §control-transform). ``observations``
    has shape ``(n_t, n_obs)`` and ``obs_variance`` broadcasts to it.

    Build with :func:`build_problem`.
    """

    forward: EulerianForward4DVar
    prior_mean: jax.Array  # S_b, (n_t,)
    prior_chol: jax.Array  # L with B = L Lᵀ, (n_t, n_t)
    observations: jax.Array  # y, (n_t, n_obs)
    obs_variance: jax.Array  # R diagonal, broadcast to (n_t, n_obs)

    def source_from_whitened(self, chi: jax.Array) -> jax.Array:
        """Map whitened control ``χ`` back to the emission signal ``S``."""
        return self.prior_mean + self.prior_chol @ chi

    def cost(self, chi: jax.Array) -> jax.Array:
        """The 4D-Var cost ``J`` at whitened control ``χ`` (scalar)."""
        q = self.source_from_whitened(chi)
        predicted = self.forward.predict(q)  # (n_t, n_obs)
        residual = predicted - self.observations
        obs_term = 0.5 * jnp.sum(residual**2 / self.obs_variance)
        prior_term = 0.5 * jnp.sum(chi**2)
        return obs_term + prior_term

    def value_and_grad(self, chi: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``(J, ∂J/∂χ)`` — the gradient is the exact adjoint via reverse-mode AD."""
        return jax.value_and_grad(self.cost)(chi)


def build_problem(
    *,
    forward: EulerianForward4DVar,
    observations: jax.Array,
    prior_mean: jax.Array,
    prior_covariance: jax.Array,
    obs_variance: jax.Array | float,
) -> FourDVarProblem:
    """Assemble a :class:`FourDVarProblem`, factorising the prior covariance.

    Args:
        forward: The differentiable transport forward.
        observations: Column observations ``y``, shape ``(n_t, n_obs)``.
        prior_mean: Source prior mean ``S_b``, shape ``(n_t,)``.
        prior_covariance: Source prior covariance ``B``, shape ``(n_t, n_t)``
            (e.g. a Matérn-3/2 temporal prior from
            :func:`plumax.lagrangian.inversion.matern32_covariance`). Cholesky
            -factorised once here.
        obs_variance: Observation-error variance ``R`` (scalar or broadcastable
            to ``observations``); ``R = R_retr + R_repr``.

    Returns:
        The configured :class:`FourDVarProblem`.
    """
    sb = jnp.asarray(prior_mean, dtype=float)
    b = jnp.asarray(prior_covariance, dtype=float)
    y = jnp.asarray(observations, dtype=float)
    if sb.ndim != 1:
        raise ValueError("build_problem: `prior_mean` must be 1-D (n_t,).")
    n_t = sb.shape[0]
    if b.shape != (n_t, n_t):
        raise ValueError(
            f"build_problem: `prior_covariance` must be ({n_t}, {n_t}), got {b.shape}."
        )
    if y.ndim != 2 or y.shape[0] != n_t:
        raise ValueError(
            f"build_problem: `observations` must be (n_t={n_t}, n_obs), got {y.shape}."
        )
    # Jitter keeps the Cholesky factor PD for smooth Matérn kernels.
    chol = jnp.linalg.cholesky(b + 1e-9 * jnp.eye(n_t))

    # Resolve obs_variance to R, the (n_t, n_obs) diagonal. A length-n_t 1-D
    # input is treated as one variance *per overpass* (R_t), so reshape it to a
    # column before broadcasting — otherwise numpy would align it with the
    # trailing n_obs axis. A length-n_obs 1-D input stays per-receptor, and a
    # scalar broadcasts to everything. When n_t == n_obs a bare 1-D vector is
    # genuinely ambiguous between the two, so reject it and require an explicit
    # 2-D shape rather than silently picking one and changing the objective.
    r_in = np.asarray(obs_variance, dtype=float)
    n_obs = y.shape[1]
    if r_in.ndim == 1 and n_t == n_obs and r_in.shape[0] == n_t:
        raise ValueError(
            f"build_problem: `obs_variance` of length {n_t} is ambiguous when "
            f"n_t == n_obs == {n_t} (per-overpass vs per-receptor). Pass an "
            f"explicit ({n_t}, 1) column for per-overpass R_t, a (1, {n_obs}) "
            f"row for per-receptor, or a full ({n_t}, {n_obs}) array."
        )
    if r_in.ndim == 1 and r_in.shape[0] == n_t:
        r_in = r_in[:, None]
    if np.any(r_in <= 0.0):
        raise ValueError("build_problem: `obs_variance` entries must be > 0.")
    try:
        r = jnp.broadcast_to(jnp.asarray(r_in), y.shape)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"build_problem: `obs_variance` shape {np.shape(obs_variance)} is not "
            f"broadcastable to observations {y.shape}."
        ) from exc
    return FourDVarProblem(
        forward=forward,
        prior_mean=sb,
        prior_chol=chol,
        observations=y,
        obs_variance=r,
    )


@dataclass(frozen=True)
class FourDVarResult:
    """Outcome of a 4D-Var minimisation.

    Attributes:
        source: MAP emission time series ``S* = S_b + L χ*``, shape ``(n_t,)``.
        whitened: Optimal whitened control ``χ*``, shape ``(n_t,)``.
        cost: Final cost value ``J(χ*)``.
        n_steps: Optimiser iterations consumed (``-1`` if not reported).
        posterior: Optional :class:`PosteriorCovariance` around the MAP, attached
            when :func:`solve_4dvar` is called with ``compute_posterior=True``.
    """

    source: jax.Array
    whitened: jax.Array
    cost: float
    n_steps: int
    posterior: PosteriorCovariance | None = None


def solve_4dvar(
    problem: FourDVarProblem,
    *,
    initial_source: jax.Array | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 100,
    solver: optx.AbstractMinimiser | None = None,
    compute_posterior: bool = False,
) -> FourDVarResult:
    """Minimise the 4D-Var cost with L-BFGS in whitened space.

    Optimises ``χ`` (whitened control), so the prior is the identity Gaussian
    and the gradient — the exact discrete adjoint of the FV solver — flows
    through ``jax.value_and_grad`` of :meth:`FourDVarProblem.cost`.

    Args:
        problem: The assembled 4D-Var problem.
        initial_source: Optional first-guess emission signal ``S₀`` (defaults to
            the prior mean, i.e. ``χ₀ = 0``).
        rtol / atol: Optimiser tolerances.
        max_steps: Maximum optimiser iterations.
        solver: Optional optimistix minimiser (defaults to ``optx.LBFGS``).
        compute_posterior: When ``True``, also evaluate the Gauss-Newton Laplace
            :func:`posterior_covariance` at the MAP and attach it to the result.

    Returns:
        The :class:`FourDVarResult` with the MAP source and diagnostics.
    """
    import optimistix as optx

    n_t = problem.prior_mean.shape[0]
    if initial_source is None:
        chi0 = jnp.zeros(n_t)
    else:
        s0 = jnp.asarray(initial_source, dtype=float)
        # χ₀ = L⁻¹ (S₀ − S_b) — solve the triangular system rather than invert.
        chi0 = jax.scipy.linalg.solve_triangular(
            problem.prior_chol, s0 - problem.prior_mean, lower=True
        )

    if solver is None:
        solver = optx.LBFGS(rtol=rtol, atol=atol)
    sol = optx.minimise(
        lambda chi, _args: problem.cost(chi),
        solver,
        chi0,
        max_steps=max_steps,
        throw=False,
    )
    chi_star = sol.value
    posterior = posterior_covariance(problem, chi_star) if compute_posterior else None
    return FourDVarResult(
        source=problem.source_from_whitened(chi_star),
        whitened=chi_star,
        cost=float(problem.cost(chi_star)),
        n_steps=int(sol.stats.get("num_steps", -1)),
        posterior=posterior,
    )


# ── posterior covariance (Gauss-Newton Laplace approximation) ─────────────────


@dataclass(frozen=True)
class PosteriorCovariance:
    """Laplace (Gauss-Newton) posterior covariance of a 4D-Var solution.

    A Gaussian approximation ``N(S*, P_S)`` to the posterior, built from the
    Gauss-Newton Hessian of the whitened-space cost at the MAP ``χ*``
    (design §posterior-covariance). The whitened-space Hessian is

        H_GN = I + J̃ᵀ R⁻¹ J̃,   J̃ = ∂(predicted_obs)/∂χ,

    so the whitened posterior covariance is ``P_χ = H_GN⁻¹`` and — because
    ``S = S_b + L χ`` — the source-space covariance is ``P_S = L P_χ Lᵀ``.

    Attributes:
        whitened_covariance: ``P_χ``, shape ``(n_t, n_t)``.
        source_covariance: ``P_S``, shape ``(n_t, n_t)``.
    """

    whitened_covariance: jax.Array  # P_χ, (n_t, n_t)
    source_covariance: jax.Array  # P_S, (n_t, n_t)

    @property
    def source_std(self) -> jax.Array:
        """Marginal source-space posterior standard deviations ``√diag(P_S)``."""
        return jnp.sqrt(jnp.diagonal(self.source_covariance))


def _gauss_newton_hessian(
    problem: FourDVarProblem, whitened_map: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Assemble ``(H_GN, J̃)`` at the MAP whitened control ``χ*``.

    The Jacobian ``J̃ = ∂(predicted_obs)/∂χ`` of the flattened whitened-space
    observation prediction is taken with :func:`jax.jacrev`. Forward-mode
    (``jacfwd``) would be better-shaped here — ``n_state = n_t`` is small while
    the observation vector is large — but the diffrax FV solve exposes a
    reverse-mode adjoint (``custom_vjp``) and rejects forward-mode JVPs, so
    reverse-mode is the only differentiation path through the transport model.
    The Gauss-Newton Hessian is then ``H_GN = I + J̃ᵀ diag(1/R) J̃``.
    """

    def predict_whitened(chi: jax.Array) -> jax.Array:
        return problem.forward.predict(problem.source_from_whitened(chi)).reshape(-1)

    jac = jax.jacrev(predict_whitened)(whitened_map)  # (n_obs_flat, n_state)
    r_inv = (1.0 / problem.obs_variance).reshape(-1)  # (n_obs_flat,)
    n_state = whitened_map.shape[0]
    h_gn = jnp.eye(n_state) + jac.T @ (r_inv[:, None] * jac)
    return h_gn, jac


def posterior_covariance(
    problem: FourDVarProblem, whitened_map: jax.Array
) -> PosteriorCovariance:
    """Gauss-Newton Laplace posterior covariance around a 4D-Var MAP.

    Builds the Gauss-Newton Hessian ``H_GN = I + J̃ᵀ R⁻¹ J̃`` of the whitened
    cost at ``whitened_map`` and inverts it with :mod:`gaussx` (the Hessian is
    wrapped as a symmetric positive-semidefinite
    :class:`lineax.MatrixLinearOperator` and inverted via :func:`gaussx.inv`).
    The whitened posterior covariance is ``P_χ = H_GN⁻¹`` and the source-space
    covariance is ``P_S = L P_χ Lᵀ`` for the prior Cholesky factor ``L``.

    Args:
        problem: The assembled 4D-Var problem.
        whitened_map: The MAP whitened control ``χ*`` (e.g.
            :attr:`FourDVarResult.whitened`), shape ``(n_t,)``.

    Returns:
        The :class:`PosteriorCovariance` holding ``P_χ`` and ``P_S``.
    """
    chi = jnp.asarray(whitened_map, dtype=float)
    h_gn, _ = _gauss_newton_hessian(problem, chi)
    hessian_op = lx.MatrixLinearOperator(
        h_gn, tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    )
    p_chi = gx.inv(hessian_op).as_matrix()
    # Symmetrise to clean up asymmetric round-off before the congruence map.
    p_chi = 0.5 * (p_chi + p_chi.T)
    chol = problem.prior_chol  # L with B = L Lᵀ
    p_source = chol @ p_chi @ chol.T
    return PosteriorCovariance(
        whitened_covariance=p_chi,
        source_covariance=0.5 * (p_source + p_source.T),
    )


def laplace_sample(
    problem: FourDVarProblem,
    whitened_map: jax.Array,
    key: jax.Array,
    n_samples: int,
) -> jax.Array:
    """Draw source-space samples from the Laplace posterior ``N(S*, P_S)``.

    Sampling happens in whitened space — ``χ ~ N(χ*, P_χ)`` via a Cholesky
    factor of ``P_χ`` — and each draw is mapped through
    :meth:`FourDVarProblem.source_from_whitened`, which is exactly the affine
    map that turns ``N(χ*, P_χ)`` into ``N(S*, P_S)``.

    Args:
        problem: The assembled 4D-Var problem.
        whitened_map: The MAP whitened control ``χ*``, shape ``(n_t,)``.
        key: A ``jax.random`` PRNG key.
        n_samples: Number of posterior draws to return.

    Returns:
        Source-space samples, shape ``(n_samples, n_t)``.
    """
    if not _is_traced(n_samples) and int(n_samples) <= 0:
        raise ValueError("laplace_sample: `n_samples` must be a positive integer.")
    chi_star = jnp.asarray(whitened_map, dtype=float)
    post = posterior_covariance(problem, chi_star)
    n_t = chi_star.shape[0]
    # Jitter keeps the whitened-covariance Cholesky PD against round-off.
    chol_chi = jnp.linalg.cholesky(
        post.whitened_covariance + 1e-12 * jnp.eye(n_t, dtype=chi_star.dtype)
    )
    noise = jax.random.normal(key, (int(n_samples), n_t), dtype=chi_star.dtype)
    chi_samples = chi_star[None, :] + noise @ chol_chi.T
    return jax.vmap(problem.source_from_whitened)(chi_samples)


__all__ = [
    "ColumnObservationOperator",
    "EulerianForward4DVar",
    "FourDVarProblem",
    "FourDVarResult",
    "PosteriorCovariance",
    "build_forward",
    "build_problem",
    "laplace_sample",
    "posterior_covariance",
    "solve_4dvar",
]
