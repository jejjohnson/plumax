"""Tier III — strong-constraint 4D-Var inversion over the Eulerian transport model.

Recovers a time-resolved emission signal from a sequence of column observations
by directly minimising the strong-constraint 4D-Var cost
([design](https://github.com/jejjohnson/plumax/blob/main/docs/design/03_tier3_eulerian.md#tier3-cost)):

    J(S) = ½‖S − S_b‖²_B  +  ½ Σ_t ‖H_t c(S, t) − y_t‖²_{R_t}

where ``c(S, t)`` is the 3-D tracer field produced by the finite-volume solver
:func:`~plumax.les_fvm.simulate_eulerian_dispersion` for emission signal ``S``,
``H_t`` is the column observation operator (vertical integral → optional spatial
sampling), ``S_b`` / ``B`` the source prior, and ``R_t`` the observation-error
covariance at overpass ``t``. By default the initial field is pinned at zero and
only the source is inverted; a :class:`BackgroundControl` adds the IC / background
term ``½‖c_0 − c_b‖²_{B_c}`` of the three-term design cost and per-face lateral
boundary scaling factors to the control.

The solve is a **vardax analysis** (the ``assim`` extra,
``pip install 'plumax[assim]'``). The control is a source, not an initial
state, so the cost above is ``vardax.ThreeDVar`` with the source–receptor map
``𝒢(S) = vec H c(S, ·)`` as its observation operator
(:meth:`FourDVarProblem.source_receptor_operator`), minimised by L-BFGS. The key
enabler (design §adjoint) is that **JAX reverse-mode autodiff through the diffrax
FV solve is the exact discrete adjoint** — there is no hand-written adjoint
model. The diffrax solve defines only a VJP, so the operator is linearised in
reverse mode (``CallableObs(jac="bwd", materialise=True)``). The incremental
variant (design §incremental), Gauss–Newton outer and CG inner iterations, is
``vardax.IncrementalFourDVar`` over the same map
(:meth:`FourDVarProblem.incremental_analysis`).

The control vector is either the **time-resolved scalar emission rate** ``S = q``
at a known source location (:class:`TemporalControl`, or the ``prior_mean`` and
``prior_covariance`` arguments of :func:`build_problem`), or a **per-cell surface
flux map** with a Matérn-3/2 prior (:class:`SpatialControl`). The analysis is
written against a flat control vector, so both share one solver.

Control-variable transform (design §control-transform): optimisation runs in
**whitened space** ``χ = L⁻¹ (S − S_b)`` where ``B = L Lᵀ``, through
``vardax.LinearParameterization(linear_basis(L, variance=1))``. The prior term
is then the identity ``½‖χ‖²``, which is what makes the problem well-conditioned;
the Matérn-3/2 temporal prior from :mod:`plumax.lagrangian.inversion` supplies
``B``. The observations are whitened too (``y / σ`` with ``R = I``), so the
tiny concentration-scale variances never enter a linear solve. The posterior is
vardax's Gauss–Newton Laplace covariance of that linear control.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from plumax._src.errors import PayloadError
from plumax._src.scalar_emission import GaussianEmission, gaussian_total
from plumax._vardax import require_vardax
from plumax.gauss_plume.observation import ColumnObservation
from plumax.lagrangian.inversion import _is_traced, matern32_covariance
from plumax.les_fvm.boundary import (
    LATERAL_FACES,
    build_default_concentration_bc,
    scale_dirichlet_faces,
)
from plumax.les_fvm.diffusion import make_eddy_diffusivity
from plumax.les_fvm.dynamics import EulerianDispersionRHS
from plumax.les_fvm.grid import PlumeGrid3D, make_grid
from plumax.les_fvm.source import (
    GaussianSource,
    SurfaceFluxSource,
    make_gaussian_source,
)
from plumax.les_fvm.wind import uniform_wind_field


if TYPE_CHECKING:
    import optimistix as optx
    import vardax


# ── control layouts ──────────────────────────────────────────────────────────


#: Tags of a covariance operator or a covariance factor built from one.
_PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


@dataclass(frozen=True)
class TemporalControl:
    """Prior of the scalar emission time series at a known source.

    The control of :func:`build_problem` when the forward has a source location.

    Attributes:
        prior_mean: Source prior mean ``S_b``, shape ``(n_t,)``.
        prior_covariance: Source prior covariance ``B``, shape ``(n_t, n_t)``.
    """

    prior_mean: jax.Array
    prior_covariance: jax.Array


@dataclass(frozen=True, eq=False)
class SpatialControl:
    """Per-cell surface flux control with a Matérn-3/2 prior.

    q(s, t) = Σ_k φ_k(t) q_k(s) on the active cells and 0 elsewhere, with
    independent coefficient fields sharing the spatial covariance,

        q_k(s) ~ N(a_k m(s), B_s),   B_s = Σ_σ ρ Σ_σ,   Φ a = 1.

    The coefficients a make the prior mean of the flux m(s) at every save time,
    so the time basis must span the constant. The flux at the save times then
    has covariance Φ Φᵀ ⊗ B_s, so a dense temporal prior B_t = L_t L_tᵀ is the
    basis Φ = L_t.

    The correlation ρ depends on ``separable``, and the two are different
    kernels. With d = s − s′ and h(r) = (1 + √3 r / ℓ) exp(−√3 r / ℓ):

    - ``separable=True`` is the product kernel ρ = h(|d_y|) h(|d_x|), stored as
      ``gaussx.Kronecker(ρ_y, ρ_x)`` when every cell is active.
    - ``separable=False`` is the isotropic kernel ρ = h(‖d‖), the dense
      :func:`~plumax.lagrangian.inversion.matern32_covariance`.

    The prior is Gaussian, so the MAP flux is not constrained to be positive.

    Attributes:
        mask: Active T-cells, shape ``(ny, nx)``. Flat control vectors list them
            row-major over ``(y, x)``, one block per basis function.
        prior_mean: Prior mean flux m [kg m⁻² s⁻¹], shape ``(ny, nx)``.
        length_scale_m: Correlation length ℓ [m].
        prior_std: Prior standard deviation σ [kg m⁻² s⁻¹], shape ``(ny, nx)``.
        time_basis: Basis Φ at the save times, shape ``(n_t, K)``, linearly
            interpolated between them. ``None`` makes the flux constant in time.
        separable: Use the product kernel rather than the isotropic one.

    Raises:
        ValueError: If the shapes disagree, no cell is active, ``length_scale_m``
            or an active ``prior_std`` is not positive, an input is not finite,
            or the basis does not span the constant.

    Examples:
        Two active cells of a 2 × 2 grid, constant in time:

        >>> import jax.numpy as jnp
        >>> control = SpatialControl(
        ...     mask=jnp.array([[True, False], [False, True]]),
        ...     prior_mean=jnp.ones((2, 2)),
        ...     length_scale_m=100.0,
        ...     prior_std=0.5 * jnp.ones((2, 2)),
        ... )
        >>> control.size
        2
        >>> control.to_field(jnp.array([2.0, 3.0])).tolist()
        [[[2.0, 0.0], [0.0, 3.0]]]
    """

    mask: jax.Array
    prior_mean: jax.Array
    length_scale_m: float
    prior_std: jax.Array
    time_basis: jax.Array | None = None
    separable: bool = True

    def __post_init__(self) -> None:
        shape = np.shape(self.mask)
        if len(shape) != 2:
            raise ValueError(f"SpatialControl: `mask` must be 2-D, got {shape}.")
        for name in ("prior_mean", "prior_std"):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(
                    f"SpatialControl: `{name}` must have the mask's shape {shape}, "
                    f"got {np.shape(getattr(self, name))}."
                )
        active = np.asarray(self.mask, dtype=bool)
        if not active.any():
            raise ValueError("SpatialControl: `mask` has no active cell.")
        if not self.length_scale_m > 0.0:
            raise ValueError("SpatialControl: `length_scale_m` must be > 0.")
        mean = np.asarray(self.prior_mean, dtype=float)[active]
        std = np.asarray(self.prior_std, dtype=float)[active]
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(std))):
            raise ValueError("SpatialControl: priors must be finite on active cells.")
        if np.any(std <= 0.0):
            raise ValueError("SpatialControl: `prior_std` must be > 0 on active cells.")
        if self.time_basis is not None:
            self._mean_coefficients()

    @property
    def n_active(self) -> int:
        """Number of active cells."""
        return int(np.count_nonzero(np.asarray(self.mask, dtype=bool)))

    @property
    def n_basis(self) -> int:
        """Number K of basis functions."""
        return 1 if self.time_basis is None else int(np.shape(self.time_basis)[1])

    @property
    def size(self) -> int:
        """Length of the flat control vector, n_active · K."""
        return self.n_active * self.n_basis

    def _mean_coefficients(self) -> jax.Array:
        """Coefficients a with Φ a = 1, shape ``(K,)``."""
        if self.time_basis is None:
            return jnp.ones(1, dtype=float)
        basis = np.asarray(self.time_basis, dtype=float)
        if basis.ndim != 2:
            raise ValueError("SpatialControl: `time_basis` must be 2-D (n_t, K).")
        constant = np.ones(basis.shape[0])
        coefficients, *_ = np.linalg.lstsq(basis, constant, rcond=None)
        # 1 ∈ span(Φ): otherwise no coefficient mean makes the flux mean m(s)
        if not np.allclose(basis @ coefficients, constant, rtol=0.0, atol=1e-8):
            raise ValueError(
                "SpatialControl: `time_basis` must span the constant, so that the "
                "prior mean of the flux is `prior_mean` at every save time."
            )
        return jnp.asarray(coefficients, dtype=float)

    def flat_prior_mean(self) -> jax.Array:
        """Prior mean μ_q of the flat control, shape ``(size,)``."""
        mean = jnp.asarray(self.prior_mean, dtype=float)[np.asarray(self.mask, bool)]
        # μ_q = a ⊗ m
        return jnp.kron(self._mean_coefficients(), mean)

    def to_field(self, flat: jax.Array) -> jax.Array:
        """Map a flat control vector onto the grid, with 0 on inactive cells.

        Args:
            flat: Flat control values, shape ``(..., size)``.

        Returns:
            The coefficient fields, shape ``(..., K, ny, nx)``.
        """
        ny, nx = np.shape(self.mask)
        rows, columns = np.nonzero(np.asarray(self.mask, dtype=bool))
        per_cell = einx.rearrange(
            "... (basis cell) -> ... basis cell", flat, basis=self.n_basis
        )
        field_shape = (*per_cell.shape[:-1], ny, nx)
        return (
            jnp.zeros(field_shape, dtype=flat.dtype)
            .at[..., rows, columns]
            .set(per_cell)
        )

    def flux_function(
        self, flat: jax.Array, save_times: jax.Array
    ) -> Callable[[jax.Array], jax.Array]:
        """The flux t -> q(·, t) of a flat control, on the ``(ny, nx)`` grid."""
        fields = self.to_field(flat)
        if self.time_basis is None:
            constant_flux = fields[0]

            def flux(t: jax.Array) -> jax.Array:
                del t
                return constant_flux

            return flux

        basis = einx.rearrange(
            "time basis -> basis time", jnp.asarray(self.time_basis, dtype=flat.dtype)
        )

        def flux(t: jax.Array) -> jax.Array:
            # φ_k(t), linear between the save times
            basis_at_t = jax.vmap(lambda phi: jnp.interp(t, save_times, phi))(basis)
            # q(s, t) = Σ_k φ_k(t) q_k(s)
            return einx.dot("basis, basis y x -> y x", basis_at_t, fields)

        return flux

    def _correlation_factors(
        self, y: jax.Array, x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """The 1-D correlation matrices ρ_y and ρ_x of the product kernel."""
        rho_y = matern32_covariance(y, variance=1.0, length_scale=self.length_scale_m)
        rho_x = matern32_covariance(x, variance=1.0, length_scale=self.length_scale_m)
        return rho_y, rho_x

    def _active_correlation(self, y: jax.Array, x: jax.Array) -> jax.Array:
        """Dense correlation matrix ρ over the active cells."""
        active = np.flatnonzero(np.asarray(self.mask, dtype=bool))
        if self.separable:
            rho_y, rho_x = self._correlation_factors(y, x)
            return jnp.kron(rho_y, rho_x)[np.ix_(active, active)]
        grid_y, grid_x = jnp.meshgrid(y, x, indexing="ij")
        coordinates = jnp.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)
        return matern32_covariance(
            coordinates[active], variance=1.0, length_scale=self.length_scale_m
        )

    def _is_kronecker(self) -> bool:
        # a mask breaks the product structure of the matrix over the active cells
        return self.separable and bool(np.all(np.asarray(self.mask, dtype=bool)))

    def _scale(self) -> lx.DiagonalLinearOperator:
        # I_K ⊗ Σ_σ over the flat control
        std = jnp.asarray(self.prior_std, dtype=float)[np.asarray(self.mask, bool)]
        return lx.DiagonalLinearOperator(jnp.tile(std, self.n_basis))

    def _over_basis(
        self, block: lx.AbstractLinearOperator
    ) -> lx.AbstractLinearOperator:
        # I_K ⊗ block: the coefficient fields are independent
        if self.n_basis == 1:
            return block
        identity = lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct((self.n_basis,), block.out_structure().dtype)
        )
        return gx.Kronecker(identity, block)

    def prior_covariance(self, y: jax.Array, x: jax.Array) -> lx.AbstractLinearOperator:
        """Prior covariance of the flat control, I_K ⊗ Σ_σ ρ Σ_σ.

        Args:
            y: T-cell centres along y [m], shape ``(ny,)``.
            x: T-cell centres along x [m], shape ``(nx,)``.

        Returns:
            The covariance, with ρ a ``gaussx.Kronecker(ρ_y, ρ_x)`` when the
            kernel is separable and every cell is active, dense otherwise.
        """
        if self._is_kronecker():
            rho_y, rho_x = self._correlation_factors(y, x)
            rho = gx.Kronecker(
                lx.MatrixLinearOperator(rho_y, _PSD_TAGS),
                lx.MatrixLinearOperator(rho_x, _PSD_TAGS),
            )
        else:
            rho = lx.MatrixLinearOperator(self._active_correlation(y, x), _PSD_TAGS)
        # B = (I_K ⊗ Σ_σ) (I_K ⊗ ρ) (I_K ⊗ Σ_σ)
        covariance = self._scale() @ self._over_basis(rho) @ self._scale()
        return lx.TaggedLinearOperator(covariance, lx.positive_semidefinite_tag)

    def prior_sqrt(self, y: jax.Array, x: jax.Array) -> lx.AbstractLinearOperator:
        """Lower-triangular square root S of :meth:`prior_covariance`, S Sᵀ = B.

        Factorises each block with ``gaussx.safe_cholesky``, whose adaptive
        jitter survives a smooth Matérn on a fine grid; a Kronecker block needs
        only its two 1-D factors.

        Args:
            y: T-cell centres along y [m], shape ``(ny,)``.
            x: T-cell centres along x [m], shape ``(nx,)``.

        Returns:
            S = (I_K ⊗ Σ_σ) (I_K ⊗ chol ρ).
        """
        if self._is_kronecker():
            rho_y, rho_x = self._correlation_factors(y, x)
            root = gx.Kronecker(
                lx.MatrixLinearOperator(_cholesky(rho_y)),
                lx.MatrixLinearOperator(_cholesky(rho_x)),
            )
        else:
            rho = self._active_correlation(y, x)
            root = lx.MatrixLinearOperator(_cholesky(rho))
        # S = (I_K ⊗ Σ_σ) (I_K ⊗ chol ρ)
        return self._scale() @ self._over_basis(root)


def _cholesky(covariance: jax.Array) -> jax.Array:
    # adaptive jitter: a fixed kick fails on badly scaled Matérn priors
    operator = lx.MatrixLinearOperator(covariance, _PSD_TAGS)
    return gx.safe_cholesky(operator)


@dataclass(frozen=True, eq=False)
class BackgroundControl:
    """Initial concentration and per-face inflow scaling in the control.

    The flat control becomes (S, δc₀, α), with priors independent of the source's,

        c₀ = c_b + δc₀,   δc₀ ~ N(0, B_c),   α_f ~ N(α̅, σ_α²) for each face f,

    and every Dirichlet lateral boundary value c_f of the forward becomes
    α_f c_f (:func:`~plumax.les_fvm.boundary.scale_dirichlet_faces`). The factors
    are ordered (west, east, south, north). This is the per-face Gaussian of the
    design page; a per-face Matérn along the boundary is its open v2.

    Attributes:
        c_b: Background initial concentration c_b on the interior cells [kg m⁻³],
            shape ``(nz, ny, nx)``.
        B_c: Covariance of δc₀, a lineax operator on the flattened interior.
        bc_scale_prior: Mean α̅ and standard deviation σ_α of each factor.

    Raises:
        ValueError: If ``c_b`` is not a finite 3-D array, ``B_c`` has the wrong
            size, or ``bc_scale_prior`` has no positive standard deviation.

    Examples:
        A background on a 2 × 3 × 4 interior adds 24 + 4 entries to the control:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import lineax as lx
        >>> covariance = lx.DiagonalLinearOperator(jnp.ones(24))
        >>> background = BackgroundControl(jnp.zeros((2, 3, 4)), covariance)
        >>> background.size
        28
    """

    c_b: jax.Array
    B_c: lx.AbstractLinearOperator
    bc_scale_prior: tuple[float, float] = (1.0, 0.1)

    def __post_init__(self) -> None:
        c_b = np.asarray(self.c_b, dtype=float)
        if c_b.ndim != 3 or not np.all(np.isfinite(c_b)):
            raise ValueError(
                "BackgroundControl: `c_b` must be a finite (nz, ny, nx) array."
            )
        if self.B_c.in_size() != c_b.size or self.B_c.out_size() != c_b.size:
            raise ValueError(
                f"BackgroundControl: `B_c` must act on {c_b.size} interior cells, "
                f"got {self.B_c.in_size()} → {self.B_c.out_size()}."
            )
        mean, std = self.bc_scale_prior
        if not (np.isfinite(mean) and np.isfinite(std) and std > 0.0):
            raise ValueError(
                "BackgroundControl: `bc_scale_prior` needs a finite mean and a "
                "finite standard deviation > 0."
            )

    @property
    def size(self) -> int:
        """Length of the control's (δc₀, α) part."""
        return int(np.size(self.c_b)) + _N_FACES

    def split(self, flat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Split a flat control into (S, δc₀ as a field, α)."""
        n_source = flat.shape[-1] - self.size
        increment = flat[..., n_source:-_N_FACES]
        increment_field = increment.reshape(
            (*increment.shape[:-1], *np.shape(self.c_b))
        )
        return flat[..., :n_source], increment_field, flat[..., -_N_FACES:]

    def flat_prior_mean(self) -> jax.Array:
        """Prior mean of the (δc₀, α) part, (0, α̅)."""
        increment_mean = jnp.zeros(np.size(self.c_b), dtype=float)
        scale_mean = jnp.full(_N_FACES, self.bc_scale_prior[0], dtype=float)
        return jnp.concatenate([increment_mean, scale_mean])


#: Lateral faces with a boundary scaling factor.
_N_FACES = len(LATERAL_FACES)


# ── differentiable forward: emission signal → column-observation time series ──


@dataclass(frozen=True)
class ColumnObservationOperator:
    """Sample the column-integrated field at fixed receptor cells.

    ``H_t`` in the 4D-Var cost: vertically integrate the 3-D concentration to a
    column (``∑_z c · dz``), then gather the values at a set of receptor pixels.
    With ``receptor_index = None`` every interior column is an observation.
    A thin wrapper over the shared
    :class:`~plumax.gauss_plume.observation.ColumnObservation` (identity
    averaging kernel, ``h_k = dz``), whose module docstring states the units.

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
        nz = concentration.shape[0]
        layer_weights = jnp.full(nz, self.dz, dtype=concentration.dtype)
        observation = ColumnObservation(
            layer_weights=layer_weights, receptor_index=self.receptor_index
        )
        return observation.from_field(concentration)


@dataclass(frozen=True)
class EulerianForward4DVar:
    """Differentiable map from an emission time series to column observations.

    Bundles the fixed transport configuration (grid, wind, diffusivity, BCs,
    source geometry, save times) so that :meth:`predict` is a pure function of
    the control vector ``q`` — the time-resolved emission rate at the save
    times, or the flat per-cell flux of a :class:`SpatialControl` — and is
    therefore ``jax.grad`` / ``jax.linearize``-able for 4D-Var.

    Build with :func:`build_forward` rather than directly.

    Attributes:
        rhs_factory: ``(q_series, bc_scale=None) -> EulerianDispersionRHS``
            closing over the static transport configuration; the emission rate
            (linearly interpolated between save times) and the optional
            per-face boundary scaling are its only free inputs.
        observation_op: The column ``H_t``.
        save_times: Observation/integration times [s], shape ``(n_t,)``.
        initial_concentration: Padded initial field ``(Nz, Ny, Nx)``.
        solver_kwargs: diffrax solve settings (solver, dt0, tolerances).
        plume_grid: The analysis grid; its T-cells place a spatial prior.
        control: The spatial control the flat control vector is laid out by;
            ``None`` for the emission time series.
        background: The initial-concentration and boundary-scaling part of the
            flat control; ``None`` pins both.
    """

    rhs_factory: Callable[..., EulerianDispersionRHS]
    observation_op: ColumnObservationOperator
    save_times: jax.Array
    initial_concentration: jax.Array
    t0: float
    t1: float
    solver_kwargs: dict = field(default_factory=dict)
    plume_grid: PlumeGrid3D | None = None
    control: SpatialControl | None = None
    background: BackgroundControl | None = None

    def with_background(self, background: BackgroundControl) -> EulerianForward4DVar:
        """This forward over the flat control (S, δc₀, α) of ``background``."""
        return dataclasses.replace(self, background=background)

    def concentration_history(self, q_series: jax.Array) -> jax.Array:
        """Solve the FV transport for ``q_series``; return interior fields.

        Returns the ``(n_t, nz, ny, nx)`` interior concentration history (ghost
        rings stripped). Differentiable in ``q_series`` via the diffrax solve.
        """
        import diffrax

        flat = jnp.asarray(q_series)
        if self.background is None:
            rhs = self.rhs_factory(flat)
            initial_concentration = self.initial_concentration
        else:
            source, increment, bc_scale = self.background.split(flat)
            rhs = self.rhs_factory(source, bc_scale)
            # c₀ = c_b + δc₀ on the interior; the first RHS call fills the ghosts
            interior = jnp.asarray(self.background.c_b, flat.dtype) + increment
            initial_concentration = self.initial_concentration.at[1:-1, 1:-1, 1:-1].set(
                interior
            )
        kw = {
            "solver": diffrax.Heun(),
            "dt0": 1.0,
            "stepsize_controller": diffrax.ConstantStepSize(),
            "max_steps": 100_000,
            **self.solver_kwargs,
        }
        # diffrax's StepTo controller prescribes its own step points and requires
        # dt0=None; the default dt0=1.0 above would otherwise make the solve
        # fail, so drop it whenever a StepTo controller is in effect.
        if isinstance(kw["stepsize_controller"], diffrax.StepTo):
            kw["dt0"] = None
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            kw["solver"],
            t0=self.t0,
            t1=self.t1,
            dt0=kw["dt0"],
            y0=initial_concentration,
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
    source_location: tuple[float, float, float] | None = None,
    uniform_wind: tuple[float, float, float],
    control: SpatialControl | None = None,
    eddy_diffusivity: float | tuple[float, float] = 1.0,
    source_radius: float | None = None,
    advection_scheme: str = "upwind1",
    receptor_index: jax.Array | None = None,
    solver_kwargs: dict | None = None,
    bc_x: str | tuple = ("dirichlet", "outflow"),
    bc_y: str | tuple = "periodic",
) -> EulerianForward4DVar:
    """Assemble a differentiable :class:`EulerianForward4DVar`.

    The transport configuration mirrors
    :func:`~plumax.les_fvm.simulate_eulerian_dispersion` but is frozen except
    for the emission time series, which becomes the 4D-Var control vector.
    A uniform wind and scalar / anisotropic constant diffusivity keep the v1
    inversion fast and the twin experiment exactly reproducible.

    Args:
        domain_x: ``(min, max, n_interior)`` along x.
        domain_y: ``(min, max, n_interior)`` along y.
        domain_z: ``(min, max, n_interior)`` along z.
        save_times: Observation/integration times [s], shape ``(n_t,)``.
        source_location: ``(x, y, z)`` release point [m] of the emission time
            series. Exactly one of this and ``control`` is given.
        uniform_wind: Constant ``(u, v, w)`` [m/s].
        control: Per-cell surface flux control, injected through
            :class:`~plumax.les_fvm.source.SurfaceFluxSource`.
        eddy_diffusivity: Constant scalar ``K`` or ``(K_h, K_z)`` [m²/s].
        source_radius: Gaussian source radius [m] (defaults to ``2·max(dx,…)``).
        advection_scheme: Reconstruction scheme (``"upwind1"`` is the robust,
            cheap default for inversion; WENO variants are available).
        receptor_index: Optional ``(n_obs, 2)`` ``(iy, ix)`` receptor cells.
        solver_kwargs: Overrides for the diffrax solve.
        bc_x: West and east boundary conditions, as
            :func:`~plumax.les_fvm.boundary.build_default_concentration_bc`
            takes them; ``(("dirichlet", c), "outflow")`` prescribes an inflow
            concentration c on the west face.
        bc_y: South and north boundary conditions, likewise.

    Returns:
        A configured :class:`EulerianForward4DVar`.
    """
    # Validate the time grid up front: t0/t1, SaveAt(ts=...) and jnp.interp all
    # assume ≥ 2 strictly-increasing save times. Catching it here gives a clear
    # error instead of an opaque diffrax / interpolation failure later.
    if (source_location is None) == (control is None):
        raise ValueError(
            "build_forward: pass exactly one of `source_location` and `control`."
        )
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

    import diffrax

    from plumax.les_fvm.simulate import _max_diffusivity, stable_step_bound

    # Reject an invalid diffusivity regardless of the solver / controller: a
    # negative (anti-diffusion) or non-finite K is unstable or guard-defeating
    # even on the adaptive path, so validate it before any branch.
    _k_h, _k_z = _max_diffusivity(eddy)

    # Stability guard for the fixed-step 4D-Var solve. The default path is
    # Heun + ConstantStepSize + dt0=1 (see `concentration_history`); an
    # over-large step silently blows up *inside* the optimisation loop, where
    # it is hardest to diagnose. The explicit CFL bound only applies to an
    # explicit solver taking a fixed step, so the guard runs only when: (a) the
    # controller is fixed — ConstantStepSize (largest step = dt0) or StepTo
    # (largest step = widest prescribed interval) — and (b) the solver is
    # explicit. Adaptive controllers are error-controlled and implicit solvers
    # are stable at these steps, so both are skipped.
    _kw = dict(solver_kwargs or {})
    _solver = _kw.get("solver", diffrax.Heun())
    _controller = _kw.get("stepsize_controller", diffrax.ConstantStepSize())
    _max_step: float | None = None
    if isinstance(_controller, diffrax.ConstantStepSize):
        _max_step = float(_kw.get("dt0", 1.0))
    elif isinstance(_controller, diffrax.StepTo):
        _ts = np.asarray(_controller.ts, dtype=float)
        _max_step = float(np.max(np.diff(_ts))) if _ts.size >= 2 else 0.0
    if _max_step is not None and not isinstance(
        _solver, diffrax.AbstractImplicitSolver
    ):
        _bound = stable_step_bound(
            dx=plume_grid.dx,
            dy=plume_grid.dy,
            dz=plume_grid.dz,
            max_u=abs(float(uniform_wind[0])),
            max_v=abs(float(uniform_wind[1])),
            max_w=abs(float(uniform_wind[2])),
            k_h=_k_h,
            k_z=_k_z,
        )
        if _max_step > _bound:
            raise ValueError(
                f"build_forward: fixed step {_max_step:g} s exceeds the stable "
                f"step ~{_bound:g} s for the fixed-step 4D-Var solve on this "
                f"grid+wind+K. Reduce the step (solver_kwargs 'dt0' or StepTo "
                f"'ts'), soften K, or pass an adaptive stepsize_controller."
            )

    horizontal_bc, vertical_bc = build_default_concentration_bc(
        bc_x=bc_x, bc_y=bc_y, bc_z=("neumann", "neumann")
    )
    save_times = jnp.asarray(save_times, dtype=plume_grid.x.dtype)

    nz, ny, nx = plume_grid.shape
    if control is not None:
        n_layers, *surface_shape = plume_grid.interior_shape
        _check_control_on_grid(control, tuple(surface_shape), save_times.shape[0])

        def source_factory(q_series: jax.Array) -> SurfaceFluxSource:
            flux_fn = control.flux_function(q_series, save_times)
            return SurfaceFluxSource(flux_fn, plume_grid.dz, n_layers)
    else:
        # Build the source *geometry* once, with a placeholder unit rate. This
        # runs the (eager) source-location bounds check and fixes the static
        # ``density`` profile outside the differentiated path; only the emission
        # rate ``q`` flows through the trace, so the per-call factory just swaps
        # ``emission_fn`` and never re-runs the host-side ``float(...)`` validation.
        base_source = make_gaussian_source(
            plume_grid=plume_grid,
            emission_rate=1.0,
            source_location=source_location,
            source_radius=source_radius,
        )

        def source_factory(q_series: jax.Array) -> GaussianSource:
            return GaussianSource(
                emission_fn=_interpolated_emission(q_series, save_times),
                density=base_source.density,
            )

    def rhs_factory(
        q_series: jax.Array, bc_scale: jax.Array | None = None
    ) -> EulerianDispersionRHS:
        source = source_factory(q_series)
        scaled_bc = (
            horizontal_bc
            if bc_scale is None
            else scale_dirichlet_faces(horizontal_bc, bc_scale)
        )
        return EulerianDispersionRHS(
            plume_grid=plume_grid,
            wind_field=wind_field,
            eddy_diffusivity=eddy,
            source=source,
            horizontal_bc=scaled_bc,
            vertical_bc=vertical_bc,
            advection_scheme=advection_scheme,
        )

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
        plume_grid=plume_grid,
        control=control,
    )


def _check_control_on_grid(
    control: SpatialControl, grid_shape: tuple[int, int], n_times: int
) -> None:
    if np.shape(control.mask) != grid_shape:
        raise ValueError(
            f"build_forward: `control.mask` must be (ny, nx) = {grid_shape}, got "
            f"{np.shape(control.mask)}."
        )
    if control.time_basis is not None and np.shape(control.time_basis)[0] != n_times:
        raise ValueError(
            f"build_forward: `control.time_basis` must have one row per save time "
            f"({n_times}), got {np.shape(control.time_basis)[0]}."
        )


# ── 4D-Var cost in whitened control space ────────────────────────────────────


class _StaticForward:
    """Identity dynamics: a source control does not evolve over the window."""

    dt = 1.0

    def step(self, state: jax.Array, dt: float) -> jax.Array:
        del dt
        return state


@dataclass(frozen=True)
class FourDVarProblem:
    """A strong-constraint 4D-Var problem over a flat source control.

    The control is a scalar emission time series or the per-cell flux of a
    :class:`SpatialControl`, optionally stacked with a :class:`BackgroundControl`;
    ``n_state`` below is its length. Holds the differentiable forward, the prior ``(S_b, B = L Lᵀ)``,
    and the observations ``(y_t, R_t)``. The cost is evaluated in whitened space
    ``χ = L⁻¹ (S − S_b)`` so the prior term is the identity ``½‖χ‖²`` and the
    Hessian is well-conditioned (design §control-transform). ``observations``
    has shape ``(n_t, n_obs)`` and ``obs_variance`` broadcasts to it.

    Build with :func:`build_problem`.
    """

    forward: EulerianForward4DVar
    prior_mean: jax.Array  # S_b, (n_state,)
    prior_chol: jax.Array  # L with B = L Lᵀ, (n_state, n_state)
    observations: jax.Array  # y, (n_t, n_obs)
    obs_variance: jax.Array  # R diagonal, broadcast to (n_t, n_obs)
    control: SpatialControl | None = None
    prior_sqrt_op: lx.AbstractLinearOperator | None = None

    @property
    def n_source(self) -> int:
        """Length of the source part of the flat control."""
        background = self.forward.background
        n_extra = 0 if background is None else background.size
        return self.prior_mean.shape[0] - n_extra

    def prior_sqrt_operator(self) -> lx.AbstractLinearOperator:
        """The prior square root L as an operator, structured where it can be."""
        if self.prior_sqrt_op is not None:
            return self.prior_sqrt_op
        return lx.MatrixLinearOperator(self.prior_chol)

    def source_from_whitened(self, chi: jax.Array) -> jax.Array:
        """Map whitened control ``χ`` back to the flat control, ``S_b + L χ``."""
        return self.prior_mean + self.prior_chol @ chi

    def cost(self, chi: jax.Array) -> jax.Array:
        """The 4D-Var cost ``J`` at whitened control ``χ`` (scalar).

        A diagnostic: the solve minimises the same cost inside
        ``vardax.ThreeDVar``, not through this method.
        """
        q = self.source_from_whitened(chi)
        predicted = self.forward.predict(q)  # (n_t, n_obs)
        residual = predicted - self.observations
        obs_term = 0.5 * jnp.sum(residual**2 / self.obs_variance)
        prior_term = 0.5 * jnp.sum(chi**2)
        return obs_term + prior_term

    def value_and_grad(self, chi: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``(J, ∂J/∂χ)`` — the gradient is the exact adjoint via reverse-mode AD."""
        return jax.value_and_grad(self.cost)(chi)

    def source_receptor_operator(self) -> vardax.CallableObs:
        """The whitened source–receptor map as a vardax observation operator.

        ``𝒢(S) = vec(forward.predict(S)) / σ`` on a ``(1, n_t)`` source row,
        so the observation-error covariance is the identity. The diffrax
        solve defines only a VJP, whose ``while_loop`` JAX cannot transpose,
        so the operator linearises by forming the Jacobian with one VJP per
        observation (``jac="bwd", materialise=True``), so the cost scales with
        the number of observations rather than the length of the control.

        Returns:
            A ``vardax.CallableObs`` ``(1, n_state) → (1, n_t·n_obs)``.

        Raises:
            ImportError: Without the ``assim`` extra.
        """
        vdx = require_vardax("FourDVarProblem.source_receptor_operator")
        sigma = jnp.sqrt(self.obs_variance)  # (n_t, n_obs)
        predict = self.forward.predict

        def whitened_prediction(source: jax.Array) -> jax.Array:
            # 𝒢(S) / σ, flattened over (time, receptor)
            return einx.rearrange("t o -> 1 (t o)", predict(source[0]) / sigma)

        return vdx.CallableObs(fn=whitened_prediction, jac="bwd", materialise=True)

    def incremental_analysis(
        self,
        *,
        outer_iters: int = 5,
        inner_iters: int = 50,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> tuple[vardax.IncrementalFourDVar, vardax.Batch1D]:
        """The Gauss–Newton analysis of this problem, and its one-window batch.

        ``vardax.IncrementalFourDVar`` with an identity forward over a
        one-time window, because the control is a source rather than an initial
        state, over the whitened source–receptor map with ``R = I``. Each outer
        iteration forms the Jacobian with one VJP per observation
        (``jac="bwd"``) and solves I + (J̃ L)ᵀ (J̃ L) in the control-variable
        transform by CG.

        Args:
            outer_iters: Gauss–Newton outer iterations.
            inner_iters: CG iterations per outer iteration.
            rtol: Relative tolerance of the CG solves.
            atol: Absolute tolerance of the CG solves.

        Returns:
            ``(model, batch)``: ``model(batch)`` is the MAP flat control, shape
            ``(1, n_state)``.

        Raises:
            ImportError: Without the ``assim`` extra.
        """
        vdx = require_vardax("FourDVarProblem.incremental_analysis")
        sigma = jnp.sqrt(self.obs_variance)
        predict = self.forward.predict

        def whitened_prediction(state: jax.Array) -> jax.Array:
            # 𝒢(q) / σ, flattened over (time, receptor)
            return einx.rearrange("t o -> (t o)", predict(state) / sigma)

        observations = self.observations / sigma
        n_flat = observations.size
        dtype = self.prior_mean.dtype
        prior_sqrt = self.prior_sqrt_operator()
        model = vdx.IncrementalFourDVar(
            forward=_StaticForward(),
            obs_op=whitened_prediction,
            prior_mean=self.prior_mean,
            prior_cov_op=prior_sqrt @ prior_sqrt.T,
            obs_cov_op=lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((n_flat,), dtype)
            ),
            prior_sqrt_op=prior_sqrt,
            config=vdx.IncrementalConfig(
                n_outer=outer_iters,
                n_inner=inner_iters,
                cg_rtol=rtol,
                cg_atol=atol,
                covariance_tol=atol,
                jac="bwd",
            ),
        )
        batch = vdx.Batch1D(
            input=einx.rearrange("t o -> 1 1 (t o)", observations),
            mask=jnp.ones((1, 1, n_flat), dtype=dtype),
        )
        return model, batch

    def analysis(
        self,
        *,
        minimiser: optx.AbstractMinimiser | None = None,
        max_steps: int = 100,
        initial_whitened: jax.Array | None = None,
    ) -> tuple[vardax.ThreeDVar, vardax.Batch1D]:
        """The 3D-Var analysis of this problem, and its one-window batch.

        ``vardax.ThreeDVar`` over :meth:`source_receptor_operator`, with the
        whitened control ``S = S_b + L χ``, ``½‖χ‖²``, as a
        ``vardax.LinearParameterization`` and ``R = I`` on the whitened
        observations ``y / σ``.

        Args:
            minimiser: An optimistix minimiser. Default
                ``optx.LBFGS(rtol=1e-6, atol=1e-8)``.
            max_steps: Iteration cap.
            initial_whitened: First guess ``χ₀``. Default ``0`` (the prior
                mean).

        Returns:
            ``(model, batch)``: ``model(batch)`` is the MAP source, shape
            ``(1, 1, n_state)``, ``model.analyse(batch)`` adds the minimiser's
            diagnostics, and ``model.posterior(batch)`` is its Laplace
            posterior.

        Raises:
            ImportError: Without the ``assim`` extra.
        """
        import optimistix as optx

        vdx = require_vardax("FourDVarProblem.analysis")
        n_state = self.prior_mean.shape[0]
        dtype = self.prior_mean.dtype
        basis = vdx.linear_basis(self.prior_chol, variance=1.0)
        if initial_whitened is None:
            control = vdx.LinearParameterization(
                basis, background=self.prior_mean[None]
            )
        else:
            control = _warm_start_control()(
                basis,
                background=self.prior_mean[None],
                start=jnp.asarray(initial_whitened),
            )
        observations = self.observations / jnp.sqrt(self.obs_variance)
        n_flat = observations.size

        def row_operator(op: lx.AbstractLinearOperator, n: int):
            return lx.FunctionLinearOperator(
                lambda x: op.mv(x[0])[None],
                jax.ShapeDtypeStruct((1, n), dtype),
                frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
            )

        psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        model = vdx.ThreeDVar(
            obs_op=self.source_receptor_operator(),
            prior_mean=self.prior_mean[None],
            # unused by the solve and the posterior (the control's prior is
            # ½‖χ‖²); B = L Lᵀ for consistency
            prior_cov_op=row_operator(
                lx.MatrixLinearOperator(self.prior_chol @ self.prior_chol.T, psd),
                n_state,
            ),
            obs_cov_op=lx.IdentityLinearOperator(
                jax.ShapeDtypeStruct((1, n_flat), dtype)
            ),
            minimiser=vdx.OptimistixMinimiser(
                minimiser or optx.LBFGS(rtol=1e-6, atol=1e-8), max_steps=max_steps
            ),
            parameterization=control,
            cov_solver=lx.AutoLinearSolver(well_posed=True),
        )
        batch = vdx.Batch1D(
            input=einx.rearrange("t o -> 1 1 (t o)", observations),
            mask=jnp.ones((1, 1, n_flat), dtype=dtype),
        )
        return model, batch


def build_problem(
    *,
    forward: EulerianForward4DVar,
    observations: jax.Array,
    prior_mean: jax.Array | None = None,
    prior_covariance: jax.Array | None = None,
    obs_variance: jax.Array | float,
    control: SpatialControl | TemporalControl | None = None,
    background: BackgroundControl | None = None,
) -> FourDVarProblem:
    """Assemble a :class:`FourDVarProblem`, factorising the prior covariance.

    Args:
        forward: The differentiable transport forward.
        observations: Column observations ``y``, shape ``(n_t, n_obs)``.
        prior_mean: Source prior mean ``S_b``, shape ``(n_t,)``, for the emission
            time series. Give either this and ``prior_covariance``, or ``control``.
        prior_covariance: Source prior covariance ``B``, shape ``(n_t, n_t)``
            (e.g. a Matérn-3/2 temporal prior from
            :func:`plumax.lagrangian.inversion.matern32_covariance`). Cholesky
            -factorised once here.
        obs_variance: Observation-error variance ``R`` (scalar or broadcastable
            to ``observations``); ``R = R_retr + R_repr``.
        control: The control and its prior: the same :class:`SpatialControl`
            that ``forward`` was built with, or a :class:`TemporalControl`.
        background: Adds the initial concentration and the per-face boundary
            scaling to the control, with a block-diagonal prior. ``None`` pins
            both, and leaves the problem as it was without this argument.

    Returns:
        The configured :class:`FourDVarProblem`.
    """
    if isinstance(control, SpatialControl):
        problem = _build_spatial_problem(
            forward, observations, prior_mean, prior_covariance, obs_variance, control
        )
    else:
        problem = _build_temporal_problem(
            forward, observations, prior_mean, prior_covariance, obs_variance, control
        )
    return problem if background is None else _with_background(problem, background)


def _with_background(
    problem: FourDVarProblem, background: BackgroundControl
) -> FourDVarProblem:
    """Stack (δc₀, α) onto a problem's control, with a block-diagonal prior."""
    grid = problem.forward.plume_grid
    if grid is None:
        raise ValueError("build_problem: `background` needs a forward with its grid.")
    if np.shape(background.c_b) != grid.interior_shape:
        raise ValueError(
            f"build_problem: `background.c_b` must be {grid.interior_shape}, got "
            f"{np.shape(background.c_b)}."
        )
    covariance_c = background.B_c.as_matrix()
    # L = blockdiag(L_S, L_c, σ_α I), so B = blockdiag(B_S, B_c, σ_α² I)
    prior_sqrt = gx.BlockDiag(
        problem.prior_sqrt_operator(),
        lx.MatrixLinearOperator(_cholesky(covariance_c)),
        lx.DiagonalLinearOperator(jnp.full(_N_FACES, background.bc_scale_prior[1])),
    )
    return dataclasses.replace(
        problem,
        forward=problem.forward.with_background(background),
        prior_mean=jnp.concatenate([problem.prior_mean, background.flat_prior_mean()]),
        prior_chol=prior_sqrt.as_matrix(),
        prior_sqrt_op=prior_sqrt,
    )


def _build_temporal_problem(
    forward: EulerianForward4DVar,
    observations: jax.Array,
    prior_mean: jax.Array | None,
    prior_covariance: jax.Array | None,
    obs_variance: jax.Array | float,
    control: TemporalControl | None,
) -> FourDVarProblem:
    """:func:`build_problem` for the emission time series."""
    if forward.control is not None:
        raise ValueError(
            "build_problem: `forward` has a spatial control; pass the same "
            "SpatialControl as `control`."
        )
    if isinstance(control, TemporalControl):
        if prior_mean is not None or prior_covariance is not None:
            raise ValueError(
                "build_problem: pass either `control` or `prior_mean` and "
                "`prior_covariance`, not both."
            )
        prior_mean = control.prior_mean
        prior_covariance = control.prior_covariance
    if prior_mean is None or prior_covariance is None:
        raise ValueError(
            "build_problem: pass `control`, or both `prior_mean` and "
            "`prior_covariance`."
        )
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
    # Reject non-finite inputs: a NaN would flow through the cost / posterior as
    # a finite-`source` solve with a NaN objective and covariance (the source-
    # only divergence check in `solve_4dvar` cannot catch that).
    for _name, _arr in (
        ("prior_mean", sb),
        ("prior_covariance", b),
        ("observations", y),
    ):
        if not bool(jnp.all(jnp.isfinite(_arr))):
            raise ValueError(
                f"build_problem: `{_name}` must be finite (contains NaN/inf)."
            )
    # Adaptive-jitter Cholesky: gx.safe_cholesky escalates the jitter on failure
    # (JIT-compatible), succeeding for badly-scaled Matérn priors where a fixed
    # 1e-9 kick would not.
    chol = gx.safe_cholesky(
        lx.MatrixLinearOperator(
            b, tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
        )
    )

    return FourDVarProblem(
        forward=forward,
        prior_mean=sb,
        prior_chol=chol,
        observations=y,
        obs_variance=_resolve_obs_variance(obs_variance, y),
    )


def _build_spatial_problem(
    forward: EulerianForward4DVar,
    observations: jax.Array,
    prior_mean: jax.Array | None,
    prior_covariance: jax.Array | None,
    obs_variance: jax.Array | float,
    control: SpatialControl,
) -> FourDVarProblem:
    """:func:`build_problem` for a per-cell flux control."""
    if prior_mean is not None or prior_covariance is not None:
        raise ValueError(
            "build_problem: pass either `control` or `prior_mean` and "
            "`prior_covariance`, not both."
        )
    if forward.control is not control or forward.plume_grid is None:
        raise ValueError(
            "build_problem: `forward` must be built with this same SpatialControl."
        )
    y = jnp.asarray(observations, dtype=float)
    n_t = forward.save_times.shape[0]
    if y.ndim != 2 or y.shape[0] != n_t:
        raise ValueError(
            f"build_problem: `observations` must be (n_t={n_t}, n_obs), got {y.shape}."
        )
    if not bool(jnp.all(jnp.isfinite(y))):
        raise ValueError("build_problem: `observations` must be finite (NaN/inf).")
    prior_sqrt = control.prior_sqrt(forward.plume_grid.y, forward.plume_grid.x)
    return FourDVarProblem(
        forward=forward,
        prior_mean=control.flat_prior_mean(),
        prior_chol=prior_sqrt.as_matrix(),
        observations=y,
        obs_variance=_resolve_obs_variance(obs_variance, y),
        control=control,
        prior_sqrt_op=prior_sqrt,
    )


def _resolve_obs_variance(obs_variance: jax.Array | float, y: jax.Array) -> jax.Array:
    """``R``, the ``(n_t, n_obs)`` diagonal, from any accepted ``obs_variance``."""
    n_t, n_obs = y.shape
    # Resolve obs_variance to R, the (n_t, n_obs) diagonal. A length-n_t 1-D
    # input is treated as one variance *per overpass* (R_t), so reshape it to a
    # column before broadcasting — otherwise numpy would align it with the
    # trailing n_obs axis. A length-n_obs 1-D input stays per-receptor, and a
    # scalar broadcasts to everything. When n_t == n_obs a bare 1-D vector is
    # genuinely ambiguous between the two, so reject it and require an explicit
    # 2-D shape rather than silently picking one and changing the objective.
    r_in = np.asarray(obs_variance, dtype=float)
    if r_in.ndim == 1 and n_t == n_obs and r_in.shape[0] == n_t:
        raise ValueError(
            f"build_problem: `obs_variance` of length {n_t} is ambiguous when "
            f"n_t == n_obs == {n_t} (per-overpass vs per-receptor). Pass an "
            f"explicit ({n_t}, 1) column for per-overpass R_t, a (1, {n_obs}) "
            f"row for per-receptor, or a full ({n_t}, {n_obs}) array."
        )
    if r_in.ndim == 1 and r_in.shape[0] == n_t:
        r_in = r_in[:, None]
    if not np.all(np.isfinite(r_in)) or np.any(r_in <= 0.0):
        # NaN passes ``<= 0`` (every comparison is False), so check finiteness
        # explicitly — a NaN R would give a NaN cost and posterior.
        raise ValueError(
            "build_problem: `obs_variance` entries must be finite and > 0."
        )
    try:
        return jnp.broadcast_to(jnp.asarray(r_in), y.shape)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"build_problem: `obs_variance` shape {np.shape(obs_variance)} is not "
            f"broadcastable to observations {y.shape}."
        ) from exc


@dataclass(frozen=True)
class BackgroundAnalysis:
    """The MAP of the :class:`BackgroundControl` part of a 4D-Var control.

    Attributes:
        initial_concentration: c₀* = c_b + δc₀* on the interior [kg m⁻³], shape
            ``(nz, ny, nx)``.
        bc_scale: Per-face factors α*, ordered (west, east, south, north).
    """

    initial_concentration: jax.Array
    bc_scale: jax.Array


@dataclass(frozen=True)
class FourDVarResult:
    """Outcome of a 4D-Var minimisation.

    Attributes:
        source: MAP emission time series ``S* = S_b + L χ*``, shape ``(n_t,)``.
        whitened: Optimal whitened control ``χ*``, of the whole flat control.
        cost: Final cost value ``J(χ*)``.
        n_steps: Optimiser iterations consumed (``-1`` if not reported).
        posterior: Optional :class:`PosteriorCovariance` around the MAP, attached
            when :func:`solve_4dvar` is called with ``compute_posterior=True``.
        converged: optimistix's strict success flag (``result ==
            successful``), as ``vardax.MinimiseResult.converged`` reports
            it. Appended after ``posterior`` to
            preserve the positional constructor contract. L-BFGS on these
            stiff problems frequently exhausts ``max_steps`` while still
            returning a good, finite MAP, so ``converged is False`` with a
            finite ``source`` is common and not by itself alarming. The
            genuine failure to watch for is a non-finite ``source`` (a
            diverged solve), which :func:`solve_4dvar` warns about loudly.
        prior_mean: Source prior mean ``S_b`` the solve used, shape ``(n_t,)``
            (``None`` when built by hand).
        prior_covariance: Source prior covariance ``B = L Lᵀ``, shape
            ``(n_t, n_t)`` (``None`` when built by hand).
        control: The :class:`SpatialControl` of a per-cell flux solve, whose
            ``source`` is a flat vector of length ``control.size``, mapped by
            :attr:`source_field`. ``None`` for the emission time series.
        background: MAP of the initial concentration and the boundary scaling,
            when the problem has a :class:`BackgroundControl`.

    For an emission time series, the result also carries the Tier V per-event
    payload (``population.PerEventPosterior``) for the event emission rate, the
    time mean Q = wᵀS with w = 1/n_t: its Laplace posterior N(wᵀS*, wᵀP_S w) and
    its prior N(wᵀS_b, wᵀBw) are exact images of the Gaussian source laws. That
    time mean means nothing over cells, so a per-cell result raises
    ``PayloadError`` there.
    """

    source: jax.Array
    whitened: jax.Array
    cost: float
    n_steps: int
    posterior: PosteriorCovariance | None = None
    converged: bool = True
    prior_mean: jax.Array | None = None
    prior_covariance: jax.Array | None = None
    control: SpatialControl | None = None
    background: BackgroundAnalysis | None = None

    @property
    def source_field(self) -> jax.Array:
        """MAP flux coefficient fields [kg m⁻² s⁻¹], shape ``(K, ny, nx)``.

        Raises:
            PayloadError: For an emission time series.
        """
        if self.control is None:
            raise PayloadError(
                "FourDVarResult.source_field needs a SpatialControl solve."
            )
        return self.control.to_field(self.source)

    def _require_time_series(self) -> None:
        if self.control is not None:
            raise PayloadError(
                "FourDVarResult: the Tier V payload is the time mean of an emission "
                "time series; this result is a per-cell flux control."
            )

    @property
    def emission_rate(self) -> jax.Array:
        """Time-mean MAP emission rate Q* = wᵀS*."""
        self._require_time_series()
        return jnp.mean(self.source)

    @property
    def emission_std(self) -> jax.Array:
        """Laplace posterior sd of Q, √(wᵀP_S w); NaN without the posterior."""
        self._require_time_series()
        if self.posterior is None:
            return jnp.full_like(self.emission_rate, jnp.nan)
        return self._mean_law(self.source, self.posterior.source_covariance).std

    def samples(self, key: jax.Array, n: int) -> jax.Array:
        """n draws of Q from the Laplace posterior N(wᵀS*, wᵀP_S w).

        The same law as averaging :func:`laplace_sample` source draws over
        time, without re-solving for the covariance.

        Raises:
            PayloadError: Without the posterior (``compute_posterior=False``), or
                for a per-cell flux control.
        """
        self._require_time_series()
        if self.posterior is None:
            raise PayloadError(
                "FourDVarResult.samples needs the Laplace posterior; solve with "
                "compute_posterior=True."
            )
        law = self._mean_law(self.source, self.posterior.source_covariance)
        return law.sample(key, n)

    def prior_logpdf(self, q: jax.Array) -> jax.Array:
        """log π(Q) of the time-mean source prior, N(wᵀS_b, wᵀBw).

        Raises:
            PayloadError: If the result was built without its prior, or for a
                per-cell flux control.
        """
        self._require_time_series()
        if self.prior_mean is None or self.prior_covariance is None:
            raise PayloadError(
                "FourDVarResult.prior_logpdf: this result carries no prior; "
                "build it with solve_4dvar."
            )
        return self._mean_law(self.prior_mean, self.prior_covariance).log_prob(q)

    @staticmethod
    def _mean_law(mean: jax.Array, covariance: jax.Array) -> GaussianEmission:
        # w = 1/n_t, so Q = wᵀS is the time mean
        weights = jnp.full_like(mean, 1.0 / mean.shape[0])
        return gaussian_total(mean, covariance, weights)


def solve_4dvar(
    problem: FourDVarProblem,
    *,
    method: Literal["lbfgs", "gauss_newton"] = "lbfgs",
    initial_source: jax.Array | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 100,
    solver: optx.AbstractMinimiser | None = None,
    compute_posterior: bool = False,
    outer_iters: int = 5,
    inner_iters: int = 50,
) -> FourDVarResult:
    """Minimise the 4D-Var cost in whitened space, via vardax.

    ``method="lbfgs"`` runs :meth:`FourDVarProblem.analysis`:
    ``vardax.ThreeDVar`` over the whitened control ``χ``, so the prior is the
    identity Gaussian and the gradient is the exact discrete adjoint of the FV
    solver.

    ``method="gauss_newton"`` runs :meth:`FourDVarProblem.incremental_analysis`:
    ``vardax.IncrementalFourDVar``, Gauss–Newton outer iterations with CG inner
    iterations over the same whitened cost. The forward is affine in the
    control, so the linearised cost is the cost itself and one outer iteration
    with a converged CG solve reaches the MAP. Each outer iteration forms the
    whole Jacobian, one VJP per observation, so it costs more than L-BFGS when
    the observations are many and the problem is well conditioned.

    The observation-space dual (PSAS) is not offered: it belongs to vardax
    (jejjohnson/vardax#72).

    Args:
        problem: The assembled 4D-Var problem.
        method: The minimiser.
        initial_source: Optional first-guess emission signal ``S₀`` (defaults to
            the prior mean, i.e. ``χ₀ = 0``), for L-BFGS only:
            ``IncrementalFourDVar`` always starts from the prior mean.
        rtol: Relative tolerance of the optimiser, or of the CG solves.
        atol: Absolute tolerance of the optimiser, or of the CG solves.
        max_steps: Maximum L-BFGS iterations.
        solver: Optional optimistix minimiser (defaults to ``optx.LBFGS``), for
            L-BFGS only.
        compute_posterior: When ``True``, also evaluate the Gauss-Newton Laplace
            :func:`posterior_covariance` at the MAP and attach it to the result.
        outer_iters: Gauss–Newton outer iterations.
        inner_iters: CG iterations per Gauss–Newton outer iteration.

    Returns:
        The :class:`FourDVarResult` with the MAP source and diagnostics.
        ``n_steps`` counts L-BFGS iterations or Gauss–Newton outer iterations.
        ``converged`` is optimistix's verdict for L-BFGS, and for Gauss–Newton
        whether ‖∇𝒥(χ*)‖ ≤ atol + rtol ‖∇𝒥(0)‖, because vardax reports none.

    Raises:
        ValueError: For an unknown ``method``, ``"psas"`` included, or an option
            the method cannot honour.
        ImportError: Without the ``assim`` extra (``pip install
            'plumax[assim]'``).
    """
    if method == "gauss_newton":
        if initial_source is not None or solver is not None:
            raise ValueError(
                "solve_4dvar: `initial_source` and `solver` apply to L-BFGS only; "
                "vardax.IncrementalFourDVar starts from the prior mean and solves "
                "with CG."
            )
        flat_map, chi_star, cost, n_steps, converged = _solve_gauss_newton(
            problem, outer_iters, inner_iters, rtol, atol
        )
    elif method == "lbfgs":
        flat_map, chi_star, cost, n_steps, converged = _solve_lbfgs(
            problem, initial_source, rtol, atol, max_steps, solver
        )
    elif method == "psas":
        raise ValueError(
            "solve_4dvar: dual PSAS belongs to vardax and is not available yet "
            "(https://github.com/jejjohnson/vardax/issues/72)."
        )
    else:
        raise ValueError(f"solve_4dvar: unknown method {method!r}.")
    finite = bool(jnp.all(jnp.isfinite(flat_map)))
    # `throw=False` hides the optimiser outcome, so surface optimistix's own
    # verdict (`result == successful`). L-BFGS on these stiff transport
    # problems often exhausts `max_steps` while still returning a good, finite
    # MAP, so a bare `converged is False` is common and not alarming on its
    # own; a non-finite MAP is the hazard to flag.
    if not finite:
        warnings.warn(
            "solve_4dvar: the optimiser returned a non-finite source; the "
            "solve diverged. The posterior is skipped (it would be NaN). Check "
            "the prior conditioning, `obs_variance`, and `initial_source`.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Skip the posterior on a non-finite MAP: the Gauss-Newton precision at a
    # NaN control would attach a NaN covariance.
    posterior = (
        _laplace_posterior(problem, flat_map) if compute_posterior and finite else None
    )
    n_source = problem.n_source
    source_chol = problem.prior_chol[:n_source, :n_source]
    return FourDVarResult(
        source=flat_map[:n_source],
        whitened=chi_star,
        cost=cost,
        n_steps=n_steps,
        converged=converged,
        posterior=posterior,
        prior_mean=problem.prior_mean[:n_source],
        prior_covariance=source_chol @ source_chol.T,
        control=problem.control,
        background=_background_analysis(problem, flat_map),
    )


def _background_analysis(
    problem: FourDVarProblem, flat_map: jax.Array
) -> BackgroundAnalysis | None:
    background = problem.forward.background
    if background is None:
        return None
    _, increment, bc_scale = background.split(flat_map)
    # c₀* = c_b + δc₀*
    initial_concentration = jnp.asarray(background.c_b, increment.dtype) + increment
    return BackgroundAnalysis(initial_concentration, bc_scale)


def _solve_lbfgs(
    problem: FourDVarProblem,
    initial_source: jax.Array | None,
    rtol: float,
    atol: float,
    max_steps: int,
    solver: optx.AbstractMinimiser | None,
) -> tuple[jax.Array, jax.Array, float, int, bool]:
    """L-BFGS through ``vardax.ThreeDVar``: (q*, χ*, cost, steps, converged)."""
    import optimistix as optx

    chi0 = None
    if initial_source is not None:
        s0 = jnp.asarray(initial_source, dtype=float)
        # χ₀ = L⁻¹ (S₀ − S_b) — solve the triangular system rather than invert.
        n_source = problem.n_source
        first_guess = problem.prior_mean.at[:n_source].set(s0)
        chi0 = jax.scipy.linalg.solve_triangular(
            problem.prior_chol, first_guess - problem.prior_mean, lower=True
        )
    model, batch = problem.analysis(
        minimiser=solver or optx.LBFGS(rtol=rtol, atol=atol),
        max_steps=max_steps,
        initial_whitened=chi0,
    )
    analysis, outcome = model.analyse(batch)
    return (
        analysis[0, 0],
        outcome.value[0],
        float(outcome.cost[0]),
        int(outcome.num_steps[0]),
        bool(outcome.converged[0]),
    )


def _solve_gauss_newton(
    problem: FourDVarProblem,
    outer_iters: int,
    inner_iters: int,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array, float, int, bool]:
    """Gauss–Newton through ``IncrementalFourDVar``: (q*, χ*, cost, steps, converged)."""
    model, batch = problem.incremental_analysis(
        outer_iters=outer_iters, inner_iters=inner_iters, rtol=rtol, atol=atol
    )
    flat_map = model(batch)[0]
    # χ* = L⁻¹ (q* − μ_q): vardax returns the state, not its control
    chi_star = jax.scipy.linalg.solve_triangular(
        problem.prior_chol, flat_map - problem.prior_mean, lower=True
    )
    cost, gradient = problem.value_and_grad(chi_star)
    _, initial_gradient = problem.value_and_grad(jnp.zeros_like(chi_star))
    # ‖∇𝒥(χ*)‖ ≤ atol + rtol ‖∇𝒥(0)‖
    threshold = atol + rtol * jnp.linalg.norm(initial_gradient)
    converged = bool(jnp.linalg.norm(gradient) <= threshold)
    return flat_map, chi_star, float(cost), outer_iters, converged


@functools.cache
def _warm_start_control() -> type:
    """``vardax.LinearParameterization`` that starts from a given ``χ₀``.

    Built on first use, since vardax is optional. The subclass keeps
    ``isinstance(..., LinearParameterization)``, which vardax's Laplace
    covariance relies on.
    """
    vdx = require_vardax("solve_4dvar")

    class WarmStartControl(vdx.LinearParameterization):
        start: jax.Array

        def init_control(self, *, key: jax.Array | None = None) -> jax.Array:
            del key
            return self.start.astype(self.background.dtype)

    return WarmStartControl


# ── posterior covariance (Gauss-Newton Laplace approximation) ─────────────────


@dataclass(frozen=True)
class PosteriorCovariance:
    """Laplace (Gauss-Newton) posterior covariance of a 4D-Var solution.

    A Gaussian approximation ``N(S*, P_S)`` to the posterior, built from the
    Gauss-Newton Hessian of the whitened-space cost at the MAP ``χ*``
    (design §posterior-covariance). The whitened-space Hessian is

        H_GN = I + J̃ᵀ R⁻¹ J̃,   J̃ = ∂(predicted_obs)/∂χ,

    so the whitened posterior covariance is ``P_χ = H_GN⁻¹`` and — because
    ``S = S_b + L χ`` — the source-space covariance is ``P_S = L P_χ Lᵀ``. With
    a :class:`BackgroundControl`, ``P_S`` is the source block of the covariance
    of the whole flat control.

    Attributes:
        whitened_covariance: ``P_χ``, shape ``(n_state, n_state)``.
        source_covariance: ``P_S``, shape ``(n_source, n_source)``.
        control: The :class:`SpatialControl` of a per-cell flux problem.
        control_covariance: Covariance of the whole flat control, shape
            ``(n_state, n_state)``, when it has a :class:`BackgroundControl`.
    """

    whitened_covariance: jax.Array  # P_χ, (n_state, n_state)
    source_covariance: jax.Array  # P_S, (n_source, n_source)
    control: SpatialControl | None = None
    control_covariance: jax.Array | None = None

    @property
    def source_std(self) -> jax.Array:
        """Marginal source-space posterior standard deviations ``√diag(P_S)``."""
        return jnp.sqrt(jnp.diagonal(self.source_covariance))

    @property
    def bc_scale_std(self) -> jax.Array:
        """Posterior sd of the per-face factors, ordered (west, east, south, north).

        Raises:
            PayloadError: Without a :class:`BackgroundControl`.
        """
        if self.control_covariance is None:
            raise PayloadError(
                "PosteriorCovariance.bc_scale_std needs a BackgroundControl problem."
            )
        return jnp.sqrt(jnp.diagonal(self.control_covariance)[-_N_FACES:])

    @property
    def std_field(self) -> jax.Array:
        """Marginal posterior sd as fields [kg m⁻² s⁻¹], shape ``(K, ny, nx)``.

        Raises:
            PayloadError: For an emission time series.
        """
        if self.control is None:
            raise PayloadError(
                "PosteriorCovariance.std_field needs a SpatialControl problem."
            )
        return self.control.to_field(self.source_std)


def _vardax_laplace(problem: FourDVarProblem, source: jax.Array) -> vardax.Posterior:
    """``vardax.Posterior`` of the Gauss–Newton Laplace approximation at ``source``.

    ``vardax.LaplaceCovariance`` on the linear control, so ``mean = S*`` and
    ``cov = P_S = L (I + J̃ᵀ J̃)⁻¹ Lᵀ`` with ``J̃ = ∂(𝒢/σ)/∂χ``, both on the
    ``(1, n_t)`` analysis state.
    """
    vdx = require_vardax("posterior_covariance")
    model, batch = problem.analysis()
    return vdx.LaplaceCovariance(solver=lx.AutoLinearSolver(well_posed=True))(
        source[None], model, batch
    )


def _laplace_posterior(
    problem: FourDVarProblem, source: jax.Array
) -> PosteriorCovariance:
    """vardax's Gauss–Newton Laplace covariance at the MAP source.

    ``P_S`` is materialised column by column from the vardax posterior;
    ``P_χ = L⁻¹ P_S L⁻ᵀ`` by two triangular solves.
    """
    n_state = source.shape[0]
    posterior = _vardax_laplace(problem, source)
    p_source = jax.vmap(lambda e: posterior.cov.mv(e[None])[0])(
        jnp.eye(n_state, dtype=source.dtype)
    )
    p_source = gx.symmetrize(p_source)
    chol = problem.prior_chol
    half = jax.scipy.linalg.solve_triangular(chol, p_source, lower=True)
    p_chi = jax.scipy.linalg.solve_triangular(chol, half.T, lower=True)
    n_source = problem.n_source
    has_background = problem.forward.background is not None
    return PosteriorCovariance(
        whitened_covariance=gx.symmetrize(p_chi),
        source_covariance=p_source[:n_source, :n_source],
        control=problem.control,
        control_covariance=p_source if has_background else None,
    )


def posterior_covariance(
    problem: FourDVarProblem, whitened_map: jax.Array
) -> PosteriorCovariance:
    """Gauss-Newton Laplace posterior covariance around a 4D-Var MAP.

    vardax's ``LaplaceCovariance`` of the whitened control at
    ``whitened_map``: the Gauss–Newton precision ``I + J̃ᵀ R⁻¹ J̃`` in ``χ``,
    mapped to the source as ``P_S = L P_χ Lᵀ`` for the prior Cholesky factor
    ``L``.

    Args:
        problem: The assembled 4D-Var problem.
        whitened_map: The MAP whitened control ``χ*`` (e.g.
            :attr:`FourDVarResult.whitened`), shape ``(n_state,)``.

    Returns:
        The :class:`PosteriorCovariance` holding ``P_χ`` and ``P_S``.

    Raises:
        ImportError: Without the ``assim`` extra.
    """
    chi = jnp.asarray(whitened_map, dtype=float)
    return _laplace_posterior(problem, problem.source_from_whitened(chi))


def laplace_sample(
    problem: FourDVarProblem,
    whitened_map: jax.Array,
    key: jax.Array,
    n_samples: int,
) -> jax.Array:
    """Draw source-space samples from the Laplace posterior ``N(S*, P_S)``.

    The draws come from ``vardax.Posterior.sample`` on the Gauss–Newton
    Laplace posterior at the MAP source, ``S = S* + P_S^½ ξ`` with
    ``ξ ~ N(0, I)``; ``P_S`` is the covariance of :func:`posterior_covariance`.

    Args:
        problem: The assembled 4D-Var problem.
        whitened_map: The MAP whitened control ``χ*``, shape ``(n_state,)``.
        key: A ``jax.random`` PRNG key.
        n_samples: Number of posterior draws to return.

    Returns:
        Source-space samples, shape ``(n_samples, n_t)``, or the flux fields
        ``(n_samples, K, ny, nx)`` of a :class:`SpatialControl` problem.
    """
    if not _is_traced(n_samples) and int(n_samples) <= 0:
        raise ValueError("laplace_sample: `n_samples` must be a positive integer.")
    chi_star = jnp.asarray(whitened_map, dtype=float)
    source = problem.source_from_whitened(chi_star)
    posterior = _vardax_laplace(problem, source)
    # S⁽ᵉ⁾ = S* + P_S^½ ξ⁽ᵉ⁾, ξ⁽ᵉ⁾ ~ N(0, I), on the (1, n_state) analysis state
    draws = posterior.sample(key, int(n_samples))
    source_draws = draws[:, 0, : problem.n_source]
    if problem.control is not None:
        return problem.control.to_field(source_draws)
    return source_draws


__all__ = [
    "BackgroundAnalysis",
    "BackgroundControl",
    "ColumnObservationOperator",
    "EulerianForward4DVar",
    "FourDVarProblem",
    "FourDVarResult",
    "PosteriorCovariance",
    "SpatialControl",
    "TemporalControl",
    "build_forward",
    "build_problem",
    "laplace_sample",
    "posterior_covariance",
    "solve_4dvar",
]
