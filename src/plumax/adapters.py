"""Data-assimilation adapters for the ``pipekit_cycle`` protocols.

``pipekit_cycle`` decomposes data assimilation into three runtime-checkable
protocols — ``ForwardModel`` (predict), ``ObservationOperator`` (compare),
``AnalysisStep`` (update). plumax ships adapters that satisfy them
**structurally**: the classes below expose exactly the right methods /
properties, so ``isinstance(obj, ForwardModel)`` succeeds, *without* plumax
importing ``pipekit_cycle`` at runtime. This keeps the dependency arrow
pointing the right way — assimilation drivers depend on plumax, not the
reverse.

- :class:`EulerianForwardModel` — wraps an
  :class:`~plumax.les_fvm.EulerianDispersionRHS` as a ``ForwardModel``,
  advancing a 3-D concentration field one explicit-Euler step per call.
- :class:`RadtranObservationOperator` — wraps the normalised Beer–Lambert
  forward model as an ``ObservationOperator`` mapping a scalar plume
  enhancement (ΔVMR) to a normalised radiance spectrum, with the analytic
  Jacobian as its tangent-linear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from plumax.radtran.forward import forward_nonlinear_normalized


if TYPE_CHECKING:
    import xarray as xr

    from plumax.les_fvm import EulerianDispersionRHS


class EulerianForwardModel:
    """Expose an Eulerian dispersion RHS as a ``pipekit_cycle.ForwardModel``.

    The adapter advances a tracer field with a single explicit-Euler step,
    ``c ← c + dt · RHS(t0, c)``. It is the minimal forward operator a cycling
    driver needs; richer time integration (diffrax) is available through
    :func:`~plumax.les_fvm.simulate_eulerian_dispersion`.

    Args:
        rhs: The tendency module ``RHS(t, concentration) -> dC/dt``.
        dt: Default integration step [s].
        t0: Time at which the (autonomous-in-practice) RHS is evaluated [s].
    """

    def __init__(self, rhs: EulerianDispersionRHS, dt: float, t0: float = 0.0) -> None:
        self.rhs = rhs
        self._dt = dt
        self.t0 = t0

    def step(self, state: Any, dt: float) -> Any:
        """Advance ``state`` by ``dt`` with one explicit-Euler step."""
        tendency = self.rhs(jnp.asarray(self.t0, dtype=jnp.asarray(state).dtype), state)
        return state + dt * tendency

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_signature(self) -> None:
        return None


class RadtranObservationOperator:
    """Expose the normalised Beer–Lambert forward as an ``ObservationOperator``.

    ``H(ΔVMR)`` answers "what normalised radiance spectrum would a plume
    enhancement of ``ΔVMR`` produce?". The innovation in a retrieval is then
    ``observed_spectrum - H(ΔVMR_forecast)``. :meth:`linearize` returns the
    analytic Jacobian ``d(L_norm)/d(ΔVMR)`` the forward model already computes.

    Args:
        ds: Absorption-cross-section LUT dataset.
        nu_obs: Observation wavenumbers [cm⁻¹].
        T_K: Layer temperature [K].
        p_atm: Layer pressure [atm].
        vmr_background: Background gas volume mixing ratio.
        path_length_cm: Optical path length [cm].
        amf: Air-mass factor.
        var: LUT cross-section variable name.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        nu_obs: np.ndarray,
        *,
        T_K: float,
        p_atm: float,
        vmr_background: float,
        path_length_cm: float,
        amf: float,
        var: str = "absorption_cross_section",
    ) -> None:
        self.ds = ds
        self.nu_obs = np.asarray(nu_obs, dtype=float)
        self.T_K = T_K
        self.p_atm = p_atm
        self.vmr_background = vmr_background
        self.path_length_cm = path_length_cm
        self.amf = amf
        self.var = var

    def _forward(self, delta_vmr: float):
        return forward_nonlinear_normalized(
            self.ds,
            self.nu_obs,
            T_K=self.T_K,
            p_atm=self.p_atm,
            vmr_background=self.vmr_background,
            delta_vmr=float(delta_vmr),
            path_length_cm=self.path_length_cm,
            amf=self.amf,
            var=self.var,
        )

    def __call__(self, state: Any) -> np.ndarray:
        """Return the normalised radiance spectrum for enhancement ``state``."""
        return self._forward(state).radiance

    def linearize(self, state: Any) -> np.ndarray:
        """Return the tangent-linear ``d(L_norm)/d(ΔVMR)`` at ``state``."""
        return self._forward(state).jacobian


__all__ = ["EulerianForwardModel", "RadtranObservationOperator"]
