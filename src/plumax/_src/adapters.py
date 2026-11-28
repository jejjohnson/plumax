"""pipekit_cycle protocol adapters over the forward models (public API: ``plumax.adapters``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from radtrax.slab import forward_nonlinear_normalized


if TYPE_CHECKING:
    import xarray as xr


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

    def _forward(self, state: float):
        return forward_nonlinear_normalized(
            self.ds,
            self.nu_obs,
            T_K=self.T_K,
            p_atm=self.p_atm,
            vmr_background=self.vmr_background,
            delta_vmr=float(state),
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


__all__ = ["RadtranObservationOperator"]
