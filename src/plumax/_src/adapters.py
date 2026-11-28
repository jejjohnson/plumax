"""pipekit_cycle protocol adapters over the forward models (public API: ``plumax.adapters``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from plumax.coupled.rtm import RadianceObservationOperator


if TYPE_CHECKING:
    import xarray as xr


class RadtranObservationOperator(eqx.Module):
    """Expose the normalised Beer–Lambert forward as an ``ObservationOperator``.

    ``H(ΔVMR)`` answers "what normalised radiance spectrum would a plume
    enhancement of ``ΔVMR`` produce?". The innovation in a retrieval is then
    ``observed_spectrum - H(ΔVMR_forecast)``. It is the one-receptor case of
    :class:`plumax.coupled.RadianceObservationOperator` without an SRF, taken
    in ``ΔVMR`` rather than column mass, and :meth:`linearize` is its
    ``jax.jacfwd``.

    Args:
        ds: Absorption-cross-section LUT dataset.
        nu_obs: Observation wavenumbers [cm⁻¹].
        T_K: Layer temperature [K].
        p_atm: Layer pressure [atm].
        path_length_cm: Optical path length [cm].
        amf: Air-mass factor.
        var: LUT cross-section variable name.

    Attributes:
        operator: The slab operator on ``nu_obs``.
    """

    operator: RadianceObservationOperator

    def __init__(
        self,
        ds: xr.Dataset,
        nu_obs: Float[ArrayLike, " G"],
        *,
        T_K: float,
        p_atm: float,
        path_length_cm: float,
        amf: float,
        var: str = "absorption_cross_section",
    ) -> None:
        self.operator = RadianceObservationOperator(
            ds,
            nu_obs=nu_obs,
            T_K=T_K,
            p_atm=p_atm,
            path_length_cm=path_length_cm,
            amf=amf,
            var=var,
        )

    def __call__(self, state: Any) -> jax.Array:
        """Return the normalised radiance spectrum for enhancement ``state``."""
        return self.operator.predict_delta_vmr(state)

    def linearize(self, state: Any) -> jax.Array:
        """Return the tangent-linear ``d(L_norm)/d(ΔVMR)`` at ``state``."""
        return jax.jacfwd(self.operator.predict_delta_vmr)(
            jnp.asarray(state, dtype=float)
        )


__all__ = ["RadtranObservationOperator"]
