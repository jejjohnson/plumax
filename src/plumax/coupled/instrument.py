"""Per-instrument observation specification for the coupled forward.

Tier IV fuses observations from several satellites (TROPOMI, EMIT, Tanager,
GHGSat, …) into one inversion. Each instrument has its own receptor geometry,
averaging kernel, observation time, and error budget; the design
([Tier IV §multi-instrument-fusion](https://github.com/jejjohnson/plumax/blob/main/docs/design/05_tier4_coupled.md#tier4-multi-instrument-fusion))
is explicit that the joint operator is a **list of per-instrument forwards** —
each kept at native resolution — not a single pre-regridded forward.

An :class:`Instrument` bundles everything the coupled forward needs to turn a
modelled column field into that instrument's observation vector and to weight
its residuals in the cost:

- **receptors** — the ``(x, y)`` ground locations it samples [m];
- **averaging_kernel** — the per-receptor scalar AK applied to the column
  enhancement (the design's ``A_inst``; scalar-per-pixel is the L2 default,
  generalisable to a full operator later);
- **observation error** — ``R = R_retr + R_repr + R_align`` assembled per
  receptor (design eq for ``ε_inst``);
- **observation_time** — the overpass time [s], used by time-resolved transport
  tiers (Tier II/III); Tier I is steady-state and ignores it.

``bias`` (the per-instrument additive offset) is *not* stored here — it is a
first-class **state** element solved for in :mod:`plumax.coupled.fusion`, so it
lives with the inference, not the fixed instrument geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from plumax.lagrangian.inversion import _is_traced


@dataclass(frozen=True)
class Instrument:
    """Fixed observation geometry + error budget for one satellite instrument.

    Attributes:
        name: Instrument identifier (e.g. ``"TROPOMI"``).
        receptors: Ground sample locations ``(n_obs, 2)`` as ``(x, y)`` [m].
        averaging_kernel: Per-receptor scalar AK ``A_inst`` applied to the column
            enhancement, shape ``(n_obs,)``. Defaults to all-ones (identity AK).
        retrieval_variance: Per-receptor L2 retrieval error ``R_retr``, shape
            ``(n_obs,)``.
        representation_variance: Model-vs-footprint mismatch ``R_repr`` (scalar
            or ``(n_obs,)``). Default ``0``.
        alignment_variance: Temporal-misalignment error ``R_align`` (scalar or
            ``(n_obs,)``). Default ``0``.
        observation_time: Overpass time [s] (used by time-resolved transport
            tiers; ignored by the steady-state Tier I forward). Default ``0``.
    """

    name: str
    receptors: jax.Array
    averaging_kernel: jax.Array | None = None
    retrieval_variance: jax.Array | float = 1.0
    representation_variance: jax.Array | float = 0.0
    alignment_variance: jax.Array | float = 0.0
    observation_time: float = 0.0

    def __post_init__(self) -> None:
        rec = np.asarray(self.receptors, dtype=float)
        if rec.ndim != 2 or rec.shape[1] != 2:
            raise ValueError(
                f"Instrument {self.name!r}: `receptors` must be (n_obs, 2); "
                f"got shape {rec.shape}."
            )
        if self.averaging_kernel is not None:
            ak = np.asarray(self.averaging_kernel, dtype=float)
            if ak.shape != (rec.shape[0],):
                raise ValueError(
                    f"Instrument {self.name!r}: `averaging_kernel` must have shape "
                    f"({rec.shape[0]},); got {ak.shape}."
                )

    @property
    def n_obs(self) -> int:
        """Number of receptor pixels."""
        return int(np.asarray(self.receptors).shape[0])

    @property
    def ak(self) -> jax.Array:
        """Averaging kernel as an array (identity ones if unset)."""
        if self.averaging_kernel is None:
            return jnp.ones(self.n_obs)
        return jnp.asarray(self.averaging_kernel, dtype=float)

    def observation_variance(self) -> jax.Array:
        """Total per-receptor observation variance ``R_retr + R_repr + R_align``.

        Each term broadcasts to ``(n_obs,)``. The design flags omitting the
        representation / alignment terms as a source of overconfident posteriors,
        so all three are summed here.

        Raises:
            ValueError: if any resulting variance is non-positive (a zero/negative
                variance would blow up or invert the likelihood weight).
        """
        n = self.n_obs
        r = (
            jnp.broadcast_to(jnp.asarray(self.retrieval_variance, dtype=float), (n,))
            + jnp.broadcast_to(
                jnp.asarray(self.representation_variance, dtype=float), (n,)
            )
            + jnp.broadcast_to(jnp.asarray(self.alignment_variance, dtype=float), (n,))
        )
        # Eager positivity check, skipped under jit / grad / vmap — the fusion
        # forward calls this from inside a traced cost, where ``r`` is a tracer
        # (every jnp op inside a trace is, even on concrete inputs) and a
        # host-bool conversion would raise. Guard on ``r`` itself, using the
        # shared ``_is_traced`` helper to match the inversion modules.
        if not _is_traced(r) and bool(jnp.any(r <= 0.0)):
            raise ValueError(
                f"Instrument {self.name!r}: total observation variance must be > 0 "
                "(check retrieval / representation / alignment variances)."
            )
        return r


__all__ = ["Instrument"]
