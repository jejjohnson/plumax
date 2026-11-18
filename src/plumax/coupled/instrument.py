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

- **quality** — optional per-pixel ``flags`` (bit flags, ``0`` = clear), the
  ``mask`` that drops pixels from the likelihood and the ``variance_inflation``
  that down-weights them. :func:`plumax.coupled.quality.apply_quality` sets the
  last two from the flags and a policy.

``bias`` (the per-instrument additive offset) is *not* stored here — it is a
first-class **state** element solved for in :mod:`plumax.coupled.fusion`, so it
lives with the inference, not the fixed instrument geometry.
"""

from __future__ import annotations

from plumax.coupled._src.instrument import (
    Instrument,
)


__all__ = [
    "Instrument",
]
