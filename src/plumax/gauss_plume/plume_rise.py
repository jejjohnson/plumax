"""Briggs plume rise and the effective release height (implementation in ``plumax.gauss_plume._src.plume_rise``)."""

from __future__ import annotations

from plumax.gauss_plume._src.plume_rise import (
    STABLE_THETA_GRADIENT,
    UNSTABLE_NEUTRAL_CLASSES,
    BriggsPlumeRise,
    briggs_plume_rise,
    effective_height,
)


__all__ = [
    "STABLE_THETA_GRADIENT",
    "UNSTABLE_NEUTRAL_CLASSES",
    "BriggsPlumeRise",
    "briggs_plume_rise",
    "effective_height",
]
