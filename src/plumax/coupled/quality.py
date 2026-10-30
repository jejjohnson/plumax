"""Per-instrument quality-flag aggregator (implementation in ``plumax.coupled._src.quality``)."""

from __future__ import annotations

from plumax.coupled._src.quality import (
    FlagPolicy,
    QualityFlag,
    QualityPolicy,
    StackedObservations,
    apply_quality,
    contribution_report,
    full_weight_pixels,
    stack_observations,
)


__all__ = [
    "FlagPolicy",
    "QualityFlag",
    "QualityPolicy",
    "StackedObservations",
    "apply_quality",
    "contribution_report",
    "full_weight_pixels",
    "stack_observations",
]
