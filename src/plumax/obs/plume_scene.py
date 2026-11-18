"""One plume from a matched-filter product (implementation in ``plumax.obs._src.plume_scene``)."""

from __future__ import annotations

from plumax.obs._src.plume_scene import (
    PlumeScene,
    load_emit_plume_complex,
    matched_filter_scene,
)


__all__ = ["PlumeScene", "load_emit_plume_complex", "matched_filter_scene"]
