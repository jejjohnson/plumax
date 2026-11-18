"""Cross-plume transects (implementation in ``plumax.geo._src.transects``)."""

from __future__ import annotations

from plumax.geo._src.transects import (
    CrossPlumeProfile,
    TransectProfiles,
    cross_plume_transects,
    fit_transect_gaussians,
    fit_width_law,
    smooth_centreline,
    transect_moments,
)


__all__ = [
    "CrossPlumeProfile",
    "TransectProfiles",
    "cross_plume_transects",
    "fit_transect_gaussians",
    "fit_width_law",
    "smooth_centreline",
    "transect_moments",
]
