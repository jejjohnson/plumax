"""``wrfout`` reader onto a ``les_fvm`` analysis grid.

Reads the WRF variables the transport tiers need, destaggers them to mass
points, converts WRF's perturbation fields to physical units, and
interpolates — vertically from WRF's terrain-following mass levels to the
analysis grid's height-above-ground levels, then horizontally (bilinear)
onto the analysis T-, U- and V-points — so the result is already on the
``les_fvm`` stagger.

Horizontal frame (v1)
---------------------
WRF mass points form a regular ``DX`` × ``DY`` grid; this loader treats
that grid as a local Cartesian frame whose origin is the south-west mass
point.  The analysis grid is placed in that frame by ``origin``: an
analysis-grid coordinate ``(x, y)`` sits at WRF metres ``(x + origin[0],
y + origin[1])``, and the wind is grid-relative.

Lat/lon frame
-------------
Passing a :class:`~plumax.met.LocalFrame` as ``origin`` places the
analysis grid by lat/lon instead, so its axes are the frame's (UTM) axes
rather than WRF's. Each analysis point is mapped to WRF fractional
indices through the file's ``XLAT`` / ``XLONG``: the mass points are
projected into the frame and the bilinear map of each cell of them is
inverted, so WRF's own projection is never rebuilt and any
``MAP_PROJ`` works. The earth-relative wind (which needs ``COSALPHA`` /
``SINALPHA``) is projected onto the frame's axes at each point.

Vertical
--------
Analysis-grid ``z`` is height above ground (``make_grid`` treats ``z_min``
as the surface).  WRF heights above ground come from the geopotential
``(PH + PHB) / g`` on the staggered levels, averaged to mass levels, minus
terrain ``HGT`` — so terrain-following levels are handled per column.
Below the lowest mass level and above the highest the nearest level is
held (constant extrapolation).

Reading
-------
The file is read through :func:`xrops.atm.open_wrfout`, which destaggers
to mass points, rebuilds height above ground from the geopotential and the
terrain, converts WRF's perturbation fields and decodes the time axis. It
needs the ``io`` extra (``pip install 'plumax[io]'``). This module owns
only what is specific to the analysis grid: the vertical and horizontal
interpolation onto it, and a grid-relative wind.

All of this is NumPy: it is IO-side, not on any gradient path, and WRF
files are large.  Only the final fields become JAX arrays.
"""

from __future__ import annotations

from plumax.met._src.wrf import (
    GRAVITY as GRAVITY,
    KAPPA as KAPPA,
    P_REFERENCE as P_REFERENCE,
    THETA_BASE as THETA_BASE,
    load_wrf,
)


__all__ = [
    "load_wrf",
]
