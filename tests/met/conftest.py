"""Synthetic ``wrfout`` fixture for the ``plumax.met`` loader tests.

Writes a small NetCDF file with the WRF variable names, dimensions,
staggering and unit conventions the loader reads, populated from analytic
fields that are *linear* in every coordinate. Linear fields survive
destaggering (averaging) and linear interpolation exactly, so the tests
can compare the loaded grid against the analytic functions to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from plumax.met.wrf import GRAVITY, P_REFERENCE, THETA_BASE


@dataclass(frozen=True)
class SyntheticWrf:
    """The analytic fields a synthetic ``wrfout`` was written from."""

    path: Path
    dx: float
    dy: float
    n_x: int
    n_y: int
    z_stag: np.ndarray  # staggered level heights above ground [m]
    seconds: np.ndarray  # XTIME converted to seconds

    @property
    def x_max(self) -> float:
        return (self.n_x - 1) * self.dx

    @property
    def y_max(self) -> float:
        return (self.n_y - 1) * self.dy

    # Wind, linear in every coordinate; `t_index` is the met step.
    @staticmethod
    def u(x, y, z, t_index):
        return 2.0 + 0.01 * x + 0.02 * z + 0.5 * t_index

    @staticmethod
    def v(x, y, z, t_index):
        return 1.0 + 0.005 * y - 0.01 * z

    @staticmethod
    def w(x, y, z, t_index):
        return 0.001 * z

    @staticmethod
    def pressure(z):
        return P_REFERENCE - 8.0 * z

    @staticmethod
    def theta(z):
        return THETA_BASE + 0.003 * z

    @staticmethod
    def pbl_height(x, y):
        return 500.0 + 0.1 * x


def write_synthetic_wrfout(
    path: Path,
    *,
    n_x: int = 12,
    n_y: int = 10,
    n_lev: int = 6,
    n_time: int = 3,
    dx: float = 50.0,
    dy: float = 50.0,
    dz: float = 20.0,
    with_xtime: bool = True,
) -> SyntheticWrf:
    """Write a flat-terrain synthetic ``wrfout`` and return its analytic spec."""
    z_stag = dz * np.arange(n_lev + 1, dtype=np.float64)
    z_mass = 0.5 * (z_stag[:-1] + z_stag[1:])
    x_mass = dx * np.arange(n_x, dtype=np.float64)
    y_mass = dy * np.arange(n_y, dtype=np.float64)
    x_stag = dx * (np.arange(n_x + 1, dtype=np.float64) - 0.5)
    y_stag = dy * (np.arange(n_y + 1, dtype=np.float64) - 0.5)
    spec = SyntheticWrf(
        path=path,
        dx=dx,
        dy=dy,
        n_x=n_x,
        n_y=n_y,
        z_stag=z_stag,
        seconds=3600.0 * np.arange(n_time, dtype=np.float64),
    )

    def grid(x, y, z):
        # (n_lev, n_y, n_x) coordinate blocks in WRF's (level, south_north, west_east) order
        return np.meshgrid(z, y, x, indexing="ij")

    u = np.empty((n_time, n_lev, n_y, n_x + 1))
    v = np.empty((n_time, n_lev, n_y + 1, n_x))
    w = np.empty((n_time, n_lev + 1, n_y, n_x))
    for t in range(n_time):
        u[t] = spec.u(*grid(x_stag, y_mass, z_mass)[::-1], t)
        v[t] = spec.v(*grid(x_mass, y_stag, z_mass)[::-1], t)
        w[t] = spec.w(*grid(x_mass, y_mass, z_stag)[::-1], t)
    zz, yy, xx = grid(x_mass, y_mass, z_mass)
    theta_pert = np.broadcast_to(spec.theta(zz) - THETA_BASE, (n_time, *zz.shape))
    pb = np.broadcast_to(spec.pressure(zz), (n_time, *zz.shape))
    phb = np.broadcast_to(
        GRAVITY * z_stag[:, None, None] * np.ones((1, n_y, n_x)),
        (n_time, n_lev + 1, n_y, n_x),
    )
    pblh = np.broadcast_to(spec.pbl_height(xx[0], yy[0]), (n_time, n_y, n_x))
    times = np.array(
        [f"2024-06-01_{h:02d}:00:00".encode() for h in range(n_time)], dtype="S19"
    )

    ds = xr.Dataset(
        {
            "U": (("Time", "bottom_top", "south_north", "west_east_stag"), u),
            "V": (("Time", "bottom_top", "south_north_stag", "west_east"), v),
            "W": (("Time", "bottom_top_stag", "south_north", "west_east"), w),
            "T": (("Time", "bottom_top", "south_north", "west_east"), theta_pert),
            "P": (
                ("Time", "bottom_top", "south_north", "west_east"),
                np.zeros_like(pb),
            ),
            "PB": (("Time", "bottom_top", "south_north", "west_east"), pb),
            "PH": (
                ("Time", "bottom_top_stag", "south_north", "west_east"),
                np.zeros_like(phb),
            ),
            "PHB": (("Time", "bottom_top_stag", "south_north", "west_east"), phb),
            "HGT": (("Time", "south_north", "west_east"), np.zeros((n_time, n_y, n_x))),
            "PBLH": (("Time", "south_north", "west_east"), pblh),
            "XTIME": (
                ("Time",),
                spec.seconds / 60.0,
                {
                    "units": "minutes since 2024-06-01 00:00:00",
                    "description": "minutes since simulation start",
                },
            ),
            "Times": (("Time",), times),
        },
        attrs={"DX": dx, "DY": dy, "TITLE": "synthetic wrfout for plumax tests"},
    )
    if not with_xtime:
        ds = ds.drop_vars("XTIME")
    ds.to_netcdf(path)
    return spec


@pytest.fixture(scope="session")
def synthetic_wrf(tmp_path_factory: pytest.TempPathFactory) -> SyntheticWrf:
    path = tmp_path_factory.mktemp("met") / "wrfout_d01_synthetic.nc"
    return write_synthetic_wrfout(path)
