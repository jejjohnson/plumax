"""Test the LagrangianDispersion pipekit operator wrapper."""

from __future__ import annotations

import xarray as xr
from pipekit import Operator

from plumax.lagrangian import HomogeneousTurbulence
from plumax.operators import LagrangianDispersion


def test_lagrangian_operator_runs_and_is_operator():
    op = LagrangianDispersion(
        turbulence=HomogeneousTurbulence(1.0, 1.0, 0.6, 30.0, 30.0, 20.0),
        domain_x=(-50.0, 400.0, 45),
        domain_y=(-120.0, 120.0, 24),
        domain_z=(0.0, 150.0, 15),
        n_particles=1500,
        t_end=120.0,
        dt=1.0,
    )
    assert isinstance(op, Operator)
    assert op.forbid_in_yaml is True
    ds = op(
        {
            "emission_rate": 1.0,
            "source_location": (0.0, 0.0, 20.0),
            "wind_speed": 5.0,
            "wind_direction": 270.0,
        }
    )
    assert isinstance(ds, xr.Dataset)
    assert ds["concentration"].dims == ("x", "y", "z")
    assert float(ds["concentration"].max()) > 0.0


def test_lagrangian_operator_config_reports_keys():
    op = LagrangianDispersion(
        turbulence=HomogeneousTurbulence.isotropic(1.0, 30.0),
        domain_x=(0.0, 100.0, 10),
        domain_y=(0.0, 100.0, 10),
        domain_z=(0.0, 100.0, 10),
    )
    cfg = op.get_config()
    assert cfg["n_particles"] == 5000
    assert cfg["domain_x"] == (0.0, 100.0, 10)
