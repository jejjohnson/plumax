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


def test_lagrangian_dispersion_config_keys():
    # Regression for #52: the config payload must reflect the real constructor
    # surface — no bogus `stability_class` key (a GaussianPlume copy-paste), and
    # the previously-dropped pbl_height / background_conc / seed present.
    import inspect

    op = LagrangianDispersion(
        turbulence=HomogeneousTurbulence.isotropic(1.0, 30.0),
        domain_x=(0.0, 100.0, 10),
        domain_y=(0.0, 100.0, 10),
        domain_z=(0.0, 100.0, 10),
        pbl_height=800.0,
        background_conc=1e-9,
        seed=7,
    )
    cfg = op.get_config()
    ctor_params = set(inspect.signature(LagrangianDispersion.__init__).parameters)
    # Every reported key is a real constructor parameter.
    assert set(cfg) <= ctor_params
    assert "stability_class" not in cfg
    # The real fields round-trip.
    assert cfg["pbl_height"] == 800.0
    assert cfg["background_conc"] == 1e-9
    assert cfg["seed"] == 7
