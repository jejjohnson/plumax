"""Tests for the pipekit Operator wrappers in plumax.operators."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pipekit import Lambda, Operator, Sequential

from plumax.operators import EulerianDispersion, GaussianPlume, GaussianPuff


@pytest.fixture
def small_domain():
    return {
        "domain_x": (-100.0, 1000.0, 56),
        "domain_y": (-300.0, 300.0, 31),
        "domain_z": (0.0, 100.0, 11),
    }


def test_gaussian_plume_apply_returns_dataset(small_domain):
    plume = GaussianPlume(stability_class="D", **small_domain)
    ds = plume(
        {
            "emission_rate": 1.0,
            "source_location": (0.0, 0.0, 10.0),
            "wind_speed": 5.0,
            "wind_direction": 270.0,
        }
    )
    assert isinstance(ds, xr.Dataset)
    assert ds["concentration"].dims == ("x", "y", "z")
    # A positive emission must produce a positive peak concentration.
    assert float(ds["concentration"].max()) > 0.0


def test_gaussian_plume_is_operator_and_composes(small_domain):
    plume = GaussianPlume(stability_class="D", **small_domain)
    assert isinstance(plume, Operator)
    # Composes with the pipekit primitives via `|`.
    pipe = plume | Lambda(lambda ds: float(ds["concentration"].max()), name="peak")
    assert isinstance(pipe, Sequential)
    peak = pipe(
        {
            "emission_rate": 2.0,
            "source_location": (0.0, 0.0, 10.0),
            "wind_speed": 4.0,
            "wind_direction": 270.0,
        }
    )
    assert peak > 0.0


def test_gaussian_plume_config_round_trip(small_domain):
    plume = GaussianPlume(stability_class="E", background_conc=0.1, **small_domain)
    cfg = plume.get_config()
    assert cfg["stability_class"] == "E"
    assert cfg["background_conc"] == 0.1
    # Config is JSON-primitive -> state round-trips through Operator.from_state.
    rebuilt = Operator.from_state(plume.state)
    assert isinstance(rebuilt, GaussianPlume)
    assert rebuilt.stability_class == "E"
    assert rebuilt.domain_x == small_domain["domain_x"]


def test_gaussian_puff_apply_returns_time_resolved_field(small_domain):
    time_array = np.array([0.0, 30.0, 60.0])
    puff = GaussianPuff(
        stability_class="D",
        time_array=time_array,
        release_frequency=0.1,
        **small_domain,
    )
    assert puff.forbid_in_yaml is True
    ds = puff(
        {
            "emission_rate": 1.0,
            "source_location": (0.0, 0.0, 10.0),
            "wind_speed": np.array([5.0, 5.0, 5.0]),
            "wind_direction": np.array([270.0, 270.0, 270.0]),
        }
    )
    assert isinstance(ds, xr.Dataset)
    assert "time" in ds.dims
    assert ds.sizes["time"] == time_array.size


def test_eulerian_operator_config_reports_keys():
    op = EulerianDispersion(
        domain_x=(0.0, 100.0, 10),
        t_start=0.0,
        t_end=10.0,
        save_interval=5.0,
    )
    assert op.forbid_in_yaml is True
    cfg = op.get_config()
    assert cfg["config_keys"] == ["domain_x", "save_interval", "t_end", "t_start"]
