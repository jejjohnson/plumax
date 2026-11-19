"""The eleven plumax.geo operators keep the pipekit Operator and GeoArray contracts.

Runs ``geoops.testing.check_operator`` on each operator over a small scene
built from the shared toy carriers. Needs the ``[geo]`` extra (in the dev
group).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest


pytest.importorskip("geoops")
from geoops.carrier import Coupling, reach_of
from geoops.testing import check_operator

from plumax import geo
from plumax.operators import (
    GaussianPlume,
    GaussianPuff,
    LagrangianDispersion,
)
from tests.geo._carriers import toy_geoarray
from tests.geo.test_vetting import _carriers, _operator, notebook_scene


def _scene() -> tuple[Any, Any, Any]:
    """An ``(H, W)`` column field with a bright bar, its mask and its labels."""
    values = np.zeros((10, 15))
    values[4:6, 2:13] = 5.0
    mask = values > 0
    enhancement = toy_geoarray(values, fill_value=-9999, with_fill_pixels=True)
    return (
        enhancement,
        toy_geoarray(mask, fill_value=None),
        toy_geoarray(mask.astype(np.int32), fill_value=None),
    )


def _cases() -> dict[str, tuple[Any, tuple[Any, ...]]]:
    enhancement, mask, labels = _scene()
    wind = {"wind_u": 1.0, "wind_v": 0.0}
    return {
        "PlumeMask": (geo.PlumeMask(threshold=1.0, min_area_px=1), (enhancement,)),
        "PlumeContours": (geo.PlumeContours(min_area_px=1), (mask,)),
        "PlumeFootprint": (geo.PlumeFootprint(min_area_m2=0.0), (mask,)),
        "WindAdvectionCone": (
            geo.WindAdvectionCone(source=(0.0, 0.0), max_distance=100.0, **wind),
            (mask,),
        ),
        "IMEEstimate": (geo.IMEEstimate(wind_speed=2.0), (enhancement, mask)),
        "CrossSectionalFlux": (
            geo.CrossSectionalFlux(source=(0.0, 0.0), **wind),
            (enhancement, mask),
        ),
        "ColumnToMass": (geo.ColumnToMass(gas="CO2", units_in="ppm_m"), (enhancement,)),
        "PlumeShapeFilter": (geo.PlumeShapeFilter(min_area_px=1), (labels,)),
        "PlumeColumnStats": (geo.PlumeColumnStats(), (labels, enhancement)),
        "PlumeQNDFeatures": (geo.PlumeQNDFeatures(), (labels, enhancement)),
    }


_CASES = _cases()


@pytest.mark.slow
@pytest.mark.parametrize("name", _CASES)
def test_plume_operator_keeps_the_contract(name: str) -> None:
    op, inputs = _CASES[name]
    check_operator(op, *inputs)
    assert type(op).backends == ("numpy",)
    assert isinstance(type(op).coupling, Coupling)
    reach_of(op)


@pytest.mark.slow
def test_plume_vetting_keeps_the_contract() -> None:
    scene = notebook_scene()
    cube, mf = _carriers(scene)
    op = _operator(scene, n_null=0)
    check_operator(op, cube, mf, scene.masks)
    assert type(op).backends == ("numpy",)
    assert type(op).forbid_in_yaml is True
    reach_of(op)


def test_plume_vetting_declares_its_contract() -> None:
    assert geo.PlumeVetting.backends == ("numpy",)
    assert geo.PlumeVetting.forbid_in_yaml is True
    assert isinstance(geo.PlumeVetting.coupling, Coupling)


@pytest.mark.parametrize(
    "cls",
    [geo.PlumeVetting, GaussianPlume, GaussianPuff, LagrangianDispersion],
)
def test_constructors_are_keyword_only(cls: type) -> None:
    parameters = list(inspect.signature(cls).parameters.values())
    assert all(p.kind is not p.POSITIONAL_OR_KEYWORD for p in parameters)
