"""Run the docstring examples of the modules written to the house style."""

from __future__ import annotations

import doctest
import importlib

import pytest


DOCTESTED_MODULES = (
    "plumax.coupled._src.quality",
    "plumax.gauss_plume._src.likelihoods",
    "plumax.gauss_plume._src.multi_source",
    "plumax.gauss_plume._src.observation",
    "plumax.gauss_plume._src.plume_rise",
    "plumax.lagrangian._src.met_ensemble",
    "plumax.lagrangian._src.wind_interp",
    "plumax.met._src.frames",
    "plumax.met._src.pbl",
    "plumax.met._src.regrid",
    "plumax.met._src.surface_layer",
    "plumax.met._src.time",
    "plumax.obs._src.ingest",
    "plumax.population._src.importance",
    "plumax.population._src.integrate",
    "plumax.population._src.marks",
    "plumax.population._src.processes",
    "plumax.priors._src.background",
)


@pytest.mark.parametrize("name", DOCTESTED_MODULES)
def test_docstring_examples(name: str) -> None:
    pytest.importorskip("pyproj")
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0, f"{name} has no examples"
    assert result.failed == 0
