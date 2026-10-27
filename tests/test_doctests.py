"""Run the docstring examples of the modules written to the house style."""

from __future__ import annotations

import doctest
import importlib

import pytest


DOCTESTED_MODULES = (
    "plumax.met._src.frames",
    "plumax.met._src.pbl",
    "plumax.met._src.regrid",
    "plumax.met._src.surface_layer",
    "plumax.met._src.time",
    "plumax.obs._src.ingest",
    "plumax.priors._src.background",
)


@pytest.mark.parametrize("name", DOCTESTED_MODULES)
def test_docstring_examples(name: str) -> None:
    pytest.importorskip("pyproj")
    module = importlib.import_module(name)
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0, f"{name} has no examples"
    assert result.failed == 0
