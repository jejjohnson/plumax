"""The radiative-transfer subpackages moved to radtrax (plumax#133).

Each old path raises an ImportError naming its new home instead of
re-exporting, so no name has two homes.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("plumax.radtran", "radtrax.slab"),
        ("plumax.radtran.forward", "radtrax.slab"),
        ("plumax.hapi_lut", "radtrax.spectroscopy.hitran"),
        ("plumax.assimilation", "radtrax.retrieval.var3d"),
        ("plumax.matched_filter", "radtrax.retrieval.matched_filter"),
        ("plumax.matched_filter.core", "radtrax.retrieval.matched_filter"),
    ],
)
def test_old_path_names_new_home(old: str, new: str) -> None:
    with pytest.raises(ImportError, match=new.replace(".", r"\.")):
        importlib.import_module(old)


def test_package_root_no_longer_exports_them() -> None:
    import plumax

    for name in ("radtran", "hapi_lut"):
        assert name not in plumax.__all__
