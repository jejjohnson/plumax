"""Gate P5: the base install of plumax pulls in no geo carrier.

Run with the interpreter of a fresh environment that has ``plumax``
installed without extras (``uv pip install .``); CI does exactly that in
``.github/workflows/base-install.yml``. The check fails if

- a geo package (``georeader``, ``geoops``, ``geotoolz``) is importable, so
  something in the base dependencies pulled it in; or
- ``import plumax`` or ``import plumax.geo`` fails, or importing an operator
  module of ``plumax.geo`` fails with anything but an ``ImportError`` that
  names the ``[geo]`` extra.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys


#: Packages the base install must not contain (package-boundaries rule 3).
FORBIDDEN = ("georeader", "geoops", "geotoolz")

#: The modules that need the [geo] extra and must say so when it is missing.
EXTRA_ONLY = ("plumax.geo.plume", "plumax.geo.plume_array")


def main() -> int:
    """Return 0 when the gate holds, 1 (with the reasons printed) otherwise."""
    problems = [
        f"{name} is installed in the base environment"
        for name in FORBIDDEN
        if importlib.util.find_spec(name) is not None
    ]
    try:
        importlib.import_module("plumax")
        importlib.import_module("plumax.geo")
    except ImportError as err:
        problems.append(f"import plumax failed: {err}")
    for name in EXTRA_ONLY:
        try:
            importlib.import_module(name)
            problems.append(f"{name} imported without the [geo] extra")
        except ImportError as err:
            if "plumax[geo]" not in str(err):
                problems.append(f"import {name} failed: {err}")
    for problem in problems:
        print(f"::error::{problem}")
    if not problems:
        print(f"base install OK: plumax imports, {len(EXTRA_ONLY)} modules need [geo]")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
