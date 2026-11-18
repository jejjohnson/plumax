"""The package layout: logic lives under ``_src``, public modules re-export (#288).

A module outside a ``_src`` package defines no function or class, except
the ``__getattr__`` and ``__dir__`` of a lazy facade (PEP 562). Every
package follows it since #289–#297.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "plumax"
#: lazy facades define these at module level
FACADE_HOOKS = frozenset({"__getattr__", "__dir__"})


def _public_modules() -> list[Path]:
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).parts
        if "_src" in parts or "__pycache__" in parts:
            continue
        modules.append(path)
    return modules


def _definitions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.name not in FACADE_HOOKS
    ]


@pytest.mark.parametrize(
    "path", _public_modules(), ids=lambda p: str(p.relative_to(PACKAGE))
)
def test_public_modules_only_re_export(path: Path) -> None:
    assert _definitions(path) == [], (
        f"{path.relative_to(PACKAGE)} defines logic; move it under _src and "
        "re-export it"
    )
