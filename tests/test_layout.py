"""The package layout: logic lives under ``_src``, public modules re-export (#288).

A module outside a ``_src`` package defines no function or class, except
the ``__getattr__`` and ``__dir__`` of a lazy facade (PEP 562). The
packages still to move are listed in ``NOT_YET_MOVED``; each of #289–#297
removes its own entry.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "plumax"
#: lazy facades define these at module level
FACADE_HOOKS = frozenset({"__getattr__", "__dir__"})
#: packages whose public modules still hold logic, by the issue that moves them
NOT_YET_MOVED = {
    "population": 295,
    "les_fvm": 296,
    "geo": 297,
}


def _public_modules() -> list[Path]:
    modules = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).parts
        if "_src" in parts or "__pycache__" in parts:
            continue
        if len(parts) > 1 and parts[0] in NOT_YET_MOVED:
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


def test_the_exceptions_still_need_their_move() -> None:
    # an exception whose package has moved must leave the list
    for package in NOT_YET_MOVED:
        modules = [
            path
            for path in sorted((PACKAGE / package).rglob("*.py"))
            if "_src" not in path.relative_to(PACKAGE).parts
        ]
        assert any(_definitions(path) for path in modules), (
            f"{package} has no logic outside _src any more; remove it from "
            "NOT_YET_MOVED"
        )
