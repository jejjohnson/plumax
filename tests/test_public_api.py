"""Public-API hygiene: annotations evaluate (#55); the root lists one set of sub-packages (#54).

``from __future__ import annotations`` keeps annotations as strings, so a
bare jaxtyping dimension (``Float[Array, N]``) or a non-string shape
(``Float[Array, ...]``) only fails once something evaluates it:
``typing.get_type_hints``, a runtime type checker or the docs build.
Names imported under ``if TYPE_CHECKING:`` are supplied by running those
imports, so only genuinely unresolvable annotations fail.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import typing
from types import ModuleType

import pytest

import plumax


def _public_modules() -> list[ModuleType]:
    modules = []
    for info in pkgutil.walk_packages(plumax.__path__, "plumax."):
        try:
            modules.append(importlib.import_module(info.name))
        except ImportError:
            continue  # moved-to-radtrax stubs and missing optional extras
    return modules


def _type_checking_namespace(module: ModuleType) -> dict[str, object]:
    """Run the module's ``if TYPE_CHECKING:`` imports into a namespace."""
    namespace: dict[str, object] = {}
    source = inspect.getsource(module)
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name | ast.Attribute)
            and ast.unparse(node.test).endswith("TYPE_CHECKING")
        ):
            imports = [
                s for s in node.body if isinstance(s, ast.Import | ast.ImportFrom)
            ]
            code = compile(
                ast.Module(body=imports, type_ignores=[]), module.__name__, "exec"
            )
            exec(
                code,
                {"__name__": module.__name__, "__package__": module.__package__},
                namespace,
            )
    return namespace


def _annotated(module: ModuleType) -> list[tuple[str, object]]:
    targets = []
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
            continue
        if inspect.isfunction(obj):
            targets.append((name, obj))
        elif inspect.isclass(obj):
            targets.append((name, obj))
            targets.extend(
                (f"{name}.{attr}", fn)
                for attr, fn in vars(obj).items()
                if inspect.isfunction(fn) and not attr.startswith("_")
            )
    return targets


@pytest.mark.parametrize("module", _public_modules(), ids=lambda m: m.__name__)
def test_annotations_evaluate(module: ModuleType) -> None:
    localns = _type_checking_namespace(module)
    failures = []
    for qualname, obj in _annotated(module):
        try:
            typing.get_type_hints(obj, localns=localns)
        except Exception as err:  # report every failure at once
            failures.append(f"{qualname}: {type(err).__name__}: {err}")
    assert not failures, "\n".join(failures)


def _advertised_subpackages() -> list[str]:
    """Sub-package names listed as ``- ``name`` :`` in the package docstring."""
    return re.findall(r"^- ``(\w+)``\s+:", plumax.__doc__ or "", flags=re.MULTILINE)


def test_advertised_subpackages_importable() -> None:
    advertised = _advertised_subpackages()
    assert sorted(advertised) == sorted(plumax.__all__)
    for name in advertised:
        assert isinstance(getattr(plumax, name), ModuleType), name
