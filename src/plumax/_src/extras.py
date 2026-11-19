"""Lazy import of an optional extra, failing with a message that names it."""

from __future__ import annotations

import importlib
from types import ModuleType

from plumax._src.errors import MissingExtraError


def require_extra(
    module: str,
    *,
    extra: str,
    caller: str,
    purpose: str,
    submodules: tuple[str, ...] = (),
) -> ModuleType:
    """Import an optional module, or raise ``MissingExtraError`` naming its extra.

    Args:
        module: The module to import, dotted (``"pyproj"``,
            ``"xtremax.point_processes"``).
        extra: The plumax extra that installs it (``"io"``, ``"assim"``).
        caller: The public function or class that needs it, for the message.
        purpose: What the caller does with it, completing the sentence
            ``"<caller> <purpose>; install it with ..."``.
        submodules: Submodules of ``module`` to import as well, for packages
            that do not load them on their own (``("curvature",)``).

    Returns:
        The imported ``module``.

    Raises:
        MissingExtraError: ``module`` or one of ``submodules`` cannot be
            imported. It subclasses ``ImportError``.

    Examples:
        >>> require_extra("json", extra="io", caller="f", purpose="reads JSON").__name__
        'json'
        >>> require_extra(
        ...     "not_a_module", extra="io", caller="load_x", purpose="reads X"
        ... )
        Traceback (most recent call last):
        ...
        plumax._src.errors.MissingExtraError: load_x reads X; install it with `pip install 'plumax[io]'`
    """
    try:
        imported = importlib.import_module(module)
        for submodule in submodules:
            importlib.import_module(f"{module}.{submodule}")
    except ImportError as exc:
        raise MissingExtraError(
            f"{caller} {purpose}; install it with `pip install 'plumax[{extra}]'`"
        ) from exc
    return imported
