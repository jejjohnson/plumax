"""Lazy access to xtremax, the ``population`` extra.

The Tier V point-process layer runs on ``xtremax.point_processes``. xtremax is
optional, so ``import plumax.population`` never imports it; the functions that
build or evaluate a process call :func:`require_xtremax` when they run.
"""

from __future__ import annotations

from types import ModuleType


def require_xtremax(caller: str) -> ModuleType:
    """Import ``xtremax.point_processes``, or raise an ``ImportError`` naming the extra.

    Args:
        caller: The public function that needs xtremax, for the message.

    Returns:
        The ``xtremax.point_processes`` module.

    Raises:
        ImportError: If xtremax is not installed.
    """
    try:
        import xtremax.point_processes as point_processes
    except ImportError as exc:
        raise ImportError(
            f"{caller} builds on xtremax point processes; install them with "
            "`pip install 'plumax[population]'`."
        ) from exc
    return point_processes
