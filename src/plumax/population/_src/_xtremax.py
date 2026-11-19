"""Lazy access to xtremax, the ``population`` extra.

The Tier V point-process layer runs on ``xtremax.point_processes``. xtremax is
optional, so ``import plumax.population`` never imports it; the functions that
build or evaluate a process call :func:`require_xtremax` when they run.
"""

from __future__ import annotations

from types import ModuleType

from plumax._src.extras import require_extra


def require_xtremax(caller: str) -> ModuleType:
    """Import ``xtremax.point_processes``, or raise a ``MissingExtraError`` naming the extra.

    Args:
        caller: The public function that needs xtremax, for the message.

    Returns:
        The ``xtremax.point_processes`` module.

    Raises:
        MissingExtraError: If xtremax is not installed.
    """
    return require_extra(
        "xtremax.point_processes",
        extra="population",
        caller=caller,
        purpose="builds on xtremax point processes",
    )
