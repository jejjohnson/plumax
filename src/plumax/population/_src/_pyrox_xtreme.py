"""Lazy access to pyrox-xtreme, the ``population`` extra.

The Tier V point-process layer runs on ``pyrox_xtreme.point_processes``.
pyrox-xtreme is optional, so ``import plumax.population`` never imports it; the
functions that build or evaluate a process call :func:`require_pyrox_xtreme`
when they run.
"""

from __future__ import annotations

from types import ModuleType

from plumax._src.extras import require_extra


def require_pyrox_xtreme(caller: str) -> ModuleType:
    """Import ``pyrox_xtreme.point_processes``, or raise a ``MissingExtraError`` naming the extra.

    Args:
        caller: The public function that needs pyrox-xtreme, for the message.

    Returns:
        The ``pyrox_xtreme.point_processes`` module.

    Raises:
        MissingExtraError: If pyrox-xtreme is not installed.
    """
    return require_extra(
        "pyrox_xtreme.point_processes",
        extra="population",
        caller=caller,
        purpose="builds on pyrox-xtreme point processes",
    )
