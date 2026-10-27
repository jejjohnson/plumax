"""Lazy access to vardax, the ``assim`` extra.

The Tier II and Tier III source inversions are vardax analyses over plumax's
source–receptor maps. vardax is optional, so ``import plumax`` never imports
it; the inversion functions call :func:`require_vardax` when they run.
"""

from __future__ import annotations

from types import ModuleType


def require_vardax(caller: str) -> ModuleType:
    """Import vardax, or raise an ``ImportError`` naming the extra.

    Args:
        caller: The public function that needs vardax, for the message.

    Returns:
        The ``vardax`` module.

    Raises:
        ImportError: If vardax is not installed.
    """
    try:
        import vardax
    except ImportError as exc:
        raise ImportError(
            f"{caller} solves the inversion with vardax; install it with "
            "`pip install 'plumax[assim]'`."
        ) from exc
    return vardax
