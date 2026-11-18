"""Lazy access to vardax and argminax, the ``assim`` extra.

The Tier II and Tier III source inversions are vardax analyses over plumax's
source–receptor maps, and the Tier III Laplace posterior is an argminax
``Laplace``. Both are optional, so ``import plumax`` imports neither; the
inversion functions call :func:`require_vardax` or :func:`require_argminax`
when they run.
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


def require_argminax(caller: str) -> ModuleType:
    """Import argminax, or raise an ``ImportError`` naming the extra.

    argminax builds the Laplace posterior at a 4D-Var MAP; it ships in the
    same ``assim`` extra as vardax.

    Args:
        caller: The public function that needs argminax, for the message.

    Returns:
        The ``argminax`` module, with ``argminax.curvature`` loaded.

    Raises:
        ImportError: If argminax is not installed.
    """
    try:
        import argminax
        import argminax.curvature  # a submodule, not loaded by the package
    except ImportError as exc:
        raise ImportError(
            f"{caller} builds the Laplace posterior with argminax; install it "
            "with `pip install 'plumax[assim]'`."
        ) from exc
    return argminax
