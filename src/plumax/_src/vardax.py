"""Lazy access to vardax and argminax, the ``assim`` extra.

The Tier II and Tier III source inversions are vardax analyses over plumax's
source–receptor maps, and the Tier III Laplace posterior is an argminax
``Laplace``. Both are optional, so ``import plumax`` imports neither; the
inversion functions call :func:`require_vardax` or :func:`require_argminax`
when they run.
"""

from __future__ import annotations

from types import ModuleType

from plumax._src.extras import require_extra


def require_vardax(caller: str) -> ModuleType:
    """Import vardax, or raise a ``MissingExtraError`` naming the extra.

    Args:
        caller: The public function that needs vardax, for the message.

    Returns:
        The ``vardax`` module.

    Raises:
        MissingExtraError: If vardax is not installed.
    """
    return require_extra(
        "vardax",
        extra="assim",
        caller=caller,
        purpose="solves the inversion with vardax",
    )


def require_argminax(caller: str) -> ModuleType:
    """Import argminax, or raise a ``MissingExtraError`` naming the extra.

    argminax builds the Laplace posterior at a 4D-Var MAP; it ships in the
    same ``assim`` extra as vardax.

    Args:
        caller: The public function that needs argminax, for the message.

    Returns:
        The ``argminax`` module, with ``argminax.curvature`` loaded.

    Raises:
        MissingExtraError: If argminax is not installed.
    """
    return require_extra(
        "argminax",
        extra="assim",
        caller=caller,
        purpose="builds the Laplace posterior with argminax",
        submodules=("curvature",),
    )
