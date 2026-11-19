"""Lazy access to vardax and argminax, the ``assim`` extra.

The Tier II and Tier III source inversions are vardax analyses over plumax's
source–receptor maps, and the Tier III Laplace posterior is an argminax
``Laplace``. Both are optional, so ``import plumax`` imports neither; the
inversion functions call :func:`require_vardax` or :func:`require_argminax`
when they run.
"""

from __future__ import annotations

from types import ModuleType

import einx
import gaussx as gx
import jax
import jax.numpy as jnp
import lineax as lx

from plumax._src.extras import require_extra


#: The tags of a covariance operator: symmetric and positive semi-definite.
PSD_TAGS = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


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


def row_operator(
    operator: lx.AbstractLinearOperator, size: int, dtype: jnp.dtype
) -> lx.FunctionLinearOperator:
    """Wrap an operator on ``(size,)`` vectors into one on ``(1, size)`` rows.

    vardax applies its covariances to ``(1, size)`` rows; this adapts an
    operator written for plain vectors.

    Args:
        operator: The operator M on ``(size,)`` vectors.
        size: The vector length.
        dtype: The dtype of the row the operator is applied to.

    Returns:
        A function operator, tagged symmetric and positive semi-definite, that
        maps a ``(1, size)`` row to M x as a ``(1, size)`` row.
    """

    def apply_to_row(row: jax.Array) -> jax.Array:
        # y = M x, kept as a (1, size) row
        product = operator.mv(row[0])
        return einx.id("dim -> 1 dim", product)

    return lx.FunctionLinearOperator(
        apply_to_row, jax.ShapeDtypeStruct((1, size), dtype), PSD_TAGS
    )


def dense_covariance(
    operator: lx.AbstractLinearOperator,
    size: int,
    dtype: jnp.dtype,
    *,
    rows: bool = True,
) -> jax.Array:
    """Materialise a covariance operator as a symmetrised ``(size, size)`` matrix.

    Applies the operator to each basis vector with ``jax.vmap`` and
    symmetrises the stacked columns with ``gaussx.symmetrize``.

    Args:
        operator: The covariance operator, such as a vardax posterior's ``cov``.
        size: The state size D.
        dtype: The dtype of the basis vectors.
        rows: Whether the operator acts on ``(1, size)`` rows (vardax's
            default) rather than on ``(size,)`` vectors.

    Returns:
        The symmetrised matrix, shape ``(size, size)``.
    """

    def column(basis_vector: jax.Array) -> jax.Array:
        # C e_k, on a (1, size) row when the operator takes rows
        if not rows:
            return operator.mv(basis_vector)
        basis_row = einx.id("param -> 1 param", basis_vector)
        column_row = operator.mv(basis_row)
        return einx.id("1 param -> param", column_row)

    return gx.symmetrize(jax.vmap(column)(jnp.eye(size, dtype=dtype)))
