"""Eager-or-traced validation: whether a value is traced, and checks that respect it."""

from __future__ import annotations

import numpy as np
from pipekit_ops import is_traced

from plumax._src.errors import ParameterError


def _is_traced(*values: object) -> bool:
    """Return ``True`` if any value is, or holds, a JAX tracer.

    Used to skip the eager value-validation ``if`` checks under ``jit`` / ``grad``
    / ``vmap``, where converting a traced scalar to a Python ``bool`` would raise
    ``TracerBoolConversionError``. A tuple, list or dict counts as traced when
    any of its leaves is (:func:`pipekit_ops.is_traced`). Validation still runs
    for concrete inputs, so ordinary callers keep the helpful error messages.
    """
    return any(is_traced(value) for value in values)


def require_positive(
    name: str,
    value: object,
    *,
    caller: str | None = None,
    allow_zero: bool = False,
    error: type[Exception] = ParameterError,
) -> None:
    """Fail unless every element of ``value`` is > 0 (≥ 0 with ``allow_zero``).

    A traced value skips the check, so the caller stays usable under ``jit``,
    ``grad`` and ``vmap``; a concrete one fails with a message that names the
    caller and the argument. NaN fails, since NaN > 0 is false.

    Args:
        name: The argument's name, as the caller's signature spells it.
        value: A scalar or an array.
        caller: The public function or class the check guards, for the message.
        allow_zero: Accept 0 (the bound becomes ≥ 0).
        error: The exception raised; a subclass of ``ValueError`` by default.

    Raises:
        ParameterError: Some element of a concrete ``value`` is out of range
            (or ``error``, when given).

    Examples:
        >>> require_positive("wind_speed", 2.0, caller="simulate_plume")
        >>> require_positive("wind_speed", 0.0, caller="simulate_plume")
        Traceback (most recent call last):
        ...
        plumax._src.errors.ParameterError: simulate_plume: `wind_speed` must be > 0 (got 0.0)
        >>> require_positive("jitter", 0.0, allow_zero=True)
    """
    if _is_traced(value):
        return
    array = np.asarray(value)
    # ok = ∀ elements: value > 0 (or ≥ 0)
    ok = np.all(array >= 0.0) if allow_zero else np.all(array > 0.0)
    if bool(ok):
        return
    bound = "≥ 0" if allow_zero else "> 0"
    prefix = f"{caller}: " if caller else ""
    raise error(f"{prefix}`{name}` must be {bound} (got {value!r})")
