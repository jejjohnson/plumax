"""Whether a value is being traced by a JAX transform, for eager-only checks."""

from __future__ import annotations

import jax
import jax.core


def _is_traced(*values: object) -> bool:
    """Return ``True`` if any value is a JAX tracer (i.e. we're inside a transform).

    Used to skip the eager value-validation ``if`` checks under ``jit`` / ``grad``
    / ``vmap``, where converting a traced scalar to a Python ``bool`` would raise
    ``TracerBoolConversionError``. Validation still runs for concrete inputs, so
    ordinary callers keep the helpful error messages.
    """
    return any(isinstance(value, jax.core.Tracer) for value in values)
