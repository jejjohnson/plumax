"""Briggs-McElroy-Pooler dispersion coefficients (JAX).

The Briggs parameterisation expresses the lateral and vertical dispersion
coefficients of a steady-state Gaussian plume as

    σ_y(x) = a_y · x · (1 + b_y · x)^{c_y}       [m]
    σ_z(x) = a_z · x · (1 + b_z · x)^{c_z}       [m]

with one parameter set per Pasquill-Gifford stability class A-F.

Public surface
--------------
- ``BRIGGS_DISPERSION_PARAMS`` : mapping {stability class → jnp.ndarray(6,)}
- ``STABILITY_CLASSES``        : canonical ordering ('A', ..., 'F')
- ``calculate_briggs_dispersion`` : JIT-compiled σ_y, σ_z evaluator
- ``get_dispersion_params``    : validated dict lookup
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
from jax import jit

from plumax._src.errors import ParameterError


STABILITY_CLASSES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")


# Format: [a_y, b_y, c_y, a_z, b_z, c_z]
BRIGGS_DISPERSION_PARAMS: dict[str, jnp.ndarray] = {
    "A": jnp.array([0.22, 0.0001, -0.5, 0.20, 0.0, 0.0]),
    "B": jnp.array([0.16, 0.0001, -0.5, 0.12, 0.0, 0.0]),
    "C": jnp.array([0.11, 0.0001, -0.5, 0.08, 0.0002, -0.5]),
    "D": jnp.array([0.08, 0.0001, -0.5, 0.06, 0.0015, -0.5]),
    "E": jnp.array([0.06, 0.0001, -0.5, 0.03, 0.0003, -1.0]),
    "F": jnp.array([0.04, 0.0001, -0.5, 0.016, 0.0003, -1.0]),
}


def check_stability(
    stability_class: str,
    *,
    caller: str,
    table: Mapping[str, jnp.ndarray] = BRIGGS_DISPERSION_PARAMS,
) -> None:
    """Fail unless ``stability_class`` is a key of ``table`` (one of A–F).

    Args:
        stability_class: The Pasquill–Gifford class.
        caller: The public function the check guards, for the message.
        table: The parameter table the class indexes.

    Raises:
        ParameterError: ``stability_class`` is not in ``table``. It subclasses
            ``ValueError``.

    Examples:
        >>> check_stability("D", caller="simulate_plume")
        >>> check_stability("G", caller="simulate_plume")
        Traceback (most recent call last):
        ...
        plumax._src.errors.ParameterError: simulate_plume: `stability_class` must be one of ('A', 'B', 'C', 'D', 'E', 'F'), got 'G'
    """
    if stability_class not in table:
        raise ParameterError(
            f"{caller}: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )


def lookup_stability(
    table: Mapping[str, jnp.ndarray], stability_class: str, *, caller: str
) -> jnp.ndarray:
    """The parameter vector of ``stability_class`` in ``table``, after checking it.

    Args:
        table: A table of six parameters per class, such as
            :data:`BRIGGS_DISPERSION_PARAMS`.
        stability_class: The Pasquill–Gifford class.
        caller: The public function the lookup serves, for the message.

    Returns:
        The class's parameter vector, shape (6,).

    Raises:
        ParameterError: ``stability_class`` is not in ``table``.
    """
    check_stability(stability_class, caller=caller, table=table)
    return table[stability_class]


def get_dispersion_params(stability_class: str) -> jnp.ndarray:
    """Return the Briggs parameter vector for a stability class.

    Parameters
    ----------
    stability_class : str
        One of ``'A'``, ``'B'``, ``'C'``, ``'D'``, ``'E'``, ``'F'``.

    Returns
    -------
    params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Raises
    ------
    ValueError
        If ``stability_class`` is not one of A-F.
    """
    return lookup_stability(
        BRIGGS_DISPERSION_PARAMS, stability_class, caller="get_dispersion_params"
    )


def stacked_params(table: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
    """The six parameters of every class, stacked in :data:`STABILITY_CLASSES` order.

    Returns:
        Shape (6, 6): one row per class A–F.
    """
    return jnp.stack([table[name] for name in STABILITY_CLASSES])


@jit
def calculate_briggs_dispersion(
    distance: jnp.ndarray,
    params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate σ_y(x), σ_z(x) from Briggs parameters.

    σ_i(x) = a_i · x · (1 + b_i · x)^{c_i} for i ∈ {y, z}.

    Parameters
    ----------
    distance : jnp.ndarray
        Downwind distance from source [m]. Negative / zero values are clamped
        to 1 m so that the coefficient evaluation stays finite; upwind points
        are masked elsewhere in the plume model.
    params : jnp.ndarray, shape (6,)
        ``[a_y, b_y, c_y, a_z, b_z, c_z]``.

    Returns
    -------
    sigma_y : jnp.ndarray
        Lateral dispersion coefficient [m], same shape as ``distance``.
    sigma_z : jnp.ndarray
        Vertical dispersion coefficient [m], same shape as ``distance``.
    """
    (
        scale_y,
        shift_y,
        exponent_y,
        scale_z,
        shift_z,
        exponent_z,
    ) = params
    downwind = jnp.maximum(distance, 1.0)
    # σ_y = a_y x (1 + b_y x)^{c_y}
    crosswind_spread = (
        scale_y * downwind * jnp.power(1.0 + shift_y * downwind, exponent_y)
    )
    # σ_z = a_z x (1 + b_z x)^{c_z}
    vertical_spread = (
        scale_z * downwind * jnp.power(1.0 + shift_z * downwind, exponent_z)
    )
    return crosswind_spread, vertical_spread
