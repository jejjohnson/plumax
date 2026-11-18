"""The coupled solve from several starting controls.

The image-fit cost has several minima in the source position (and the
direction), so one start can end in the wrong one. :func:`solve_coupled_multistart`
runs :func:`~plumax.coupled.solve_coupled` from each start with the same prior,
so every start minimises the same cost and the costs are comparable, and keeps
the converged start with the lowest cost.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from plumax._src.errors import ControlBlockError, ConvergenceError
from plumax.coupled._src.inversion import CoupledProblem, CoupledResult, solve_coupled


@dataclasses.dataclass(frozen=True)
class MultistartResult:
    """Every start's outcome, and the one that won.

    Attributes:
        best: Result of the converged start with the lowest cost.
        index: Position j* of that start.
        costs: Cost 𝒥(θ*_j) at each start's minimum.
        converged: Whether each start converged.
        map_estimates: MAP control of each start, by block.
    """

    best: CoupledResult
    index: int
    costs: Float[np.ndarray, " start"]
    converged: Bool[np.ndarray, " start"]
    map_estimates: list[dict[str, jax.Array]]


def solve_coupled_multistart(
    problem: CoupledProblem,
    starts: Mapping[str, Float[Array, "start ..."]],
    **solve_options: Any,
) -> MultistartResult:
    """The coupled solve from each start, keeping the lowest cost.

    θ*_j = argmin 𝒥(θ) from θ⁽⁰⁾_j;  j* = argmin_j 𝒥(θ*_j) over the converged j.

    Algorithm:
        1. Check that every block of ``starts`` has the same number J of starts.
        2. For each j, solve from the j-th start of each block; blocks not in
           ``starts`` start at the prior mean.
        3. Keep the converged start with the lowest cost.

    The starts run one after another in a Python loop; each is a full
    :func:`~plumax.coupled.solve_coupled`.

    Args:
        problem: The coupled problem; its prior is the same for every start.
        starts: Starting values by block, with a leading axis of J starts.
        **solve_options: Passed to every :func:`~plumax.coupled.solve_coupled`
            (``laplace``, ``rtol``, ``max_steps``, ...).

    Returns:
        The :class:`MultistartResult`.

    Raises:
        ControlBlockError: If ``starts`` is empty, its blocks differ in their
            number of starts, or a block is not in the prior.
        ConvergenceError: If no start converges.
        ValueError: If ``solve_options`` holds ``initial``.
    """
    if "initial" in solve_options:
        raise ValueError(
            "solve_coupled_multistart: give the starts in `starts`, not `initial`."
        )
    if not starts:
        raise ControlBlockError("solve_coupled_multistart: `starts` is empty.")
    values = {name: jnp.asarray(value, dtype=float) for name, value in starts.items()}
    counts = {
        name: value.shape[0] if value.ndim else 0 for name, value in values.items()
    }
    num_starts = next(iter(counts.values()))
    if num_starts == 0 or any(count != num_starts for count in counts.values()):
        raise ControlBlockError(
            "solve_coupled_multistart: every block of `starts` needs the same "
            f"leading number of starts, at least one; got {counts}."
        )

    # --- one solve per start ---
    results = []
    for start_index in range(num_starts):
        # θ⁽⁰⁾_j
        initial = {name: value[start_index] for name, value in values.items()}
        results.append(solve_coupled(problem, initial=initial, **solve_options))

    # --- the lowest converged cost ---
    costs = np.array([result.cost for result in results], dtype=float)
    converged = np.array([result.converged for result in results], dtype=bool)
    if not bool(converged.any()):
        raise ConvergenceError(
            f"solve_coupled_multistart: none of the {num_starts} starts converged; "
            "raise `max_steps` or check the starts."
        )
    # j* = argmin_j 𝒥(θ*_j) over the converged j
    index = int(np.argmin(np.where(converged, costs, np.inf)))
    return MultistartResult(
        best=results[index],
        index=index,
        costs=costs,
        converged=converged,
        map_estimates=[dict(result.map_estimate) for result in results],
    )


__all__ = ["MultistartResult", "solve_coupled_multistart"]
