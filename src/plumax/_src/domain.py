"""The ``(start, stop, n)`` domains of the forward models: checks, axes and meshes."""

from __future__ import annotations

from collections.abc import Sequence

import einx
import numpy as np
from jaxtyping import Bool, Float, Int

from plumax._src.tracing import require_positive


#: A model domain along one axis: ``(start, stop, n_points)`` [m, m, -].
Domain = tuple[float, float, int]


def check_domain(name: str, domain: Sequence[float], *, caller: str) -> None:
    """Fail unless ``domain`` is ``(start, stop, n_points)`` with start < stop and n ≥ 2.

    Args:
        name: The argument's name (``"domain_x"``), for the message.
        domain: The domain to check.
        caller: The public function the check guards, for the message.

    Raises:
        ValueError: The tuple has the wrong length, start ≥ stop, or n < 2.

    Examples:
        >>> check_domain("domain_x", (0.0, 100.0, 11), caller="simulate_plume")
        >>> check_domain("domain_x", (0.0, 100.0, 1), caller="simulate_plume")
        Traceback (most recent call last):
        ...
        ValueError: simulate_plume: `domain_x` requires n_points ≥ 2 (got n=1)
    """
    if len(domain) != 3:
        raise ValueError(
            f"{caller}: `{name}` must be (start, stop, n_points); "
            f"got length {len(domain)}"
        )
    start, stop, num_points = domain
    if not (start < stop):
        raise ValueError(
            f"{caller}: `{name}` requires start < stop "
            f"(got start={start!r}, stop={stop!r})"
        )
    if int(num_points) < 2:
        raise ValueError(
            f"{caller}: `{name}` requires n_points ≥ 2 (got n={num_points!r})"
        )


def check_source(source_location: Sequence[float], *, caller: str) -> None:
    """Fail unless ``source_location`` is ``(x, y, z)`` with z ≥ 0.

    Args:
        source_location: The source position [m].
        caller: The public function the check guards, for the message.

    Raises:
        ValueError: The tuple has the wrong length.
        ParameterError: The source height is negative.
    """
    if len(source_location) != 3:
        raise ValueError(
            f"{caller}: `source_location` must be (x, y, z); "
            f"got length {len(source_location)}"
        )
    require_positive(
        "source_location[2]", source_location[2], caller=caller, allow_zero=True
    )


def node_axis(domain: Domain) -> np.ndarray:
    """The ``n`` grid nodes of a domain, from start to stop inclusive.

    Examples:
        >>> node_axis((0.0, 10.0, 3)).tolist()
        [0.0, 5.0, 10.0]
    """
    start, stop, num_points = domain
    return np.linspace(start, stop, int(num_points))


def edge_axis(domain: Domain) -> np.ndarray:
    """The ``n + 1`` cell edges of a domain of ``n`` cells, from start to stop.

    Examples:
        >>> edge_axis((0.0, 10.0, 2)).tolist()
        [0.0, 5.0, 10.0]
    """
    start, stop, num_cells = domain
    return np.linspace(start, stop, int(num_cells) + 1)


def containing_cell(
    coordinate: Float[np.ndarray, "*shape"] | float, edges: Float[np.ndarray, " edge"]
) -> tuple[Int[np.ndarray, "*shape"], Bool[np.ndarray, "*shape"]]:
    """Index of the cell holding each coordinate, and whether it is inside.

    k = #{m : e_m ≤ x} − 1, inside when 0 ≤ k < n_cells,

    so cell k is the half-open [e_k, e_{k+1}): a coordinate on an inner edge
    belongs to the cell above it, and one on the last edge is outside. Each
    caller applies its own rule to the coordinates outside.

    Args:
        coordinate: Coordinates x.
        edges: The ``n_cells + 1`` ascending cell edges e.

    Returns:
        ``(index, inside)``; ``index`` is −1 or ``n_cells`` where outside.

    Examples:
        >>> edges = np.array([0.0, 25.0, 50.0])
        >>> index, inside = containing_cell(np.array([-1.0, 0.0, 25.0, 50.0]), edges)
        >>> index.tolist(), inside.tolist()
        ([-1, 0, 1, 2], [False, True, True, False])
    """
    # k = #{m : e_m ≤ x} − 1
    index = np.asarray(np.searchsorted(edges, coordinate, side="right") - 1)
    inside = (index >= 0) & (index < edges.size - 1)
    return index, np.asarray(inside)


def cell_centres(edges: np.ndarray) -> np.ndarray:
    """The centres of the cells between consecutive edges.

    Examples:
        >>> cell_centres(np.array([0.0, 5.0, 10.0])).tolist()
        [2.5, 7.5]
    """
    # c_i = (e_i + e_{i+1}) / 2
    return 0.5 * (edges[:-1] + edges[1:])


def flat_mesh(
    easting: np.ndarray, northing: np.ndarray, height: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every (x, y, z) node of a 3-D grid as three flat arrays, z fastest.

    Examples:
        >>> x, y, z = flat_mesh(np.arange(2.0), np.arange(3.0), np.arange(4.0))
        >>> x.shape, float(z[1])
        ((24,), 1.0)
    """
    easting_grid, northing_grid, height_grid = np.meshgrid(
        easting, northing, height, indexing="ij"
    )
    return (
        einx.id("x y z -> (x y z)", easting_grid),
        einx.id("x y z -> (x y z)", northing_grid),
        einx.id("x y z -> (x y z)", height_grid),
    )
