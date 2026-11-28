"""The shared eager-or-traced validation helpers in ``plumax._src.tracing``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.errors import ParameterError, PlumaxError
from plumax._src.tracing import require_positive


def test_require_positive_names_the_caller_and_the_argument() -> None:
    with pytest.raises(
        ParameterError, match=r"simulate_plume: `wind_speed` must be > 0"
    ):
        require_positive("wind_speed", 0.0, caller="simulate_plume")


def test_require_positive_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        require_positive("std", -1.0)


def test_require_positive_allows_zero_when_asked() -> None:
    require_positive("jitter", 0.0, allow_zero=True)
    with pytest.raises(ParameterError, match="≥ 0"):
        require_positive("jitter", -1e-9, allow_zero=True)


def test_require_positive_checks_every_element_and_rejects_nan() -> None:
    with pytest.raises(ParameterError):
        require_positive("variance", np.array([1.0, 0.0]))
    with pytest.raises(ParameterError):
        require_positive("variance", float("nan"))


def test_require_positive_raises_the_given_error() -> None:
    with pytest.raises(PlumaxError):
        require_positive("tile_m", 0.0, error=PlumaxError)


def test_require_positive_skips_traced_values() -> None:
    @jax.jit
    def halve(value: jax.Array) -> jax.Array:
        require_positive("value", value)
        return value / 2.0

    assert float(halve(jnp.asarray(-2.0))) == -1.0


@pytest.mark.parametrize(
    "pack",
    [lambda v: (v, 2.0), lambda v: [1.0, v], lambda v: {"x": v}],
    ids=["tuple", "list", "dict"],
)
def test_require_positive_skips_a_container_that_holds_a_tracer(pack) -> None:
    @jax.jit
    def halve(value: jax.Array) -> jax.Array:
        require_positive("value", pack(value))
        return value / 2.0

    assert float(halve(jnp.asarray(-2.0))) == -1.0


def test_require_positive_still_checks_a_concrete_container() -> None:
    with pytest.raises(ParameterError):
        require_positive("pair", (1.0, -2.0))
