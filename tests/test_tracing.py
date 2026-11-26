"""The shared eager-or-traced validation helpers in ``plumax._src.tracing``."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.errors import ParameterError, PlumaxError
from plumax._src.tracing import require, require_positive


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


def test_require_raises_eagerly_on_a_concrete_condition() -> None:
    with pytest.raises(ParameterError, match="z0 must be > 0"):
        require(jnp.asarray(-1.0) > 0.0, "z0 must be > 0")


def test_require_returns_the_value_when_the_condition_holds() -> None:
    value = jnp.asarray([1.0, 2.0])
    assert require(value > 0.0, "positive", value=value) is value


def test_require_attaches_the_check_under_jit() -> None:
    @eqx.filter_jit
    def checked(value: jax.Array) -> jax.Array:
        return require(value > 0.0, "value must be > 0", value=value)

    assert float(checked(jnp.asarray(1.0))) == 1.0
    with pytest.raises(eqx.EquinoxRuntimeError, match="value must be > 0"):
        checked(jnp.asarray(-1.0))


def test_require_needs_a_value_for_a_traced_condition() -> None:
    @jax.jit
    def unchecked(value: jax.Array) -> jax.Array:
        require(value > 0.0, "value must be > 0")
        return value

    with pytest.raises(TypeError, match="needs `value`"):
        unchecked(jnp.asarray(1.0))
