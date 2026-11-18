"""``Instrument`` is an equinox Module: its arrays are pytree leaves (#313)."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import Instrument
from plumax.obs import ColumnObservation


def _instrument() -> Instrument:
    return Instrument(
        name="EMIT",
        receptors=np.zeros((3, 2)),
        averaging_kernel=np.full(3, 0.5),
        retrieval_variance=np.array([1.0, 2.0, 4.0]),
        observation_time=60.0,
    )


def test_arrays_are_leaves_and_metadata_is_static():
    leaves = jax.tree_util.tree_leaves(_instrument())
    # the receptors, the kernel and the variance; the three defaults are floats
    assert sum(isinstance(leaf, np.ndarray) for leaf in leaves) == 3
    assert not any(isinstance(leaf, str) for leaf in leaves)
    assert 60.0 not in [leaf for leaf in leaves if isinstance(leaf, float)]


def test_column_observation_passes_through_filter_jit_whole():
    observation = ColumnObservation(
        instrument=_instrument(),
        enhancement=jnp.array([1.0, 2.0, 3.0]),
        time=np.datetime64("NaT"),
    )

    @eqx.filter_jit
    def weighted_sum(o: ColumnObservation) -> jax.Array:
        # ∑ₙ Aₙ yₙ / σₙ²
        weighted = o.instrument.ak * o.enhancement / o.instrument.observation_variance()
        return jnp.sum(weighted)

    expected = np.sum(0.5 * np.array([1.0, 2.0, 3.0]) / np.array([1.0, 2.0, 4.0]))
    assert float(weighted_sum(observation)) == pytest.approx(expected, rel=1e-6)


def test_built_inside_a_trace():
    @eqx.filter_jit
    def count(receptors: jax.Array) -> int:
        return Instrument(name="EMIT", receptors=receptors).n_obs

    assert count(jnp.zeros((4, 2))) == 4


def test_replace_reruns_the_shape_check():
    with pytest.raises(ValueError, match="averaging_kernel"):
        dataclasses.replace(_instrument(), averaging_kernel=np.ones(2))
    with pytest.raises(ValueError, match=r"\(n_obs, 2\)"):
        Instrument(name="EMIT", receptors=np.zeros(3))
