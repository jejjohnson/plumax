"""The Tier II footprint emulator, its calibration and inference on it (#90)."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax._src.errors import ParameterError
from plumax.lagrangian import (
    FootprintEmulator,
    FootprintTrainingSet,
    HomogeneousTurbulence,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    footprint_operator,
    generate_training_set,
    init_footprint_emulator,
    linear_gaussian_inversion,
    matern32_covariance,
    train_footprint_emulator,
    uniform_wind,
)


TURBULENCE = HomogeneousTurbulence(0.5, 0.5, 0.3, 30.0, 30.0, 10.0)
RECEPTORS = jnp.array([[0.0, y, 10.0] for y in (-225.0, -75.0, 75.0, 225.0)])
DOMAIN_X = (-1500.0, 300.0, 6)
DOMAIN_Y = (-450.0, 450.0, 4)
FOOTPRINT_KWARGS = {"t_back": 300.0, "dt": 10.0, "pbl_height": 200.0}
N_CELLS = DOMAIN_X[2] * DOMAIN_Y[2]


def _operational_met(key: jax.Array) -> jax.Array:
    """Mean wind (u, v) at the toy facility: 3 m/s from the west, ±10 %, ±6°."""
    speed_key, direction_key = jax.random.split(key)
    speed = 3.0 * jnp.exp(0.1 * jax.random.normal(speed_key))
    direction = jnp.deg2rad(6.0) * jax.random.normal(direction_key)
    return jnp.stack([speed * jnp.cos(direction), speed * jnp.sin(direction)])


def _wind(context: jax.Array):
    return uniform_wind(float(context[0]), float(context[1]))


def _untrained(n_cells: int = 5) -> FootprintEmulator:
    network = eqx.nn.MLP(4, n_cells, 8, 2, key=jax.random.PRNGKey(0))
    return FootprintEmulator(
        network,
        jnp.array([3.0, 0.0, 0.0, 0.0]),
        jnp.array([0.3, 0.3, 100.0, 100.0]),
        jnp.asarray(0.2),
    )


@pytest.fixture(scope="module")
def trained():
    """Emulator trained on 96 cases, and 8 held-out cases at more particles."""
    common = {
        "met_sampler": _operational_met,
        "wind_from_context": _wind,
        "receptors": RECEPTORS,
        "turbulence": TURBULENCE,
        "domain_x": DOMAIN_X,
        "domain_y": DOMAIN_Y,
        **FOOTPRINT_KWARGS,
    }
    data = generate_training_set(
        jax.random.PRNGKey(0), n_cases=96, n_particles=500, **common
    )
    held_out = generate_training_set(
        jax.random.PRNGKey(1), n_cases=8, n_particles=1000, **common
    )
    emulator = init_footprint_emulator(
        data, width=32, depth=2, key=jax.random.PRNGKey(2)
    )
    emulator = train_footprint_emulator(emulator, data, max_steps=500)
    emulated = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
        emulator, held_out.met_context, RECEPTORS
    )
    return emulator, held_out, emulated


def test_generate_training_set_shapes():
    data = generate_training_set(
        jax.random.PRNGKey(0),
        met_sampler=_operational_met,
        wind_from_context=_wind,
        receptors=RECEPTORS[:2],
        turbulence=TURBULENCE,
        domain_x=(-600.0, 0.0, 3),
        domain_y=(-200.0, 200.0, 2),
        n_cases=2,
        n_particles=50,
        **FOOTPRINT_KWARGS,
    )
    assert data.met_context.shape == (2, 2)
    assert data.footprints.shape == (2, 2, 6)
    assert bool((data.footprints >= 0.0).all())
    assert float(data.footprints.sum()) > 0.0


def test_generate_training_set_rejects_no_cases():
    with pytest.raises(ParameterError):
        generate_training_set(
            jax.random.PRNGKey(0),
            met_sampler=_operational_met,
            wind_from_context=_wind,
            receptors=RECEPTORS,
            turbulence=TURBULENCE,
            domain_x=DOMAIN_X,
            domain_y=DOMAIN_Y,
            n_cases=0,
        )


@settings(max_examples=20, deadline=None)
@given(
    u=st.floats(-10.0, 10.0),
    v=st.floats(-10.0, 10.0),
    x=st.floats(-1e4, 1e4),
    y=st.floats(-1e4, 1e4),
)
def test_emulated_rows_are_non_negative(u, v, x, y):
    row = _untrained()(jnp.array([u, v]), jnp.array([x, y]))
    assert row.shape == (5,)
    assert bool((row >= 0.0).all())


def test_emulated_footprint_matches_loop_jit_and_vmap():
    emulator = _untrained()
    context = jnp.array([3.1, 0.2])
    looped = jnp.stack([emulator(context, r[:2]) for r in RECEPTORS])
    eager = emulated_footprint(emulator, context, RECEPTORS)
    jitted = eqx.filter_jit(emulated_footprint)(emulator, context, RECEPTORS)
    contexts = jnp.stack([context, context + 0.5])
    batched = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
        emulator, contexts, RECEPTORS
    )
    np.testing.assert_allclose(eager, looped, rtol=1e-12)
    np.testing.assert_allclose(jitted, looped, rtol=1e-12)
    np.testing.assert_allclose(batched[0], looped, rtol=1e-12)


def test_emulator_gradient_matches_finite_differences():
    emulator = _untrained()
    context = jnp.array([3.1, 0.2])

    def total(c: jax.Array) -> jax.Array:
        return emulated_footprint(emulator, c, RECEPTORS).sum()

    gradient = jax.grad(total)(context)
    step = 1e-6
    central = jnp.stack(
        [
            (total(context + step * e) - total(context - step * e)) / (2 * step)
            for e in jnp.eye(2)
        ]
    )
    np.testing.assert_allclose(gradient, central, rtol=1e-6)


def test_training_reduces_the_residual():
    contexts = jax.random.uniform(jax.random.PRNGKey(0), (6, 2))
    footprints = 0.1 + _toy_footprints(contexts)
    data = FootprintTrainingSet(contexts, RECEPTORS[:3], footprints)
    initial = init_footprint_emulator(data, width=8, depth=1, key=jax.random.PRNGKey(1))
    trained = train_footprint_emulator(initial, data, max_steps=50)

    def residual(emulator: FootprintEmulator) -> float:
        emulated = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
            emulator, contexts, RECEPTORS[:3]
        )
        return float(emulator_relative_error(emulated, footprints))

    assert residual(trained) < 0.5 * residual(initial)


def _toy_footprints(contexts: jax.Array) -> jax.Array:
    """Smooth toy footprints (case, 3 obs, 5 cells) from the contexts."""
    weights = jnp.linspace(0.5, 1.5, 5)
    rows = contexts[:, :1, None] * weights + contexts[:, 1:, None]
    return jnp.broadcast_to(rows, (contexts.shape[0], 3, 5))


def test_init_rejects_empty_network():
    data = _untrained_data()
    with pytest.raises(ParameterError):
        init_footprint_emulator(data, width=0, key=jax.random.PRNGKey(0))


def _untrained_data():
    return FootprintTrainingSet(
        met_context=jnp.array([[3.0, 0.0], [3.2, 0.1]]),
        receptors=RECEPTORS[:2],
        footprints=jnp.full((2, 2, 5), 0.1),
    )


def test_relative_error_is_zero_for_the_reference():
    reference = jnp.arange(12.0).reshape(2, 6) + 1.0
    assert float(emulator_relative_error(reference, reference)) == 0.0


def test_adjoint_calibration():
    pytest.importorskip("vardax")
    emulator = _untrained()
    context = jnp.array([3.1, 0.2])
    emulated = emulated_footprint(emulator, context, RECEPTORS)
    # ⟨F_emu x, y⟩ = ⟨x, F_emuᵀ y⟩ through the operator the inversions use
    operator = footprint_operator(emulated)
    source = jax.random.normal(jax.random.PRNGKey(3), (1, 5))
    obs = jax.random.normal(jax.random.PRNGKey(4), (1, 4))
    (adjoint,) = jax.linear_transpose(operator.fn, source)(obs)
    forward_pairing = jnp.vdot(operator.fn(source), obs)
    adjoint_pairing = jnp.vdot(source, adjoint)
    np.testing.assert_allclose(forward_pairing, adjoint_pairing, rtol=1e-12)
    # vardax's calibration gate: passes on the emulator's own map, fails at +20 %
    assert_emulator_calibrated(
        emulator,
        emulated[None],
        met_context=context[None],
        receptors=RECEPTORS,
        key=jax.random.PRNGKey(5),
    )
    with pytest.raises(AssertionError):
        assert_emulator_calibrated(
            emulator,
            1.2 * emulated[None],
            met_context=context[None],
            receptors=RECEPTORS,
            key=jax.random.PRNGKey(5),
        )


@pytest.mark.slow
def test_residual_below_5pct(trained):
    _, held_out, emulated = trained
    error = emulator_relative_error(emulated, held_out.footprints)
    assert float(error) < 0.05


@pytest.mark.slow
def test_trained_emulator_is_adjoint_calibrated(trained):
    pytest.importorskip("vardax")
    emulator, held_out, _ = trained
    assert_emulator_calibrated(
        emulator,
        held_out.footprints,
        met_context=held_out.met_context,
        receptors=RECEPTORS,
        key=jax.random.PRNGKey(6),
    )


@pytest.mark.slow
def test_step4_matches_step2_posterior(trained):
    vardax = pytest.importorskip("vardax")
    import lineax as lx

    _, held_out, emulated = trained
    x_edges = jnp.linspace(DOMAIN_X[0], DOMAIN_X[1], DOMAIN_X[2] + 1)
    y_edges = jnp.linspace(DOMAIN_Y[0], DOMAIN_Y[1], DOMAIN_Y[2] + 1)
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = jnp.meshgrid(x_centres, y_centres, indexing="ij")
    coordinates = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
    prior_mean = jnp.full(N_CELLS, 1e-6)
    prior_cov = matern32_covariance(
        coordinates, variance=(1e-6) ** 2, length_scale=400.0
    )
    # synthetic source: a 3e-6 kg m⁻² s⁻¹ hotspot upwind of the receptors
    true_source = prior_mean.at[N_CELLS // 2].set(3e-6)
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    for reference, emulated_case in zip(held_out.footprints, emulated, strict=True):
        noise_std = 0.02 * float(jnp.max(reference @ true_source))
        noise = noise_std * jax.random.normal(
            jax.random.PRNGKey(7), (RECEPTORS.shape[0],)
        )
        observation = reference @ true_source + noise
        obs_variance = jnp.full(RECEPTORS.shape[0], noise_std**2)
        inversion = {
            "prior_mean": prior_mean,
            "prior_cov": prior_cov,
            "obs_variance": obs_variance,
        }
        # Step 2 on the particle footprint, Step 4 on the emulated one
        oracle = linear_gaussian_inversion(reference, observation, **inversion)
        fast = linear_gaussian_inversion(emulated_case, observation, **inversion)
        vardax.assert_posterior_agreement(
            vardax.Posterior(mean=fast.mean),
            vardax.Posterior(
                mean=oracle.mean, cov=lx.MatrixLinearOperator(oracle.covariance, psd)
            ),
            tolerance_sigma=1.0,
        )
        np.testing.assert_allclose(fast.std, oracle.std, rtol=0.1)
