"""The Tier II footprint emulator, its calibration and inference on it (#90)."""

from __future__ import annotations

import zlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st


pipekit_train = pytest.importorskip("pipekit_train")

from plumax._src.errors import ParameterError
from plumax.lagrangian import (
    FootprintEmulator,
    FootprintSimulator,
    HomogeneousTurbulence,
    MetContextPrior,
    assert_emulator_calibrated,
    emulated_footprint,
    emulator_relative_error,
    ensemble_footprints,
    footprint_dataset,
    footprint_operator,
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


def _simulator(n_particles: int, receptors: jax.Array = RECEPTORS, **grid):
    return FootprintSimulator(
        wind_from_context=_wind,
        receptors=receptors,
        turbulence=TURBULENCE,
        domain_x=grid.get("domain_x", DOMAIN_X),
        domain_y=grid.get("domain_y", DOMAIN_Y),
        n_particles=n_particles,
        **FOOTPRINT_KWARGS,
    )


def _stack(dataset) -> tuple[jax.Array, jax.Array]:
    pairs = list(dataset)
    contexts = jnp.stack([jnp.asarray(context) for context, _ in pairs])
    footprints = jnp.stack([jnp.asarray(footprint) for _, footprint in pairs])
    return contexts, footprints


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Emulator trained on 96 cases (12 for early stopping), 8 held out at 2× particles.

    The pipekit-train splits of 120 cases at 500 particles give 96 training
    and 12 validation cases; the held-out cases are the test split of 80 other
    seeds at 1000 particles.
    """
    cache = str(tmp_path_factory.mktemp("footprint_cache"))
    simulator = _simulator(500)
    data = footprint_dataset(simulator, _operational_met, n_cases=120, cache_dir=cache)
    val = footprint_dataset(
        simulator, _operational_met, n_cases=120, cache_dir=cache, split="val"
    )
    held_out = footprint_dataset(
        _simulator(1000),
        _operational_met,
        n_cases=80,
        seed=1000,
        cache_dir=cache,
        split="test",
    )
    emulator = init_footprint_emulator(
        data, RECEPTORS, width=32, depth=2, key=jax.random.PRNGKey(2)
    )
    emulator = train_footprint_emulator(emulator, data, RECEPTORS, val_dataset=val)
    met_context, footprints = _stack(held_out)
    emulated = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
        emulator, met_context, RECEPTORS
    )
    return emulator, met_context, footprints, emulated


@pytest.mark.slow
def test_footprint_dataset_shapes_and_cache(tmp_path):
    simulator = _simulator(
        50, RECEPTORS[:2], domain_x=(-600.0, 0.0, 3), domain_y=(-200.0, 200.0, 2)
    )
    # 3 cases: the 80 % train split holds 2 of them
    data = footprint_dataset(
        simulator, _operational_met, n_cases=3, cache_dir=str(tmp_path)
    )
    met_context, footprints = _stack(data)
    assert met_context.shape == (2, 2)
    assert footprints.shape == (2, 2, 6)
    assert bool((footprints >= 0.0).all())
    assert float(footprints.sum()) > 0.0
    # a second dataset on the same cache reads the cached footprints back
    again = footprint_dataset(
        simulator, _operational_met, n_cases=3, cache_dir=str(tmp_path)
    )
    np.testing.assert_array_equal(_stack(again)[1], footprints)


@pytest.mark.slow
def test_simulator_matches_the_particle_model_seeded_by_context():
    simulator = _simulator(
        50, RECEPTORS[:1], domain_x=(-600.0, 0.0, 3), domain_y=(-200.0, 200.0, 2)
    )
    context = MetContextPrior(_operational_met)(seed=4)
    first = simulator.step(context, simulator.dt)
    np.testing.assert_array_equal(simulator.step(context, simulator.dt), first)
    other = simulator.step(context + 0.01, simulator.dt)
    assert not bool(jnp.allclose(other, first))
    assert first.shape == (1, 6)
    reference = ensemble_footprints(
        [_wind(context)],
        RECEPTORS[:1],
        TURBULENCE,
        (-600.0, 0.0, 3),
        (-200.0, 200.0, 2),
        n_particles=50,
        seed=zlib.crc32(np.asarray(context).tobytes()),
        **FOOTPRINT_KWARGS,
    )
    np.testing.assert_array_equal(reference[0], first)


def test_footprint_dataset_rejects_no_cases(tmp_path):
    with pytest.raises(ParameterError):
        footprint_dataset(
            _simulator(10), _operational_met, n_cases=0, cache_dir=str(tmp_path)
        )


def test_simulator_rejects_no_particles():
    with pytest.raises(ParameterError):
        _simulator(0)


@pytest.mark.slow
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


@pytest.mark.slow
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


class _ToyForward:
    """Smooth toy footprints (3 obs, 5 cells) from a context, a ForwardModel."""

    dt = 1.0

    def step(self, state: jax.Array, dt: float) -> jax.Array:
        del dt
        weights = jnp.linspace(0.5, 1.5, 5)
        row = 0.1 + state[0] * weights + state[1]
        return jnp.broadcast_to(row, (3, 5))


def _uniform_context(key: jax.Array) -> jax.Array:
    return jax.random.uniform(key, (2,))


def _toy_dataset():
    return pipekit_train.SimulationDataset(
        forward_model=_ToyForward(),
        prior=MetContextPrior(_uniform_context),
        n_samples=10,
    )


@pytest.mark.slow
def test_training_reduces_the_residual():
    data = _toy_dataset()
    contexts, footprints = _stack(data)
    receptors = RECEPTORS[:3]
    initial = init_footprint_emulator(
        data, receptors, width=8, depth=1, key=jax.random.PRNGKey(1)
    )
    trained = train_footprint_emulator(
        initial, data, receptors, max_steps=100, batch_size=4
    )

    def residual(emulator: FootprintEmulator) -> float:
        emulated = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
            emulator, contexts, receptors
        )
        return float(emulator_relative_error(emulated, footprints))

    assert residual(trained) < 0.5 * residual(initial)
    # only the network is trained; the standardisation and the scale stay fixed
    np.testing.assert_array_equal(trained.input_mean, initial.input_mean)
    np.testing.assert_array_equal(trained.input_scale, initial.input_scale)
    np.testing.assert_array_equal(trained.footprint_scale, initial.footprint_scale)


@pytest.mark.slow
def test_early_stopping_ends_training_early():
    data = _toy_dataset()
    receptors = RECEPTORS[:3]
    initial = init_footprint_emulator(
        data, receptors, width=8, depth=1, key=jax.random.PRNGKey(1)
    )
    # a zero learning rate never improves the validation MSE
    stopped = train_footprint_emulator(
        initial,
        data,
        receptors,
        val_dataset=data.with_split("val"),
        max_steps=10_000,
        batch_size=4,
        learning_rate=0.0,
        patience=1,
        eval_every_n_steps=1,
    )
    leaves = jax.tree.leaves(eqx.filter(stopped, eqx.is_inexact_array))
    initial_leaves = jax.tree.leaves(eqx.filter(initial, eqx.is_inexact_array))
    for leaf, initial_leaf in zip(leaves, initial_leaves, strict=True):
        np.testing.assert_array_equal(leaf, initial_leaf)


def test_init_rejects_empty_network():
    with pytest.raises(ParameterError):
        init_footprint_emulator(
            _toy_dataset(), RECEPTORS[:3], width=0, key=jax.random.PRNGKey(0)
        )


def test_relative_error_is_zero_for_the_reference():
    reference = jnp.arange(12.0).reshape(2, 6) + 1.0
    assert float(emulator_relative_error(reference, reference)) == 0.0


@pytest.mark.slow
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
    _, _, footprints, emulated = trained
    error = emulator_relative_error(emulated, footprints)
    assert float(error) < 0.05


@pytest.mark.slow
def test_trained_emulator_is_adjoint_calibrated(trained):
    pytest.importorskip("vardax")
    emulator, met_context, footprints, _ = trained
    assert_emulator_calibrated(
        emulator,
        footprints,
        met_context=met_context,
        receptors=RECEPTORS,
        key=jax.random.PRNGKey(6),
    )


@pytest.mark.slow
def test_step4_matches_step2_posterior(trained):
    vardax = pytest.importorskip("vardax")
    import lineax as lx

    _, _, footprints, emulated = trained
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
    for reference, emulated_case in zip(footprints, emulated, strict=True):
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
