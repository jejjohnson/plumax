"""The per-instrument quality-flag aggregator feeding R and the masks (#105)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from plumax._src.errors import ParameterError
from plumax.coupled import (
    FlagPolicy,
    Instrument,
    QualityFlag,
    QualityPolicy,
    apply_quality,
    build_coupled_forward,
    contribution_report,
    fuse_observations,
    stack_observations,
)


class _LineSource:
    """A stub backend: response 1 + x / 100 at each receptor."""

    background = 0.0

    def column_response(self, instrument):
        return 1.0 + jnp.asarray(instrument.receptors)[:, 0] / 100.0


def _receptors(xs):
    return np.column_stack([np.asarray(xs, dtype=float), np.zeros(len(xs))])


FLAGS_A = jnp.array([0, QualityFlag.CLOUD, 0, QualityFlag.AOD])
FLAGS_B = jnp.array([QualityFlag.NOT_CONVERGED, 0, 0])
PRIOR_MEAN = jnp.zeros(3)
PRIOR_COVARIANCE = jnp.diag(jnp.array([4.0, 1.0, 1.0]))


def _instruments():
    a = Instrument("A", _receptors([0.0, 50.0, 100.0, 150.0]), retrieval_variance=0.5)
    b = Instrument("B", _receptors([20.0, 80.0, 140.0]), retrieval_variance=2.0)
    return a, b


def _observations(forward):
    clean = forward.predict(1.5, [0.2, -0.3])
    # the flagged pixels are off by a lot
    return [
        clean[0] + jnp.array([0.0, 5.0, 0.0, 0.0]),
        clean[1] + jnp.array([-4.0, 0, 0]),
    ]


def _fuse(instruments, observations, **kwargs):
    forward = build_coupled_forward(_LineSource(), instruments)
    return fuse_observations(
        forward,
        observations,
        prior_mean=PRIOR_MEAN,
        prior_covariance=PRIOR_COVARIANCE,
        **kwargs,
    )


def test_mask_equals_infinite_inflation():
    a, b = _instruments()
    observations = _observations(build_coupled_forward(_LineSource(), [a, b]))
    mask = QualityPolicy(FlagPolicy.MASK)
    inflate = QualityPolicy(FlagPolicy.INFLATE, inflate_factor=1e12)
    masked = _fuse(
        [apply_quality(a, FLAGS_A, mask), apply_quality(b, FLAGS_B, mask)],
        observations,
    )
    inflated = _fuse(
        [apply_quality(a, FLAGS_A, inflate), apply_quality(b, FLAGS_B, inflate)],
        observations,
    )
    np.testing.assert_allclose(masked.mean, inflated.mean, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        masked.covariance, inflated.covariance, rtol=1e-6, atol=1e-9
    )
    # only the cloud and convergence flags reject: the AOD pixel stays
    assert masked.pixels_used.tolist() == [3, 2]
    assert contribution_report(
        [apply_quality(a, FLAGS_A), apply_quality(b, FLAGS_B)]
    ) == {"A": 3, "B": 2}


def test_policy_through_fuse_observations_matches_apply_quality():
    a, b = _instruments()
    observations = _observations(build_coupled_forward(_LineSource(), [a, b]))
    flagged = [
        Instrument("A", a.receptors, retrieval_variance=0.5, flags=FLAGS_A),
        Instrument("B", b.receptors, retrieval_variance=2.0, flags=FLAGS_B),
    ]
    through_fusion = _fuse(flagged, observations, quality=QualityPolicy())
    up_front = _fuse([apply_quality(inst) for inst in flagged], observations)
    np.testing.assert_allclose(through_fusion.mean, up_front.mean, rtol=1e-12)
    # flags alone, without a policy, change nothing
    unscreened = _fuse(flagged, observations)
    plain = _fuse([a, b], observations)
    np.testing.assert_allclose(unscreened.mean, plain.mean, rtol=1e-12)


def test_fully_flagged_instrument_dropped():
    a, b = _instruments()
    observations = _observations(build_coupled_forward(_LineSource(), [a, b]))
    cloudy = apply_quality(b, jnp.full(3, QualityFlag.CLOUD))
    # a NaN under the mask must not reach the posterior
    observations[1] = jnp.full(3, jnp.nan)
    both = _fuse([a, cloudy], observations)
    assert contribution_report([a, cloudy]) == {"A": 4, "B": 0}
    assert both.pixels_used.tolist() == [4, 0]
    assert bool(jnp.all(jnp.isfinite(both.mean)))
    alone = fuse_observations(
        build_coupled_forward(_LineSource(), [a]),
        observations[:1],
        prior_mean=PRIOR_MEAN[:2],
        prior_covariance=PRIOR_COVARIANCE[:2, :2],
    )
    np.testing.assert_allclose(both.mean[:2], alone.mean, rtol=1e-12)
    # B's bias stays at its prior
    np.testing.assert_allclose(both.biases[1], 0.0, atol=1e-15)
    np.testing.assert_allclose(both.bias_std[1], 1.0, rtol=1e-12)


def test_vardax_consumes_the_mask_and_diagonal_r():
    vdx = pytest.importorskip("vardax")
    a, b = _instruments()
    screened = [
        apply_quality(a, FLAGS_A),
        apply_quality(b, FLAGS_B, QualityPolicy(FlagPolicy.INFLATE, 50.0)),
    ]
    forward = build_coupled_forward(_LineSource(), screened)
    observations = _observations(forward)
    observations[0] = observations[0].at[1].set(jnp.nan)
    stacked = stack_observations(screened, observations)
    assert float(stacked.values[1]) == 0.0

    # y = G θ over θ = [Q, bias_A, bias_B]
    design = jnp.zeros((7, 3))
    design = (
        design.at[:4, 0].set(forward.responses[0]).at[4:, 0].set(forward.responses[1])
    )
    design = design.at[:4, 1].set(1.0).at[4:, 2].set(1.0)
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    # B on a (1, 3) control row
    prior_cov_op = lx.FunctionLinearOperator(
        lambda x: x @ PRIOR_COVARIANCE, jax.ShapeDtypeStruct((1, 3), float), psd
    )
    model = vdx.ThreeDVar(
        obs_op=vdx.CallableObs(fn=lambda theta: theta @ design.T),
        prior_mean=PRIOR_MEAN[None],
        prior_cov_op=prior_cov_op,
        obs_cov_op=stacked.obs_cov_op(),
        cov_solver=lx.AutoLinearSolver(well_posed=True),
    )
    batch = stacked.batch()
    analysis = model(batch)[0, 0]
    laplace = vdx.LaplaceCovariance(solver=lx.AutoLinearSolver(well_posed=True))(
        analysis[None], model, batch
    )
    covariance = jax.vmap(lambda e: laplace.cov.mv(e[None])[0])(jnp.eye(3))
    closed_form = fuse_observations(
        forward,
        observations,
        prior_mean=PRIOR_MEAN,
        prior_covariance=PRIOR_COVARIANCE,
    )
    np.testing.assert_allclose(analysis, closed_form.mean, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(covariance, closed_form.covariance, rtol=1e-8)


def test_quality_defaults_and_validation():
    a, _ = _instruments()
    assert a.observation_mask().tolist() == [1.0] * 4
    assert contribution_report([a]) == {"A": 4}
    once = apply_quality(a, FLAGS_A)
    twice = apply_quality(once)
    assert twice.observation_mask().tolist() == once.observation_mask().tolist()
    clear = apply_quality(a)
    assert clear.observation_mask().tolist() == [1.0] * 4
    with pytest.raises(ParameterError):
        QualityPolicy(inflate_factor=0.5)
    with pytest.raises(ValueError, match="repeat"):
        contribution_report([a, a])
    with pytest.raises(ValueError, match="flags"):
        Instrument("A", a.receptors, flags=jnp.zeros(2, dtype=int))
    with pytest.raises(ValueError, match="observation vectors"):
        stack_observations([a], [])
