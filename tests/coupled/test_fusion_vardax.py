"""``fuse_observations`` as a vardax state-space OI analysis (#188).

The fusion BLUE used to be hand-rolled in information form with two
``jnp.linalg.inv`` calls. It now runs ``vardax.OptimalInterpolation(form="state")``.
These tests pin it to the numbers the hand-rolled code returned and to a dense
closed form,

    P* = (B⁻¹ + Gᵀ R⁻¹ G)⁻¹,   θ* = θ_b + P* Gᵀ R⁻¹ (y − G θ_b − c),

on a tiny two-instrument problem with a correlated prior, a masked NaN pixel,
variance inflation and a non-zero background.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import CoupledForward, Instrument, PlumeSource, fuse_observations


pytest.importorskip("vardax")


# fuse_observations before #188 (hand-rolled information form, float64)
_PINNED = {
    1: (
        [1.6860995020531266, 0.22473192293625122, 0.22508441851498437],
        [
            [0.06618892778589354, -0.05021017981772721, -0.03876319931402573],
            [-0.05021017981772721, 0.05638197421419106, 0.030079107929562084],
            [-0.03876319931402573, 0.030079107929562084, 0.04769865749083248],
        ],
    ),
    2: (
        [
            1.75991136856091,
            -0.12562325760119486,
            0.21960355687835809,
            0.2526978537096668,
        ],
        [
            [
                0.08249858677522104,
                -0.037347768714622046,
                -0.046038807567280265,
                -0.02429946926308374,
            ],
            [
                -0.037347768714622046,
                0.07947033141899612,
                -0.006963508088081535,
                -0.025878500645795863,
            ],
            [
                -0.046038807567280265,
                -0.006963508088081535,
                0.056568801932823105,
                0.029951152875658034,
            ],
            [
                -0.02429946926308374,
                -0.025878500645795863,
                0.029951152875658034,
                0.05415966031784949,
            ],
        ],
    ),
}


def _case(n_param):
    source = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 5),
        background=0.7,
    )
    first = Instrument(
        "A",
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        retrieval_variance=0.04,
        representation_variance=0.01,
        variance_inflation=np.array([1.0, 2.0, 1.0]),
    )
    second = Instrument(
        "B",
        np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]),
        retrieval_variance=0.09,
        mask=np.array([1.0, 0.0, 1.0, 1.0]),
    )
    if n_param == 1:
        responses = (jnp.array([0.5, 1.5, 0.8]), jnp.array([0.3, 2.0, 1.1, 0.6]))
        prior_mean = jnp.array([1.0, 0.0, 0.1])
        prior_cov = jnp.array([[4.0, 0.3, 0.0], [0.3, 0.25, 0.05], [0.0, 0.05, 0.16]])
    else:
        responses = (
            jnp.array([[0.5, 0.1], [1.5, 0.4], [0.8, 0.9]]),
            jnp.array([[0.3, 0.2], [2.0, 0.1], [1.1, 0.7], [0.6, 1.2]]),
        )
        prior_mean = jnp.array([1.0, 0.5, 0.0, 0.1])
        prior_cov = (
            jnp.diag(jnp.array([4.0, 2.0, 0.25, 0.16]))
            .at[0, 1]
            .set(0.5)
            .at[1, 0]
            .set(0.5)
        )
    forward = CoupledForward(
        source=source, instruments=(first, second), responses=responses
    )
    observations = [jnp.array([1.9, 3.4, 2.2]), jnp.array([1.5, jnp.nan, 2.9, 1.8])]
    return forward, observations, prior_mean, prior_cov


def _dense_closed_form(forward, observations, prior_mean, prior_cov):
    """θ* and P* with explicit inverses, masked pixels dropped."""
    first, second = forward.instruments
    columns = [np.asarray(r) for r in forward.responses]
    response = np.concatenate(
        [c[:, None] if c.ndim == 1 else c for c in columns], axis=0
    )
    indicator = np.zeros((7, 2))
    indicator[:3, 0] = 1.0
    indicator[3:, 1] = 1.0
    design = np.concatenate([response, indicator], axis=1)
    variance = np.concatenate(
        [first.observation_variance(), second.observation_variance()]
    )
    keep = np.concatenate([first.observation_mask(), second.observation_mask()]) > 0
    obs = np.concatenate([np.asarray(o) for o in observations])
    design, variance, obs = design[keep], variance[keep], obs[keep]
    prior_mean = np.asarray(prior_mean)
    # P* = (B⁻¹ + Gᵀ R⁻¹ G)⁻¹
    covariance = np.linalg.inv(
        np.linalg.inv(np.asarray(prior_cov)) + design.T @ (design / variance[:, None])
    )
    # θ* = θ_b + P* Gᵀ R⁻¹ (y − c − G θ_b)
    residual = obs - 0.7 - design @ prior_mean
    mean = prior_mean + covariance @ (design.T @ (residual / variance))
    return mean, covariance


@pytest.mark.parametrize("n_param", [1, 2])
def test_matches_the_hand_rolled_posterior(n_param):
    forward, observations, prior_mean, prior_cov = _case(n_param)
    posterior = fuse_observations(
        forward, observations, prior_mean=prior_mean, prior_covariance=prior_cov
    )
    mean, covariance = _PINNED[n_param]
    np.testing.assert_allclose(np.asarray(posterior.mean), mean, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(posterior.covariance), covariance, rtol=1e-9)


@pytest.mark.parametrize("n_param", [1, 2])
def test_matches_the_dense_closed_form(n_param):
    forward, observations, prior_mean, prior_cov = _case(n_param)
    posterior = fuse_observations(
        forward, observations, prior_mean=prior_mean, prior_covariance=prior_cov
    )
    mean, covariance = _dense_closed_form(forward, observations, prior_mean, prior_cov)
    np.testing.assert_allclose(np.asarray(posterior.mean), mean, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(posterior.covariance), covariance, rtol=1e-9)
    # the derived fields read the same posterior
    np.testing.assert_allclose(
        np.asarray(posterior.bias_std), np.sqrt(np.diag(covariance))[n_param:]
    )


def test_jit_and_grad_through_the_vardax_solve():
    forward, observations, prior_mean, prior_cov = _case(1)

    def emission(first_obs):
        posterior = fuse_observations(
            forward,
            [first_obs, observations[1]],
            prior_mean=prior_mean,
            prior_covariance=prior_cov,
        )
        return posterior.emission_rate

    eager = emission(observations[0])
    jitted = jax.jit(emission)(observations[0])
    assert float(jitted) == pytest.approx(float(eager), rel=1e-12)
    # θ* is linear in y: ∂Q*/∂y_A is row 0 of P* Gᵀ R⁻¹, read off by central
    # differences
    gradient = jax.grad(emission)(observations[0])
    step = 1e-6
    basis = jnp.eye(3)
    differences = jnp.stack(
        [
            (
                emission(observations[0] + step * e)
                - emission(observations[0] - step * e)
            )
            / (2.0 * step)
            for e in basis
        ]
    )
    np.testing.assert_allclose(np.asarray(gradient), np.asarray(differences), rtol=1e-6)
