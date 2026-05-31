"""Tests for the Tier IV coupled multi-instrument fusion (v1: Tier I + AK)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from plumax.coupled import (
    CoupledForward,
    FusionPosterior,
    Instrument,
    PlumeSource,
    build_coupled_forward,
    column_response,
    default_prior,
    fuse_observations,
    predict_observation,
)
from plumax.gauss_plume.plume import simulate_plume


def _grid(nx, ny, x0, x1, y0, y1):
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.reshape(-1), gy.reshape(-1)])


def _source(background=0.0):
    return PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,  # from west → plume on +x
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 21),
        background=background,
    )


def _instrument(name="TROPOMI", retrieval_variance=1e-18):
    return Instrument(
        name,
        _grid(8, 6, 100.0, 2000.0, -300.0, 300.0),
        retrieval_variance=retrieval_variance,
    )


# ── Instrument ───────────────────────────────────────────────────────────────


def test_instrument_validation_and_variance():
    inst = Instrument(
        "X",
        _grid(3, 3, 0, 100, -50, 50),
        retrieval_variance=1.0,
        representation_variance=0.5,
        alignment_variance=0.25,
    )
    assert inst.n_obs == 9
    # R = R_retr + R_repr + R_align summed per receptor.
    np.testing.assert_allclose(np.asarray(inst.observation_variance()), 1.75)
    # Identity AK by default.
    np.testing.assert_allclose(np.asarray(inst.ak), 1.0)


def test_instrument_rejects_bad_receptors_and_ak():
    with pytest.raises(ValueError, match="receptors"):
        Instrument("bad", np.zeros((4, 3)))  # not (n, 2)
    with pytest.raises(ValueError, match="averaging_kernel"):
        Instrument("bad", _grid(2, 2, 0, 1, 0, 1), averaging_kernel=jnp.ones(3))


def test_instrument_rejects_nonpositive_variance():
    inst = Instrument("X", _grid(2, 2, 0, 1, 0, 1), retrieval_variance=0.0)
    with pytest.raises(ValueError, match="variance must be > 0"):
        inst.observation_variance()


# ── Forward + composition correctness ────────────────────────────────────────


def test_column_response_is_unit_q_column():
    # The forward is linear in Q: predict(Q) − background = Q · column_response.
    src = _source()
    inst = _instrument()
    resp = column_response(src, inst)
    assert resp.shape == (inst.n_obs,)
    pred2 = predict_observation(src, inst, emission_rate=2.0)
    np.testing.assert_allclose(np.asarray(pred2), 2.0 * np.asarray(resp), rtol=1e-6)


def test_composition_matches_bare_plume_column():
    # Composition-correctness (design §validation): identity AK → the coupled
    # forward equals the bare simulate_plume column integral at the receptors.
    src = _source()
    receptors = _grid(5, 4, 200.0, 1500.0, -150.0, 150.0)
    inst = Instrument("ID", receptors)  # identity AK
    q = 3.0
    pred = predict_observation(src, inst, emission_rate=q)

    # Reference: simulate_plume on a grid containing the receptor x/y, trapezoid
    # column over the same z, sampled at the receptors.
    xs = np.unique(receptors[:, 0])
    ys = np.unique(receptors[:, 1])
    z = np.asarray(src.column_z)
    ds = simulate_plume(
        emission_rate=q,
        source_location=src.location,
        wind_speed=src.wind_speed,
        wind_direction=src.wind_direction,
        stability_class=src.stability_class,
        domain_x=(float(xs[0]), float(xs[-1]), len(xs)),
        domain_y=(float(ys[0]), float(ys[-1]), len(ys)),
        domain_z=(float(z[0]), float(z[-1]), len(z)),
    )
    col = ds["column_concentration"].values  # (nx, ny), trapezoid over z
    x_to_i = {round(float(v), 6): i for i, v in enumerate(xs)}
    y_to_j = {round(float(v), 6): j for j, v in enumerate(ys)}
    ref = np.array(
        [
            col[x_to_i[round(float(rx), 6)], y_to_j[round(float(ry), 6)]]
            for rx, ry in receptors
        ]
    )
    np.testing.assert_allclose(np.asarray(pred), ref, rtol=1e-5, atol=1e-12)


def test_build_coupled_forward_requires_instruments():
    with pytest.raises(ValueError, match="at least one instrument"):
        build_coupled_forward(_source(), [])


def test_coupled_predict_bias_count_check():
    fwd = build_coupled_forward(_source(), [_instrument("A"), _instrument("B")])
    assert isinstance(fwd, CoupledForward)
    with pytest.raises(ValueError, match="biases"):
        fwd.predict(1.0, biases=[0.0])  # wrong count


# ── Fusion: twin recovery, fusion benefit, bias, hold-out ────────────────────


def _two_instrument_forward():
    src = _source()
    trop = Instrument(
        "TROPOMI",
        _grid(8, 6, 100.0, 2000.0, -300.0, 300.0),
        retrieval_variance=1e-18,
    )
    emit = Instrument(
        "EMIT",
        _grid(10, 8, 50.0, 1500.0, -200.0, 200.0),
        retrieval_variance=1e-18,
    )
    return build_coupled_forward(src, [trop, emit])


def test_twin_recovery_multi_instrument():
    fwd = _two_instrument_forward()
    q_true, bias_true = 2.5, [0.0, 3e-9]
    y = fwd.predict(q_true, biases=bias_true)
    mean, cov = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=10.0,
        bias_prior_std=1e-8,
    )
    post = fuse_observations(fwd, y, prior_mean=mean, prior_covariance=cov)
    assert isinstance(post, FusionPosterior)
    assert post.emission_rate == pytest.approx(q_true, rel=1e-3)
    np.testing.assert_allclose(
        np.asarray(post.biases), np.asarray(bias_true), atol=2e-10
    )
    assert post.instrument_names == ("TROPOMI", "EMIT")


def test_fusion_reduces_emission_uncertainty():
    # Two instruments must constrain Q at least as tightly as one (more data).
    fwd2 = _two_instrument_forward()
    q_true = 2.5
    y2 = fwd2.predict(q_true)
    mean2, cov2 = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=10.0,
        bias_prior_std=1e-8,
    )
    post2 = fuse_observations(fwd2, y2, prior_mean=mean2, prior_covariance=cov2)

    fwd1 = build_coupled_forward(fwd2.source, [fwd2.instruments[0]])
    mean1, cov1 = default_prior(
        n_instruments=1,
        emission_prior_mean=1.0,
        emission_prior_std=10.0,
        bias_prior_std=1e-8,
    )
    post1 = fuse_observations(fwd1, [y2[0]], prior_mean=mean1, prior_covariance=cov1)

    assert post2.emission_std <= post1.emission_std + 1e-12


def test_posterior_covariance_is_psd_and_shaped():
    fwd = _two_instrument_forward()
    y = fwd.predict(2.0)
    mean, cov = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=5.0,
        bias_prior_std=1e-8,
    )
    post = fuse_observations(fwd, y, prior_mean=mean, prior_covariance=cov)
    p = np.asarray(post.covariance)
    assert p.shape == (3, 3)  # Q + 2 biases
    np.testing.assert_allclose(p, p.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(p)) > -1e-10


def test_recovers_per_instrument_bias():
    # A genuine inter-instrument offset must be absorbed into bias_inst, not Q.
    fwd = _two_instrument_forward()
    q_true = 1.8
    biases = [2e-9, -1.5e-9]
    y = fwd.predict(q_true, biases=biases)
    mean, cov = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=10.0,
        bias_prior_std=1e-7,
    )
    post = fuse_observations(fwd, y, prior_mean=mean, prior_covariance=cov)
    assert post.emission_rate == pytest.approx(q_true, rel=1e-3)
    np.testing.assert_allclose(np.asarray(post.biases), np.asarray(biases), atol=2e-10)


def test_cross_instrument_holdout():
    # Invert on instrument 0 only, then predict instrument 1's observations from
    # the recovered Q; with a clean twin they must match (design §hold-out).
    fwd = _two_instrument_forward()
    q_true = 2.2
    y = fwd.predict(q_true)
    fwd0 = build_coupled_forward(fwd.source, [fwd.instruments[0]])
    m0, c0 = default_prior(
        n_instruments=1,
        emission_prior_mean=1.0,
        emission_prior_std=10.0,
        bias_prior_std=1e-8,
    )
    post0 = fuse_observations(fwd0, [y[0]], prior_mean=m0, prior_covariance=c0)
    # Predict the held-out instrument 1 from the recovered Q.
    held = predict_observation(fwd.source, fwd.instruments[1], post0.emission_rate)
    np.testing.assert_allclose(
        np.asarray(held), np.asarray(y[1]), rtol=1e-3, atol=1e-12
    )


# ── Fusion validation ────────────────────────────────────────────────────────


def test_fuse_observations_shape_validation():
    fwd = _two_instrument_forward()
    y = fwd.predict(2.0)
    mean, cov = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=5.0,
        bias_prior_std=1e-8,
    )
    with pytest.raises(ValueError, match="prior_mean"):
        fuse_observations(fwd, y, prior_mean=jnp.zeros(2), prior_covariance=cov)
    with pytest.raises(ValueError, match="observation vectors"):
        fuse_observations(fwd, [y[0]], prior_mean=mean, prior_covariance=cov)


def test_default_prior_rejects_nonpositive_std():
    with pytest.raises(ValueError, match="standard deviations"):
        default_prior(
            n_instruments=1,
            emission_prior_mean=1.0,
            emission_prior_std=0.0,
            bias_prior_std=1.0,
        )
