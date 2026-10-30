"""Tier II and Tier III transport backends behind the coupled forward (#101).

The grids are tiny on purpose: these check the plumbing (one column, one set of
units, one fusion harness for every tier), not transport physics.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.errors import DomainError, GridMismatchError
from plumax.coupled import (
    EulerianSource,
    Instrument,
    LagrangianSource,
    PlumeSource,
    TransportBackend,
    build_coupled_forward,
    default_prior,
    fuse_observations,
)
from plumax.gauss_plume.dispersion import get_dispersion_params
from plumax.gauss_plume.observation import ColumnObservation
from plumax.gauss_plume.plume import plume_concentration_vmap
from plumax.lagrangian import HomogeneousTurbulence, compute_footprint, uniform_wind
from plumax.les_fvm.fourdvar import build_forward


DOMAIN_X = (-200.0, 600.0, 8)
DOMAIN_Y = (-200.0, 200.0, 4)
COLUMN_Z = jnp.array([0.0, 40.0, 80.0])
TURBULENCE = HomogeneousTurbulence(0.5, 0.5, 0.3, 20.0, 20.0, 10.0)
PARTICLES = {"n_particles": 100, "t_back": 60.0, "dt": 5.0, "pbl_height": 150.0}
RECEPTORS = np.array([[150.0, 0.0], [250.0, 50.0]])


def _lagrangian(**settings) -> LagrangianSource:
    return LagrangianSource.point(
        (50.0, 0.0),
        winds=[uniform_wind(4.0, 0.0)],
        turbulence=TURBULENCE,
        domain_x=DOMAIN_X,
        domain_y=DOMAIN_Y,
        column_z=COLUMN_Z,
        **{**PARTICLES, **settings},
    )


def _eulerian_forward(receptor_index=None):
    return build_forward(
        domain_x=(0.0, 400.0, 20),
        domain_y=(-100.0, 100.0, 10),
        domain_z=(0.0, 80.0, 4),
        save_times=jnp.linspace(0.0, 40.0, 3),
        source_location=(50.0, 0.0, 20.0),
        uniform_wind=(4.0, 0.0, 0.0),
        eddy_diffusivity=2.0,
        receptor_index=receptor_index,
    )


# ── Tier I is unchanged ──────────────────────────────────────────────────────


def test_plume_source_forward_and_fusion_unchanged():
    # Reference values computed on feat/85-masked-k-multisource (20735d1),
    # before the backends existed.
    source = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=250.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 11),
        background=1e-6,
    )
    first = Instrument(
        "A",
        np.array([[300.0, 0.0], [600.0, 50.0], [900.0, -40.0]]),
        averaging_kernel=np.array([1.0, 0.9, 0.8]),
        retrieval_variance=1e-12,
    )
    second = Instrument(
        "B",
        np.array([[400.0, 20.0], [800.0, 0.0]]),
        retrieval_variance=4e-12,
        representation_variance=1e-12,
    )
    forward = build_coupled_forward(source, [first, second])
    assert isinstance(source, TransportBackend)
    np.testing.assert_allclose(
        forward.responses[0],
        [8.575728599489481e-08, 3.4502974192353653e-06, 4.6293044473406e-10],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        forward.responses[1],
        [1.1871150518534409e-06, 2.0223824341937198e-08],
        rtol=1e-12,
    )
    predicted = forward.predict(2.5, [1e-6, -2e-6])
    np.testing.assert_allclose(
        predicted[0],
        [2.214393214987237e-06, 1.0625743548088415e-05, 2.001157326111835e-06],
        rtol=1e-12,
    )
    mean, covariance = default_prior(
        n_instruments=2,
        emission_prior_mean=1.0,
        emission_prior_std=2.0,
        bias_prior_std=5e-6,
    )
    posterior = fuse_observations(
        forward, predicted, prior_mean=mean, prior_covariance=covariance
    )
    np.testing.assert_allclose(
        posterior.mean,
        [2.4545734516146243, 1.039688088648922e-06, -1.7932521646013506e-06],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        posterior.covariance,
        [
            [0.12199300579204715, -1.4191790013571596e-07, -6.694859023484863e-08],
            [-1.4191790013571596e-07, 4.940444594236586e-13, 7.788318093721083e-14],
            [-6.694859023484863e-08, 7.788318093721083e-14, 2.309468016225912e-12],
        ],
        rtol=1e-9,
    )
    assert posterior.emission_rate.shape == ()


# ── identity composition: coupled forward == bare transport forward ──────────


def test_identity_composition_all_tiers():
    instrument = Instrument("single", RECEPTORS)
    emission_rate, background = 2.0, 3e-7

    # Tier I: the bare plume, integrated over the column
    plume = PlumeSource(
        location=(50.0, 0.0, 20.0),
        wind_speed=4.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=COLUMN_Z,
        background=background,
    )
    n_levels = COLUMN_Z.shape[0]
    x = jnp.repeat(jnp.asarray(RECEPTORS[:, 0]), n_levels)
    y = jnp.repeat(jnp.asarray(RECEPTORS[:, 1]), n_levels)
    z = jnp.tile(COLUMN_Z, RECEPTORS.shape[0])
    u, v = plume.wind_uv()
    concentration = plume_concentration_vmap(
        x, y, z, 50.0, 0.0, 20.0, u, v, emission_rate, get_dispersion_params("D")
    )
    profiles = concentration.reshape(RECEPTORS.shape[0], n_levels)
    bare = jnp.trapezoid(profiles, x=COLUMN_Z, axis=1) + background
    coupled = build_coupled_forward(plume, [instrument]).predict(emission_rate)[0]
    np.testing.assert_allclose(coupled, bare, rtol=1e-12)

    # Tier II: footprints of every receptor level, ρ_air F q, then the column
    lagrangian = _lagrangian(background=background)
    pattern = np.asarray(lagrangian.emission_pattern)
    rows = []
    for index, (rx, ry, rz) in enumerate(
        (rx, ry, float(rz)) for rx, ry in RECEPTORS for rz in COLUMN_Z
    ):
        footprint, _, _ = compute_footprint(
            (rx, ry, rz),
            TURBULENCE,
            DOMAIN_X,
            DOMAIN_Y,
            wind=uniform_wind(4.0, 0.0),
            seed=index,
            **PARTICLES,
        )
        rows.append(np.asarray(footprint).reshape(-1) @ pattern)
    mixing = np.asarray(rows).reshape(RECEPTORS.shape[0], n_levels)
    bare = (
        np.trapezoid(1.2 * emission_rate * mixing, x=np.asarray(COLUMN_Z), axis=1)
        + background
    )
    coupled = build_coupled_forward(lagrangian, [instrument]).predict(emission_rate)
    assert float(np.max(bare)) > 0.0
    np.testing.assert_allclose(coupled[0], bare, rtol=1e-10)

    # Tier III: les_fvm's own column operator at the receptor cells
    eulerian = EulerianSource(forward=_eulerian_forward(), background=background)
    overpass = Instrument("single", RECEPTORS, observation_time=20.0)
    index = eulerian.time_index(overpass)
    receptor_index = ColumnObservation.for_grid(
        eulerian.plume_grid, instrument=overpass
    ).receptor_index
    bare_forward = _eulerian_forward(receptor_index=receptor_index)
    bare = bare_forward.predict(jnp.full(3, emission_rate))[index] + background
    coupled = build_coupled_forward(eulerian, [overpass]).predict(emission_rate)[0]
    assert float(jnp.max(bare)) > 0.0
    np.testing.assert_allclose(coupled, bare, rtol=1e-10)


# ── design-matrix responses widen fusion, nothing else ───────────────────────


class _MatrixSource:
    """A stub backend with a fixed design matrix (two emission parameters)."""

    background = 0.0

    def column_response(self, instrument):
        x = jnp.asarray(instrument.receptors)[:, 0]
        return jnp.stack([1.0 + x / 100.0, 2.0 - x / 200.0], axis=1)


def test_fusion_with_design_matrix_matches_closed_form():
    instruments = [
        Instrument("A", np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])),
        Instrument("B", np.array([[50.0, 0.0], [150.0, 0.0]]), retrieval_variance=2.0),
    ]
    forward = build_coupled_forward(_MatrixSource(), instruments)
    truth = jnp.array([1.5, 0.5])
    observations = forward.predict(truth, [0.1, -0.2])
    prior_mean = jnp.zeros(4)
    prior_covariance = jnp.diag(jnp.array([4.0, 4.0, 1.0, 1.0]))
    posterior = fuse_observations(
        forward,
        observations,
        prior_mean=prior_mean,
        prior_covariance=prior_covariance,
    )
    # closed form over [q₁, q₂, bias_A, bias_B]
    g = np.zeros((5, 4))
    g[:, :2] = np.concatenate([np.asarray(r) for r in forward.responses])
    g[:3, 2] = 1.0
    g[3:, 3] = 1.0
    r_inv = np.diag([1.0, 1.0, 1.0, 0.5, 0.5])
    precision = np.linalg.inv(np.asarray(prior_covariance)) + g.T @ r_inv @ g
    covariance = np.linalg.inv(precision)
    y = np.concatenate([np.asarray(o) for o in observations])
    mean = covariance @ (g.T @ r_inv @ y)
    assert posterior.emission_rate.shape == (2,)
    np.testing.assert_allclose(posterior.mean, mean, rtol=1e-10)
    np.testing.assert_allclose(posterior.covariance, covariance, rtol=1e-10)


def test_lagrangian_patterns_give_a_design_matrix():
    point = _lagrangian()
    n_cells = DOMAIN_X[2] * DOMAIN_Y[2]
    second = jnp.zeros(n_cells).at[5].set(1.0 / 10_000.0)
    patterns = jnp.stack([point.emission_pattern, second], axis=1)
    widened = dataclasses.replace(point, emission_pattern=patterns)
    instrument = Instrument("single", RECEPTORS)
    matrix = widened.column_response(instrument)
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(
        matrix[:, 0], point.column_response(instrument), rtol=1e-12
    )


# ── errors ───────────────────────────────────────────────────────────────────


def test_backend_errors():
    with pytest.raises(DomainError):
        LagrangianSource.point(
            (5_000.0, 0.0),
            winds=[uniform_wind(4.0, 0.0)],
            turbulence=TURBULENCE,
            domain_x=DOMAIN_X,
            domain_y=DOMAIN_Y,
            column_z=COLUMN_Z,
        )
    with pytest.raises(GridMismatchError):
        LagrangianSource(
            emission_pattern=jnp.ones(3),
            winds=(uniform_wind(4.0, 0.0),),
            turbulence=TURBULENCE,
            domain_x=DOMAIN_X,
            domain_y=DOMAIN_Y,
            column_z=COLUMN_Z,
        )
    with pytest.raises(ValueError, match="at least one"):
        LagrangianSource(
            emission_pattern=jnp.ones(32),
            winds=(),
            turbulence=TURBULENCE,
            domain_x=DOMAIN_X,
            domain_y=DOMAIN_Y,
            column_z=COLUMN_Z,
        )
    eulerian = EulerianSource(forward=_eulerian_forward())
    with pytest.raises(GridMismatchError, match="save times"):
        eulerian.column_response(Instrument("late", RECEPTORS, observation_time=7.0))


# ── cross-tier consistency ───────────────────────────────────────────────────


@pytest.mark.slow
def test_cross_tier_posteriors_overlap():
    # Stationary met (4 m/s along x) for all three tiers and one truth Q. Each
    # tier simulates its own observations with the same relative noise draw,
    # because the three transports differ in physics at this resolution; the
    # check is that one fusion harness serves all of them.
    truth = 2.0
    instrument = Instrument("single", RECEPTORS, observation_time=40.0)
    noise = np.array([0.6, -0.4])
    backends = {
        "I": PlumeSource(
            location=(50.0, 0.0, 20.0),
            wind_speed=4.0,
            wind_direction=270.0,
            stability_class="D",
            column_z=COLUMN_Z,
        ),
        "II": _lagrangian(n_particles=300),
        "III": EulerianSource(forward=_eulerian_forward()),
    }
    posteriors = {}
    for tier, backend in backends.items():
        response = backend.column_response(instrument)
        sigma = 0.1 * truth * float(jnp.mean(response))
        tier_instrument = Instrument(
            "single", RECEPTORS, retrieval_variance=sigma**2, observation_time=40.0
        )
        forward = build_coupled_forward(backend, [tier_instrument])
        observation = forward.predict(truth)[0] + sigma * noise
        mean, covariance = default_prior(
            n_instruments=1,
            emission_prior_mean=1.0,
            emission_prior_std=5.0,
            bias_prior_std=1e-3 * sigma,
        )
        posteriors[tier] = fuse_observations(
            forward, [observation], prior_mean=mean, prior_covariance=covariance
        )
    for tier, posterior in posteriors.items():
        z = abs(float(posterior.emission_rate) - truth) / float(posterior.emission_std)
        assert z < 3.0, (tier, z)
    tiers = list(posteriors)
    for i, first in enumerate(tiers):
        for second in tiers[i + 1 :]:
            a, b = posteriors[first], posteriors[second]
            gap = abs(float(a.emission_rate) - float(b.emission_rate))
            spread = float(jnp.sqrt(a.emission_std**2 + b.emission_std**2))
            assert gap < 2.0 * spread, (first, second, gap, spread)
