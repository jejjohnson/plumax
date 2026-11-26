"""What the data determined in a coupled fit, and how well it fits (#271).

Toy scenes only: a column plume on about a hundred 60 m pixels.
"""

from __future__ import annotations

import dataclasses
import math

import einx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from plumax.coupled import (
    CoupledPrior,
    CoupledProblem,
    FitDiagnostics,
    Instrument,
    PlumeSource,
    build_coupled_forward,
    column_response,
    fit_diagnostics,
    solve_coupled,
)
from plumax.coupled._src.diagnostics import _stacked_jacobian
from plumax.gauss_plume import ColumnPlume, ColumnPlumeSource, PlumeGeometry


PIXEL = 60.0
# w₀ = g / √12
INITIAL_WIDTH = PIXEL / math.sqrt(12.0)
LINE_DENSITY = 0.1
NOISE = 2e-4


def _receptors(length=600.0, half_width=300.0):
    xs = jnp.arange(-2 * PIXEL, length + 1.0, PIXEL)
    ys = jnp.arange(-half_width, half_width + 1.0, PIXEL)
    gx, gy = jnp.meshgrid(xs, ys, indexing="ij")
    return einx.id("x y, x y -> (x y) (1 + 1)", gx, gy)


def _source():
    plume = ColumnPlume(
        reference_distance=300.0,
        initial_width=INITIAL_WIDTH,
        upwind_scale=INITIAL_WIDTH,
        pixel_size=PIXEL,
    )
    geometry = PlumeGeometry(
        location=jnp.zeros(2),
        direction=0.1,
        reference_width=60.0,
        width_exponent=0.8,
    )
    return ColumnPlumeSource(plume=plume, geometry=geometry)


def _problem(source, instruments, observations, mean, std):
    forward = build_coupled_forward(source, instruments)
    return CoupledProblem(
        forward=forward,
        prior=CoupledPrior.from_std(mean, std),
        observations=observations,
    )


# ── a linear problem: y = q r + Ω_bg + ε ────────────────────────────────────

LINEAR_MEAN = {"background": 0.0, "line_density": 0.08}
LINEAR_STD = {"background": 1e-4, "line_density": 0.03}


def _linear(key, noise=NOISE, truth=None):
    source = _source()
    instrument = Instrument("EMIT", _receptors(), retrieval_variance=noise**2)
    truth = (
        {"background": 2e-5, "line_density": LINE_DENSITY} if truth is None else truth
    )
    # y = q r + Ω_bg + ε
    response = source.column_response(instrument)
    clean = truth["line_density"] * response + truth["background"]
    observation = clean + noise * jax.random.normal(key, clean.shape)
    problem = _problem(
        source, [instrument], {"EMIT": observation}, LINEAR_MEAN, LINEAR_STD
    )
    return problem, response, observation


def _closed_form(response, noise):
    # J = [1, r] in the layout (background, line_density)
    jacobian = np.stack([np.ones_like(response), np.asarray(response)], axis=-1)
    # F = Jᵀ Σ_ε⁻¹ J
    fisher = jacobian.T @ jacobian / noise**2
    prior_covariance = np.diag(
        [LINEAR_STD["background"] ** 2, LINEAR_STD["line_density"] ** 2]
    )
    # Σ_θ|y = (Σ_θ⁻¹ + F)⁻¹
    posterior = np.linalg.inv(np.linalg.inv(prior_covariance) + fisher)
    return fisher, prior_covariance, posterior


@pytest.fixture(scope="module")
def linear_fit():
    problem, response, observation = _linear(jax.random.PRNGKey(0))
    result = solve_coupled(problem)
    return problem, response, observation, result


@pytest.mark.slow
def test_linear_bound_and_information_are_the_closed_form(linear_fit):
    problem, response, _, result = linear_fit
    report = fit_diagnostics(problem, result)
    assert isinstance(report, FitDiagnostics)
    fisher, prior_covariance, posterior = _closed_form(response, NOISE)
    np.testing.assert_allclose(report.fisher_information, fisher, rtol=1e-10)

    # √[F⁻¹]_aa
    bound = np.sqrt(np.diagonal(np.linalg.inv(fisher)))
    np.testing.assert_allclose(
        report.cramer_rao_std["background"], bound[0], rtol=1e-10
    )
    np.testing.assert_allclose(
        report.cramer_rao_std["line_density"], bound[1], rtol=1e-10
    )
    # the Schur form (F_aa − f_ar² / F_rr)⁻¹ of the same bound
    schur = [
        1.0 / (fisher[0, 0] - fisher[0, 1] ** 2 / fisher[1, 1]),
        1.0 / (fisher[1, 1] - fisher[0, 1] ** 2 / fisher[0, 0]),
    ]
    np.testing.assert_allclose(bound**2, schur, rtol=1e-8)

    # 1 − [Σ_θ|y]_aa / [Σ_θ]_aa and d_s = tr(I − Σ_θ|y Σ_θ⁻¹)
    fraction = 1.0 - np.diagonal(posterior) / np.diagonal(prior_covariance)
    np.testing.assert_allclose(
        report.data_fraction["background"], fraction[0], rtol=1e-10
    )
    np.testing.assert_allclose(
        report.data_fraction["line_density"], fraction[1], rtol=1e-10
    )
    dofs = np.trace(np.eye(2) - posterior @ np.linalg.inv(prior_covariance))
    np.testing.assert_allclose(report.dofs, dofs, rtol=1e-10)


@pytest.mark.slow
def test_linear_pull_and_residuals_are_the_closed_form(linear_fit):
    problem, response, observation, result = linear_fit
    report = fit_diagnostics(problem, result)
    estimate = result.map_estimate
    for name in ("background", "line_density"):
        # (θ* − μ_θ) / σ_θ
        pull = (estimate[name] - LINEAR_MEAN[name]) / LINEAR_STD[name]
        np.testing.assert_allclose(report.prior_pull[name], pull, rtol=1e-10)
    # r = y − q* r − Ω_bg*
    residual = (
        observation - estimate["line_density"] * response - estimate["background"]
    )
    n_obs = residual.shape[0]
    chi2 = float(jnp.sum(jnp.square(residual))) / NOISE**2
    np.testing.assert_allclose(
        report.reduced_chi2["EMIT"], chi2 / (n_obs - report.dofs), rtol=1e-8
    )
    rms = math.sqrt(float(jnp.mean(jnp.square(residual))))
    np.testing.assert_allclose(report.rms_residual["EMIT"], rms, rtol=1e-8)


@pytest.mark.slow
def test_an_unseen_parameter_has_no_bound_and_keeps_its_prior():
    source = _source()
    receptors = _receptors()
    n_obs = receptors.shape[0]
    seen = Instrument("EMIT", receptors, retrieval_variance=NOISE**2)
    masked = Instrument(
        "GHGSat", receptors, retrieval_variance=NOISE**2, mask=jnp.zeros(n_obs)
    )
    observation = LINE_DENSITY * source.column_response(seen)
    problem = _problem(
        source,
        [seen, masked],
        {"EMIT": observation, "GHGSat": observation},
        {"line_density": 0.08, "biases": jnp.zeros(2)},
        {"line_density": 0.03, "biases": 1e-4},
    )
    report = fit_diagnostics(problem, solve_coupled(problem))
    # GHGSat's bias has a zero column of J_h
    assert math.isinf(float(report.cramer_rao_std["biases"][1]))
    assert float(report.data_fraction["biases"][1]) == 0.0
    assert math.isfinite(float(report.cramer_rao_std["biases"][0]))
    assert math.isfinite(float(report.cramer_rao_std["line_density"]))
    assert math.isnan(report.reduced_chi2["GHGSat"])


# ── the nonlinear fit: properties ───────────────────────────────────────────

FULL_MEAN = {
    "log_line_density": math.log(LINE_DENSITY),
    "log_reference_width": math.log(60.0),
    "width_exponent": 0.8,
    "direction": 0.1,
    "location": jnp.zeros(2),
    "background": 0.0,
}
FULL_STD = {
    "log_line_density": 2.0,
    "log_reference_width": 1.0,
    "width_exponent": 0.5,
    "direction": 0.3,
    "location": 60.0,
    "background": 1e-4,
}


def _full(noise=NOISE):
    source = _source()
    instrument = Instrument("EMIT", _receptors(), retrieval_variance=noise**2)
    clean = LINE_DENSITY * source.column_response(instrument)
    observation = clean + NOISE * jax.random.normal(jax.random.PRNGKey(4), clean.shape)
    return _problem(source, [instrument], {"EMIT": observation}, FULL_MEAN, FULL_STD)


@pytest.fixture(scope="module")
def full_fit():
    problem = _full()
    return problem, solve_coupled(problem, laplace=False)


@pytest.mark.slow
def test_fisher_uses_the_jacobian_of_the_stacked_prediction(full_fit):
    problem, result = full_fit
    _, unflatten = problem.prior.flatten()
    jacobian_of = _stacked_jacobian(problem, unflatten)
    theta = result.mean
    jacobian = jacobian_of(theta)

    def predict(state):
        return problem.predict_stacked(unflatten(state))

    step = 1e-6
    # (h(θ + h eₖ) − h(θ − h eₖ)) / 2h, scaled to each control
    scales = jnp.maximum(jnp.abs(theta), 1.0)
    differences = [
        (predict(theta + step * scale * unit) - predict(theta - step * scale * unit))
        / (2 * step * scale)
        for unit, scale in zip(jnp.eye(theta.shape[0]), scales, strict=True)
    ]
    expected = einx.id("param obs -> obs param", jnp.stack(differences))
    for k in range(theta.shape[0]):
        column_scale = float(jnp.max(jnp.abs(expected[:, k])))
        np.testing.assert_allclose(
            jacobian[:, k], expected[:, k], rtol=1e-5, atol=1e-5 * column_scale
        )
    np.testing.assert_allclose(jax.jit(jacobian_of)(theta), jacobian, rtol=1e-12)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**16))
@pytest.mark.slow
def test_fisher_is_symmetric_positive_semidefinite(full_fit, seed):
    problem, result = full_fit
    # anywhere near the MAP, not only at it
    shift = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), result.mean.shape)
    moved = dataclasses.replace(
        result, mean=result.mean + shift * jnp.sqrt(jnp.abs(result.mean) + 1.0)
    )
    report = fit_diagnostics(problem, moved)
    fisher = report.fisher_information
    np.testing.assert_allclose(fisher, fisher.T, rtol=1e-12, atol=0.0)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert eigenvalues.min() >= -1e-9 * eigenvalues.max()
    assert 0.0 <= report.dofs <= result.mean.shape[0]


@settings(max_examples=5, deadline=None)
@given(low=st.floats(0.5, 2.0), ratio=st.floats(1.5, 10.0))
@pytest.mark.slow
def test_dofs_rise_with_the_signal_to_noise_ratio(full_fit, low, ratio):
    _, result = full_fit
    # the same fit, read against a smaller and a larger noise
    quiet = fit_diagnostics(_full(noise=low * NOISE), result)
    loud = fit_diagnostics(_full(noise=low * ratio * NOISE), result)
    assert quiet.dofs > loud.dofs


# ── calibration ─────────────────────────────────────────────────────────────


@pytest.mark.validation
def test_reduced_chi2_is_one_on_scenes_with_the_right_noise():
    n_scenes = 200
    key = jax.random.PRNGKey(7)
    values = []
    for scene in range(n_scenes):
        key, truth_key, noise_key = jax.random.split(key, 3)
        # θ ~ N(μ_θ, Σ_θ), so E[χ²] = N − d_s exactly for the linear forward
        draw = jax.random.normal(truth_key, (2,))
        truth = {
            "background": LINEAR_MEAN["background"]
            + LINEAR_STD["background"] * draw[0],
            "line_density": LINEAR_MEAN["line_density"]
            + LINEAR_STD["line_density"] * draw[1],
        }
        problem, _, _ = _linear(noise_key, truth=truth)
        report = fit_diagnostics(problem, solve_coupled(problem, laplace=False))
        values.append(report.reduced_chi2["EMIT"])
        if (scene + 1) % 50 == 0:
            # each solve keeps its compiled code; see solve_coupled's map limit
            jax.clear_caches()
    n_obs = _receptors().shape[0]
    # sd[χ²_red] = √(2 / (N − d_s)) per scene, d_s ≈ 2
    standard_error = math.sqrt(2.0 / (n_obs - 2.0)) / math.sqrt(n_scenes)
    mean = float(np.mean(values))
    assert abs(mean - 1.0) < 3.0 * standard_error, (
        f"mean reduced χ² {mean:.4f} over {n_scenes} scenes, se {standard_error:.4f}"
    )


def _mixed_unit_problem():
    """A source position in metres beside a background in kg m⁻²."""
    source = PlumeSource(
        location=(0.0, 0.0, 10.0),
        wind_speed=5.0,
        wind_direction=270.0,
        stability_class="D",
        column_z=jnp.linspace(0.0, 200.0, 11),
    )
    x, y = np.meshgrid(
        np.linspace(150.0, 900.0, 8), np.linspace(-100.0, 100.0, 9), indexing="ij"
    )
    receptors = np.column_stack([x.ravel(), y.ravel()])
    scale = float(jnp.max(column_response(source, Instrument("A", receptors))))
    instrument = Instrument("A", receptors, retrieval_variance=(0.002 * scale) ** 2)
    prior = CoupledPrior.from_std(
        {"emission_rate": 0.1, "location": jnp.zeros(2), "background": 0.0},
        {
            "emission_rate": 0.05,
            "location": jnp.full(2, 50.0),
            "background": 0.01 * scale,
        },
    )
    return CoupledProblem(
        forward=build_coupled_forward(source, [instrument]),
        prior=prior,
        observations={"A": 0.1 * column_response(source, instrument)},
    )


def test_cramer_rao_bound_does_not_depend_on_the_units_of_a_block():
    """A background in kg m⁻² must not hide a position in metres.

    The null space used to be cut at 1e-10 of F's largest eigenvalue, in mixed
    units, so the background's 5e11 put every other eigenvalue under the cut.
    """
    problem = _mixed_unit_problem()
    report = fit_diagnostics(problem, solve_coupled(problem))
    dense = np.sqrt(np.diag(np.linalg.inv(np.asarray(report.fisher_information))))
    flat = np.concatenate(
        [np.ravel(report.cramer_rao_std[name]) for name, _ in problem.prior.layout]
    )
    assert np.all(np.isfinite(flat))
    np.testing.assert_allclose(flat, dense, rtol=1e-8)
