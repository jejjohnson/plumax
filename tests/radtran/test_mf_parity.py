"""radtran's matched filters against ``matched_filter.core`` (plumax#131).

The two stacks share the score ε̂ = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t) and its SNR
ε̂ √(tᵀ Σ⁻¹ t). Before radtran's copy is deleted, both must agree on the same
(μ, Σ, t) to 1e-10 relative in float64.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np

from plumax.matched_filter import core
from plumax.radtran.background import robust_lowrank_covariance
from plumax.radtran.gaussx_solve import (
    build_lowrank_covariance_operator,
    matched_filter_image_op,
    matched_filter_pixel_op,
    matched_filter_snr_op,
)
from plumax.radtran.matched_filter import (
    matched_filter_image,
    matched_filter_pixel,
    matched_filter_snr,
)


RTOL = 1e-10


def _scene(n_bands: int = 6, shape: tuple[int, int] = (16, 16)) -> np.ndarray:
    rng = np.random.default_rng(0)
    loc = np.linspace(0.3, 0.5, n_bands)
    base = rng.normal(loc=loc, scale=0.02, size=(*shape, n_bands))
    return np.moveaxis(base, -1, 0)  # (B, H, W), radtran's layout


def _target(n_bands: int) -> np.ndarray:
    return -np.linspace(0.01, 0.05, n_bands)


def test_gaussx_operator_path_matches_core():
    scene = _scene()
    target = _target(scene.shape[0])
    cov, mu = build_lowrank_covariance_operator(scene, rank=3)
    cube = jnp.asarray(np.moveaxis(scene, 0, -1))  # (H, W, B), core's layout

    ref_image = matched_filter_image_op(scene, mu, cov, target)
    image = np.asarray(core.apply_image(cube, jnp.asarray(mu), cov, target))
    np.testing.assert_allclose(image, ref_image, rtol=RTOL)

    ref_pixel = matched_filter_pixel_op(scene[:, 3, 5], mu, cov, target)
    pixel = float(core.apply_pixel(cube[3, 5], jnp.asarray(mu), cov, target))
    np.testing.assert_allclose(pixel, ref_pixel, rtol=RTOL)

    ref_snr = matched_filter_snr_op(np.array(0.7), cov, target)
    snr = float(core.matched_filter_snr(0.7, cov, target))
    np.testing.assert_allclose(snr, ref_snr, rtol=RTOL)


def test_dense_path_matches_core_on_a_dense_operator():
    scene = _scene()
    target = _target(scene.shape[0])
    _, mu = build_lowrank_covariance_operator(scene, rank=3)
    cov_dense, cov_inv = robust_lowrank_covariance(scene, rank=3)
    cov_op = lx.MatrixLinearOperator(
        jnp.asarray(cov_dense),
        tags=frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    cube = jnp.asarray(np.moveaxis(scene, 0, -1))

    ref_image = matched_filter_image(scene, mu, cov_inv, target)
    image = np.asarray(core.apply_image(cube, jnp.asarray(mu), cov_op, target))
    np.testing.assert_allclose(image, ref_image, rtol=RTOL)

    ref_pixel = matched_filter_pixel(scene[:, 3, 5], mu, cov_inv, target)
    pixel = float(core.apply_pixel(cube[3, 5], jnp.asarray(mu), cov_op, target))
    np.testing.assert_allclose(pixel, ref_pixel, rtol=RTOL)

    ref_snr = matched_filter_snr(np.array(0.7), cov_inv, target)
    snr = float(core.matched_filter_snr(0.7, cov_op, target))
    np.testing.assert_allclose(snr, ref_snr, rtol=RTOL)
