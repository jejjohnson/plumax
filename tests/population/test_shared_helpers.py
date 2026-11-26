"""Reference values of the helpers the intensity, POD and likelihood code share.

The numbers were measured on the code before the shared helpers replaced the
hand-written copies, in float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers
from numpyro.infer.util import log_density

from plumax.population.fitting import tmtpp_iw_model
from plumax.population.importance import EventEvidence
from plumax.population.intensity import CoalMineVentilationIntensity
from plumax.population.marks import LognormalMark
from plumax.population.pod import LogisticPOD
from plumax.population.processes import padded_event_buffer


pytest.importorskip("pyrox_xtreme")

WINDOW = (0.0, 10.0)


def _log_pod(q):
    return -jnp.log1p(1.0 / q**2)


def _tmtpp_log_density(intensity, params):
    event_times, mask = padded_event_buffer([1.3, 4.7, 8.2], WINDOW, max_events=6)
    # a padded buffer: three rows of evidence the mask drops
    q_samples = jnp.array([[3.0, 5.0]] * 6) * jnp.arange(1, 7)[:, None]
    evidence = EventEvidence(log_weights=jnp.zeros((6, 2)), q_samples=q_samples)
    model = handlers.substitute(tmtpp_iw_model, data=params)
    density, _ = log_density(
        model,
        (event_times, mask, evidence),
        {
            "mark": LognormalMark(),
            "intensity": intensity,
            "window": WINDOW,
            "log_pod": _log_pod,
            "support": (1e-4, 1e4),
        },
        {},
    )
    return float(density)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("intensity", "params", "reference"),
    [
        ("constant", {"lambda_0": 0.6}, -22.547910961331006),
        (
            "diurnal",
            {"lambda_0": 1.2, "amplitude": 0.5, "phase_hours": 14.0},
            # re-recorded for #437: the kernel now peaks at phase_hours, not 6 h
            # later (it was -28.39781776118004)
            -27.984796619319567,
        ),
    ],
)
def test_tmtpp_log_density_matches_the_reference(intensity, params, reference):
    params = {**params, "mark.mu": 0.4, "mark.sigma": 0.7}
    with jax.enable_x64(True):
        value = _tmtpp_log_density(intensity, params)
    np.testing.assert_allclose(value, reference, rtol=1e-12)


def test_shift_gate_kernel_matches_the_reference():
    kernel = CoalMineVentilationIntensity(
        lambda_extraction=14.008841028276679,
        lambda_maintenance=0.6648146268452466,
        lambda_idle=1.1424766881173876,
    )
    with jax.enable_x64(True):
        t = jnp.linspace(0.0, 400.0, 37) + 0.137
        values = np.asarray(kernel(t))
    np.testing.assert_allclose(
        values[[3, 10, 20]],
        [11.200832244561404, 1.1531878618366416, 12.645444486302967],
        rtol=1e-12,
    )


def test_logistic_pod_matches_the_reference():
    pod = LogisticPOD(Q_50=123.18952160742565, k=0.008864195024603288)
    with jax.enable_x64(True):
        values = np.asarray(pod(jnp.logspace(-2.0, 3.0, 25)))
    np.testing.assert_allclose(
        values[[12, 14, 16]],
        [0.2565563014846394, 0.26525916394600224, 0.28884449568703296],
        rtol=1e-12,
    )
