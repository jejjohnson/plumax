"""The shared optional-extra, NUTS and observation-check helpers in ``plumax._src``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.checks import check_observations
from plumax._src.errors import MissingExtraError
from plumax._src.extras import require_extra


def test_require_extra_returns_the_module() -> None:
    module = require_extra("json", extra="io", caller="f", purpose="reads JSON")
    assert module.__name__ == "json"


def test_require_extra_names_the_extra_and_is_an_import_error() -> None:
    with pytest.raises(ImportError, match=r"pip install 'plumax\[assim\]'"):
        require_extra(
            "not_a_module", extra="assim", caller="solve_x", purpose="solves X"
        )
    with pytest.raises(MissingExtraError, match="solve_x solves X"):
        require_extra(
            "not_a_module", extra="assim", caller="solve_x", purpose="solves X"
        )


def test_require_extra_imports_submodules() -> None:
    import sys

    module = require_extra(
        "xml", extra="io", caller="f", purpose="parses XML", submodules=("dom",)
    )
    assert module.__name__ == "xml"
    assert "xml.dom" in sys.modules
    with pytest.raises(MissingExtraError):
        require_extra(
            "xml", extra="io", caller="f", purpose="parses XML", submodules=("nope",)
        )


def test_check_observations_rejects_empty_and_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="must contain ≥ 1 point"):
        check_observations("f", [], ([], [], []))
    with pytest.raises(ValueError, match="axis 'y'"):
        check_observations("f", [1.0, 2.0], ([0.0, 1.0], [0.0], [0.0, 1.0]))


@pytest.mark.slow
def test_run_nuts_samples_and_accepts_a_seed_or_a_key() -> None:
    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    from plumax._src.nuts import run_nuts

    def model(obs: jax.Array) -> None:
        loc = numpyro.sample("loc", dist.Normal(0.0, 10.0))
        numpyro.sample("obs", dist.Normal(loc, 1.0), obs=obs)

    obs = jnp.full((20,), 3.0)
    by_seed = run_nuts(model, obs, seed=0, num_warmup=50, num_samples=50)
    by_key = run_nuts(
        model, obs, seed=jax.random.PRNGKey(0), num_warmup=50, num_samples=50
    )
    assert by_seed["loc"].shape == (50,)
    np.testing.assert_array_equal(by_seed["loc"], by_key["loc"])
    assert abs(float(jnp.mean(by_seed["loc"])) - 3.0) < 1.0


@pytest.mark.slow
def test_run_nuts_runs_gibbs_over_a_discrete_site() -> None:
    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist

    from plumax._src.nuts import run_nuts

    def model(obs: jax.Array) -> None:
        switch = numpyro.sample("switch", dist.Bernoulli(0.5))
        loc = numpyro.sample("loc", dist.Normal(0.0, 10.0))
        numpyro.sample("obs", dist.Normal(loc + 5.0 * switch, 1.0), obs=obs)

    draws = run_nuts(
        model,
        jnp.zeros(5),
        seed=1,
        num_warmup=30,
        num_samples=30,
        discrete_gibbs=True,
    )
    assert set(np.unique(np.asarray(draws["switch"]))) <= {0, 1}
