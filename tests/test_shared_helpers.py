"""The shared optional-extra, NUTS and observation-check helpers in ``plumax._src``."""

from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from plumax._src.checks import check_observations
from plumax._src.errors import MissingExtraError
from plumax._src.vardax import require_argminax, require_vardax


def test_require_vardax_names_the_extra_when_vardax_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "vardax", None)
    with pytest.raises(
        MissingExtraError,
        match=(
            r"solve_x needs 'vardax', which is not installed; "
            r"install it with: pip install 'plumax\[assim\]'"
        ),
    ):
        require_vardax("solve_x")


def test_require_vardax_lets_a_broken_install_raise_its_own_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    # an installed vardax whose own import fails is not a missing extra
    (tmp_path / "vardax.py").write_text("from math import no_such_name\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "vardax", raising=False)
    with pytest.raises(ImportError, match="no_such_name") as caught:
        require_vardax("solve_x")
    assert not isinstance(caught.value, MissingExtraError)


def test_require_argminax_imports_its_submodules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("argminax")
    module = require_argminax("solve_x")
    assert module.__name__ == "argminax"
    assert "argminax.curvature" in sys.modules
    assert "argminax.solve" in sys.modules
    monkeypatch.setitem(sys.modules, "argminax.solve", None)
    with pytest.raises(MissingExtraError, match=r"'argminax\.solve'"):
        require_argminax("solve_x")


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
