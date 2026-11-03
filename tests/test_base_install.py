"""The optional stacks stay optional: ``import plumax`` needs none of them.

vardax (the ``assim`` extra), NumPyro (the ``inference`` extra) and the geo
carriers (geoops and georeader, the ``geo`` extra) are imported lazily by
the functions and operators that need them. Each check runs in a fresh
interpreter, so modules imported by other tests cannot hide a regression.
``_BLOCK`` installs an import hook that raises ``ModuleNotFoundError`` for
the named packages exactly as a missing install does. (gaussx imports NumPyro
on its own when it is installed, so "not in ``sys.modules``" is checked for
vardax only.)
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# A meta-path finder that makes `import <top>` and `import <top>.<sub>` fail
# with the same ModuleNotFoundError (name=<top>) as an uninstalled package.
_BLOCK = """
import sys

class _Missing:
    def __init__(self, *names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition(".")[0]
        if top in self.names:
            raise ModuleNotFoundError(f"No module named {top!r}", name=top)
        return None

def block(*names):
    sys.meta_path.insert(0, _Missing(*names))
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_no_vardax_without_extra():
    result = _run(
        """
        import sys
        import plumax, plumax.lagrangian, plumax.les_fvm
        assert "vardax" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


def test_imports_without_either_extra():
    result = _run(
        """
        block("vardax", "numpyro")
        import plumax, plumax.lagrangian, plumax.les_fvm, plumax.coupled
        """
    )
    assert result.returncode == 0, result.stderr


def test_noise_models_without_numpyro():
    # The likelihoods import NumPyro only to sample a site; building a noise
    # model and its plain-JAX log-density need neither extra.
    result = _run(
        """
        block("numpyro")
        import jax.numpy as jnp
        from plumax.gauss_plume import StudentTNoise
        value = StudentTNoise(scale=1.0).log_prob(jnp.zeros(2), jnp.zeros(2))
        assert value.shape == (2,)
        """
    )
    assert result.returncode == 0, result.stderr


def test_inversion_without_vardax_names_the_extra():
    result = _run(
        """
        block("vardax")
        import jax.numpy as jnp
        from plumax.lagrangian import linear_gaussian_inversion
        try:
            linear_gaussian_inversion(
                jnp.ones((2, 3)), jnp.ones(2), prior_mean=jnp.ones(3),
                prior_cov=jnp.eye(3), obs_variance=jnp.ones(2),
            )
        except ImportError as exc:
            assert "plumax[assim]" in str(exc), exc
        else:
            raise AssertionError("no ImportError")
        """
    )
    assert result.returncode == 0, result.stderr


def test_fusion_without_vardax_names_the_extra():
    result = _run(
        """
        block("vardax")
        import jax.numpy as jnp
        from plumax.coupled import (
            CoupledForward, Instrument, PlumeSource, fuse_observations,
        )
        source = PlumeSource(
            location=(0.0, 0.0, 10.0), wind_speed=5.0, wind_direction=270.0,
            stability_class="D", column_z=jnp.linspace(0.0, 200.0, 3),
        )
        instrument = Instrument("A", jnp.zeros((2, 2)))
        forward = CoupledForward(
            source=source, instruments=(instrument,), responses=(jnp.ones(2),)
        )
        try:
            fuse_observations(
                forward, [jnp.ones(2)], prior_mean=jnp.zeros(2),
                prior_covariance=jnp.eye(2),
            )
        except ImportError as exc:
            assert "plumax[assim]" in str(exc), exc
        else:
            raise AssertionError("no ImportError")
        """
    )
    assert result.returncode == 0, result.stderr


def test_no_geo_carrier_without_extra():
    result = _run(
        """
        import sys
        import plumax, plumax.geo
        loaded = {m.partition(".")[0] for m in sys.modules}
        assert not loaded & {"geoops", "georeader", "geotoolz"}, loaded
        """
    )
    assert result.returncode == 0, result.stderr


def test_geo_operator_without_extra_names_it():
    result = _run(
        """
        block("geoops", "georeader")
        import plumax
        from plumax.errors import MissingExtraError
        try:
            from plumax.geo import PlumeMask
        except MissingExtraError as exc:
            assert "plumax[geo]" in str(exc), exc
        else:
            raise AssertionError("no MissingExtraError")
        """
    )
    assert result.returncode == 0, result.stderr


def test_population_without_xtremax():
    # xtremax (the `population` extra) is imported only when a process is
    # built, and its absence is reported with the extra's name.
    result = _run(
        """
        import sys
        import plumax.population
        assert "xtremax" not in sys.modules
        block("xtremax")
        from plumax.population import make_process
        try:
            make_process("constant", (0.0, 1.0), lambda_0=1.0)
        except ImportError as exc:
            assert "plumax[population]" in str(exc), exc
        else:
            raise AssertionError("no ImportError")
        """
    )
    assert result.returncode == 0, result.stderr
