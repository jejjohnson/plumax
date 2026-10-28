"""The optional stacks stay optional: ``import plumax`` needs neither.

vardax (the ``assim`` extra) and NumPyro (the ``inference`` extra) are
imported lazily by the functions that need them. Each check runs in a fresh
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
