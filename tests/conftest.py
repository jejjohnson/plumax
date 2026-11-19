"""Top-level pytest configuration for ``plumax``.

Enables JAX 64-bit math at the earliest possible point: ``JAX_ENABLE_X64=1``
is set in the env *before* JAX has a chance to import (pytest plugins like
``pytest-jaxtyping`` can pull JAX in during plugin discovery, which happens
before this module is even loaded — but env-var reads still pick up the
value because JAX checks ``JAX_ENABLE_X64`` on first config access). We
also call ``jax.config.update`` as a belt-and-braces fallback.

The adjoint / round-trip invariants in the radtran and assimilation suites
assert agreements at ~1e-12 — well below float32 precision (~1e-7) — so a
silent regression to float32 turns multiple tests into mysteriously-failing
phantoms.
"""

from __future__ import annotations

import os


os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import pytest


jax.config.update("jax_enable_x64", True)

# Runtime shape and dtype checks for the packages written to the house style
# (jaxtyping annotations checked by beartype). Installed before anything
# imports them; the older subpackages are not covered yet.
from jaxtyping import install_import_hook


_TYPED_PACKAGES = (
    "plumax.coupled._src",
    "plumax.gauss_plume._src",
    "plumax.gauss_puff._src",
    "plumax.geo._src",
    "plumax.lagrangian._src",
    "plumax.les_fvm._src",
    "plumax.met._src",
    "plumax.obs._src",
    "plumax.population._src",
    "plumax.priors._src",
    "plumax.vetting._src",
)
# beartype configured with PEP 484's numeric tower (int passes for float)


_type_check_hook = install_import_hook(
    _TYPED_PACKAGES, "tests._type_check.pep484_beartype"
)


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_code():
    """Free JAX's compiled code after each module.

    Compiled functions stay mapped for the life of the process, so without this
    a worker grows to 5–6 GiB over the full suite and XLA can fail with
    ``LLVM compilation error: Cannot allocate memory``.
    """
    yield
    jax.clear_caches()
    # a test that switches x64 off and leaves it off turns every later test in
    # this worker into a float32 run
    assert jax.config.jax_enable_x64, "a test in this module left x64 switched off"


#: seconds an unmarked test's call may take before it fails; 0 switches the check off
_FAST_BUDGET = float(os.environ.get("PLUMAX_FAST_BUDGET") or 0.0)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Fail a passing unmarked test whose call ran longer than the fast budget.

    The fast tier is every test without a ``slow`` or ``validation`` marker.
    CI's fast job sets ``PLUMAX_FAST_BUDGET``, so a test that grows past it
    fails there and gets a marker (or a smaller problem) instead of slowing
    every run. Fixture setup is not counted.
    """
    outcome = yield
    report = outcome.get_result()
    if not _FAST_BUDGET or report.when != "call" or not report.passed:
        return
    if item.get_closest_marker("slow") or item.get_closest_marker("validation"):
        return
    if call.duration > _FAST_BUDGET:
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} took {call.duration:.1f} s, over the fast budget of "
            f"{_FAST_BUDGET:g} s: shrink it, or mark it `slow` (deterministic) "
            "or `validation` (statistical)"
        )
