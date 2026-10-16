"""Band-integrated radiative transfer.

Bridges the line-by-line :mod:`plumax.hapi_lut` absorption-LUT
family to the multispectral / hyperspectral retrieval world by providing:

- :mod:`.config`         — :class:`ObservationGeometry`, :class:`InstrumentSpec`.
- :mod:`.srf`            — :class:`SpectralResponseFunction` (gaussian /
  rectangular / triangular / custom) for band integration.
- :mod:`.forward`        — exact / Maclaurin / Taylor Beer-Lambert forward
  models and their normalised (differential) variants, each returning
  ``(radiance, Jacobian, transmittance)``.
- :mod:`.nb_lut`         — pre-tabulated normalised-brightness LUT
  ``nB_b(ΔX)`` plus :func:`inject_plume` for scene-injection.

The matched filter (targets, background estimators and the score) lives in
:mod:`plumax.matched_filter`.

The input LUT is expected to be the :class:`xarray.Dataset` produced by
:func:`plumax.hapi_lut.build_lut_dataset`.

Notes
-----
``radtran`` is an aggregator: it re-exports symbols from its submodules but
imports them *lazily* via :pep:`562`. This means

- ``import plumax`` (which itself imports ``radtran``) does not import
  any submodule until you touch one of its names.
- ``from plumax.radtran import build_nb_lut`` triggers exactly
  one submodule import (``radtran.nb_lut``) and stops there.
- The ``__all__`` list still drives star-imports and IDE auto-complete,
  so the lazy loading is invisible to consumers.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING


# Map exported name → (submodule, attr-name). Used by ``__getattr__`` to
# route lookups to the right submodule on first access.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # config
    "ATM_TO_PA": ("config", "ATM_TO_PA"),
    "BOLTZMANN_J_PER_K": ("config", "BOLTZMANN_J_PER_K"),
    "CM3_PER_M3": ("config", "CM3_PER_M3"),
    "InstrumentSpec": ("config", "InstrumentSpec"),
    "ObservationGeometry": ("config", "ObservationGeometry"),
    "number_density_cm3": ("config", "number_density_cm3"),
    # forward
    "ForwardResult": ("forward", "ForwardResult"),
    "forward_nonlinear": ("forward", "forward_nonlinear"),
    "forward_nonlinear_normalized": ("forward", "forward_nonlinear_normalized"),
    "forward_maclaurin": ("forward", "forward_maclaurin"),
    "forward_maclaurin_normalized": ("forward", "forward_maclaurin_normalized"),
    "forward_taylor": ("forward", "forward_taylor"),
    "forward_taylor_normalized": ("forward", "forward_taylor_normalized"),
    # SRF
    "SpectralResponseFunction": ("srf", "SpectralResponseFunction"),
    # instrument (PSF + GSD) — JAX-native linear operators
    "PointSpreadFunction": ("instrument", "PointSpreadFunction"),
    "GroundSamplingDistance": ("instrument", "GroundSamplingDistance"),
    # nB-LUT / injection
    "NBLookup": ("nb_lut", "NBLookup"),
    "build_nb_lut": ("nb_lut", "build_nb_lut"),
    "lookup_nb": ("nb_lut", "lookup_nb"),
    "inject_plume": ("nb_lut", "inject_plume"),
}


def __getattr__(name: str):  # PEP 562 — lazy module attribute access
    try:
        submodule, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module 'plumax.radtran' has no attribute {name!r}"
        ) from None
    module = importlib.import_module(f"plumax.radtran.{submodule}")
    value = getattr(module, attr)
    globals()[name] = value  # cache on first access so subsequent lookups are O(1)
    return value


def __dir__() -> list[str]:  # IDE / dir() introspection sees the full surface
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # static type checkers: see the symbols as if eagerly imported
    from plumax.radtran.config import (
        ATM_TO_PA,
        BOLTZMANN_J_PER_K,
        CM3_PER_M3,
        InstrumentSpec,
        ObservationGeometry,
        number_density_cm3,
    )
    from plumax.radtran.forward import (
        ForwardResult,
        forward_maclaurin,
        forward_maclaurin_normalized,
        forward_nonlinear,
        forward_nonlinear_normalized,
        forward_taylor,
        forward_taylor_normalized,
    )
    from plumax.radtran.instrument import (
        GroundSamplingDistance,
        PointSpreadFunction,
    )
    from plumax.radtran.nb_lut import (
        NBLookup,
        build_nb_lut,
        inject_plume,
        lookup_nb,
    )
    from plumax.radtran.srf import SpectralResponseFunction


__all__ = sorted(_LAZY_EXPORTS)
