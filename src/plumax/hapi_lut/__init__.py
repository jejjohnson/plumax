"""HAPI-driven absorption cross-section look-up tables.

Submodules:
  - ``config``    — ``GasConfig`` / ``LUTGridConfig`` / ``ATMOSPHERIC_GASES``.
  - ``generator`` — single-gas fetch / compute / wrap / save pipeline.
  - ``multi``     — multi-gas orchestration (separate-per-gas + combined).
  - ``beers``     — LUT interpolation + Beer-Lambert transmittance + the
                    differential-ratio form used for plume-enhancement retrievals.

HAPI (``hitran-api`` on PyPI) is imported lazily inside the generator +
multi pipelines — importing this sub-package does not require HAPI, only
building / fetching a LUT does.
"""

from __future__ import annotations

from plumax.hapi_lut import beers, config, generator, multi
from plumax.hapi_lut.beers import (
    absorption_coefficient,
    air_mass_factor,
    beers_law_from_lut,
    interpolate_cross_section,
    number_density,
    plume_ratio_spectrum,
    transmittance,
)
from plumax.hapi_lut.config import (
    ATMOSPHERIC_GASES,
    DEFAULT_VMR_NOMINAL,
    GasConfig,
    LUTGridConfig,
)
from plumax.hapi_lut.generator import (
    HAPI_CACHE_ENV,
    build_lut_dataset,
    compute_absorption_lut,
    default_cache_dir,
    fetch_hitran_data,
    generate_single_gas_lut,
    save_lut,
)
from plumax.hapi_lut.multi import (
    create_combined_lut,
    create_multi_gas_luts,
)


__all__ = [
    "ATMOSPHERIC_GASES",
    "DEFAULT_VMR_NOMINAL",
    # generator
    "HAPI_CACHE_ENV",
    # config
    "GasConfig",
    "LUTGridConfig",
    "absorption_coefficient",
    "air_mass_factor",
    "beers",
    "beers_law_from_lut",
    "build_lut_dataset",
    "compute_absorption_lut",
    # submodules
    "config",
    "create_combined_lut",
    # multi
    "create_multi_gas_luts",
    "default_cache_dir",
    "fetch_hitran_data",
    "generate_single_gas_lut",
    "generator",
    # beers
    "interpolate_cross_section",
    "multi",
    "number_density",
    "plume_ratio_spectrum",
    "save_lut",
    "transmittance",
]
