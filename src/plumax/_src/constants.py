"""Physical constants shared across plumax, each defined once.

The constants of nature, the standard atmosphere, dry air and the trace-gas
molar masses are geophysix's (``geophysix.constants``); the names below keep
plumax's own spelling. What geophysix does not hold stays here: the von Kármán
constant, the near-surface temperature, the potential-temperature reference
pressure, and WRF's own gravity.
"""

from __future__ import annotations

from geophysix.constants import CH4, CO2, STANDARD


#: Molar mass of CH₄, M [kg mol⁻¹].
M_CH4 = CH4.molar_mass
#: Molar mass of CO₂ [kg mol⁻¹].
M_CO2 = CO2.molar_mass
#: Avogadro's number N_A [mol⁻¹] (exact, SI 2019).
AVOGADRO = STANDARD.avogadro
#: Boltzmann constant k_B [J K⁻¹] (exact, SI 2019).
BOLTZMANN = STANDARD.boltzmann
#: Standard surface pressure [Pa] (ISA sea level), for products that report none.
P_STANDARD = STANDARD.standard_pressure
#: Near-surface temperature [K] (ISA sea level), when a scene gives none.
T_STANDARD = STANDARD.standard_temperature

#: von Kármán constant κ.
KARMAN = 0.4
#: Standard gravity g [m s⁻²].
GRAVITY = STANDARD.standard_gravity
#: Gravitational acceleration WRF uses for its geopotential [m s⁻²]. WRF's own
#: convention (g = 9.81), kept so a ``wrfout`` geopotential is read back to the
#: heights WRF computed; it is not standard gravity.
WRF_GRAVITY = 9.81
#: Dry-air gas constant R_d [J kg⁻¹ K⁻¹], R / M_dry.
R_DRY = STANDARD.dry_air_gas_constant
#: Specific heat of dry air at constant pressure c_p [J kg⁻¹ K⁻¹].
CP_DRY = STANDARD.dry_air_heat_capacity
#: Dry-air gas constant over specific heat, κ_d = R_d / c_p.
KAPPA = STANDARD.kappa
#: Reference pressure of potential temperature p_ref [Pa].
P_REFERENCE = 1.0e5
