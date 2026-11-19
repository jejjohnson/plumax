"""Physical constants shared across plumax, each defined once."""

from __future__ import annotations


#: Molar mass of CH₄, M [kg mol⁻¹] (12.011 + 4 × 1.008 g mol⁻¹).
M_CH4 = 0.0160425
#: Molar mass of CO₂ [kg mol⁻¹].
M_CO2 = 0.04401
#: Avogadro's number N_A [mol⁻¹] (exact, SI 2019).
AVOGADRO = 6.02214076e23
#: Boltzmann constant k_B [J K⁻¹] (exact, SI 2019).
BOLTZMANN = 1.380649e-23
#: Standard surface pressure [Pa] (ISA sea level), for products that report none.
P_STANDARD = 101_325.0
#: Near-surface temperature [K] (ISA sea level), when a scene gives none.
T_STANDARD = 288.15

#: von Kármán constant κ.
KARMAN = 0.4
#: Standard gravity g [m s⁻²].
GRAVITY = 9.80665
#: Gravitational acceleration WRF uses for its geopotential [m s⁻²].
WRF_GRAVITY = 9.81
#: Dry-air gas constant R_d [J kg⁻¹ K⁻¹].
R_DRY = 287.04
#: Specific heat of dry air at constant pressure c_p [J kg⁻¹ K⁻¹].
CP_DRY = 1004.5
#: Dry-air gas constant over specific heat, κ_d = R_d / c_p.
KAPPA = R_DRY / CP_DRY
#: Reference pressure of potential temperature p_ref [Pa].
P_REFERENCE = 1.0e5
