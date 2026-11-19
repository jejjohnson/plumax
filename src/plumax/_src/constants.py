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
