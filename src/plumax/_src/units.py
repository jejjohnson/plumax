"""Column-enhancement unit conversions, at the scene's pressure and temperature.

A path-integrated enhancement E [ppm m] is a mixing ratio integrated along the
light path, so its amount of gas depends on the air number density along the
path, n = p / (k_B T) (the ideal-gas law):

    ΔΩ_mol = E · 10⁻⁶ · p / (k_B T N_A)   [mol m⁻²],
    ΔΩ     = M · ΔΩ_mol                    [kg m⁻²].
"""

from __future__ import annotations

import numpy as np
from geophysix.columns import convert
from geophysix.constants import Gas
from jaxtyping import Float

from plumax._src.constants import M_CH4


def ppm_m_to_column_moles(
    enhancement_ppm_m: Float[np.ndarray, "*#pixel"] | float,
    *,
    pressure_pa: float,
    temperature_k: float,
) -> Float[np.ndarray, " ..."]:
    """Molar column enhancement of a path-integrated enhancement.

    ΔΩ_mol = E · 10⁻⁶ · p / (k_B T N_A).

    Args:
        enhancement_ppm_m: Path-integrated enhancement E [ppm m].
        pressure_pa: Layer pressure p [Pa].
        temperature_k: Layer temperature T [K].

    Returns:
        ΔΩ_mol [mol m⁻²].

    Examples:
        1000 ppm·m at 1013.25 hPa and 288.15 K is 42.3 mmol m⁻²:

        >>> import numpy as np
        >>> moles = ppm_m_to_column_moles(1000.0, pressure_pa=101_325.0, temperature_k=288.15)
        >>> bool(np.isclose(moles, 42.29e-3, rtol=1e-3))
        True
    """
    # ΔΩ_mol = E · 10⁻⁶ · p / (k_B T N_A)
    return np.asarray(
        convert(
            enhancement_ppm_m,
            "ppm_m",
            "mol_m2",
            pressure=pressure_pa,
            temperature=temperature_k,
        )
    )


def ppm_m_to_column_mass(
    enhancement_ppm_m: Float[np.ndarray, "*#pixel"] | float,
    *,
    pressure_pa: float,
    temperature_k: float,
    molar_mass: float = M_CH4,
) -> Float[np.ndarray, " ..."]:
    """Column-mass enhancement of a path-integrated enhancement.

    ΔΩ = E · 10⁻⁶ · (p / (k_B T)) · M / N_A.

    Args:
        enhancement_ppm_m: Path-integrated enhancement E [ppm m].
        pressure_pa: Layer pressure p [Pa].
        temperature_k: Layer temperature T [K].
        molar_mass: Molar mass M of the gas [kg mol⁻¹]; CH₄ by default.

    Returns:
        ΔΩ [kg m⁻²].

    Examples:
        1000 ppm·m at 1013.25 hPa and 288.15 K: 42.3 mol m⁻³ · 1 mm · M:

        >>> import numpy as np
        >>> mass = ppm_m_to_column_mass(
        ...     1000.0, pressure_pa=101_325.0, temperature_k=288.15
        ... )
        >>> bool(np.isclose(mass, 42.29e-3 * 0.0160425, rtol=1e-3))
        True

        Thinner air at altitude carries less mass for the same E:

        >>> high = ppm_m_to_column_mass(
        ...     1000.0, pressure_pa=70_000.0, temperature_k=288.15
        ... )
        >>> bool(high < mass)
        True
    """
    # ΔΩ = M · E · 10⁻⁶ · p / (k_B T N_A)
    return np.asarray(
        convert(
            enhancement_ppm_m,
            "ppm_m",
            "kg_m2",
            gas=Gas("gas", molar_mass),
            pressure=pressure_pa,
            temperature=temperature_k,
        )
    )
