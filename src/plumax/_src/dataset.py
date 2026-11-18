"""The variable attributes every forward model's output dataset shares."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import xarray as xr


#: Attributes of the ``concentration`` variable [kg m⁻³].
CONCENTRATION_ATTRS: dict[str, str] = {
    "long_name": "Mass concentration",
    "units": "kg/m^3",
}

#: Attributes of the ``column_concentration`` variable [kg m⁻²].
COLUMN_ATTRS: dict[str, str] = {
    "long_name": "Column-integrated concentration",
    "units": "kg/m^2",
}


def label_concentrations(dataset: xr.Dataset) -> xr.Dataset:
    """Set the shared attributes of ``concentration`` and ``column_concentration``.

    Args:
        dataset: A forward model's output, holding both variables.

    Returns:
        The same dataset, labelled in place.
    """
    dataset["concentration"].attrs = dict(CONCENTRATION_ATTRS)
    dataset["column_concentration"].attrs = dict(COLUMN_ATTRS)
    return dataset
