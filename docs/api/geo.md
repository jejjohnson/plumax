# Plume operators (geo)

Plume post-processing on GeoTensors, behind the `[geo]` extra (`pip install 'plumax[geo]'`): the ten operators ported from geotoolz's `plume` module, as `pipekit.Operator`s built with geoops's public carrier helpers (the pattern of radtrax's `[geo]` operators). Importing `plumax.geo` does not need the extra; the operators load on first use, and a missing extra raises an `ImportError` naming it.

::: plumax.geo
    options:
      members: false

## `plume` — the operators

::: plumax.geo.plume
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - PlumeMask
        - PlumeContours
        - PlumeFootprint
        - PlumeShapeFilter
        - PlumeColumnStats
        - PlumeQNDFeatures
        - WindAdvectionCone
        - IMEEstimate
        - CrossSectionalFlux
        - ColumnToMass

## `plume_array` — the array functions

::: plumax.geo.plume_array
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - plume_mask
        - wind_advection_cone
        - convert_column_units
        - plume_length
        - pixel_area
        - pixel_centers
        - squeeze_single_band
