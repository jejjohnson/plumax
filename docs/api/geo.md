# Plume operators (geo)

Plume post-processing on GeoArrays, behind the `[geo]` extra (`pip install 'plumax[geo]'`): the ten operators ported from geotoolz's `plume` module, as `pipekit_ops.Operator`s built with geoops's public carrier helpers (the pattern of radtrax's `[geo]` operators). Importing `plumax.geo` does not need the extra; the operators load on first use, and a missing extra raises an `ImportError` naming it.

::: plumax.geo
    options:
      members: false

## `plume`: the operators

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

## `vetting`: plume vetting on carriers

`PlumeVetting` runs [plume vetting](vetting.md) (`vet_candidate` and the scene null) on every candidate of a scene and returns the scores as one `xarray.Dataset`; the decision rule is left to the caller.

::: plumax.geo.vetting
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - PlumeVetting

## `starts`: starting values of an image fit

The starting direction of a column-plume fit, from the source to the centroid of the plume mask, and the grid of source positions that [`solve_coupled_multistart`](coupled_inversion.md) starts from.

::: plumax.geo.starts
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - plume_direction
        - source_offset_grid

## `transects`: cross-plume transects

Profiles of the enhancement across the plume at regular downwind distances, by spline interpolation of the image, and each profile's line density, centre and width. `CrossPlumeProfile` returns them as one table on a GeoArray; the array functions underneath give the smoothed centreline and the width law, the starting values of a [column-plume fit](coupled_inversion.md). They are NumPy and SciPy code, outside any traced path.

::: plumax.geo.transects
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - CrossPlumeProfile
        - TransectProfiles
        - cross_plume_transects
        - transect_moments
        - fit_transect_gaussians
        - smooth_centreline
        - fit_width_law

## `plume_array`: the array functions

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
