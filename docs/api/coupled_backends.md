# Transport backends

The Tier II and Tier III transports behind `CoupledForward`. A `TransportBackend` gives each instrument's column per unit emission through the shared `ColumnObservation`: `LagrangianSource` from met-ensemble backward footprints, `EulerianSource` from the `les_fvm` finite-volume run at the overpass time. `PlumeSource` is the Tier I backend. A backend whose response depends on parameters the inversion fits subclasses `AbstractControlledSource`: it declares those parameters as named control blocks, with the emission blocks that scale its response, and evaluates the response at any value of them. `PlumeSource` declares the source position and the wind, and the column plume `gauss_plume.ColumnPlumeSource` its geometry, scaled by the line density $q = Q / U$. **Tier IV.**

::: plumax.coupled.backends
