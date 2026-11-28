# Adapters

Classes that structurally satisfy the `pipekit_cycle` `ObservationOperator` protocol. `plumax` does not import `pipekit_cycle` at runtime. The `ForwardModel` over an `EulerianDispersionRHS` is `integrix.march.DiffraxForwardModel(rhs, solver=diffrax.Euler(), dt0=dt, dt=dt)`, as the module docstring below shows.

::: plumax.adapters
