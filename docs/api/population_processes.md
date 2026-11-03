# Point processes

The intensity and probability-of-detection registries as [`xtremax.point_processes`](https://github.com/jejjohnson/xtremax) processes: an inhomogeneous Poisson ground process per intensity kernel, a POD curve as a thinning retention, and the detected-event process `ThinningProcess(MarkedTemporalPointProcess(...))`. xtremax supplies the likelihoods and samplers; plumax supplies the methane-domain kernels. Needs the `population` extra. **Tier V.**

::: plumax.population.processes
