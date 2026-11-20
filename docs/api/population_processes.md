# Point processes

The intensity and probability-of-detection registries as [`pyrox_xtreme.point_processes`](https://github.com/jejjohnson/pyrox/tree/main/packages/pyrox-xtreme) processes: an inhomogeneous Poisson ground process per intensity kernel, a POD curve as a thinning retention, and the detected-event process `ThinningProcess(MarkedTemporalPointProcess(...))`. pyrox-xtreme supplies the likelihoods and samplers; plumax supplies the methane-domain kernels. Needs the `population` extra. **Tier V.**

::: plumax.population.processes
