# Changelog

## [0.1.1](https://github.com/jejjohnson/plumax/compare/v0.1.0...v0.1.1) (2026-08-19)


### Features

* bootstrap plumax — scaffold, forward-model port, pipekit layer, and design docs ([#7](https://github.com/jejjohnson/plumax/issues/7)) ([847b106](https://github.com/jejjohnson/plumax/commit/847b106276513d5139c84b6290ba69efa0fe6ad5))
* coupled RTM observation operator — nonlinear + linearised (Beer–Lambert + SRF) ([#13](https://github.com/jejjohnson/plumax/issues/13)) ([e4b56c8](https://github.com/jejjohnson/plumax/commit/e4b56c864d00e05e82c00b9d6579ee87d9bcfd99))
* **les_fvm:** honor lateral BCs in horizontal advection & diffusion ([#59](https://github.com/jejjohnson/plumax/issues/59)) ([61ff2c9](https://github.com/jejjohnson/plumax/commit/61ff2c9a7ffb81a99ccb99556657189b1639e0fa))
* tier II model-based source inversion (Gaussian + lognormal, Matérn-3/2 prior) ([#8](https://github.com/jejjohnson/plumax/issues/8)) ([a318529](https://github.com/jejjohnson/plumax/commit/a3185298450a600cc75ff5a087817ce26ab1fe11))
* tier III 4D-Var posterior covariance (Gauss-Newton Hessian via gaussx) ([#12](https://github.com/jejjohnson/plumax/issues/12)) ([34f7e83](https://github.com/jejjohnson/plumax/commit/34f7e83b66b175b63cbd8ae905a4ed2323a03283))
* tier IV coupled multi-instrument fusion (Tier I + averaging kernel, v1) ([#10](https://github.com/jejjohnson/plumax/issues/10)) ([ec05a5f](https://github.com/jejjohnson/plumax/commit/ec05a5f127cdc4fc2f211e01b83012b10b64ac6f))
* tier V population — cross-tier catalog adapter + size distribution + point process ([#14](https://github.com/jejjohnson/plumax/issues/14)) ([a530486](https://github.com/jejjohnson/plumax/commit/a530486f16693e765dbdd12b0505c84c952d14eb))
* wire the tier III Eulerian 4D-Var loop end-to-end ([#9](https://github.com/jejjohnson/plumax/issues/9)) ([2fce40a](https://github.com/jejjohnson/plumax/commit/2fce40a3d243671b08b5e7067b34e78a8e99f997))


### Bug Fixes

* **gauss_plume/puff:** correct PG class-A σ_z, conserve puff mass, share wind solve (epic [#20](https://github.com/jejjohnson/plumax/issues/20)) ([#62](https://github.com/jejjohnson/plumax/issues/62)) ([058df9e](https://github.com/jejjohnson/plumax/commit/058df9e9ba7bc7fcd32b041b82f104a624851163))
* **lagrangian/assimilation:** footprint units + clock, DFS control space, Hanna adapter, config (epic [#23](https://github.com/jejjohnson/plumax/issues/23)) ([#63](https://github.com/jejjohnson/plumax/issues/63)) ([69437e4](https://github.com/jejjohnson/plumax/commit/69437e42b09422fe866fbe6b258a8ec4512dde3e))
* **les_fvm:** boundary-condition docs, wind staggering, CFL guard, 4D-Var convergence (epic [#19](https://github.com/jejjohnson/plumax/issues/19)) ([#58](https://github.com/jejjohnson/plumax/issues/58)) ([b940a22](https://github.com/jejjohnson/plumax/commit/b940a2208ae209f5344856eda258dc6f526b842b))
* **les_fvm:** open-wall BC consistency — cell-Dirichlet inflow + periodic seams ([#61](https://github.com/jejjohnson/plumax/issues/61)) ([99ab0fb](https://github.com/jejjohnson/plumax/commit/99ab0fbf7c26b83b2f2bbb959b94c97d10a0909f))
* **radtran/matched_filter/hapi_lut:** retrieval-stack robustness (epic [#22](https://github.com/jejjohnson/plumax/issues/22)) ([#66](https://github.com/jejjohnson/plumax/issues/66)) ([fe569c5](https://github.com/jejjohnson/plumax/commit/fe569c5bfbc75fc919f403ee3c9aa4cc029b35ea))

## Changelog
