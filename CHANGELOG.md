# Changelog

## [0.2.4](https://github.com/jejjohnson/plumax/compare/v0.2.3...v0.2.4) (2026-11-04)


### Features

* **coupled:** add masked-k multi-source handling across instruments ([#236](https://github.com/jejjohnson/plumax/issues/236)) ([0507550](https://github.com/jejjohnson/plumax/commit/0507550c91bcd2120166b7a2521574bcddbda36b))
* **les_fvm:** add a per-cell surface flux control with a matern prior ([#226](https://github.com/jejjohnson/plumax/issues/226)) ([67788e9](https://github.com/jejjohnson/plumax/commit/67788e9b617a79b013812cebbe25986e9b5fb7dc))
* **les_fvm:** add initial concentration and inflow scaling to the 4d-var control ([#227](https://github.com/jejjohnson/plumax/issues/227)) ([a92f9a1](https://github.com/jejjohnson/plumax/commit/a92f9a136c01c25fb346b80668d7bc3f775341a9))
* **les_fvm:** add monin-obukhov and smagorinsky eddy diffusivity ([#221](https://github.com/jejjohnson/plumax/issues/221)) ([efc6bf3](https://github.com/jejjohnson/plumax/commit/efc6bf3fddba59b752b04bbf394462a05ef129ac)), closes [#94](https://github.com/jejjohnson/plumax/issues/94)
* **les_fvm:** route solve_4dvar through gauss-newton on vardax ([#229](https://github.com/jejjohnson/plumax/issues/229)) ([1181678](https://github.com/jejjohnson/plumax/commit/1181678b37869131f32f7902783b425e247e6c28))
* **population:** add multi-satellite pod union and hierarchical calibration prior ([#233](https://github.com/jejjohnson/plumax/issues/233)) ([296e30d](https://github.com/jejjohnson/plumax/commit/296e30d708fbbc5a2421b1f2a9f334ec0d21ceff))
* **population:** add persistency metrics for a fitted intensity ([#222](https://github.com/jejjohnson/plumax/issues/222)) ([f3b129a](https://github.com/jejjohnson/plumax/commit/f3b129a6b122a2b2f177a484aac8fc6599356a23)), closes [#112](https://github.com/jejjohnson/plumax/issues/112)
* **population:** add simulation-based calibration and prior-swap harnesses ([#235](https://github.com/jejjohnson/plumax/issues/235)) ([c8fe472](https://github.com/jejjohnson/plumax/commit/c8fe472bbf9ea9f6dda33b20c977ca5c3a37a679))
* **population:** add total-emission estimator with missing-mass correction ([#223](https://github.com/jejjohnson/plumax/issues/223)) ([228494e](https://github.com/jejjohnson/plumax/commit/228494e3ba18569185c1856dee3361d2c50f6cc1)), closes [#113](https://github.com/jejjohnson/plumax/issues/113)


### Bug Fixes

* **ci:** stop the full test job running out of memory ([#214](https://github.com/jejjohnson/plumax/issues/214)) ([61697f5](https://github.com/jejjohnson/plumax/commit/61697f52fcaa07691d6b88faeecdbccde6617472))


### Performance Improvements

* **lagrangian:** compile the footprint simulator once ([#224](https://github.com/jejjohnson/plumax/issues/224)) ([c5dcf15](https://github.com/jejjohnson/plumax/commit/c5dcf151d44eb56a78c841016586b55f8c6c54a8))
* **lagrangian:** hold the footprint dataset cache in memory ([#216](https://github.com/jejjohnson/plumax/issues/216)) ([99605de](https://github.com/jejjohnson/plumax/commit/99605dee7826307c652b62fbb44ab5c2ce2de39f))

## [0.2.3](https://github.com/jejjohnson/plumax/compare/v0.2.2...v0.2.3) (2026-11-04)


### Features

* **coupled:** nonlinear end-to-end inversion through the radiance operator, on vardax ([#196](https://github.com/jejjohnson/plumax/issues/196)) ([9c780b8](https://github.com/jejjohnson/plumax/commit/9c780b8f3362698eac66ef05fd501b60727f9e85)), closes [#102](https://github.com/jejjohnson/plumax/issues/102)
* **coupled:** per-instrument quality-flag aggregator feeding R and the masks ([#195](https://github.com/jejjohnson/plumax/issues/195)) ([beb35b9](https://github.com/jejjohnson/plumax/commit/beb35b91b48d65bff5a97674c6edcf15049e43c5)), closes [#105](https://github.com/jejjohnson/plumax/issues/105)
* **coupled:** tier II and tier III transport backends for CoupledForward ([#194](https://github.com/jejjohnson/plumax/issues/194)) ([a63b4ff](https://github.com/jejjohnson/plumax/commit/a63b4ffde5d1bacc58efa8613e52b3f68f9dcc35)), closes [#101](https://github.com/jejjohnson/plumax/issues/101)
* **gauss_plume:** briggs plume rise for buoyant and momentum sources ([#190](https://github.com/jejjohnson/plumax/issues/190)) ([8565b7e](https://github.com/jejjohnson/plumax/commit/8565b7eceab0fb1fda22c90c66e94d3e48667fba)), closes [#82](https://github.com/jejjohnson/plumax/issues/82)
* **gauss_plume:** masked-K multi-source inference ([#191](https://github.com/jejjohnson/plumax/issues/191)) ([2a9cee9](https://github.com/jejjohnson/plumax/commit/2a9cee901b615cbedd646a5aa19e64ed0222f028))
* **gauss_plume:** pick the Pasquill–Gifford class from the Obukhov and roughness lengths (Golder 1972) ([#211](https://github.com/jejjohnson/plumax/issues/211)) ([c0bd98a](https://github.com/jejjohnson/plumax/commit/c0bd98ac572323ae29f0200bb70cc88e001fd092)), closes [#86](https://github.com/jejjohnson/plumax/issues/86)
* **gauss_plume:** student-t and heteroscedastic observation noise for Tier I inference ([#189](https://github.com/jejjohnson/plumax/issues/189)) ([0d7b2cb](https://github.com/jejjohnson/plumax/commit/0d7b2cb68f42d304ffa21706b79503fc4ab5fa24)), closes [#84](https://github.com/jejjohnson/plumax/issues/84)
* **lagrangian:** footprint emulator with vardax calibration gates and emulator-based inversion ([#202](https://github.com/jejjohnson/plumax/issues/202)) ([d3c2677](https://github.com/jejjohnson/plumax/commit/d3c26771bb4004440f3be94c1cfc227907f6d18c))
* **population:** importance-corrected TMTPP mark likelihood with ESS diagnostic and log-space quadrature ([#201](https://github.com/jejjohnson/plumax/issues/201)) ([f0d46e6](https://github.com/jejjohnson/plumax/commit/f0d46e65fcc05f1f50e68690f0b4ef7da2f9b711)), closes [#108](https://github.com/jejjohnson/plumax/issues/108)
* **population:** mark registry with lognormal, lognormal-Pareto and mixture families ([#200](https://github.com/jejjohnson/plumax/issues/200)) ([99a18c1](https://github.com/jejjohnson/plumax/commit/99a18c1becb8357cc3a0a48567f58d5708cd3eac)), closes [#109](https://github.com/jejjohnson/plumax/issues/109)
* **population:** per-event posterior payload v2 from every tier ([#199](https://github.com/jejjohnson/plumax/issues/199)) ([bb9fa86](https://github.com/jejjohnson/plumax/commit/bb9fa8615ba21da15105d18e7b27f010724cbf46)), closes [#107](https://github.com/jejjohnson/plumax/issues/107)
* **population:** run the intensity and POD registries on xtremax point processes ([#198](https://github.com/jejjohnson/plumax/issues/198)) ([6413d12](https://github.com/jejjohnson/plumax/commit/6413d12acda5f0f66cbcac09d143e16b136566ed)), closes [#106](https://github.com/jejjohnson/plumax/issues/106)


### Bug Fixes

* **lagrangian:** match the stable and neutral Hanna profiles to Stohl et al. (2005) and FLEXPART ([#210](https://github.com/jejjohnson/plumax/issues/210)) ([9ae3a22](https://github.com/jejjohnson/plumax/commit/9ae3a229510673175cb38827482a83b929948430)), closes [#209](https://github.com/jejjohnson/plumax/issues/209)

## [0.2.2](https://github.com/jejjohnson/plumax/compare/v0.2.1...v0.2.2) (2026-10-30)


### Features

* **gauss_plume:** one column + averaging-kernel observation shared by every tier ([#179](https://github.com/jejjohnson/plumax/issues/179)) ([8ce7ebf](https://github.com/jejjohnson/plumax/commit/8ce7ebf40ffa8971f6b9f91d03a81178fb93baa8)), closes [#83](https://github.com/jejjohnson/plumax/issues/83)
* **geo:** plumax.geo with the ten plume operators from geotoolz ([#177](https://github.com/jejjohnson/plumax/issues/177)) ([68ea412](https://github.com/jejjohnson/plumax/commit/68ea412728f81c70340e6aecc824a522049cfd87))
* **geo:** PlumeVetting, plume vetting on GeoTensors in plumax.geo ([#178](https://github.com/jejjohnson/plumax/issues/178)) ([7974adb](https://github.com/jejjohnson/plumax/commit/7974adbd81ae19a0aba1a01e1b4493ef88eeb9be)), closes [#157](https://github.com/jejjohnson/plumax/issues/157)
* **lagrangian:** gridded C-grid wind sampled at each particle ([#180](https://github.com/jejjohnson/plumax/issues/180)) ([654e9a5](https://github.com/jejjohnson/plumax/commit/654e9a5dc9d953e84d3ff574f4d1c6d6ecc7099f)), closes [#88](https://github.com/jejjohnson/plumax/issues/88)
* **lagrangian:** met-ensemble footprints and the inversion over them ([#181](https://github.com/jejjohnson/plumax/issues/181)) ([84eb653](https://github.com/jejjohnson/plumax/commit/84eb653d133048a657a621039f98a435c3898416)), closes [#89](https://github.com/jejjohnson/plumax/issues/89)

## [0.2.1](https://github.com/jejjohnson/plumax/compare/v0.2.0...v0.2.1) (2026-10-28)


### Features

* **inversion:** run the Tier II and Tier III source inversions on vardax ([#171](https://github.com/jejjohnson/plumax/issues/171)) ([da1c5ae](https://github.com/jejjohnson/plumax/commit/da1c5aeba405413a202807b662c446c7717dc051))
* **vetting:** add null_masks and null_pvalue, a per-scene null over the plume's shape ([#166](https://github.com/jejjohnson/plumax/issues/166)) ([2135084](https://github.com/jejjohnson/plumax/commit/213508459823c81b8392c7f4cabb64de2f54afad)), closes [#155](https://github.com/jejjohnson/plumax/issues/155)
* **vetting:** add select_seeds and background_pool, a candidate's seeds and pool ([#165](https://github.com/jejjohnson/plumax/issues/165)) ([a6bfcac](https://github.com/jejjohnson/plumax/commit/a6bfcacedf54b8ed7bfb1a0844e17da06ccb6da9))
* **vetting:** add vet_candidate and VettingResult, one candidate on radtrax's absorption-shape test ([#167](https://github.com/jejjohnson/plumax/issues/167)) ([44b1142](https://github.com/jejjohnson/plumax/commit/44b11428c12bc42393924bc109ecebbca163ed1e)), closes [#156](https://github.com/jejjohnson/plumax/issues/156)

## [0.2.0](https://github.com/jejjohnson/plumax/compare/v0.1.2...v0.2.0) (2026-10-27)


### ⚠ BREAKING CHANGES

* **met:** load_wrf needs `pip install 'plumax[io]'`, and an XTIME-only file now also needs the SIMULATION_START_DATE attribute (every wrfout carries Times).
* plumax.radtran, plumax.hapi_lut, plumax.assimilation and plumax.matched_filter raise ImportError, and the package root no longer exports radtran or hapi_lut. Use radtrax.slab (forward models, constants, nB-LUT), radtrax.instrument (SpectralResponseFunction, PointSpreadFunction, GroundSamplingDistance), radtrax.spectroscopy.hitran and radtrax.spectroscopy.tables (HITRAN tables, Beer-Lambert on a table; install radtrax[hitran]), radtrax.retrieval.var3d and radtrax.retrieval.matched_filter. The plumax[hapi] extra is gone.
* **radtran:** plumax.radtran no longer exports trimmed_mean_spectrum, robust_lowrank_covariance, matched_filter_pixel/_image/_snr, their _op variants, build_lowrank_covariance_operator or the target_* constructors; use plumax.matched_filter (background, core, target) instead.

### Features

* **background:** randomized SVD via gaussx ([#129](https://github.com/jejjohnson/plumax/issues/129)) ([63d59e4](https://github.com/jejjohnson/plumax/commit/63d59e438d10535def8249e4c5e16b8b67deb6dc))
* **coupled:** plume-to-radiance through radtrax's layered ForwardModel ([#140](https://github.com/jejjohnson/plumax/issues/140)) ([919c995](https://github.com/jejjohnson/plumax/commit/919c9951b02e80ce12eee167577c08359de040dc))
* depend on radtrax and re-point coupled ([#138](https://github.com/jejjohnson/plumax/issues/138)) ([b38a4d5](https://github.com/jejjohnson/plumax/commit/b38a4d5f8b4748284090e1c102e5f27c36b53e18))
* **met:** load ERA5 onto a LocalFrame-placed analysis grid ([#148](https://github.com/jejjohnson/plumax/issues/148)) ([64defb7](https://github.com/jejjohnson/plumax/commit/64defb7720e3d40bd604ba0d7682da406e2c1409))
* **met:** local UTM frame (LocalFrame), shared clock (TimeAxis), and WRF placement by lat/lon ([#147](https://github.com/jejjohnson/plumax/issues/147)) ([dd233fc](https://github.com/jejjohnson/plumax/commit/dd233fcf98ec18f15d37d449999f947c5d04790b))
* **met:** read wrfout through xrtoolz behind a plumax[io] extra ([#146](https://github.com/jejjohnson/plumax/issues/146)) ([030c978](https://github.com/jejjohnson/plumax/commit/030c978cad3bccc5292b8703e4a7b6a78a959d6d)), closes [#69](https://github.com/jejjohnson/plumax/issues/69)
* **met:** surface-layer similarity (Monin–Obukhov), bulk-Richardson PBL height, Hanna factory ([#149](https://github.com/jejjohnson/plumax/issues/149)) ([7d22111](https://github.com/jejjohnson/plumax/commit/7d221114914707827fd26d73dde059a963afe029))
* **obs:** ingest TROPOMI, EMIT and GHGSat L2 into coupled.Instrument ([#150](https://github.com/jejjohnson/plumax/issues/150)) ([29af24c](https://github.com/jejjohnson/plumax/commit/29af24c584daa0085a6b781a8202be5290c0cdba))
* **priors:** inventory prior mean on the analysis grid, per-tile background ([#151](https://github.com/jejjohnson/plumax/issues/151)) ([0c4b997](https://github.com/jejjohnson/plumax/commit/0c4b997b23a3a912d888519b68d008b2f4503c24))


### Miscellaneous

* **radtran:** retire radtran's matched-filter half in favour of matched_filter.core ([#137](https://github.com/jejjohnson/plumax/issues/137)) ([6e39aab](https://github.com/jejjohnson/plumax/commit/6e39aabe042e1242febd0948eb9f7c702547d8ee))
* remove radtran, hapi_lut, assimilation and matched_filter (moved to radtrax) ([#139](https://github.com/jejjohnson/plumax/issues/139)) ([ebc9d75](https://github.com/jejjohnson/plumax/commit/ebc9d75b1432b960174a86de2f1f023ff037ba79))

## [0.1.2](https://github.com/jejjohnson/plumax/compare/v0.1.1...v0.1.2) (2026-09-26)


### Features

* roadmap status refresh, pinned git dependencies, and the MetField wrfout loader ([#116](https://github.com/jejjohnson/plumax/issues/116)) ([46fd747](https://github.com/jejjohnson/plumax/commit/46fd74721150a4dbc9ca8aa492a169a8a8b515ec))
* tier V.B: Intensity, POD, and paradox modules with theory & notebooks ([#125](https://github.com/jejjohnson/plumax/issues/125)) ([7419a9e](https://github.com/jejjohnson/plumax/commit/7419a9ec35ad9db7a934757d32354ffd58e681e1))

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
