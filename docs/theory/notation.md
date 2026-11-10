# Notation

This page collects the symbols that the design and theory pages use, grouped by topic, with their units and the pages where they appear. Equations on the pages are typeset, and code comments and docstrings write the same symbols in Unicode (Q, σ_y, ΔΩ).

The pages were written at different times and for different readers, so a few letters mean different things in different places. $L$ is the boundary-layer height in Tier I, the Obukhov length on the meteorology pages and the plume length in the IME method; $\lambda$ is an event intensity on the population pages, but a chemical loss rate and an adjoint variable on the Tier III page; and $H$ is both the forward map and the effective stack height on the Tier I page. The Pages column says where each meaning applies, so read a symbol with the page it sits on. Which of these to rename is still open ([#254](https://github.com/jejjohnson/plumax/issues/254)).

## Transport and geometry

:::{table} Transport and geometry: symbols as the pages use them today.
:label: tbl-notation-transport-and-geometry

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $C$, $C(\vec x, t)$ | mass concentration of a passive tracer | kg m⁻³ | README, [Home](../index.md), [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $c$, $c(\mathbf x,t)$ | concentration used by the design pages (Tier III says mixing ratio in kg/kg; Tier I and Tier II say kg m⁻³) | kg m⁻³ or kg kg⁻¹ | [Tier I](../design/01_tier1_gaussian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md), [RTM stack](../design/04_rtm_stack.md), [Equations](equations.md) |
| $\vec x=(x,y,z)$ | position, $x$ downwind, $y$ crosswind, $z$ up | m | README, [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $x'$, $y'$ | source-aligned coordinates | m | [Tier I](../design/01_tier1_gaussian.md) |
| $\tilde x,\tilde y,\tilde z,\tilde t$ | puff-centred coordinates and puff age | m, s | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $\vec x_0=(x_0,y_0,H)$, $x_0$ | source position (puff); $x_0=(\mathrm{lat}_0, \mathrm{lon}_0)$ source location (Tier I) | m; degrees | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Tier I](../design/01_tier1_gaussian.md), [Prerequisites](../design/00_prerequisites.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\vec x_c(t)$, $x_c,y_c$ | puff centre trajectory | m | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $\vec x_s$, $r_s$, $\rho_s$, $Z$ | Gaussian-regularised source centre, radius, shape and normaliser | m, m, m⁻³, - | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $\vec u=(u,v,w)$, $\mathbf u(\mathbf x,t)$ | wind vector and components | m s⁻¹ | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Prerequisites](../design/00_prerequisites.md) |
| $u$, $\bar u$ | wind speed in the plume formulae | m s⁻¹ | README, [Home](../index.md), [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Tier I](../design/01_tier1_gaussian.md), [Equations](equations.md) |
| $U_\text{eff}$, $U_\text{catalog}$, $U_\text{target}$, $U_\text{ERA5}$ | effective, catalogue and reanalysis wind speed | m s⁻¹ | README, [V.A](../design/06_tier5a_instantaneous.md) |
| $\mathbf U$, $U=\Vert\mathbf U\Vert$ | 10 m wind vector and its speed | m s⁻¹ | [Inventory I](inventory/01_inventory_problem.md) |
| $\mathbf v=86.4\,\mathbf U$ | advection velocity in space-time kernels | km d⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\theta_\text{wind}$, $\theta_\text{met}$ | wind direction | degrees | [Tier I](../design/01_tier1_gaussian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $I_u,I_v,s$ | cumulative wind integrals and travel distance | m | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $\vec J$, $\vec J_A$ | total contaminant mass flux, advective part | kg m⁻² s⁻¹ | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md) |
| $S(\vec x,t)$ | volumetric source term | kg m⁻³ s⁻¹ | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $Q$, $Q(t)$, $Q_k(t)$, $Q_i$ | emission rate (steady, time-resolved, per source, per event) | kg s⁻¹ (design); kg/hr (MT:60); t/h, kg/h (catalogues) | README, [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Tier I](../design/01_tier1_gaussian.md), [V.B](../design/06_tier5b_point_process.md), [Tier IV](../design/05_tier4_coupled.md), [TMTPP theory](population/01_mttpp_theory.md) |
| $m$, $m_i$ | puff mass | kg | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $q(t)$ | emission rate in the source shape $S=q\rho_s$ | kg s⁻¹ | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $H$, $H_\text{eff}$, $H_\text{stack}$, $\Delta h$ | effective, physical stack height, plume rise | m | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Tier I](../design/01_tier1_gaussian.md) |
| $h$ | source height (README) | m | README, [Home](../index.md) |
| $\mathbf K=\mathrm{diag}(K_x,K_y,K_z)$, $K_h$, $K(x)$ | eddy diffusivity | m² s⁻¹ | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md), [Tier III](../design/03_tier3_eulerian.md) |
| $r$ | reduced distance $\frac1u\int_0^xK\,\mathrm d\xi$ | m² | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md) |
| $c(r,y,z)$, $a(r,y)$, $b(r,z)$ | reduced-coordinate solution and its y- and z-factors | kg m⁻³; m⁻¹ | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md) |
| $\sigma_y(x)$, $\sigma_z(x)$, $\sigma_x$, $\sigma$ | Gaussian dispersion widths (Briggs / Pasquill-Gifford) | m | README, [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md), [Tier I](../design/01_tier1_gaussian.md), [Prerequisites](../design/00_prerequisites.md) |
| $a_i,b_i,c_i$ ; $a,b$ ; $a_y,b_y,c_y,a_z,b_z,c_z$ | Briggs and PG coefficients; power-law $\sigma^2=ax^b$ | - | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md), [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $G_{\sigma^2}(s)$ | 1-D normal density | m⁻¹ | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $\xi$ | dummy integration variable in $r$ | m | [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md) |
| $\tau$ (puff age) | $\tau=t-t_0$ | s | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $t_0$, $t_i$ | release time(s) | s | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $f_\text{release}$, $\Delta t$ | puff release frequency, spacing | s⁻¹, s | [Gaussian puff](gauss_puff/00_gaussian_puff_derivation.md) |
| $\Delta t$, $\Delta x,\Delta y,\Delta z$ | time step and cell sizes | s, m | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md), [Tier II](../design/02_tier2_lagrangian.md) |
| $C_k$, $w_{k+1/2}$, $\widehat F_{k+1/2}$, $K_{z,k}$ | finite-volume cell values, face velocity, upwind flux | - | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $\partial_t(\rho c)$, $\rho(\mathbf x,t)$ | dry-air density in the conservation form | kg m⁻³ | [Tier III](../design/03_tier3_eulerian.md) |
| $\lambda$ (chemical loss) | first-order loss rate | s⁻¹ | [Tier III](../design/03_tier3_eulerian.md) |
| $\lambda_\text{chem}$, $\lambda$ (adjoint) | adjoint chemical loss and adjoint concentration | - | [Tier III](../design/03_tier3_eulerian.md) |
| $k_\text{OH}$ | OH loss rate constant | s⁻¹ | [Tier II](../design/02_tier2_lagrangian.md) |
| $\mathbf F$, $F(r,s)$ | source-receptor footprint matrix and entry | s m² kg⁻¹; $(n_\text{obs},n_\text{grid})$ | [Tier II](../design/02_tier2_lagrangian.md) |
| $\tilde{\mathbf F}=\mathrm{diag}(\mathbf q_a)\mathbf F$ | linearised Jacobian | - | [Tier II](../design/02_tier2_lagrangian.md) |
| $V_s$, $\rho_\text{air}$, $z_\text{part}$, $f_\text{PBL}$ | cell volume, air density, particle height, PBL fraction | m³, kg m⁻³, m, - | [Tier II](../design/02_tier2_lagrangian.md) |
| $\mathbf v$, $\mathbf a$, $\mathbf b$, $\mathbf W$ | turbulent velocity, drift, diffusion, Wiener process | m s⁻¹ | [Tier II](../design/02_tier2_lagrangian.md) |
| $\sigma_{ij}$, $\tau_L$ | Reynolds stress, Lagrangian timescale | m² s⁻², s | [Tier II](../design/02_tier2_lagrangian.md) |
| $\mathbf b\to\mathbf 0$ | zero-turbulence limit | - | [Tier II](../design/02_tier2_lagrangian.md) |
| $\mathbf u,\mathbf v$ (test vectors) | random vectors in the adjoint test | - | [Tier III](../design/03_tier3_eulerian.md) |
| $\mathcal T$ | atmospheric transport operator $\mu\mapsto f$ | ppb per kg h⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $C_\Delta$ | coarsening operator | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\mathbf x,\mathbf x_a,\mathbf h$ (AK page) | model profile, retrieval prior, column weights | - | [Prerequisites](../design/00_prerequisites.md) |
| $S$, $x\in S$, $A_1..A_4$ | study region, location, cells | km, km² | [Inventory I](inventory/01_inventory_problem.md) |
| $\Delta x$ | pixel size | km | [Inventory I](inventory/01_inventory_problem.md) |
| $T=[0,T_\text{end}]$, $t\in T$ | time window | d | [Inventory I](inventory/01_inventory_problem.md) |
| $X$, $A\subset X$, $\lvert A\rvert$ | index set, region, area | km² | [Inventory I](inventory/01_inventory_problem.md) |
| $\delta_x$ | point mass | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\mathbf v$ (INV) | see $\mathbf v=86.4\mathbf U$ above | km d⁻¹ | [Inventory II](inventory/02_inventory_objects.md) |
:::

## Meteorology and turbulence

:::{table} Meteorology and turbulence: symbols as the pages use them today.
:label: tbl-notation-meteorology-and-turbulence

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $L$ (PBL height) | capping inversion height | m | [Tier I](../design/01_tier1_gaussian.md) |
| $L(s,t)$ | local PBL height in the footprint | m | [Tier II](../design/02_tier2_lagrangian.md) |
| $L$, $L_\text{Obukhov}$ | Obukhov length | m | [Prerequisites](../design/00_prerequisites.md), [Tier I](../design/01_tier1_gaussian.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md) |
| $u_*$ | friction velocity | m s⁻¹ | [Prerequisites](../design/00_prerequisites.md), [Tier I](../design/01_tier1_gaussian.md) |
| $z_0$ | surface roughness | m | [Prerequisites](../design/00_prerequisites.md) |
| $\kappa$ | von Karman constant in $K_z\approx\kappa u_*z(1-z/L)$ | - | [Tier III](../design/03_tier3_eulerian.md) |
| `u,v,w,T,p,q,tke,pblh,z0,lu,hgt` | MetField variables | m/s, K, Pa, kg/kg, m²/s², m | [Prerequisites](../design/00_prerequisites.md) |
| $T$ | temperature | K | [RTM stack](../design/04_rtm_stack.md), [Prerequisites](../design/00_prerequisites.md) |
| $p$ | pressure | Pa | [RTM stack](../design/04_rtm_stack.md) |
| $q_{\text{CH}_4}$, $q_{\text{H}_2\text{O}}$ | species mixing ratios | - | [RTM stack](../design/04_rtm_stack.md) |
| $\rho_{\text{CH}_4}(z)$ | CH₄ number/mass density profile | - | [RTM stack](../design/04_rtm_stack.md) |
| $\overline{u'C'}$ | turbulent flux | kg m⁻² s⁻¹ | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md) |
| $\mathrm{AOD}$ | aerosol optical depth | - | [RTM stack](../design/04_rtm_stack.md) |
| PG class A-F | Pasquill-Gifford stability class | - | [Prerequisites](../design/00_prerequisites.md), [Gaussian plume](gauss_plume/00_gaussian_plume_derivation.md) |
| $\mathrm{SZA},\mathrm{VZA},\mathrm{RAA}$ | solar zenith, viewing zenith, relative azimuth | degrees | [RTM stack](../design/04_rtm_stack.md) |
| $z_k$, $k$, $\Delta z$ | vertical level, index, thickness | m | [Eulerian FV](les_fvm/00_eulerian_dispersion_derivation.md), [Tier II](../design/02_tier2_lagrangian.md) |
| $N_x,N_y,N_z$ | grid sizes | - | README, [Home](../index.md) |
| $T,Z,Y,X$ | MetField axis sizes in shape strings | - | [Prerequisites](../design/00_prerequisites.md) |
:::

## Observation, column and retrieval

:::{table} Observation, column and retrieval: symbols as the pages use them today.
:label: tbl-notation-observation-column-and-retrieval

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $o_k$ | sensor reading $C(x_k, y_k, z_k)+b+\varepsilon_k$ | kg m⁻³ | README, [Home](../index.md) |
| $b$ | scalar background in the README example | kg m⁻³ | README |
| $\varepsilon_k\sim\mathcal N(0,s^2)$, $s$ | noise and its standard deviation | kg m⁻³ | README, [Home](../index.md) |
| $\boldsymbol\varepsilon$, $\varepsilon(x,y)$, $\boldsymbol\varepsilon_t$, $\boldsymbol\varepsilon_\text{inst}$ | observation noise | as $\mathbf y$ | [Tier I](../design/01_tier1_gaussian.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\sigma_\text{retr}(x,y)$ | per-pixel retrieval standard deviation | ppb | [Tier I](../design/01_tier1_gaussian.md), [Tier II](../design/02_tier2_lagrangian.md) |
| $\sigma_r$, $\varepsilon_r$ | per-pixel retrieval precision and noise | ppb | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\mathbf R$, $\mathbf R_\text{retr}$, $\mathbf R_\text{repr}$, $\mathbf R_\text{align}$, $\mathbf R_t$, $\mathbf R_\text{inst}$ | observation-error covariance and its parts | $D_y\times D_y$ | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf S_\varepsilon$ | retrieval measurement-error covariance | $D_y\times D_y$ | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf y_\text{model}(x,y)$, $y_\text{obs}$, $\hat y$ | simulated and observed column; AK-applied column | ppb | [Tier I](../design/01_tier1_gaussian.md), [Prerequisites](../design/00_prerequisites.md) |
| $\mathbf y$, $\mathbf y_t$, $\mathbf y_\text{inst}$ | observations | - | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $y_\text{patch}$, $C,H_\text{patch},W_\text{patch}$ | image patch input to the amortized predictor | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\mathbf y_\text{radiance}$ | spectrum | - | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf A$, $\mathbf A_t$, $\mathbf A_\text{inst}$ | satellite averaging-kernel matrix | $D_x\times D_x$ | [Prerequisites](../design/00_prerequisites.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md), [RTM stack](../design/04_rtm_stack.md) |
| $A_\text{surf}$ | surface albedo | - | [RTM stack](../design/04_rtm_stack.md) |
| $\mathrm{col}_z$ | vertical column operator | - | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md) |
| $\mathbf c_\text{bg}$, $c_\text{bg}(x,y)$ | regional background | ppb | [Tier I](../design/01_tier1_gaussian.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf H$, $\mathbf H_t$, $\mathbf H_\text{inst}$, $\mathbf H_\text{obs}$ | observation operator | - | [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md), [RTM stack](../design/04_rtm_stack.md) |
| $H$ (T1 forward map) | forward map $(Q, x_0, H_\text{eff}, \dots)\to y_\text{model}$ | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\mathbf x$ (retrieval state) | $(\text{profile}_{\text{CH}_4}, \text{profile}_{\text{H}_2\text{O}}, A_\text{surf}, \text{AOD}, p_\text{surf, offset})$ | - | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf x_a$, $\mathbf S_a$ | retrieval prior mean and covariance | - | [RTM stack](../design/04_rtm_stack.md) |
| $F(\mathbf x)$ | RTM forward model | radiance | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf K_k=\partial F/\partial\mathbf x$ | RTM Jacobian | $D_y\times D_x$ | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf G_k$ | gain matrix | $D_x\times D_y$ | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf S^*$, $\Delta\mathbf S$ | posterior covariance, contraction | - | [RTM stack](../design/04_rtm_stack.md) |
| $\mathrm{DOFs}=\mathrm{tr}(\mathbf A)$, $H$ (Shannon) | degrees of freedom, information content | - | [RTM stack](../design/04_rtm_stack.md) |
| $\delta\chi^2_k$, $k_\text{max}$ | convergence criterion, iteration cap | - | [RTM stack](../design/04_rtm_stack.md) |
| $\nu$ (wavenumber), $L(\nu)$ | radiance spectrum | W m⁻² sr⁻¹ cm | [RTM stack](../design/04_rtm_stack.md) |
| $F_\text{solar}$, $\tau$, $\tau_\text{total}$, $\sigma(\nu,T,p)$ | solar irradiance, optical depth, two-way depth, cross-section | - | [RTM stack](../design/04_rtm_stack.md) |
| $L_\text{TIR}$, $\varepsilon_\text{surf}$, $B(T,\nu)$ | thermal radiance, emissivity, Planck | - | [RTM stack](../design/04_rtm_stack.md) |
| $T_\text{gas}$, $R_\text{surf}$, $S_\text{scatt}$ | factorised LUT terms | - | [RTM stack](../design/04_rtm_stack.md) |
| $\Delta\Omega(x,y)$, $\Delta\Omega_i$ | column enhancement | kg m⁻² | README |
| $M$, $A_i$, $\mathrm{IME}$ | plume mask, pixel area, integrated mass enhancement | -, m², kg | README |
| $\mathbf t$ | matched-filter target spectrum | - | README |
| $\tau$ (Otsu threshold) | segmentation threshold | kg m⁻² | README |
| $\widehat{\Delta X}_\text{MF}$, $\widehat{\Delta X}$, $\Delta X$ | matched-filter and transmittance-based column | ppm·m or mol m⁻² | [Plume vetting](plume_vetting.md) |
| $R(\lambda)\approx\eta(\lambda)E(\lambda;\Delta X)$ | ratio spectrum, surface ratio, transmittance | - | [Plume vetting](plume_vetting.md) |
| $D_\text{norm}$ | normalised vetting distance | - | [Plume vetting](plume_vetting.md) |
| $I$, $O$, $I_0$, $M_c$, $M_s$, $M_\theta$ | in-pixels, out-pixels, seeds, candidate masks, null copies | - | [Plume vetting](plume_vetting.md) |
| $k=40$, $g=1$, $b=200$, $q$ | seed count, growth steps, box half-width, 99th percentile | - | [Plume vetting](plume_vetting.md) |
| $\tau_\text{bg}$ | background pool score threshold | ppm·m | [Plume vetting](plume_vetting.md) |
| $\hat p$, $S$ (copies), $S'$ | MC p-value, null copies, finite-score copies | -, count | [Plume vetting](plume_vetting.md) |
| $\theta\sim U[0°,360°)$, $\Delta$ | null rotation, shift | degrees, px | [Plume vetting](plume_vetting.md) |
| $L_\text{out}$ | out-pixel radiance | - | [Plume vetting](plume_vetting.md) |
| $\mathrm{XCH_4}$, $\Delta\mathrm{VMR}$, $\Delta\tau$ | column mole fraction, enhancement, optical depth increment | ppb | README, [Inventory I](inventory/01_inventory_problem.md), [Tier IV](../design/05_tier4_coupled.md) |
| $y(x)=\mathrm{XCH_4}(x)$, $f(x)$, $f_\text{bg}$ | raw column, enhancement, background | ppb | [Inventory II](inventory/02_inventory_objects.md), [Inventory I](inventory/01_inventory_problem.md) |
| $\lambda$ (wavelength) | spectral coordinate | nm | [Plume vetting](plume_vetting.md) |
| $\mathrm{AMF}$ | air-mass factor | - | [RTM stack](../design/04_rtm_stack.md) |
:::

## Inversion and assimilation

:::{table} Inversion and assimilation: symbols as the pages use them today.
:label: tbl-notation-inversion-and-assimilation

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $\mathbf x$ (T1/T4 state) | coupled state $(Q_{1:K}(t),x_{0,1:K},\bar u,\theta_\text{wind},\mathbf c_\text{bg},\alpha_\text{BC},\text{bias}_\text{inst},\dots)$ | - | [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf q$, $\mathbf q_a$ | source vector and prior | kg s⁻¹ per cell | [Tier II](../design/02_tier2_lagrangian.md) |
| $q_a$, $\sigma_{\log Q}$ | background inventory prior, lognormal width | - | [Prerequisites](../design/00_prerequisites.md) |
| $S$, $S_b$, $S^k$, $\delta S$ | Tier III source field, prior mean, iterate, increment | kg m⁻³ s⁻¹ | [Tier III](../design/03_tier3_eulerian.md) |
| $c_0$, $c_b(0)$ | initial concentration and its prior | - | [Tier III](../design/03_tier3_eulerian.md) |
| $\mathbf B$, $\mathbf B_\text{log}$, $\mathbf B_c$ | background (prior) covariance | - | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf x_b$ | prior mean of the coupled state | - | [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf P^*$ | posterior covariance | - | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md) |
| $\boldsymbol\chi=\mathbf B^{-1/2}(S-S_b)$ | control transform variable | - | [Tier III](../design/03_tier3_eulerian.md), [Equations](equations.md) |
| $J(S,c_0)$, $J_\text{lin}$, $J(\mathbf x)$ | 4D-Var cost | scalar | [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md), [RTM stack](../design/04_rtm_stack.md) |
| $\mathcal L$ in $\partial\mathcal L/\partial c$ | adjoint forcing (a cost, not a loss) | - | [Tier III](../design/03_tier3_eulerian.md) |
| $\lVert\cdot\rVert^2_{\mathbf R}$ | weighted norm | - | [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\mathbf K_Q$ | OU/GP kernel matrix for $Q(t)$ | - | [Tier IV](../design/05_tier4_coupled.md) |
| $\ell$ | Matérn correlation length | km | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md) |
| $\alpha_\text{face}$, $\alpha$, $\alpha_\text{BC}$ | lateral-BC scaling factor | - | [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md) |
| $\text{bias}_\text{inst}\sim\mathcal N(0,\sigma^2_\text{inst})$ | additive per-instrument bias | ppb | [Tier IV](../design/05_tier4_coupled.md) |
| $\theta$, $\mu_Q$, $\sigma_Q$, $W(t)$ | OU rate, mean, noise scale, Wiener process | s⁻¹, kg s⁻¹ | [Tier IV](../design/05_tier4_coupled.md) |
| $Q(t)\sim\mathrm{OU}$, $\mu_Q(t)$ | stochastic emission process | kg s⁻¹ | [Tier IV](../design/05_tier4_coupled.md) |
| $K=n_\text{sources}$, $K_\text{max}$, $p_k$ | source count, cap, activity probabilities | - | [Tier IV](../design/05_tier4_coupled.md) |
| $\mu_Q$, $\sigma_Q$ (T1 prior) | lognormal prior parameters on $Q$ | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\sigma_{x_0}$, $\sigma_H$, $\sigma_{\bar u,\text{met}}$, $\sigma_{\theta,\text{met}}$ | prior standard deviations | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\nu\approx5$ | Student-t degrees of freedom | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\mathrm{CV}(Q)$, $\sigma_\text{post}(Q)$ | coefficient of variation, posterior std | - | [Tier I](../design/01_tier1_gaussian.md) |
| $Q^*$, $x_0^*$, $H^*$ | synthetic truth | - | [Tier I](../design/01_tier1_gaussian.md) |
| $f_\theta$, $g_\phi$ | amortized predictor, coupled emulator | - | [Tier I](../design/01_tier1_gaussian.md), [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md), [Tier IV](../design/05_tier4_coupled.md), [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf x^k$ | RTM iterate | - | [RTM stack](../design/04_rtm_stack.md) |
| $\mathbf U,\mathbf V$ (low rank) | factors of $\mathbf F\approx\mathbf U\mathbf V^\top$, rank $r$ | - | [Tier II](../design/02_tier2_lagrangian.md) |
| $\mathbf F^\top$, $\langle\cdot,\cdot\rangle$ | adjoint of forward (test) | - | [Tier III](../design/03_tier3_eulerian.md) |
| $N_\text{met}$, $n_\text{obs}$, $n_\text{grid}$, $n_\text{iter}$, $n_\text{steps}$ | counts | - | [Tier II](../design/02_tier2_lagrangian.md), [Tier III](../design/03_tier3_eulerian.md) |
| $\mathrm{ESS}_i$, $w_s$ | importance-weight effective sample size | - | [Tier V](../design/06_tier5_population.md) |
| $\pi_\text{per-event}(Q)$ | per-event prior used at Tiers I-IV | - | [Tier V](../design/06_tier5_population.md), [V.A](../design/06_tier5a_instantaneous.md), [Equations](equations.md) |
| $L_i(Q)=p(\text{obs}_i\mid Q)$ | per-event likelihood | - | [Tier V](../design/06_tier5_population.md), [V.A](../design/06_tier5a_instantaneous.md), [V.B](../design/06_tier5b_point_process.md), [Equations](equations.md) |
| $Q_i^{(s)}$, $S$ (draws), $s$ | posterior draw and draw count | - | [Tier V](../design/06_tier5_population.md), [V.A](../design/06_tier5a_instantaneous.md) |
| $\mu_{\log Q}$, $\sigma_{\log Q}$ | lognormal summary of a per-event posterior | - | [Tier V](../design/06_tier5_population.md), [V.A](../design/06_tier5a_instantaneous.md) |
| $(\boldsymbol\mu_{xy},\boldsymbol\Sigma_{xy})$ | source-location posterior summary | - | [Tier V](../design/06_tier5_population.md) |
| $Q_\text{corrected}$, $Q_\text{catalog}$, $L_\text{plume}$ | catalogue rescaling | kg s⁻¹ | [V.A](../design/06_tier5a_instantaneous.md) |
:::

## Point processes and population

:::{table} Point processes and population: symbols as the pages use them today.
:label: tbl-notation-point-processes-and-population

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $\lambda(t)$, $\lambda_0$, $\lambda_\text{true}$, $\lambda_\text{obs}$, $\lambda_\text{facility}$ | event intensity (rate) | s⁻¹ (5B:17), events/hour (MT:24), plumes/day (ZOO:38), d⁻¹ (INV1:168) | [Tier V](../design/06_tier5_population.md), [V.B](../design/06_tier5b_point_process.md), [TMTPP theory](population/01_mttpp_theory.md), [Intensity zoo](population/02_intensity_zoo.md), [V.C](../design/06_tier5c_persistency.md), [Inventory I](inventory/01_inventory_problem.md) |
| $\lambda^*(t)$, $\mathcal H_t$ | conditional intensity, history | d⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\lambda(\mathrm d x)$, $\lambda(A)$, $\lambda(x)$, $\lambda_\ge$ | spatial source intensity measure and density, above-floor part | km⁻² | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\lambda_K$ | Poisson prior mean on source count | - | [Tier I](../design/01_tier1_gaussian.md) |
| $\Lambda(T)$, $\Lambda_\text{true}(t)$, $\Lambda(t_0,t_0+s)$ | cumulative intensity | events | [TMTPP theory](population/01_mttpp_theory.md), [V.C](../design/06_tier5c_persistency.md) |
| $\Lambda_\text{fp}(A)$ | expected false positives | - | [Inventory III](inventory/03_inventory_phases.md) |
| $N(t_1,t_2)$, $N_\text{true}(t)$, $N_\text{detected}$ | counting process | - | [V.C](../design/06_tier5c_persistency.md), [TMTPP theory](population/01_mttpp_theory.md), [V.D](../design/06_tier5d_total_emission.md) |
| $N(A)$, $N_\text{det}(A)$, $N_\text{miss}(A)$, $N_\text{obs}$ | source counts per region | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory III](inventory/03_inventory_phases.md) |
| $f(Q)$, $f_\text{true}(Q)$, $f_\text{obs}(Q)$ | mark density | 1/(kg s⁻¹) | [V.B](../design/06_tier5b_point_process.md), [TMTPP theory](population/01_mttpp_theory.md) |
| $f(\mathrm du)$ | above-floor mark law | - | [Inventory III](inventory/03_inventory_phases.md) |
| $\mathbb E[Q]$, $\mathbb E[P_d]$, $\mathbb E[Q\mid\text{detected}]$ | mean mark, mean POD | - | [V.D](../design/06_tier5d_total_emission.md), [TMTPP theory](population/01_mttpp_theory.md) |
| $P_d(Q)$, $P_d^k$, $P_d^\cup$ | probability of detection, per-satellite, union | - | [V.B](../design/06_tier5b_point_process.md), [V.D](../design/06_tier5d_total_emission.md), [V.A](../design/06_tier5a_instantaneous.md) |
| $Q_{50}$, $k$ (Hill steepness), $Q_{50,\text{inst}}$, $k_\text{inst}$ | Hill POD parameters | kg s⁻¹, - | [V.B](../design/06_tier5b_point_process.md) |
| $Q_\text{min}$, $Q_\text{break}$ | power-law and lognormal-Pareto parameters | kg s⁻¹ | [V.B](../design/06_tier5b_point_process.md) |
| $\alpha$ (power-law exponent), $\mathrm{Pareto}(\alpha,Q_\text{break})$ | mark tail index | - | [V.B](../design/06_tier5b_point_process.md), [V.A](../design/06_tier5a_instantaneous.md) |
| $\mu,\sigma^2$ (LN), $\pi_k$ (mixture weights) | lognormal and mixture parameters | - | [V.B](../design/06_tier5b_point_process.md) |
| $\mu$, $\alpha$, $\beta$ (Hawkes) | baseline, boost, decay | d⁻¹, d⁻¹, d⁻¹ | [V.B](../design/06_tier5b_point_process.md), [Intensity zoo](population/02_intensity_zoo.md), [V.C](../design/06_tier5c_persistency.md) |
| $\varphi$ | Hawkes kernel (INV) or phase of diurnal sinusoid (5B) | - | [Inventory II](inventory/02_inventory_objects.md), [V.B](../design/06_tier5b_point_process.md) |
| $A$, $T_\text{day}$, $\tau$ (5B) | sinusoid amplitude, period, blowdown decay time | d⁻¹, d, d | [V.B](../design/06_tier5b_point_process.md) |
| $t_\text{open}$, $t_\text{prev}$, $t_\text{last}$, $h(t-t_\text{last})$ | clock markers, renewal hazard | d | [V.B](../design/06_tier5b_point_process.md), [V.C](../design/06_tier5c_persistency.md), [Intensity zoo](population/02_intensity_zoo.md) |
| $\mathcal{GP}(\mu,K)$, $K(t,t')$ | log-Gaussian Cox process | - | [V.B](../design/06_tier5b_point_process.md), [Intensity zoo](population/02_intensity_zoo.md) |
| $z(t)$, $\omega$, $h(t)$, $\mathrm{RNN}$, $W$, $b$ | ODE-state and neural intensity | - | [Intensity zoo](population/02_intensity_zoo.md) |
| $g(t)$ | deterministic intensity | - | [Intensity zoo](population/02_intensity_zoo.md) |
| $\Delta t$ (wait), $\mathbb E[\Delta t\mid t_0]$ | waiting time to next event | h | [V.C](../design/06_tier5c_persistency.md), [Persistency](population/03_persistency.md) |
| $\mathbb P(N(t_1,t_2)\ge1)$, $P(\cdot)$ | occurrence probability | - | [V.C](../design/06_tier5c_persistency.md), [Persistency](population/03_persistency.md) |
| $S(t\mid t_0)$, $h(t)$, $f_\text{time}$ | survival, hazard, wait-time density | - | [Persistency](population/03_persistency.md) |
| $M_\text{total}(T)$, $M_\text{naive}$, $M_\text{event}$, $\mathbb E[D]$ | total mass, biased estimator, mass per event, duration | kg | [V.D](../design/06_tier5d_total_emission.md), [Persistency](population/03_persistency.md) |
| $Q_\text{crit}$ | exceedance threshold | kg/hr | [Persistency](population/03_persistency.md) |
| $p$, $n$, $k$, $X$ (Binomial/Poisson proof) | trial probability, slices, count, variable | - | [Persistency](population/03_persistency.md) |
| $\log L(\lambda,f,P_d)$ | TMTPP log-likelihood | - | [Tier V](../design/06_tier5_population.md), [V.B](../design/06_tier5b_point_process.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\mathcal D$ | set of detections | - | [Tier V](../design/06_tier5_population.md), [V.D](../design/06_tier5d_total_emission.md) |
| $K$ (satellites) | number of satellites | - | [V.D](../design/06_tier5d_total_emission.md) |
| $K_i$ | number of sources in overpass $i$ | - | [V.A](../design/06_tier5a_instantaneous.md) |
| $\operatorname{LogNormal}$, $\operatorname{LN}$, $\operatorname{Gamma}(\alpha_0+n,\beta_0+AT)$ | families; Gamma-Poisson update | - | [V.B](../design/06_tier5b_point_process.md), README, [V.C](../design/06_tier5c_persistency.md) |
| $A\cdot T$ | exposure (area times duration) | km² d | README |
| $\mathrm{Bernoulli}$, $\mathrm{Poisson}$, $\mathrm{Beta}$ | families | - | [Inventory II](inventory/02_inventory_objects.md), [Inventory III](inventory/03_inventory_phases.md) |
| $\mu$ (measure), $\mu_\text{pt}$, $\mu_\text{df}$, $\mu_\text{atomic}$, $\mu_\text{ac}$, $\mu_\text{sc}$, $\bar\mu$ | emission measure and parts | kg h⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $s_k$, $s_0$, $u_k=s_k/s_0$, $\bar s_k=\pi_ks_k$ | on-state rate, reference, dimensionless rate, time-averaged rate | kg h⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\pi_k$ | persistence (fraction of time on) | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $G(\mathrm d\pi\mid u)$, $G(A)$ | persistence law; fraction of basin total | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\rho(\mathrm du)$ | Lévy measure over dimensionless rates | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\nu(\mathrm dx,\mathrm du)$ | space-rate intensity measure | - | [Inventory II](inventory/02_inventory_objects.md) |
| $\sigma,\tau$ (tempered stable), $\theta_d$, $\theta_\text{PY}$, $\theta$ (Laplace) | Lévy-measure and Pitman-Yor parameters | - | [Inventory II](inventory/02_inventory_objects.md) |
| $B_0$, $c$ (beta process) | base measure, concentration | - | [Inventory II](inventory/02_inventory_objects.md) |
| $Z_{ik}$, $Z_j$, $Z_i$, $c_j$ | on/off indicators, candidate sites | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md), [Inventory III](inventory/03_inventory_phases.md) |
| $p_i(x,u)$, $p(x,s)$, $\bar p(x)$, $p_\text{camp}$, $p_\text{next}$, $p_\text{cloud}$ | detection probabilities | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory III](inventory/03_inventory_phases.md) |
| $R=\rho([u_\text{floor},\infty))$ | mass above detection floor | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory III](inventory/03_inventory_phases.md) |
| $u_\text{floor}$, $s_\text{min}$ | detection floor | -, kg h⁻¹ | [Inventory I](inventory/01_inventory_problem.md), [Inventory III](inventory/03_inventory_phases.md) |
| $e(x)$ | diffuse flux density | kg h⁻¹ km⁻² | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $h(x)$, $H(A)$, $H=(500,\dots)$ | bottom-up emission density; prior expected emission | kg h⁻¹ km⁻², kg h⁻¹ | [Inventory III](inventory/03_inventory_phases.md), [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $m(x)$, $k(x,x')$, $\sigma_f$, $\ell$, $\mathbf m$, $\mathbf K$ | GP mean, kernel, amplitude, length, finite-dim mean/kernel matrix | ppb | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $m_g$, $k_g$, $g=\log\lambda$ | log-density GP mean and kernel | - | [Inventory I](inventory/01_inventory_problem.md), [Inventory III](inventory/03_inventory_phases.md) |
| $\nu_t$, $\Sigma(x)=A(x)A^\mathsf T$ | Student-t dof, matrix-GP covariance | - | [Inventory II](inventory/02_inventory_objects.md) |
| $m_k$, $M(t)$, $t_k$ | event mass, cumulative mass | kg | [Inventory I](inventory/01_inventory_problem.md), [Inventory II](inventory/02_inventory_objects.md) |
| $\ell_t$, $\mathrm dW$, $\nu=p+\tfrac12$ | OU timescale, Brownian, Matérn smoothness | d | [Inventory II](inventory/02_inventory_objects.md) |
| $\mathbf 1^\mathsf T\Sigma\mathbf 1/n^2$ | stacked noise variance | - | [Inventory II](inventory/02_inventory_objects.md) |
| $\eta_k$ | log-normal per-plume rate noise | - | [Inventory II](inventory/02_inventory_objects.md), [Inventory III](inventory/03_inventory_phases.md) |
| $k_S\cdot k_T$, $V_k(t)$ | separable kernel, stick-breaking weight | - | [Inventory II](inventory/02_inventory_objects.md) |
| $\mathcal H$ | DA observation operator (named for contrast) | - | [Inventory II](inventory/02_inventory_objects.md) |
| `λ_true`, `λ_obs`, `E[P_d]`, `f_true(Q)`, `t_i`, `Q_i`, `N_true(t)` (backticks, Unicode) | plain-text spellings of the above | - | [TMTPP theory](population/01_mttpp_theory.md), [Persistency](population/03_persistency.md) |
:::

## Inventory priors

:::{table} Inventory priors: symbols as the pages use them today.
:label: tbl-notation-inventory-priors

| Symbol | Meaning | Units or shape | Pages |
| --- | --- | --- | --- |
| $q_a$ | background emission inventory | - | [Prerequisites](../design/00_prerequisites.md) |
| $\mathbf B$ | spatial prior covariance built from the inventory | - | [Prerequisites](../design/00_prerequisites.md) |
| $\sigma_{\log Q}\approx0.5$ | inventory uncertainty | - | [Prerequisites](../design/00_prerequisites.md) |
| $S_b$, $\mathbf q_a$ | inventory-derived prior mean | - | [Tier III](../design/03_tier3_eulerian.md), [Tier II](../design/02_tier2_lagrangian.md) |
:::

## Emission inventory

The inventory pages fix their own symbols for the whole section, with rates in kg h⁻¹ and times in days.

:::{table} Emission inventory: the symbols of the inventory pages, fixed for the whole section.
:label: tbl-notation-inventory

| Symbol | Meaning | Units |
|---|---|---|
| $S$ | spatial domain | - |
| $x \in S$ | location (easting, northing) | km, km |
| $T = [0, T_{\mathrm{end}}]$ | time window | d |
| $t \in T$ | a time | d |
| $X$ | generic index set: $S$, $T$, or $S \times T$ | - |
| $A \subset X$ | a region or interval | - |
| $\lvert A \rvert$ | area (or length) of $A$ | $\mathrm{km^2}$ (d) |
| $\delta_x$ | point mass at $x$ | - |
| $N(A)$ | number of point sources in $A$ | - |
| $\lambda(A)$, $\lambda(x)$ | expected count in $A$; source density | -, $\mathrm{km^{-2}}$ |
| $\lambda(t)$ | event rate | $\mathrm{d^{-1}}$ |
| $s_k$ | on-state rate of point source $k$ | $\mathrm{kg\,h^{-1}}$ |
| $s_0$ | reference rate, $1\ \mathrm{kg\,h^{-1}}$ | $\mathrm{kg\,h^{-1}}$ |
| $\pi_k$ | persistence (fraction of time on) of source $k$ | - |
| $G(\mathrm{d}\pi \mid u)$ | distribution of persistence given on-state rate | - |
| $p_{\mathrm{camp}}$ | probability a source is detected at least once over a campaign | - |
| $\lambda_{\ge}$, $R$ | count intensity of sources above the floor $u_{\mathrm{floor}}$; $R = \rho([u_{\mathrm{floor}},\infty))$ | $\mathrm{km^{-2}}$, - |
| $e(x)$ | diffuse flux density | $\mathrm{kg\,h^{-1}\,km^{-2}}$ |
| $\mu(A)$ | total emission rate from $A$ (point + diffuse) | $\mathrm{kg\,h^{-1}}$ |
| $\mu_{\mathrm{pt}}$, $\mu_{\mathrm{df}}$ | atomic and diffuse parts of $\mu$ | $\mathrm{kg\,h^{-1}}$ |
| $H(A)$ | prior expected emission from $A$ | $\mathrm{kg\,h^{-1}}$ |
| $u_k$ | dimensionless rate $s_k/s_0$ | - |
| $\rho(\mathrm{d}u)$ | Lévy measure over dimensionless rates $u$ | - |
| $G(A)$ | fraction of basin total from $A$ | - |
| $f(x)$ | $\mathrm{XCH_4}$ enhancement | ppb |
| $m(x)$, $k(x,x')$ | mean, covariance kernel of a field | ppb, $\mathrm{ppb^2}$ |
| $g(x)$ | log source density, $\log\lambda(x)$ | - |
| $\Delta x$, $\Delta t$ | pixel size, revisit interval | km, d |
| $C_\Delta$ | coarsening operator: $\mu \mapsto$ pixel totals | - |
| $\mathcal{T}$ | atmospheric transport operator: $\mu \mapsto f$ | ppb per $\mathrm{kg\,h^{-1}}$ |
| $\mathbf{U}$, $U$ | 10 m wind vector (east, north); its speed $U = \lVert\mathbf{U}\rVert$ | $\mathrm{m\,s^{-1}}$ |
| $p(x,s)$ | detection probability | - |
| $s_{\min}$ | detection limit | $\mathrm{kg\,h^{-1}}$ |
| $c_j$, $Z_j$ | candidate site $j$; its emitting indicator | km km, - |
| $m_k$, $M(t)$ | event mass; cumulative mass by $t$ | kg |
| $\lambda^*(t)$, $\mathcal{H}_t$ | conditional intensity; history before $t$ | $\mathrm{d^{-1}}$, - |
:::
