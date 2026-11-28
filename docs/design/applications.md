# Applications of the four questions beyond trace gases

28 Nov 2026 · J. Emmanuel Johnson

## Context

plumax was designed for methane, and its README now says "plumes of anything". This page asks how far that claim reaches. It first states what the four questions (segment, quantify, attribute, persist) mean mathematically and in practice, then walks through candidate applications and labels each one by how well it fits. It informs the [roadmap](roadmap.md) and it does no implementation: its output is the list of follow-up reports in section 6.

The code was read at plumax `b14cd8b` (27 Nov 2026). Sensors, products and outside models are named from general knowledge, and none was checked against a product guide for this page.

## 1. What the library hopes to achieve

plumax answers four questions about something that a source releases into a moving fluid and that a sensor later sees as an image. This section fixes the picture and then takes the questions one at a time.

### 1.A The picture

A tracer has concentration c(𝐱, t) in kg m⁻³ at position 𝐱 and time t. A fluid with velocity 𝐮 carries it, turbulence with diffusivity K spreads it, a loss rate ℓ removes it, and a source s adds it.

```math
\frac{\partial c}{\partial t} + \nabla \cdot (\mathbf{u}\, c) = \nabla \cdot (K\, \nabla c) - \ell\, c + s(\mathbf{x}, t; \boldsymbol{\theta})
```

Here 𝐮 is in m s⁻¹, K in m² s⁻¹, ℓ in s⁻¹ and s in kg m⁻³ s⁻¹. The source parameters 𝛉 hold what we want to know: the location 𝐱ₛ, the rate Q in kg s⁻¹ and the timing. A point source is s = Q δ(𝐱 − 𝐱ₛ). The tracer is passive, which means it does not change 𝐮.

A sensor does not see c. It sees an image 𝐲 with N pixels, which is an observation operator ℋ applied to c, on top of a background 𝐛 and noise 𝛆.

```math
\mathbf{y} = \mathbf{b} + \mathcal{H}(c) + \boldsymbol{\varepsilon}
```

For methane, ℋ integrates c along the line of sight to a column in kg m⁻², and the retrieval that precedes it belongs to radtrax. The enhancement is Δ = 𝐲 − 𝐛, and everything below works on Δ.

### 1.B Segment: where is it?

**Mathematically,** segmentation is a test at every pixel of whether the enhancement is larger than the noise allows. With noise standard deviation σ and a threshold κ, the mask is

```math
m_i = \mathbb{1}\!\left[\frac{y_i - b_i}{\sigma_i} > \kappa\right], \qquad i = 1, \dots, N
```

and the plume is the connected set Ω of pixels with mᵢ = 1 that reaches back to a plausible source.

**Practically,** this is background estimation, thresholding, connected components, contours and vetting. The README names `PlumeMask`, `PlumeContours` and `PlumeFootprint` in `plumax.geo`, and `plumax.vetting` holds the candidate tests.

**In application,** the mask is the detection itself. It says whether an event happened, where it is and how large it looks, and every later question starts from it.

### 1.C Quantify: how much?

**Mathematically,** quantification estimates the rate Q with its uncertainty. The cheap route is a mass balance. The integrated mass enhancement sums the column enhancement over the mask, with pixel area Aᵢ in m², and the rate follows from an effective wind speed U and a plume length L (Varon et al., 2018).

```math
\mathrm{IME} = \sum_{i \in \Omega} \Delta_i\, A_i \quad [\mathrm{kg}], \qquad Q = \frac{U\, \mathrm{IME}}{L} \quad [\mathrm{kg\, s^{-1}}]
```

The full route inverts the transport model. With 𝒢 the map from source parameters to the predicted image (the transport equation followed by ℋ), the answer is a posterior.

```math
p(\boldsymbol{\theta} \mid \mathbf{y}) \propto p(\mathbf{y} \mid \mathcal{G}(\boldsymbol{\theta}))\; p(\boldsymbol{\theta})
```

**Practically,** the README names `IMEEstimate`, `CrossSectionalFlux` and `infer_emission_rate`. Every transport tier exposes the same `forward(params, met) → observations`, so one sampler serves all of them.

**In application,** the rate is what an operator, a regulator or an inventory needs: a number in kg s⁻¹ with an interval, not a picture.

### 1.D Attribute: from where?

**Mathematically,** attribution splits the observed enhancement among S candidate sources. Transport is linear in the source rate for a passive tracer, so with 𝐪 the vector of S rates and 𝐇 the N × S source-receptor matrix,

```math
\mathbf{y} = \mathbf{b} + \mathbf{H}\, \mathbf{q} + \boldsymbol{\varepsilon}, \qquad H_{ij} = \frac{\partial y_i}{\partial q_j}
```

where column j of 𝐇 is the image that a unit emission from source j would produce. A Gaussian prior on 𝐪 gives a Gaussian posterior in closed form. When the location is unknown, 𝐱ₛ joins 𝛉 and the problem is no longer linear.

**Practically,** the README names `compute_footprint` (the backward run that builds 𝐇), `linear_gaussian_inversion`, `solve_4dvar` and `fuse_observations`.

**In application,** attribution turns a rate into responsibility: which facility, which ship, which vent.

### 1.E Persist: how often?

**Mathematically,** events are points (tₖ, Qₖ) of a marked point process with intensity λ(t) in events per second and a rate distribution f(Q). The sensor only reports an event with a detection probability P_d(Q), so the catalogue follows a thinned process.

```math
\lambda_{\mathrm{obs}}(t, Q) = \lambda(t)\; f(Q)\; P_d(Q)
```

Undoing the thinning gives the true event rate, and with τ the event duration in seconds the total mass released over a window T is

```math
E = \mathbb{E}[Q\, \tau] \int_T \lambda(t)\, \mathrm{d}t \quad [\mathrm{kg}]
```

**Practically,** the README names `EmissionCatalog`, `fit_poisson_rate`, `population.pod`, `population.persistency` and `total_emission`, built on the point processes of pyrox-xtreme.

**In application,** this is the step from one scene to a population: how often a source emits, when to look again, and what the unseen events add up to.

### 1.F The fit test

An application fits when four conditions hold.

1. **A source.** Something is released, and a few parameters (place, rate, timing) describe the release.
2. **A known carrier.** A flow we can obtain carries the tracer from the source to the sensor, and the tracer does not change that flow.
3. **An image.** A sensor maps an enhancement over a background, so there is something to segment.
4. **Recurrence.** Events repeat, so a catalogue and a rate mean something.

The labels below follow from the test.

- **Perfect fit.** All four conditions hold and each question keeps its meaning. The model may need an extra term.
- **Medium fit.** One condition is weak, so one question changes meaning or loses its data.
- **Poor fit.** There is no source. Only segment and persist apply.

## 2. Applications

The applications run from the ones plumax nearly covers today to the ones it should leave alone. The table gives the label and the reason, and the entries below give the motivation, objective, data and model for each.

| # | Application | Fit | Why |
| --- | --- | --- | --- |
| 2.1 | Combustion and industrial gases | perfect | add a loss term |
| 2.2 | Volcanic SO₂ and ash | perfect | add injection height |
| 2.3 | Wildfire smoke | perfect | moving line source |
| 2.4 | Oil spills | perfect | ocean carrier |
| 2.5 | River sediment plumes | perfect | add settling |
| 2.6 | Thermal discharges | perfect | tracer is heat |
| 2.7 | Mineral dust | medium | source is an area |
| 2.8 | Ammonia | medium | coarse images |
| 2.9 | Radionuclide releases | medium | point data |
| 2.10 | Marine plastics | medium | weak imaging |
| 2.11 | Algal blooms | medium | grows in place |
| 2.12 | Groundwater contaminants | medium | no image |
| 2.13 | Eddies, heatwaves, floods | poor | no source |

### 2.1 Combustion and industrial gases (NO₂, SO₂, CO₂, CO) · perfect fit

**Motivation.** Power plants, smelters, cities and ships emit these gases, and reported inventories are often late or incomplete.

**Objective.** Estimate the emission rate of each facility or city and track it over time.

**Data.** Column maps from imaging spectrometers (TROPOMI for NO₂, SO₂ and CO; OCO-2, OCO-3 and the planned CO2M for CO₂), with reanalysis winds.

**Model.** The existing Gaussian, Lagrangian and Eulerian tiers, with a first-order loss ℓ = 1/τ for the chemical lifetime τ. NO₂ lives for hours, so ℓ cannot be dropped, and the classic method fits Q and τ together from the downwind decay (Beirle et al., 2011).

**The four questions.** All unchanged. Quantify gains one parameter, the lifetime.

### 2.2 Volcanic SO₂ and ash · perfect fit

**Motivation.** Ash closes airspace, and the SO₂ flux is a main sign of a volcano's state.

**Objective.** Estimate the eruption source term: mass flux and injection height as functions of time.

**Data.** SO₂ columns from ultraviolet and infrared sounders (TROPOMI, IASI), ash from geostationary infrared imagers, plume heights from lidar or stereo.

**Model.** Lagrangian or Eulerian transport with the release spread over a vertical profile, a loss term for SO₂ conversion to sulphate, and a settling velocity for ash. Flux methods from satellite SO₂ are surveyed in Theys et al. (2013).

**The four questions.** All unchanged. Attribute is easy (the vent is known) and the injection height takes its place as the unknown.

### 2.3 Wildfire smoke · perfect fit

**Motivation.** Smoke is a leading air-quality hazard, and fire emission inventories disagree widely.

**Objective.** Estimate aerosol and CO emission rates per fire and compare them with estimates from fire radiative power.

**Data.** Aerosol optical depth and CO columns (VIIRS, TROPOMI), active-fire detections and fire radiative power for the source geometry (Kaiser et al., 2012).

**Model.** Lagrangian or Eulerian transport with a source that is a moving line along the fire front, an injection height from plume rise, and deposition. plumax already has a Briggs plume rise for stacks in `gauss_plume`.

**The four questions.** All unchanged. The source has an extent and moves, which the source parameterisation must allow.

### 2.4 Oil spills · perfect fit

**Motivation.** Spills from ships, platforms and pipelines damage coasts, and illegal bilge dumping goes unpunished when no one can name the ship.

**Objective.** Map the slick, estimate the spilled volume, name the source and forecast the landfall.

**Data.** Synthetic-aperture radar (Sentinel-1), where oil damps the capillary waves and shows dark, optical sun glint, ship positions from AIS, and currents and winds from ocean and weather models.

**Model.** Two-dimensional Lagrangian drift at the sea surface, with velocity 𝐮 = 𝐮_current + α 𝐮_wind, where α is the windage factor, about 3 % by the common rule of thumb (unverified). Weathering (evaporation, emulsification) is a loss term. OpenDrift's OpenOil is the reference model (Dagestad et al., 2018).

**The four questions.** All unchanged in meaning. Attribute is the backward run from the slick to the ship tracks. Persist separates chronic sources (natural seeps, leaking wells) from single events.

**New.** This is the first case where the carrier is the ocean, so the `met` input must accept currents from sesmax.

### 2.5 River sediment plumes · perfect fit

**Motivation.** Rivers deliver sediment, nutrients and pollutants to the coast, and the plume sets water clarity for reefs, seagrass and fisheries.

**Objective.** Estimate the sediment flux at the river mouth and where it settles.

**Data.** Ocean-colour turbidity and suspended-matter products (Sentinel-2, Sentinel-3, Landsat), river discharge gauges, coastal currents.

**Model.** Surface or shallow three-dimensional transport with a settling velocity that removes sediment, and a buoyant fresh layer near the mouth.

**The four questions.** All unchanged. The source is known, so attribute matters only where several rivers meet.

### 2.6 Thermal discharges · perfect fit

**Motivation.** Power plants and industry release cooling water, and permits limit the temperature rise in the receiving water.

**Objective.** Estimate the heat flux of an outfall and the area held above a temperature threshold.

**Data.** Thermal-infrared surface temperature at tens of metres (Landsat, ECOSTRESS), with air temperature and wind for the surface heat loss.

**Model.** Surface advection and diffusion of heat, where the tracer is temperature excess, Q is in watts and the loss term is the exchange with the atmosphere.

**The four questions.** All unchanged, with Q in W and not in kg s⁻¹.

### 2.7 Mineral dust · medium fit

**Motivation.** Dust affects health, solar power and aviation, and its sources shift with drought and land use.

**Objective.** Map the emitting areas and their flux.

**Data.** Aerosol optical depth and dust indices from geostationary and polar imagers.

**Model.** Atmospheric transport with settling and deposition.

**Why medium.** The source is a wide area whose strength depends on the wind itself, so "quantify" becomes a flux map and not a rate at a point, and segmentation has no sharp plume to find.

### 2.8 Ammonia · medium fit

**Motivation.** Ammonia from livestock and fertiliser forms fine particles and is poorly inventoried.

**Objective.** Estimate emissions of feedlots and fertiliser plants.

**Data.** Infrared sounder columns (IASI, CrIS) at pixels of roughly ten kilometres or more.

**Model.** Atmospheric transport with a lifetime of hours and strong deposition.

**Why medium.** Single overpasses rarely show a plume at that resolution. The usual method averages many overpasses in a wind-rotated frame, so segment works on a temporal mean and persist loses its events.

### 2.9 Radionuclide and chemical accident releases · medium fit

**Motivation.** After an accident the source term is unknown, and every dose estimate depends on it.

**Objective.** Reconstruct the release rate over time and forecast deposition.

**Data.** Station networks of air concentration and deposition, with few or no images.

**Model.** Lagrangian transport with radioactive decay and wet and dry deposition. Stohl et al. (2012) reconstructed the Fukushima source term this way, with a linear inversion of the kind in 1.D.

**Why medium.** Quantify and attribute fit exactly, but the data are points, so there is nothing to segment, and events are too rare for persist.

### 2.10 Marine plastics and debris · medium fit

**Motivation.** Rivers and coastal cities feed plastic to the ocean, and clean-up and policy need to know which ones.

**Objective.** Attribute floating debris to its river or coastal source.

**Data.** Sparse detections of floating patches in optical imagery, beach surveys, river input models (Lebreton et al., 2017).

**Model.** Lagrangian surface drift with windage and beaching, the same machinery as 2.4.

**Why medium.** Transport and attribution fit, but imaging of debris is weak and patchy, so segment and quantify rest on little data.

### 2.11 Algal blooms and sargassum · medium fit

**Motivation.** Harmful blooms close fisheries, and sargassum landings cost Caribbean coasts every year.

**Objective.** Map the bloom, forecast its path and link it to a nutrient source.

**Data.** Ocean-colour chlorophyll and floating-algae indices.

**Model.** Surface transport with a growth term, which is a loss rate ℓ of negative sign that depends on light and nutrients.

**Why medium.** The flow carries the bloom, but most of the mass grows in place. "Quantify" stops meaning an emission rate, and "attribute" points at a nutrient input and not at the origin of the biomass.

### 2.12 Groundwater contaminants · medium fit

**Motivation.** Leaks from tanks, landfills and industry contaminate aquifers, and liability depends on identifying the leak.

**Objective.** Locate the source and reconstruct its release history.

**Data.** Concentrations at monitoring wells, over years.

**Model.** Advection and dispersion in a porous medium under Darcy flow, with sorption and decay. Source identification here is a long-studied inverse problem (Atmadja and Bagtzoglou, 2001).

**Why medium.** The equation of 1.A and the inversions of 1.C and 1.D apply, but there is no image, the carrier is not a geophysical fluid the stack models, and the time scale is years.

### 2.13 Eddies, heatwaves and flood extent · poor fit

**Motivation.** These are the anomalies that first suggested widening plumax, and each is an object in an image with a lifetime.

**Why poor.** None has a source. An eddy is a structure of the flow, a heatwave is an exceedance of a climatology, and a flood is shallow-water dynamics. There is no rate to invert and nothing to attribute, so the transport tiers and the inversions would go unused. Segment and persist do apply, and that shared part is better placed outside plumax (recommendation in section 5).

## 3. Findings

1. **The README already claims any passive tracer.** It lists methane, CO₂, NO₂, SO₂, smoke and heat. Checked in `README.md` @ `b14cd8b`.
2. **The transport tiers have no loss, deposition, settling or windage term.** A search of `src/` for `decay`, `lifetime`, `half_life`, `deposition`, `settling` and `windage` finds only the turbulence memory of the puff and particle models. Checked with `git grep` @ `b14cd8b`.
3. **No tier has a moving or time-tracked source.** The same search for `moving source` and `source_track` finds nothing. Checked with `git grep` @ `b14cd8b`. Extended (area or line) sources were not searched for (unverified).
4. **The carrier input is atmospheric only.** `plumax.met` holds `era5`, `wrf`, `pbl` and `surface_layer`, and nothing for ocean currents. Checked in the tree @ `b14cd8b`.
5. **The observation side is methane only.** The package docstring describes `plumax.obs` as TROPOMI, EMIT and GHGSat methane products. Checked in `src/plumax/__init__.py` @ `b14cd8b`.
6. **Four additions cover every perfect and medium fit above.** They are a loss or growth rate ℓ, a settling velocity with deposition, sources that move or have extent, and a carrier that is not atmospheric wind. This follows from section 2 and is an analysis, not a test.
7. **Segment and persist look independent of the tracer.** They act on an enhancement image and on a catalogue. Whether `plumax.geo` and `plumax.population` assume methane units anywhere was not checked (unverified).

## 4. What it builds on

The physics that the applications add should come from the stack, and plumax should keep the source, the inversion and the catalogue.

| Need | Owner | State |
| --- | --- | --- |
| Particles, drift, windage | tracerax | base dependency; windage unverified |
| Eulerian grid and operators | gridax | base dependency |
| Ocean currents as the carrier | sesmax | unverified |
| Loss, settling, deposition terms | tracerax and plumax tiers | missing (finding 2) |
| Retrieval of the image from radiance | radtrax | methane today |
| Point processes | pyrox-xtreme | `population` extra |
| Object metrics for 2.13 | diagnostix | unverified |

The dependency states come from `pyproject.toml` @ `b14cd8b`. The rows marked unverified need a survey of each package's `origin/main` before any design is written.

## 5. Recommendations

- **Adopt: the loss term ℓ first.** It is one term in the transport equation, it serves 2.1, 2.2, 2.4, 2.6 and 2.9, and it makes NO₂ a second gas with public data.
- **Adopt: one atmospheric demonstration other than methane.** Volcanic SO₂ is the cleanest, since the source location is known and the signal is strong.
- **Adopt: oil spills as the first ocean case.** All four questions carry over, and it forces the carrier contract with sesmax that 2.5, 2.6, 2.10 and 2.11 also need.
- **Skip: eddies, heatwaves and floods inside plumax.** They have no source. Their shared part (mask, objects, tracks, catalogue) belongs in diagnostix and the extremes packages.
- **Skip for now: groundwater and station-only inversions.** They reuse the inversion maths but none of the imaging or the carriers.

Recommended first: the loss term with an NO₂ or SO₂ example. Alternative: go straight to oil spills, which costs more but proves the ocean coupling.

## 6. Follow-up reports

- [ ] `[Design] which tracers and carriers plumax covers` — settles the scope statement and the fit test of 1.F as the boundary.
- [ ] `feat(transport): first-order loss rate on every tier` — ℓ in the puff, particle and Eulerian models, with a decay test against the analytic solution.
- [ ] `feat(transport): settling velocity and deposition` — for ash, dust and sediment.
- [ ] `[Design] sources that move or have extent` — line, area and track sources behind the same `forward(params, met)`.
- [ ] `[Design] a carrier contract that accepts ocean currents` — what `met` must provide, and what sesmax supplies.
- [ ] `research(oil): map OpenDrift's OpenOil onto tracerax and plumax` — drift, windage and weathering, with a gap table.
- [ ] `[Design] where object tracking and event catalogues for non-source anomalies live` — the shared layer for 2.13.

## References

These are listed from memory, without DOIs, and were not checked against the publishers' pages.

- Atmadja, J. and Bagtzoglou, A. C. (2001). State of the art report on mathematical methods for groundwater pollution source identification. *Environmental Forensics*, 2.
- Beirle, S., Boersma, K. F., Platt, U., Lawrence, M. G. and Wagner, T. (2011). Megacity emissions and lifetimes of nitrogen oxides probed from space. *Science*, 333.
- Dagestad, K.-F., Röhrs, J., Breivik, Ø. and Ådlandsvik, B. (2018). OpenDrift v1.0: a generic framework for trajectory modelling. *Geoscientific Model Development*, 11.
- Kaiser, J. W. et al. (2012). Biomass burning emissions estimated with a global fire assimilation system based on observed fire radiative power. *Biogeosciences*, 9.
- Lebreton, L. C. M. et al. (2017). River plastic emissions to the world's oceans. *Nature Communications*, 8.
- Stohl, A. et al. (2012). Xenon-133 and caesium-137 releases into the atmosphere from the Fukushima Dai-ichi nuclear power plant: determination of the source term, atmospheric dispersion, and deposition. *Atmospheric Chemistry and Physics*, 12.
- Theys, N. et al. (2013). Volcanic SO₂ fluxes derived from satellite data: a survey using OMI, GOME-2, IASI and MODIS. *Atmospheric Chemistry and Physics*, 13.
- Varon, D. J. et al. (2018). Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes. *Atmospheric Measurement Techniques*, 11.
- plumax, `README.md`, `pyproject.toml` and `src/plumax/` @ `b14cd8b`.

## Relationships

- Parent: none
- Related: the [roadmap](roadmap.md), and the sesmax, plumax and radtrax contracts in [geostack#328](https://github.com/jejjohnson/geostack/issues/328)
