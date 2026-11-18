# Tier I: the image-fitted plume

**Forward model:** the closed-form column of a steady Gaussian plume, with a width law and a meandering centreline fitted to the image.

This page fits a plume model directly to one column-enhancement image and reads off the emission rate. The data constrain the mass per metre of downwind travel, $q = Q / U$, and the plume's shape; the wind speed $U$ enters once, after the fit. The design was settled in [#266](https://github.com/jejjohnson/plumax/issues/266) and the work is tracked in [#265](https://github.com/jejjohnson/plumax/issues/265).

:::{important} Decision (17 Nov 2026)
The image fit is a transport backend whose parameters are named control blocks of the existing `coupled.solve_coupled`. The solver, the quality masks, the Laplace posterior and the Tier V payload do not change. A standalone fit with its own cost, optimiser and result type was considered and not chosen.
:::

---

(image-fit-why)=
## Why fit the image

Every image-based rate in plumax needs a wind before it starts. `IMEEstimate` and `CrossSectionalFlux` take a wind speed and a mask and report a fixed 50 % uncertainty. `solve_coupled` fits a 3-D Briggs plume with a fixed stability class, summed over height levels by the trapezoid rule.

An imaging spectrometer sees more than that. At 60 m pixels it resolves how fast the plume widens and where it bends, and the column it sees does not depend on the vertical spread at all. JPL's SPIIRE ([emit-sds/SPIIRE](https://github.com/emit-sds/SPIIRE) @ `ae9a54d`, Apache-2.0) fits exactly this model to EMIT and AVIRIS-3 scenes. This page brings the same model into plumax on top of the machinery Tier IV already has, so the fit gets the coupled solver's whitened control, quality masks, Laplace posterior and per-event payload without new solver code.

---

(image-fit-symbols)=
## Symbols

:::{table} Symbols of the image fit. The pages of Tier III and Tier V use $q$ for other quantities; on this page it is the line density only.
:label: tbl-image-fit-symbols

| Symbol | Meaning | Shape, units |
| --- | --- | --- |
| $\vect{s}_n$, $\vect{s}_0$ | centre of pixel $n$, source position, in a local metre frame | $\mathbb{R}^2$, m |
| $\varphi$ | plume direction, counter-clockwise from east, pointing downwind | rad |
| $d_n$, $c_n$ | downwind and crosswind distance of pixel $n$ from the source | m |
| $Q$, $U$ | emission rate, effective wind speed | kg s⁻¹, m s⁻¹ |
| $q = Q / U$ | line density: the mass per metre of downwind travel | kg m⁻¹ |
| $w(d)$ | crosswind standard deviation of the plume at distance $d$ | m |
| $w_0$, $w_\text{ref}$, $d_\text{ref}$, $b$ | initial width, width at the reference distance, reference distance, growth exponent | m, m, m, – |
| $m(d)$, $\vect{m}$ | centreline offset at distance $d$, and its values at the $K$ knots $d_k = k \Delta_d$ | m, $\mathbb{R}^K$ m |
| $\Delta\Omega_n$ | modelled column-mass enhancement of pixel $n$ | kg m⁻² |
| $y_n$, $\sigma_n$, $\omega_n$ | observed enhancement, its noise standard deviation, its quality mask | kg m⁻², kg m⁻², 0 or 1 |
| $A_n$ | sensitivity of pixel $n$: the score recovered per unit of true enhancement | – |
| $\Omega_\text{bg}$ | background column offset | kg m⁻² |
| $\gvect{\theta}$ | the stacked control | $\mathbb{R}^{D_\theta}$ |
| $\gvect{\mu}_\theta$, $\gmat{\Sigma}_\theta$ | prior mean and covariance of the control | $\mathbb{R}^{D_\theta}$, $D_\theta \times D_\theta$ |
:::

---

(image-fit-model)=
## The model

(image-fit-frame)=
### Plume frame

The plume frame is a rotation about the source, with unit vectors $\vect{e}_\parallel = (\cos\varphi, \sin\varphi)$ along the plume and $\vect{e}_\perp = (-\sin\varphi, \cos\varphi)$ across it:

$$
d_n = (\vect{s}_n - \vect{s}_0)^\top \vect{e}_\parallel
$$ (eq-image-fit-downwind)

$$
c_n = (\vect{s}_n - \vect{s}_0)^\top \vect{e}_\perp
$$ (eq-image-fit-crosswind)

(image-fit-column)=
### Column of a reflected plume

The steady Gaussian plume with ground reflection has concentration

$$
C(d, c, z) = \frac{Q}{2\pi U w \sigma_z} \exp\left(-\frac{c^2}{2 w^2}\right) \left[\exp\left(-\frac{(z - H)^2}{2\sigma_z^2}\right) + \exp\left(-\frac{(z + H)^2}{2\sigma_z^2}\right)\right]
$$ (eq-image-fit-reflected-plume)

with vertical spread $\sigma_z$ and release height $H$. The bracket is a normal density and its mirror image about the ground, so it integrates to $\sqrt{2\pi} \sigma_z$ over $z \ge 0$ for every $H$ and $\sigma_z$. The full column is therefore

$$
\Delta\Omega(d, c) = \int_0^\infty C \mathrm{d}z = \frac{q}{\sqrt{2\pi} w(d)} \exp\left(-\frac{(c - m(d))^2}{2 w(d)^2}\right) \text{ for } d > 0
$$ (eq-image-fit-column)

and zero for $d \le 0$, where the centreline offset $m(d)$ shifts the ridge sideways. The column needs no stability class, no release height and no vertical grid.

(image-fit-width)=
### Width law

The width grows as a power law above an initial width:

$$
w(d)^2 = w_0^2 + w_\text{ref}^2 \left(\frac{d}{d_\text{ref}}\right)^{2b}
$$ (eq-image-fit-width)

With $w_0 = 0$ this is SPIIRE's $\sigma_y = a (d / 1000)^b$ with $a = w_\text{ref}$. The initial width removes the singularity at the source, and $w_0 = g / \sqrt{12}$ stands in for the average over a pixel of side $g$, whose variance is $g^2 / 12$. Point evaluation is still a sample, not an average: a Gaussian of width $w$ sampled every $g$ sums to $1 / g$ only up to an aliasing error of about $2 \exp(-2\pi^2 w^2 / g^2)$, which is 0.4 at $w = g / \sqrt{12}$ and below $10^{-8}$ once $w \ge g$. Near the source, where the plume is narrower than a pixel, the exact crosswind pixel average is the option to use.

(image-fit-centreline)=
### Centreline

The centreline is linear between knots, starting at zero at the source. With $m_0 = 0$ at $d_0 = 0$ and $k = \lfloor d / \Delta_d \rfloor$:

$$
m(d) = m_k + (m_{k+1} - m_k) \frac{d - d_k}{\Delta_d} \text{ for } 0 \le d < d_K
$$ (eq-image-fit-centreline)

and $m(d) = m_K$ beyond the last knot. Shifting a Gaussian does not change its integral, so for every $d > 0$ the crosswind integral of the column is the line density, whatever the width and the bends:

$$
\int_{-\infty}^{\infty} \Delta\Omega(d, c) \mathrm{d}c = q
$$ (eq-image-fit-mass-flux)

(image-fit-observation)=
### Observation

The matched filter recovers a share $A_n$ of the true enhancement, so the model is multiplied by it before it meets the data:

$$
y_n = A_n \Delta\Omega_n(\gvect{\theta}) + \Omega_\text{bg} + \varepsilon_n
$$ (eq-image-fit-observation)

with $\varepsilon_n \sim \mathcal{N}(0, \sigma_n^2)$. The sensitivity and the noise come from radtrax's matched filter ([radtrax#311](https://github.com/jejjohnson/radtrax/issues/311), [radtrax#312](https://github.com/jejjohnson/radtrax/issues/312)) or from the product's own rasters. They enter `coupled.Instrument` as its averaging kernel and retrieval variance, which are the slots that exist today.

(image-fit-control)=
### Control

The control stacks the blocks the prior names:

$$
\gvect{\theta} = (\log q, \log w_\text{ref}, b, \varphi, \vect{s}_0, \Omega_\text{bg}, \vect{m}) \in \mathbb{R}^{7 + K}
$$ (eq-image-fit-control)

---

(image-fit-wind)=
## Why the wind is not a control

$Q$ and $U$ enter the column only as $q = Q / U$. Write the prediction in terms of $(\log Q, \log U)$: the two columns of the Jacobian are $+q \vect{r}$ and $-q \vect{r}$, with $\vect{r}$ the response per unit line density. The Fisher information of the pair is

$$
\mat{F}_{QU} = q^2 \vect{r}^\top \gmat{\Sigma}_\varepsilon^{-1} \vect{r} \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}
$$ (eq-image-fit-wind-fisher)

which has rank one: only $\log Q - \log U = \log q$ is constrained, and with both blocks present their split is the prior's alone. The column plume therefore declares a `log_line_density` block and no `wind` block. The wind is applied after the solve, independent of the fit, as a lognormal speed of mean $U$ and standard deviation $s_U$:

$$
\sigma_U^2 = \log(1 + s_U^2 / U^2)
$$ (eq-image-fit-wind-variance)

$$
\mu_U = \log U - \sigma_U^2 / 2
$$ (eq-image-fit-wind-mean)

$$
\log Q \sim \mathcal{N}(\mu_q + \mu_U, \sigma_q^2 + \sigma_U^2)
$$ (eq-image-fit-rate)

where $\mu_q$ and $\sigma_q^2$ are the Laplace mean and variance of $\log q$. For $\sigma_q = 0.15$ and a wind of 3.5 ± 1.0 m s⁻¹, $\sigma_U = 0.28$ and the standard deviation of $\log Q$ is 0.32, so the wind carries about three quarters of the variance. A catalogue event that stores $q$ needs no wind rescaling at all ([#111](https://github.com/jejjohnson/plumax/issues/111)).

---

(image-fit-inference)=
## Inference

(image-fit-cost)=
### Cost and posterior

The cost is the one `solve_coupled` minimises today:

$$
\mathcal{J}(\gvect{\theta}) = \frac{1}{2} \Vert \gvect{\theta} - \gvect{\mu}_\theta \Vert^2_{\gmat{\Sigma}_\theta^{-1}} + \frac{1}{2} \sum_n \omega_n \frac{(y_n - A_n \Delta\Omega_n(\gvect{\theta}) - \Omega_\text{bg})^2}{\sigma_n^2}
$$ (eq-image-fit-cost)

It is minimised by L-BFGS in the whitened control $\gvect{\theta} = \gvect{\mu}_\theta + \mat{L} \gvect{\chi}$, with $\gmat{\Sigma}_\theta = \mat{L} \mat{L}^\top$, so every block has unit prior scale whatever its units. With $\mat{J}_h$ the Jacobian of the predictions and $\gmat{\Sigma}_\varepsilon = \op{diag}(\sigma_n^2 / \omega_n)$, the Fisher information and the Laplace posterior are

$$
\mat{F} = \mat{J}_h^\top \gmat{\Sigma}_\varepsilon^{-1} \mat{J}_h
$$ (eq-image-fit-fisher)

$$
\gmat{\Sigma}_{\theta \mid y} = (\gmat{\Sigma}_\theta^{-1} + \mat{F})^{-1}
$$ (eq-image-fit-laplace)

(image-fit-diagnostics)=
### What the data determined

The prior-free bound on one parameter $a$, with all others also unknown, is the inverse of a Schur complement of $\mat{F}$, which is what SPIIRE reports:

$$
\mathbb{V}[a] \ge [\mat{F}^{-1}]_{aa}
$$ (eq-image-fit-crb)

The share of each parameter set by the data compares the posterior with the prior:

$$
\text{data fraction}_a = 1 - \frac{[\gmat{\Sigma}_{\theta \mid y}]_{aa}}{[\gmat{\Sigma}_\theta]_{aa}}
$$ (eq-image-fit-data-fraction)

A knot beyond the visible plume has no data: its row of $\mat{F}$ is zero and its posterior is its conditional prior. SPIIRE, which has no prior on the knots, inverts only a leading block of its Hessian instead; the prior below makes the full posterior well defined.

(image-fit-stages)=
### Two stages and a multistart

A small rotation $\delta\varphi$ moves the ridge by $d \delta\varphi$ at distance $d$, which is the centreline $m(d) = d \delta\varphi$, so the direction and a linear trend in the knots are nearly the same parameter. The fit runs in two stages, as SPIIRE's does:

1. **Straight.** Fit $(\log q, \log w_\text{ref}, b, \varphi, \vect{s}_0, \Omega_\text{bg})$ from a 3 × 3 grid of source positions one pixel apart, and keep the lowest cost. The starting direction points from the catalogue's pseudo-origin to the plume's centroid.
2. **Bent.** Hold the direction, start the knots from the smoothed transect centreline, and refit with the centreline block released.

The cost has several minima in the source position, which is why stage 1 starts from a grid rather than one point.

On a synthetic 3 km plume whose centreline is a sine of 200 m amplitude and 3 km wavelength, the straight stage turns the direction by 0.27 rad towards the brighter first bend and underestimates $q$ by 9 %. Starting its knots from zero, the bent stage recovers them to 4 m without noise. At a peak signal-to-noise ratio of 10 the knot error is 32 m, which is what the knots' own posterior standard deviations predict (31 m), and $q$ is within 1 % of the truth. The error grows downwind as the plume widens and fades, so the posterior standard deviation of each knot, not a fixed tolerance, says how well the bend is known.

(image-fit-prior)=
### Centreline prior

The knots have a Matérn-3/2 Gaussian-process prior pinned to zero at the source. With $r = \vert d - d' \vert$, length scale $\ell$ and amplitude $\sigma_m$:

$$
k(d, d') = \sigma_m^2 \left(1 + \frac{\sqrt{3} r}{\ell}\right) \exp\left(-\frac{\sqrt{3} r}{\ell}\right)
$$ (eq-image-fit-matern)

$$
k_0(d, d') = k(d, d') - \frac{k(d, 0) k(0, d')}{k(0, 0)}
$$ (eq-image-fit-pinned-kernel)

and $\vect{m} \sim \mathcal{N}(\vect{0}, \mat{K}_{DD})$ with $[\mat{K}_{DD}]_{jk} = k_0(d_j, d_k)$. For $\ell$ = 1 km and $\sigma_m$ = 300 m the prior standard deviation is 84 m at 180 m and 148 m at 360 m, rising to 300 m far downwind. SPIIRE's box bounds at the same two distances are 60 m and 120 m, so the defaults allow a similar freedom near the source. Both defaults are starting values to be tuned on the validation scenes. A random walk from the source, $k(d, d') = \kappa^2 \min(d, d')$, stays available as the alternative.

---

(image-fit-numerics)=
## Numerics

- The upwind cut at $d_n = 0$ makes the cost discontinuous in $\vect{s}_0$ and $\varphi$. The default replaces it by a logistic ramp $1 / (1 + \exp(-d_n / \ell_u))$ with $\ell_u = w_0$; the hard cut stays for the oracle tests. The ramp removes mass near the source: the crosswind integral is $q / (1 + \exp(-d / \ell_u))$, so the mass-flux invariant holds to 1e-3 only beyond about $6.9 \ell_u$, 120 m for EMIT.
- $q$ and $w_\text{ref}$ are fitted in logs; $b$ is fitted directly, with its prior keeping it near the observed range of 0.5 to 1.
- $d_\text{ref}$ is set to half the plume length, which decorrelates $\log w_\text{ref}$ from $b$; 1000 m is the default until a plume length is known.
- The exponent is evaluated as $-\frac{1}{2} ((c - m) / w)^2$, never with a separate $w^2$ in the denominator; both it and the amplitude are safe in float32 once $w \ge w_0$.
- The power is computed as $\exp(2 b \log(d / d_\text{ref}))$ on $d$ clipped at 1 mm, so the gradient in $b$ is finite at the source.
- Distances come from a metre-based local frame, never from a Web Mercator warp. SPIIRE takes its pixel size after warping to EPSG:3857, which inflates distances by $1 / \cos(\text{latitude})$, 1.31 at 40° N, and $q$ inherits the factor.
- SPIIRE intends to drop the source pixel from its cost, but the line that does so (`cost.at[pnt].set(0.)`) discards its result. With $w_0 > 0$ there is no singular pixel to drop.
- The knot spacing is at least two pixels, so each knot has data on both sides; SPIIRE uses three.
- The Laplace covariance costs one forward-mode product per control, 40 for a 33-knot plume, and each forward is one exponential per pixel.

---

(image-fit-api)=
## API

(image-fit-modules)=
### Modules

The fixed choices of the model are fields of a callable module, and the fitted variables travel together in their own module, which goes into `__call__`.

:::{table} Where each piece lives, and the issue that adds it.
:label: tbl-image-fit-modules

| File | Symbols | Issue |
| --- | --- | --- |
| `gauss_plume/_src/column.py` | `PlumeGeometry`, `ColumnPlume`, `ColumnPlumeSource`, `plume_frame`, `power_law_width`, `unit_column`, `centreline_offset` | [#267](https://github.com/jejjohnson/plumax/issues/267), [#270](https://github.com/jejjohnson/plumax/issues/270) |
| `gauss_plume/_src/centreline_prior.py` | `CentrelinePrior` | [#270](https://github.com/jejjohnson/plumax/issues/270) |
| `coupled/_src/backends.py` | `AbstractControlledSource` | [#268](https://github.com/jejjohnson/plumax/issues/268) |
| `coupled/_src/inversion.py` | `CoupledPrior.from_blocks`, `CoupledResult.with_wind`, `solve_coupled(initial=...)` | [#268](https://github.com/jejjohnson/plumax/issues/268), [#270](https://github.com/jejjohnson/plumax/issues/270), [#271](https://github.com/jejjohnson/plumax/issues/271) |
| `coupled/_src/multistart.py`, `coupled/_src/diagnostics.py` | `solve_coupled_multistart`, `FitDiagnostics`, `fit_diagnostics` | [#271](https://github.com/jejjohnson/plumax/issues/271) |
| `geo/_src/transects.py`, `geo/_src/starts.py` | `CrossPlumeProfile` and its array functions, `plume_direction`, `source_offset_grid` | [#269](https://github.com/jejjohnson/plumax/issues/269), [#271](https://github.com/jejjohnson/plumax/issues/271) |
| `obs/_src/plume_scene.py` | `PlumeScene`, `load_emit_plume_complex`, `matched_filter_scene` | [#272](https://github.com/jejjohnson/plumax/issues/272) |
:::

(image-fit-call-site)=
### Call site

One EMIT plume, cropped to a 200 × 200 pixel window at 60 m, with 33 knots 180 m apart. The single solve below shows every block in one place; in practice it runs as the two stages above.

```python
import jax.numpy as jnp
import plumax.coupled as cpl
import plumax.gauss_plume as gp
import plumax.obs as obs

# Shapes: N_Ωy = 40,000 pixels; K = 33 centreline knots; D_θ = 40 controls

# proposed (#272): Aₙ and σₙ come with the scene
scene = obs.load_emit_plume_complex(
    "scene_CH4ENH.nc",
    "plume_complexes.json",
    plume_id="CH4_PlumeComplex-1234",
    frame=frame,
    sensitivity_path="scene_CH4SENS.nc",
    uncertainty_path="scene_CH4UNCERT.nc",
)

# proposed (#267, #270)
# w₀ = g / √12 for a pixel of side g = 60 m
initial_width = 60.0 / 12.0**0.5
# w(d)² = w₀² + w_ref² (d / d_ref)^(2b),  m(d) linear between knots d_k = k Δ_d
plume = gp.ColumnPlume(
    initial_width=initial_width, upwind_scale=initial_width, knot_spacing=180.0
)
centreline_0 = jnp.zeros(33)
geometry = gp.PlumeGeometry(
    location=scene.origin,
    direction=scene.direction,
    reference_width=jnp.asarray(150.0),
    width_exponent=jnp.asarray(0.9),
    centreline=centreline_0,
)
source = gp.ColumnPlumeSource(plume=plume, geometry=geometry)

# today (plumax)
forward = cpl.build_coupled_forward(source, [scene.instrument])

# proposed (#270)
# d_k = k Δ_d
knot_index = jnp.arange(1, 34)
knot_distance = 180.0 * knot_index                        # (K,)
centreline_prior = gp.CentrelinePrior(length_scale=1000.0, amplitude=300.0)
# K_DD = [k₀(d_j, d_k)]_jk
knot_covariance = centreline_prior(knot_distance)         # (K,) → (K, K)

# θ = (log q, log w_ref, b, φ, s₀, Ω_bg, m)
prior_mean = {
    "log_line_density": jnp.log(0.1),
    "log_reference_width": jnp.log(150.0),
    "width_exponent": 0.9,
    "direction": scene.direction,
    "location": scene.origin,
    "background": 0.0,
    "centreline": centreline_0,
}
prior_covariance = {
    "log_line_density": 3.0,
    "log_reference_width": 1.0,
    "width_exponent": 0.3,
    "direction": jnp.deg2rad(15.0),
    "location": 60.0,
    "background": 1e-4,
    "centreline": knot_covariance,
}
# proposed (#270)
# Σ_θ = blockdiag(σ², …, K_DD)
prior = cpl.CoupledPrior.from_blocks(mean=prior_mean, covariance=prior_covariance)

# today (plumax)
observations = {"EMIT": scene.enhancement}
problem = cpl.CoupledProblem(forward=forward, prior=prior, observations=observations)

# today (plumax)
# θ* = argmin 𝒥(θ),  Σ_θ|y = (Σ_θ⁻¹ + J_hᵀ Σ_ε⁻¹ J_h)⁻¹
result = cpl.solve_coupled(problem)

# proposed (#268)
# Q = q U,  𝕍[log Q] = 𝕍[log q] + 𝕍[log U]
rate = result.with_wind(speed=3.5, speed_std=1.0)
```

(image-fit-ownership)=
### Ownership

1. **plumax** owns the column plume, its control blocks, the centreline prior, the transect diagnostics and the ingest of a plume scene. It never computes a sensitivity from radiance.
2. **radtrax** owns the matched-filter sensitivity and noise maps that become $A_n$ and $\sigma_n$. It never imports plumax.
3. **vardax and argminax** own the minimiser and the Laplace covariance. They never import plumax or radtrax.
4. **geotoolz** (`xrproducts`, `geoops.qa`) owns reading the product files and decoding the quality masks ([geotoolz#742](https://github.com/jejjohnson/geotoolz/issues/742), [geotoolz#743](https://github.com/jejjohnson/geotoolz/issues/743)). It never imports plumax or radtrax.

---

(image-fit-gates)=
## Gates

:::{table} The gates that close the image fit. The thresholds of G3 and G6 are proposals, to be revisited once the first results exist.
:label: tbl-image-fit-gates

| Gate | Passes when | Issue |
| --- | --- | --- |
| G1 column identity | the closed form, with the Briggs crosswind width, no meander and a hard cut, equals the trapezoid column of `plume_concentration` on 2,000 levels from 0 to 5 km to 1e-4 relative | [#267](https://github.com/jejjohnson/plumax/issues/267) |
| G2 mass flux | with the hard upwind cut, the crosswind sum of $\Delta\Omega$ times the pixel side equals $q$, with and without meander: to 1e-10 relative at every distance with the pixel average, and to 1e-3 relative with point evaluation wherever the width is at least one pixel. The logistic ramp scales the sum by $1 / (1 + e^{-d / \ell_u})$, which is within 1e-3 of 1 only beyond $\ell_u \log 999 \approx 6.9 \ell_u$ | [#267](https://github.com/jejjohnson/plumax/issues/267), [#270](https://github.com/jejjohnson/plumax/issues/270) |
| G3 recovery | on 200 synthetic scenes, the truth of $\log q$ lies inside the 1σ Laplace interval in a fraction consistent with 68.3 % at 3 binomial standard errors | [#268](https://github.com/jejjohnson/plumax/issues/268) |
| G4 wind degeneracy | the Fisher matrix of $(\log Q, \log U)$ has a second singular value below 1e-10 of the first | [#268](https://github.com/jejjohnson/plumax/issues/268) |
| G5 linear limit | with only a linear `line_density` block and one bias free, the solve equals `fuse_observations` to 1e-8 | [#268](https://github.com/jejjohnson/plumax/issues/268) |
| G6 controlled release | the metered rate lies inside the 2σ interval of $Q = q U$ for at least 80 % of the catalogue's airborne controlled-release scenes | [#273](https://github.com/jejjohnson/plumax/issues/273) |
:::

---

(image-fit-risks)=
## Risks

- **Several minima in the source position and the direction.** Fallback: the multistart grid of stage 1, widened if the costs of neighbouring starts disagree.
- **The width law and the meander trade off.** A wide straight plume and a narrow bending one can fit equally well. Fallback: the two stages, with the width blocks tightened around stage 1 when the centreline is released.
- **The Laplace posterior is a poor description of a short plume.** Fallback: the NumPyro path with the same forward, for scenes whose data fraction of $\log q$ is below 0.5.
- **Opening the block table changes `CoupledPrior`'s validation, and `PlumeSource` becomes an `eqx.Module`.** Fallback: `PlumeSource` declares today's `location` and `wind` blocks and keeps its fields, so every existing call site is unchanged.

---

(image-fit-references)=
## References

- SPIIRE: [emit-sds/SPIIRE](https://github.com/emit-sds/SPIIRE) @ `ae9a54d` (Apache-2.0). The forward model is in `plume_models.py`, the cost in `objective_functions_for_2d_inversion.py`, the staged fit and the Hessian bound in `emission_estimate_2d.py`, and the transects in `compute_cross_plume_slices.py`.
- Varon et al. (2018), *Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes*, AMT, doi:10.5194/amt-11-5673-2018, for the integrated mass enhancement and cross-sectional flux this fit is compared with.
- Rasmussen and Williams (2006), *Gaussian Processes for Machine Learning*, ch. 4, for the Matérn kernel.
