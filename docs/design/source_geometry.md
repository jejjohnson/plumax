# Source geometries: lines, areas, volumes and tracks

28 Nov 2026 · J. Emmanuel Johnson

## Decision

Recommended: add a small family of source geometries in a new `plumax.sources` module (`Point`, `Line`, `Area`, `Volume`, `Track`), and let every tier take one in place of the bare `(x, y, z)` tuple. Each geometry offers two views of itself, a quadrature rule of weighted points and a cover of grid cells, and each tier consumes the view that suits it. The emission rate stays one total Q in kg s⁻¹, so the inversions and the catalogue do not change.

Alternative: keep the tiers as they are and let the user rasterise a shape into the two tiers that already take a grid.

Status: open.

## Question

How does plumax describe a source that is a line, a polygon, a volume or a moving point, so that all tiers accept it through one type and the rest of the library stays linear in the rate?

## Motivation

The [applications page](applications.md) lists six perfect fits beyond trace gases, and four of them have sources that are not points: a fire front is a moving line, an oil slick and a landfill are areas, a ship is a track, and an eruption column is a vertical line. plumax today forces each of these into a point or into a hand-built raster.

The code was read at plumax `bc0c396`, tracerax `65d0c22`, functax `b1086f4`, integrix `dc3279f`, gridax `aa19982` and geotoolz `543ad57`, all on 28 Nov 2026.

## User stories

1. **A fire front.** As a smoke analyst I have the fire perimeter as a polyline from active-fire detections. I want the smoke column that a line release along it produces, and the emission rate of each kilometre of front.
2. **An oil slick.** As a spill responder I have a slick polygon from a radar image. I want to release tracer over the polygon and drift it forward, and to run it backward to the ships that crossed it.
3. **A landfill.** As a methane analyst I have a landfill outline. I want one emission rate for the whole site without pretending it is a point, because the plume near the site is wider than a point plume.
4. **A ship.** As an analyst of ship emissions I have a position track from AIS. I want the plume of a source that moves while it emits.
5. **An eruption.** As a volcanologist I want the release spread over a vertical line from the vent to the injection height.

## What exists today

Two tiers take a gridded source, and four take a point. No tier takes a geometry.

| Tier | Source argument | Extended source |
| --- | --- | --- |
| `gauss_plume` | `source_x, source_y, source_z` | K points only |
| `gauss_puff` | `source_location` tuple | none |
| `lagrangian` forward | `source_location` tuple | none |
| `lagrangian` backward | `emission_pattern` per cell | any raster |
| `les_fvm` | `GaussianSource`, `SurfaceFluxSource` | any density |
| `coupled` | follows its backend | as above |

**Findings in the code.**

1. **The point is enforced.** `check_source` in `src/plumax/_src/domain.py` fails unless `source_location` has length three. `plume_concentration`, `simulate_puff` and `simulate_lagrangian` all take the point as scalars or a tuple. Checked @ `bc0c396`.
2. **`GaussianSource` already holds any density.** Its fields are `emission_fn` and a `density` of shape `(nz, ny, nx)` that integrates to one. Only its constructor, `make_gaussian_source`, is specific to a point. Checked in `src/plumax/les_fvm/_src/source.py` @ `bc0c396`.
3. **`SurfaceFluxSource` is a surface map.** It takes a flux `q(y, x, t)` in kg m⁻² s⁻¹ and puts it in the lowest layer, so it serves a ground-level area and nothing elevated. Same file.
4. **`LagrangianSource.emission_pattern` is general.** It is a flux per footprint cell, with an optional second axis of one pattern per emission parameter. Only the `point` constructor exists. Checked in `src/plumax/coupled/_src/backends.py` @ `bc0c396`.
5. **`MaskedSources` sums K point plumes.** It has one rate, one position and one activity per slot, which is the closest thing to an extended source in the analytic tiers. Checked in `src/plumax/gauss_plume/_src/multi_source.py` @ `bc0c396`.
6. **The forward particle run seeds with `at_point`.** tracerax also ships `along_line`, `in_box` and `release_times`, which plumax does not use for sources. Checked in `src/plumax/lagrangian/_src/concentration.py` @ `bc0c396` and `src/tracerax/_src/particles/seed.py` @ `65d0c22`.
7. **The steady particle estimator assumes one release in a steady flow.** `steady_concentration` in tracerax states that it needs N particles released at once and a steady velocity. A moving source breaks that assumption. Checked @ `65d0c22`.

## Background math

A source geometry is a set Γ in space with a measure, and a source is that geometry times a rate.

### A geometry as a normalised measure

Let Γ be a point, a curve, a surface patch or a volume, and let |Γ| be its measure: 1 for a point, a length in m, an area in m² or a volume in m³. Let Q(t) be the total rate in kg s⁻¹ and let w(𝛏) be a profile on Γ that integrates to one. The source term of the transport equation is

```math
s(\mathbf{x}, t) = Q(t) \int_\Gamma w(\boldsymbol{\xi})\, \delta(\mathbf{x} - \boldsymbol{\xi})\, \mathrm{d}\mu(\boldsymbol{\xi}), \qquad \int_\Gamma w\, \mathrm{d}\mu = 1
```

where 𝛏 runs over Γ, μ is the length, area or volume measure, and δ is the Dirac delta. A uniform release has w = 1/|Γ|. The rate per unit of the geometry is Q/|Γ|, in kg m⁻¹ s⁻¹ for a line and kg m⁻² s⁻¹ for an area, and it is derived, never stored.

### The response is an integral of point responses

Transport is linear in the source for a passive tracer. Let G(𝐱; 𝛏) be the response at receptor 𝐱 to a unit point source at 𝛏, which is what every tier computes today. The response to the geometry is

```math
c(\mathbf{x}) = Q \int_\Gamma w(\boldsymbol{\xi})\, G(\mathbf{x}; \boldsymbol{\xi})\, \mathrm{d}\mu(\boldsymbol{\xi})
```

Two discretisations of this integral cover every tier.

**The points view.** A quadrature rule on Γ gives K nodes 𝛏ₖ and weights wₖ that sum to one, and the response is a weighted sum of point responses.

```math
c(\mathbf{x}) \approx Q \sum_{k=1}^{K} w_k\, G(\mathbf{x}; \boldsymbol{\xi}_k), \qquad \sum_k w_k = 1
```

Particles are the same view with random nodes: N particles drawn from w, each with weight 1/N.

**The cells view.** On a grid with cells Cⱼ of volume Vⱼ, the source enters as a density that is constant in each cell.

```math
s_j = \frac{Q\, W_j}{V_j}, \qquad W_j = \int_{\Gamma \cap C_j} w\, \mathrm{d}\mu, \qquad \sum_j W_j = 1
```

Here Wⱼ is the share of the geometry inside cell j. For a uniform area it is the area of the polygon clipped to the cell, divided by the polygon's area.

### A closed form to test against

A straight line of length L, perpendicular to a uniform wind of speed u, at height h, with total rate Q, has an exact Gaussian-plume response. With σ_y and σ_z the crosswind and vertical spreads at downwind distance x, and y the crosswind offset from the middle of the line,

```math
c(x, y, z) = \frac{Q}{L} \cdot \frac{1}{\sqrt{2\pi}\, u\, \sigma_z} \left[ e^{-\frac{(z-h)^2}{2\sigma_z^2}} + e^{-\frac{(z+h)^2}{2\sigma_z^2}} \right] \cdot \frac{1}{2} \left[ \operatorname{erf}\!\frac{L/2 - y}{\sqrt{2}\, \sigma_y} + \operatorname{erf}\!\frac{L/2 + y}{\sqrt{2}\, \sigma_y} \right]
```

The last factor tends to one for a long line, which recovers the infinite line source, and the whole expression tends to the point plume as L goes to zero.

### A moving source

A track is a point whose position 𝛏(t) depends on time, so the source is s = Q(t) δ(𝐱 − 𝛏(t)). It is a geometry in space and time and has no steady state. In the puff tier, puff k is released at time tₖ over an interval Δt, from 𝛏(tₖ), with mass

```math
m_k = Q(t_k)\, \Delta t, \qquad \mathbf{x}_k(t_k) = \boldsymbol{\xi}(t_k)
```

and the particle tier does the same with one release time and one start position per particle.

### What attribution gains

A geometry can be split into pieces Γ₁, …, Γ_S, each with its own rate qₛ. The observation stays linear, 𝐲 = 𝐛 + 𝐇 𝐪 + 𝛆, and column s of 𝐇 is the response of piece s at unit rate. So the rate per kilometre of a fire front is the same inversion plumax already runs for S point sources.

## Background numerics

**Line quadrature needs nodes closer than the plume is wide.** The integrand along the line is a Gaussian of width σ_y, so the node spacing h must resolve it. I measured the error of two rules against the closed form above, for L = 2,000 m and σ_y = 76.3 m (Briggs rural class D at x = 1,000 m), over crosswind offsets from −2,000 m to 2,000 m. The error is the largest absolute difference divided by the peak value.

| Nodes K | h / σ_y | Midpoint | Gauss–Legendre |
| --- | --- | --- | --- |
| 13 | 2.02 | 6.8 × 10⁻² | 2.4 × 10⁻¹ |
| 27 | 0.97 | 1.0 × 10⁻² | 3.8 × 10⁻⁴ |
| 54 | 0.49 | 2.4 × 10⁻³ | 4.6 × 10⁻¹⁴ |
| 108 | 0.24 | 6.0 × 10⁻⁴ | 7.6 × 10⁻¹⁵ |

These numbers come from a NumPy script run for this page on 28 Nov 2026, with its own Gaussian kernel and not plumax's. Gauss–Legendre reaches rounding error once h is at most σ_y/2, and the midpoint rule converges slowly because of the line's two ends. Under-resolved, Gauss–Legendre is worse than midpoint.

**The rule of thumb fails near the source.** σ_y goes to zero at the line, so no fixed K resolves receptors arbitrarily close to it. The default K is set from the nearest receptor, and receptors closer than that are a documented limit.

**Area cover is exact by clipping.** The share Wⱼ of a polygon in a rectangular cell is a polygon clipped to a convex polygon, which Sutherland–Hodgman computes exactly. The shares sum to one by construction, so mass is conserved to rounding.

**Gradients.** The polyline rule is differentiable in its vertices (functax states this for `polyline`), so the points view carries gradients with respect to the shape. The cells view is computed on the host in NumPy, so the shape is fixed data and only the rate is differentiated. A thin line on a grid also aliases, so the grid tiers smooth the cover with the same Gaussian radius that `make_gaussian_source` uses for a point.

**Cost.** The points view costs K point evaluations in the analytic tiers. The cells view and the particle seeding cost nothing extra at run time.

**Static shapes.** K, the number of vertices and the number of particles are Python integers, so every array shape is fixed under `jit`.

## What it builds on

Most of the machinery exists upstream, and one piece needs to be made public.

| Need | Upstream name | State |
| --- | --- | --- |
| Nodes and weights along a polyline | `functax.quad.polyline`, `CurveRule` | today |
| Rule on an interval | `functax.quad.gauss_legendre`, `Rule` | today |
| Weighted sum over nodes | `integrix.rule.integrate` | today |
| Particles at a point, on a line, in a box | `tracerax.particles.at_point`, `along_line`, `in_box` | today |
| Release times of a continuous release | `tracerax.particles.release_times` | today |
| Polygon clipped to a cell, and its area | functax `clip_polygons`, `planar_polygon_area` | private, in `_src/interp/_clip.py` |
| Particles in a polygon or on a polyline | none | missing |
| Burn a shapely geometry on a `GeoArray` | `geoops.geom.Rasterize` | today, NumPy, binary |

**Gaps, each with its owner.**

- **functax: a public polygon cover.** `clip_polygons` and `planar_polygon_area` exist but are reached only through the `Conservative` interpolator. A public `functax.quad.polygon_cover(vertices, x_edges, y_edges)` that returns the area of the polygon in each cell would serve plumax and any other grid model.
- **tracerax: two seeders.** `in_polygon` (uniform draws in a polygon) and `along_polyline` (particles shared among segments by length) belong beside `along_line` and `in_box`.
- **plumax: one private helper to retire.** `obs/_src/plume_scene.py` has its own even-odd point-in-polygon test, which should move to functax with the cover.

**What stays local.** plumax keeps the geometry family, the wiring of each tier and the `Track`. `geoops.geom.Rasterize` is not used for the cover, because it burns whole pixels on a georeferenced array and plumax needs fractions on a model grid in metres. It remains the route from a segmented mask to a polygon and back.

## API proposal

The proposal is a new module `plumax.sources`. Existing classes already use the word "source" for transport backends (`PlumeSource`, `LagrangianSource`, `EulerianSource`), so the geometries take short names and are read with the module: `sources.Line`.

### The geometries

In the block, V is the number of vertices, K the number of quadrature nodes, N the number of particles, and (N_x, N_y, N_z) the grid. The symbols are those of the maths above: |Γ| is the measure, 𝛏ₖ and wₖ the nodes and weights, Wⱼ the cell shares.

```python
# Shapes: V vertices; K nodes; N particles; (N_x, N_y, N_z) grid cells; T track times

# proposed (plumax.sources)
class AbstractSourceGeometry(eqx.Module):
    # |Γ|: 1 for a point, else m, m² or m³
    # → ()
    @property
    @abc.abstractmethod
    def measure(self) -> Float[Array, ""]: ...

    # (ξ_k, w_k), Σ_k w_k = 1
    # → nodes (K, 3), weights (K,)
    def rule(self, n: int) -> Rule: ...

    # W_j = ∫_{Γ ∩ C_j} w dμ, Σ_j W_j = 1
    # → (N_x, N_y, N_z)
    def cover(self, x_edges, y_edges, z_edges) -> Float[Array, "x y z"]: ...

    # ξ_p ~ w, m_p = M / N
    # → positions (N, 3)
    def seed(self, n: int, *, key, mass=1.0, release_time=0.0) -> ParticleState: ...

    # Γ = Γ_1 ∪ … ∪ Γ_S, pieces of equal measure
    def split(self, n: int) -> tuple[AbstractSourceGeometry, ...]: ...


# proposed (plumax.sources)
class Point(AbstractSourceGeometry):
    location: Float[Array, " 3"]

# proposed (plumax.sources)
class Line(AbstractSourceGeometry):
    vertices: Float[Array, "V 3"]

# proposed (plumax.sources)
class Area(AbstractSourceGeometry):
    ring: Float[Array, "V 2"]
    height: float = 0.0

# proposed (plumax.sources)
class Volume(AbstractSourceGeometry):
    lower: Float[Array, " 3"]
    upper: Float[Array, " 3"]

# proposed (plumax.sources)
class Track(eqx.Module):
    times: Float[Array, " T"]
    positions: Float[Array, "T 3"]

    # ξ(t), linear between the track's times
    # () → (3,)
    def position(self, t: Float[Array, ""]) -> Float[Array, " 3"]: ...
```

`Track` is not a subclass, because it has no measure and no steady response. It is accepted only by the tiers that resolve time.

### What each tier does with it

Every tier keeps its current signature and gains a `source` keyword that takes a geometry. Passing the old tuple keeps working and means `sources.Point`.

| Tier | View | What changes |
| --- | --- | --- |
| `gauss_plume` | points | sum of point plumes over `rule(n)` |
| `gauss_puff` | points, or `Track` | each release leaves from a node, or from 𝛏(tₖ) |
| `lagrangian` forward | seed | `seed` in place of `at_point` |
| `lagrangian` backward | cells | `LagrangianSource.from_geometry` builds the pattern |
| `les_fvm` | cells | `make_source` builds the density from `cover` |
| `coupled` | follows backend | `PlumeSource.location` becomes `PlumeSource.geometry` |

**`gauss_plume`.** A new `geometry_concentration` evaluates `plume_concentration` at the K nodes and sums with the weights. `plume_concentration` itself does not change.

**`gauss_puff`.** `simulate_puff` takes `source=`. For a static geometry each release time emits K puffs of mass Q Δt wₖ. For a `Track` each release leaves from 𝛏(tₖ).

**`lagrangian` forward.** `simulate_lagrangian` calls `source.seed(n_particles, key=...)`. A `Track` is refused here, because the steady estimator needs one release in a steady flow (finding 7).

**`lagrangian` backward.** `LagrangianSource.from_geometry(geometry, ...)` sets `emission_pattern` to Wⱼ / A_cell, and `point` becomes the case of a one-cell cover. With `geometry.split(S)` the pattern has S columns, which the field already allows (finding 4).

**`les_fvm`.** `make_source(plume_grid, emission_rate, geometry)` returns the existing `GaussianSource` with `density = W / V_cell`, smoothed by the source radius. The class is unchanged (finding 2), and `make_gaussian_source` becomes the `Point` case. A `Track` needs a new `MovingSource` whose density is evaluated at each time.

**`coupled`.** `PlumeSource` stores a geometry. Its fitted `location` block, of shape `(2,)`, becomes a horizontal shift of the whole geometry, so a point behaves as today.

### What does not change

The rate is still one total Q in kg s⁻¹ per geometry. `linear_gaussian_inversion`, `solve_4dvar`, `fuse_observations`, the catalogue and the persistency metrics act on rates and never see the shape.

## API usage examples

The first block of example 1 uses only shipped names and was run as written. Every block with a `proposed` line is a sketch and was not run, and its `today` names were checked by name and signature in the repos at the commits above.

### 1. A line source in the Gaussian plume

This is the closed-form case of the maths section: L = 2,000 m across a 5 m s⁻¹ wind, total rate Q = 2 kg s⁻¹, with P receptors.

```python
import jax
import jax.numpy as jnp
import functax.quad as fq
import integrix.rule as ir
from plumax.gauss_plume.plume import plume_concentration

# Shapes: V = 2 vertices; K = 54 nodes; P receptors

# user
# Γ: a line from (0, −1000, 2) to (0, 1000, 2) [m]
# → (V, 3)
front = jnp.array([[0.0, -1000.0, 2.0], [0.0, 1000.0, 2.0]])

# today (functax)
# (ξ_k, Δs_k), Σ_k Δs_k = |Γ| = L
# (V, 3) → nodes (K, 3), weights (K,)
curve = fq.polyline(front, 54, rule="gauss_legendre")

# w_k = Δs_k / L
# → (K,)
weights = curve.weights / curve.weights.sum()

# today* (plumax)
# G(x; ξ_k), the point plume of unit rate from node k
# (K, 3) → (K, P)
unit_plumes = jax.vmap(
    lambda node: plume_concentration(
        x, y, z, node[0], node[1], node[2], wind_u, wind_v, 1.0, dispersion_params
    )
)(curve.nodes)

# today (integrix)
# c(x) = Q Σ_k w_k G(x; ξ_k)
# (K, P) → (P,)
concentration = 2.0 * ir.integrate(weights, unit_plumes, axis=0).value
```

This block was run on 28 Nov 2026 in float64 with plumax `d00f3eb`, functax 0.0.15 and integrix 0.0.13, with Briggs class D and 401 receptors 1,000 m downwind. The result matches the closed form to 4.5 × 10⁻¹⁴ of its peak, and K = 54 matches K = 400 to 1.2 × 10⁻¹³.

With the proposal, the same result is one call.

```python
# Shapes: K = 54 nodes; P receptors

# proposed (plumax.sources)
front = sources.Line(vertices=front_vertices)

# proposed (gauss_plume)
# c(x) = Q Σ_k w_k G(x; ξ_k)
# → (P,)
concentration = geometry_concentration(
    receptors, front, wind_u, wind_v, emission_rate=2.0,
    dispersion_params=dispersion_params, n_nodes=54,
)
```

### 2. Particles released along a fire front

```python
# Shapes: N = 20,000 particles; V vertices; (N_x, N_y, N_z) grid cells

# proposed (plumax.sources)
front = sources.Line(vertices=perimeter_vertices)

# proposed (lagrangian)
# ξ_p ~ uniform on Γ; C = Q R / (N V_cell)
# → concentration (N_x, N_y, N_z)
smoke = simulate_lagrangian(
    emission_rate=40.0, source=front, turbulence=turbulence,
    domain_x=domain_x, domain_y=domain_y, domain_z=domain_z,
    n_particles=20_000, wind=wind,
)
```

Inside, for a `Line` of one segment, `seed` is tracerax's `along_line`.

```python
# today (tracerax)
# x_p = start + (p + ½)/N (stop − start), m_p = M / N
# → positions (N, 3)
state = along_line(front.vertices[0], front.vertices[1], 20_000)
```

### 3. An oil slick as an area source for a backward inversion

```python
# Shapes: V ring vertices; (N_x, N_y) footprint cells; N_y obs receptors

# proposed (plumax.sources)
slick = sources.Area(ring=slick_ring, height=0.0)

# proposed (coupled)
# q_j = W_j / A_cell, Σ_j W_j = 1
# → emission_pattern (N_x N_y,)
backend = LagrangianSource.from_geometry(
    slick, winds=winds, turbulence=turbulence,
    domain_x=domain_x, domain_y=domain_y, column_z=column_z,
)
```

### 4. The rate of each piece of a line

```python
# Shapes: S = 8 pieces; N_y obs receptors

# proposed (plumax.sources)
# Γ = Γ_1 ∪ … ∪ Γ_S
pieces = front.split(8)

# proposed (coupled)
# H[:, s] = response of piece s at unit rate
# → (N_y, S)
response = jnp.stack(
    [backend_for(piece).column_response(instrument) for piece in pieces], axis=-1
)

# today* (plumax)
# p(q | y) = 𝒩(μ_q, Σ_q), y = H q + ε
# (N_y, S) → mean (S,), covariance (S, S)
posterior = linear_gaussian_inversion(response, observations, prior_mean, prior_cov, obs_cov)
```

### 5. A ship as a moving source in the puff tier

```python
# Shapes: T = 120 track times; N_t output times; (N_x, N_y, N_z) grid cells

# proposed (plumax.sources)
# ξ(t), linear between AIS fixes
ship = sources.Track(times=fix_times, positions=fix_positions)

# proposed (gauss_puff)
# puff k: m_k = Q Δt, released from ξ(t_k)
# → concentration (N_t, N_x, N_y, N_z)
plume = simulate_puff(
    emission_rate=0.05, source=ship, wind_speed=wind_speed,
    wind_direction=wind_direction, stability_class="D",
    domain_x=domain_x, domain_y=domain_y, domain_z=domain_z, time_array=time_array,
)
```

## Options considered

| Criterion | A. Geometry type | B. Per-tier constructors | C. User rasters |
| --- | --- | --- | --- |
| One description for all tiers | yes | no | no |
| Analytic tiers covered | yes | yes | no |
| Gradient in the shape | points view | points view | no |
| New public names | 6 | about 12 (estimate) | 0 |
| Upstream work | 2 gaps | 2 gaps | none |

**A. A geometry type (recommended).** One object is written once and passed to any tier, as in the examples above.

**B. Per-tier constructors.** Each tier grows its own `line`, `area` and `track` builders, such as `LagrangianSource.line(...)` and `make_line_source(...)`. The same shape is then described six ways, and a cross-tier comparison needs six conversions.

**C. User rasters.** Nothing is added, and the user burns shapes into `emission_pattern` and `SurfaceFluxSource`. This leaves the analytic tiers and the forward particle run on points, and moves mass conservation to the user.

**Rejected.** A rate per unit of geometry (kg m⁻¹ s⁻¹) as the stored parameter, because it changes units between geometries and breaks the catalogue, which compares rates across events.

## Gates

| Gate | Passes when |
| --- | --- |
| G1 normalisation | `rule(n).weights` and `cover(...)` each sum to 1 within 10⁻¹² in float64, for every geometry |
| G2 point unchanged | every tier called with `sources.Point` equals today's call with the tuple, to 10⁻¹² relative |
| G3 finite line | `geometry_concentration` matches the erf closed form to 10⁻⁸ relative with h ≤ σ_y/2 and Gauss–Legendre |
| G4 point limit | a `Line` of length 1 mm matches the point plume to 10⁻⁶ relative at 1 km |
| G5 additivity | the response of a geometry equals the sum of the responses of `split(8)`, to 10⁻¹⁰ relative |
| G6 cover is exact | the cover of an axis-aligned rectangle equals the analytic overlap areas to 10⁻¹² |
| G7 mass in `les_fvm` | the domain mass after time t, before any outflow, equals Q t to 10⁻⁶ relative |
| G8 track at rest | a `Track` with constant position equals `Point` in `gauss_puff`, to 10⁻¹² relative |
| G9 gradients | ∂c/∂Q and ∂c/∂(vertex) of the points view match finite differences to 10⁻⁵ relative |

## Risks

- **Near-field error.** Quadrature fails for receptors closer than the node spacing. Fallback: raise K from the nearest receptor distance, and document the limit.
- **The cover is not differentiable in the shape.** A fit of the outline through a grid tier gets a zero gradient. Fallback: fit the shape in the points view, or fit only a shift and a scale.
- **Polygons with holes or self-intersections.** Sutherland–Hodgman takes a simple polygon. Fallback: accept simple rings only and refuse the rest with an error.
- **A moving source in `les_fvm` costs one density evaluation per step.** Fallback: ship `Track` for the puff and particle tiers first.
- **Name collision.** `plumax.sources` next to `PlumeSource` and `LagrangianSource` may confuse. Fallback: name the module `plumax.release`.

## Follow-up reports

- [ ] `feat(sources): the geometry family with rule, cover, seed and split` — `Point`, `Line`, `Area`, `Volume`; gates G1, G5, G6.
- [ ] `feat(quad): a public polygon cover of grid cells` — in functax, from `clip_polygons` and `planar_polygon_area`; blocks `Area.cover`.
- [ ] `feat(particles): in_polygon and along_polyline seeders` — in tracerax; blocks `Area.seed` and multi-segment `Line.seed`.
- [ ] `feat(gauss_plume): geometry_concentration over a source rule` — gates G2, G3, G4, G9.
- [ ] `feat(lagrangian): seed the forward run and build footprints from a geometry` — gate G2.
- [ ] `feat(les_fvm): make_source from a geometry's cover` — gates G2, G7.
- [ ] `feat(gauss_puff): static geometries and Track releases` — gate G8.
- [ ] `feat(coupled): PlumeSource holds a geometry` — the `location` block as a shift.
- [ ] `[Design] a time-resolved particle estimator for moving sources` — what replaces the steady estimator for a `Track`.

## References

- plumax, `src/plumax/` @ `bc0c396`: `_src/domain.py`, `gauss_plume/_src/plume.py`, `gauss_plume/_src/multi_source.py`, `gauss_puff/_src/puff.py`, `lagrangian/_src/concentration.py`, `les_fvm/_src/source.py`, `coupled/_src/backends.py`, `coupled/_src/forward.py`.
- tracerax, `src/tracerax/_src/particles/seed.py` and `_src/bin/concentration.py` @ `65d0c22`.
- functax, `src/functax/quad.py`, `_src/quad/_curve.py` and `_src/interp/_clip.py` @ `b1086f4`.
- integrix, `src/integrix/rule.py` @ `dc3279f`.
- geotoolz, `packages/geotoolz-ops/src/geoops/geom/_src/operators.py` @ `543ad57`.
- The finite line source formula is the standard crosswind integral of the Gaussian plume. It is derived here from the point plume and no textbook was consulted for this page.

## Relationships

- Parent: none
- Related: the [applications page](applications.md), whose follow-up "sources that move or have extent" this page answers
