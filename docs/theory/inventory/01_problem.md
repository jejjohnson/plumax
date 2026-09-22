# Part I — The real problem and the real data

Before any mathematics: what is physically out there, what the instruments return, and the one running example carried through every later page.

## I.1 The task, in three phases {#i-1}

```mermaid
flowchart LR
    A["<b>Phase A</b><br/>DISCOVERY<br/><i>random: locations</i>"] --> B["<b>Phase B</b><br/>MONITORING<br/><i>random: state over time</i>"] --> C["<b>Phase C</b><br/>ESTIMATION<br/><i>random: rates and their sum</i>"]
```

| Phase | Question | What is random |
|---|---|---|
| **A — Discovery** | Where are the sources we don't yet know about? | **Locations** |
| **B — Monitoring** | For sources we know, when are they emitting? | **State over time** |
| **C — Estimation** | How much does each emit, and what is the total? | **Rates and their sum** |

The three phases need different mathematical objects because a different thing is uncertain in each. [Part II](02_objects.md) builds those objects; [Part III](03_phases.md) maps them back onto the phases.

## I.2 What is actually out there {#i-2}

Methane sources in an oil-and-gas basin come in two physical kinds, and the distinction is one of *scale*:

<div class="grid cards" markdown>

-   :material-circle-medium:{ .lg .middle } **Point sources**

    ---

    A single failed valve, an unlit flare, a tank vent. Spatial extent of metres; a compact, high-contrast plume. Rates from tens of $\mathrm{kg\,h^{-1}}$ to several $\mathrm{t\,h^{-1}}$. Often **intermittent**: on for hours, off for days.

-   :material-texture-box:{ .lg .middle } **Area (diffuse) sources**

    ---

    Thousands of small leaks across an old field, a landfill face, a wetland. Extent of kilometres; a broad, low-contrast enhancement. Individually tiny, collectively large. Usually **persistent**.

</div>

Two empirical facts drive everything downstream:

1. **Rates are heavy-tailed.** A few percent of point sources ("super-emitters") carry a large share of the total.
2. **Whether something is a point or an area source depends on what you look with.** A cluster of twenty small pads inside one coarse pixel is, to that instrument, an area source. To a fine imager it is twenty points.

## I.3 What the instruments give {#i-3}

Every instrument is characterised by four numbers.[^specs]

| Quantity | Symbol | Coarse mapper | Fine imager | Units |
|---|:---:|:---:|:---:|---|
| pixel size | $\Delta x$ | $\approx 5$–$7$ | $\approx 0.03$ | km |
| revisit | $\Delta t$ | $\approx 1$ | $\approx 5$ (or tasked) | d |
| per-source detection limit | $s_{\min}$ | $\approx$ several $\times 10^3$ | $\approx 10^2$–$10^3$ | $\mathrm{kg\,h^{-1}}$ |
| per-pixel precision | $\sigma_r$ | $\approx 10$ | $\approx 100+$ | ppb |

Consequences:

=== "Coarse mapper"

    Sees almost every point source in the [§I.2](#i-2) example only as part of a pixel **aggregate**. It sees area sources well, after averaging many days.

=== "Fine imager"

    Resolves individual point plumes but revisits rarely, so it **samples** an intermittent source rather than observing it continuously.

[^specs]: Values are indicative orders of magnitude; check current mission specifications before relying on them.

## I.4 Data products {#i-4}

| Product | What it is | Units |
|---|---|---|
| L2 $\mathrm{XCH_4}$ | column-mean mole fraction per pixel | ppb |
| enhancement $f(x)$ | $\mathrm{XCH_4}$ minus local background | ppb |
| plume mask | pixels flagged as plume | — |
| rate estimate | per-plume flux from mask + wind (e.g. IME method) | $\mathrm{kg\,h^{-1}}$ |
| wind $\mathbf{U}$ | 10 m wind vector from reanalysis | $\mathrm{m\,s^{-1}}$ |
| bottom-up inventory | gridded prior emission density | $\mathrm{kg\,h^{-1}\,km^{-2}}$ |
| infrastructure database | candidate site coordinates $c_j$ (pads, compressors) | km, km |

!!! info "The infrastructure database changes the maths"
    It turns some of Phase A's "unknown locations" into "known candidate locations with unknown state". That is a different mathematical object ([§II.4.4](02_objects.md#ii-4-4 "Beta process — persistence prior")).

## I.5 The running example {#i-5}

A $40\,\mathrm{km} \times 40\,\mathrm{km}$ study region $S$ in the Permian Basin, partitioned into four $20\,\mathrm{km} \times 20\,\mathrm{km}$ cells $A_1,\dots,A_4$.

<figure>
  <svg viewBox="-24 -8 372 360" width="380" role="img" aria-label="Map of the 40 km by 40 km study region with four cells, four point sources and one diffuse field" style="max-width:100%;font-family:var(--md-text-font-family);font-size:13px">
    <rect x="176" y="40" width="128" height="100" rx="6" style="fill:var(--md-accent-fg-color);fill-opacity:.18;stroke:var(--md-accent-fg-color);stroke-dasharray:4 3"/>
    <text x="240" y="118" text-anchor="middle" style="fill:var(--md-default-fg-color--light);font-style:italic">field D · 200 km²</text>
    <rect x="0" y="0" width="320" height="320" style="fill:none;stroke:var(--md-default-fg-color);stroke-width:1.5"/>
    <line x1="160" y1="0" x2="160" y2="320" style="stroke:var(--md-default-fg-color--light)"/>
    <line x1="0" y1="160" x2="320" y2="160" style="stroke:var(--md-default-fg-color--light)"/>
    <g style="fill:var(--md-default-fg-color--light);font-weight:700">
      <text x="10" y="20">A₁</text><text x="170" y="20">A₂</text>
      <text x="10" y="180">A₃</text><text x="170" y="180">A₄</text>
    </g>
    <g style="fill:var(--md-primary-fg-color);stroke:var(--md-default-bg-color);stroke-width:1.5">
      <circle cx="96" cy="64" r="5.5"/>
      <circle cx="248" cy="40" r="4"/>
      <circle cx="112" cy="264" r="10"/>
      <circle cx="280" cy="224" r="4"/>
    </g>
    <g style="fill:var(--md-default-fg-color)">
      <text x="104" y="60">1 · 120</text>
      <text x="256" y="36">2 · 35</text>
      <text x="98" y="268" text-anchor="end">3 · 410</text>
      <text x="288" y="228">4 · 60</text>
    </g>
    <g style="fill:var(--md-default-fg-color--light);font-size:11px">
      <text x="0" y="338">0</text><text x="160" y="338" text-anchor="middle">20</text><text x="320" y="338" text-anchor="end">40 km</text>
    </g>
  </svg>
  <figcaption>The study region S. Dots are the point sources, sized by on-state rate (kg h⁻¹); the dashed patch is the diffuse old field D inside A₂.</figcaption>
</figure>

**Point sources** (true, unknown to us at the start):

| $k$ | $x_k$ (km) | $s_k$ on ($\mathrm{kg\,h^{-1}}$) | $\pi_k$ | in |
|:---:|:---:|:---:|:---:|:---:|
| 1 | (12, 32) | 120 | 0.9 | $A_1$ |
| 2 | (31, 35) | 35 | 1.0 | $A_2$ |
| 3 | (14, 7) | 410 | 0.3 | $A_3$ |
| 4 | (35, 12) | 60 | 1.0 | $A_4$ |

$\pi_k$ is the fraction of time source $k$ is on. Source 3 is a large intermittent emitter: a blowdown that fires roughly one day in three.

**Area source:** an old field $D \subset A_2$ of $200\,\mathrm{km^2}$ with flux density $e(x) \approx 2\ \mathrm{kg\,h^{-1}\,km^{-2}}$, giving $400\ \mathrm{kg\,h^{-1}}$ diffuse.

**True totals:**

=== "Time-averaged (what an inventory wants)"

    $$
    \begin{aligned}
    \text{point, time-averaged } (\pi_k s_k): &\quad 108 + 35 + 123 + 60 = 326\ \mathrm{kg\,h^{-1}}\\
    \text{diffuse:} &\quad 400\ \mathrm{kg\,h^{-1}}\\
    \text{basin, time-averaged:} &\quad 726\ \mathrm{kg\,h^{-1}}
    \end{aligned}
    $$

    ```mermaid
    %%{init: {"themeVariables": {"pie1": "#00695c", "pie2": "#26a69a", "pie3": "#80cbc4", "pie4": "#b2dfdb", "pie5": "#ffb74d", "pieStrokeColor": "#ffffff", "pieOuterStrokeColor": "#9e9e9e", "pieTitleTextColor": "#8a8a8a", "pieLegendTextColor": "#8a8a8a", "pieSectionTextColor": "#ffffff"}}}%%
    pie showData title Basin total, time-averaged (kg/h)
        "Source 1 (0.9 × 120)" : 108
        "Source 2 (1.0 × 35)" : 35
        "Source 3 (0.3 × 410)" : 123
        "Source 4 (1.0 × 60)" : 60
        "Diffuse field D" : 400
    ```

=== "Instantaneous (all sources on)"

    $$
    \begin{aligned}
    \text{point, instantaneous, all on:} &\quad 120 + 35 + 410 + 60 = 625\ \mathrm{kg\,h^{-1}}\\
    \text{diffuse:} &\quad 400\ \mathrm{kg\,h^{-1}}\\
    \text{basin, instantaneous:} &\quad 1025\ \mathrm{kg\,h^{-1}}
    \end{aligned}
    $$

**Instruments:**

| | $\Delta x$ | revisit | $s_{\min}$ | What it sees here |
|---|---|---|---|---|
| Coarse mapper | $\approx 7$ km | daily | $\approx 5000\ \mathrm{kg\,h^{-1}}$ | **no** single point source |
| Fine imager | $\approx 0.03$ km | every 5 d | $\approx 100\ \mathrm{kg\,h^{-1}}$ | sources **1 and 3**, when they are on and the scene is clear |

## I.6 Notation and units {#i-6}

Fixed for the whole section.

??? abstract "Full symbol table (click to expand)"

    | Symbol | Meaning | Units |
    |---|---|---|
    | $S$ | spatial domain | — |
    | $x \in S$ | location (easting, northing) | km, km |
    | $T = [0, T_{\mathrm{end}}]$ | time window | d |
    | $t \in T$ | a time | d |
    | $X$ | generic index set: $S$, $T$, or $S \times T$ | — |
    | $A \subset X$ | a region or interval | — |
    | $\lvert A \rvert$ | area (or length) of $A$ | $\mathrm{km^2}$ (d) |
    | $\delta_x$ | point mass at $x$ | — |
    | $N(A)$ | number of point sources in $A$ | — |
    | $\lambda(A)$, $\lambda(x)$ | expected count in $A$; source density | —, $\mathrm{km^{-2}}$ |
    | $\lambda(t)$ | event rate | $\mathrm{d^{-1}}$ |
    | $s_k$ | on-state rate of point source $k$ | $\mathrm{kg\,h^{-1}}$ |
    | $s_0$ | reference rate, $1\ \mathrm{kg\,h^{-1}}$ | $\mathrm{kg\,h^{-1}}$ |
    | $\pi_k$ | persistence (fraction of time on) of source $k$ | — |
    | $G(\mathrm{d}\pi \mid u)$ | distribution of persistence given on-state rate | — |
    | $p_{\mathrm{camp}}$ | probability a source is detected at least once over a campaign | — |
    | $e(x)$ | diffuse flux density | $\mathrm{kg\,h^{-1}\,km^{-2}}$ |
    | $\mu(A)$ | total emission rate from $A$ (point + diffuse) | $\mathrm{kg\,h^{-1}}$ |
    | $\mu_{\mathrm{pt}}$, $\mu_{\mathrm{df}}$ | atomic and diffuse parts of $\mu$ | $\mathrm{kg\,h^{-1}}$ |
    | $H(A)$ | prior expected emission from $A$ | $\mathrm{kg\,h^{-1}}$ |
    | $u_k$ | dimensionless rate $s_k/s_0$ | — |
    | $\rho(\mathrm{d}u)$ | Lévy measure over dimensionless rates $u$ | — |
    | $G(A)$ | fraction of basin total from $A$ | — |
    | $f(x)$ | $\mathrm{XCH_4}$ enhancement | ppb |
    | $m(x)$, $k(x,x')$ | mean, covariance kernel of a field | ppb, $\mathrm{ppb^2}$ |
    | $g(x)$ | log source density, $\log\lambda(x)$ | — |
    | $\Delta x$, $\Delta t$ | pixel size, revisit interval | km, d |
    | $C_\Delta$ | coarsening operator: $\mu \mapsto$ pixel totals | — |
    | $\mathcal{T}$ | atmospheric transport operator: $\mu \mapsto f$ | ppb per $\mathrm{kg\,h^{-1}}$ |
    | $\mathbf{U}$, $U$ | 10 m wind vector (east, north); its speed $U = \lVert\mathbf{U}\rVert$ | $\mathrm{m\,s^{-1}}$ |
    | $p(x,s)$ | detection probability | — |
    | $s_{\min}$ | detection limit | $\mathrm{kg\,h^{-1}}$ |
    | $c_j$, $Z_j$ | candidate site $j$; its emitting indicator | km km, — |
    | $m_k$, $M(t)$ | event mass; cumulative mass by $t$ | kg |
    | $\lambda^*(t)$, $\mathcal{H}_t$ | conditional intensity; history before $t$ | $\mathrm{d^{-1}}$, — |

!!! note "Two conventions"
    - Inside $\rho(\mathrm{d}u)$ and Gamma shape parameters, rates are **dimensionless**: $u = s/s_0$, and physical totals carry the factor $s_0$ back.
    - Wind enters space–time kernels as the velocity vector $\mathbf{v} = 86.4\,\mathbf{U}$, converting $\mathrm{m\,s^{-1}}$ to $\mathrm{km\,d^{-1}}$.
