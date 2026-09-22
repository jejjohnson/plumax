# Part III — Mapping the objects onto the three phases

Each phase gets the same four headings: **what is random**, the **object** from [Part II](02_objects.md) that describes it, the **data** it consumes, and what falls out.

## III.1 Phase A — Discovery {#iii-1}

**What is random**
:   Locations of sources not in the database

**Object**
:   Thinned log-Gaussian Cox process ([§II.7](02_objects.md#ii-7) + [§II.9](02_objects.md#ii-9))

**Data**
:   Fine-imager scenes over searched cells; the infrastructure database; the bottom-up inventory as the GP mean $m_g$

Discovery counts **sources**, so it needs a finite number of them. Keep $\lambda$ as the CRM base measure of [§II.4](02_objects.md#ii-4) (the same one [§III.3](#iii-3) integrates against), and give each source two marks: its **on-state** rate $s_0 u$, with Lévy measure $\rho(\mathrm{d}u)$, and its persistence $\pi$, with conditional distribution $G(\mathrm{d}\pi \mid u)$. Count only sources above a size floor $u_{\mathrm{floor}}$ (well below $s_{\min}/s_0$). Their count intensity and normalised rate distribution are

$$
\lambda_{\ge}(\mathrm{d}x) = R\,\lambda(\mathrm{d}x),
\qquad
f(\mathrm{d}u) = \frac{\rho(\mathrm{d}u)\,\mathbf{1}[u \ge u_{\mathrm{floor}}]}{R},
\qquad
R = \rho\bigl([u_{\mathrm{floor}}, \infty)\bigr).
$$

A source at $x$ with marks $(u, \pi)$, covered by clear scenes $i = 1, \dots, n$ so far, is detected at least once with probability

$$
p_{\mathrm{camp}}(x,u,\pi) = 1 - \prod_{i=1}^{n} \bigl(1 - \pi\,p_i(x,u)\bigr),
$$

where $p_i(x,u)$ is scene $i$'s POD ([§II.9.2](02_objects.md#ii-9-2); it varies with wind, viewing geometry, and retrieval sensitivity) and on/off states are independent across scenes. With identical scenes this reduces to $1 - (1 - \pi\,p)^{n}$. Its mark average is $\bar p(x) = \iint p_{\mathrm{camp}}(x,u,\pi)\,G(\mathrm{d}\pi \mid u)\,f(\mathrm{d}u)$.

!!! abstract "The central identity"
    By the (marked) thinning theorem, on a searched cell $A$, **conditional on the intensity** $\lambda$:

    $$
    \begin{aligned}
    N_{\mathrm{det}}(A) \mid \lambda &\sim \mathrm{Poisson}\!\left(\int_A \bar p(x)\,\lambda_{\ge}(\mathrm{d}x)\right),\\
    N_{\mathrm{miss}}(A) \mid \lambda &\sim \mathrm{Poisson}\!\left(\int_A \bigl(1-\bar p(x)\bigr)\,\lambda_{\ge}(\mathrm{d}x)\right),
    \qquad \text{independent given } \lambda
    \end{aligned}
    $$

    Under the LGCP, $\lambda$ is random, and marginally the two counts are **dependent** through it. That dependence is exactly what lets detections (and empty searched cells) inform the posterior over $\lambda$, and hence the expected number missed. Write the likelihood conditionally on $\lambda$ and integrate over $\lambda$; never factor the marginal.

So the posterior over $\lambda$ updates from **both** detections **and** searched-but-empty cells, and the expected number still undiscovered (above the floor) is

$$
\int_A \bigl(1 - \bar p(x)\bigr)\,\lambda_{\ge}(\mathrm{d}x)
$$

computable, not guessed.

!!! warning "Why the floor"
    With the infinite-activity priors of [§II.4](02_objects.md#ii-4), $R \to \infty$ as $u_{\mathrm{floor}} \to 0$: the missed source **count** is infinite, even though the missed **mass** is finite. Counting questions (Phase A) therefore need the floor. Mass questions (the search objective below, and Phase C in [§III.3](#iii-3)) integrate against $\lambda\,\rho$ directly, with no floor and no factor $R$.

**Search strategy falls out.** The next cell to image is the one maximising the posterior expected *inventory* emission that is still undiscovered **and** that the next image would detect,

$$
s_0\,\mathbb{E}\!\left[\int_A \int_0^\infty\!\!\int_0^1 \pi u\,\bigl(1 - p_{\mathrm{camp}}(x,u,\pi)\bigr)\,\pi\,p_{\mathrm{next}}(x,u)\;G(\mathrm{d}\pi \mid u)\,\rho(\mathrm{d}u)\,\lambda(\mathrm{d}x)\right]
$$

Here $\pi u$ weights each source by its time-averaged contribution, not its on-state rate, and $\pi\,p_{\mathrm{next}}$ is the chance the next scene catches it on and detects it. Keeping $\pi$ inside one integral preserves the dependence between past misses and the next detection, which share the same $\pi$; averaging them over $\pi$ separately would misrank intermittent bright sources against persistent faint ones. The objective weights high posterior density (Cox correlation from neighbours) against what the instrument can actually see.

```mermaid
flowchart LR
    prior["LGCP prior<br/>on λ"] --> scene["image cell A"]
    scene -->|"detections + empty searched cells"| post["posterior on λ<br/>(neighbours move via k_g)"]
    post -->|"argmax expected undiscovered emission"| scene
```

???+ example "Running example"
    Choose the floor so that $\lambda_{\ge}(A_1) \approx 15$, the count of [§II.3](02_objects.md#ii-3). With $\bar p \approx 0.3$, the prior mean of detections is $0.3 \times 15 = 4.5$. After one clear fine-imager scene over $A_1$ detecting source 1: $N_{\mathrm{obs}}(A_1) = 1$ against that prior mean of 4.5 detectable, so the posterior $\lambda(A_1)$ drops; the Cox prior propagates a milder drop to $A_3$; $A_2$ and $A_4$ are unsearched and unchanged. Expected undiscovered in $A_1 \approx (1 - \bar p)\,\hat\lambda_{\ge}(A_1)$ with mark-averaged $\bar p \approx 0.3$.

## III.2 Phase B — Monitoring {#iii-2}

**What is random**
:   The on/off state and timing of *known* sources

**Object**
:   Per-site Beta–Bernoulli persistence ([§II.4.4](02_objects.md#ii-4-4)), a temporal point process of initiations ([§II.10.2](02_objects.md#ii-10-2)), or an on/off switching process ([§II.10.4](02_objects.md#ii-10-4)), observed through the thinned clock ([§II.9.3](02_objects.md#ii-9-3), [§II.10.6](02_objects.md#ii-10-6))

**Data**
:   The sequence of (overpass, cloud, detected, rate) for each known site

**Model choice by physics.**

| Physical behaviour | Model | Key parameter |
|---|---|---|
| scheduled venting | renewal | $h$ peaked at 7 d |
| leak until repair | two-state (semi-)Markov switching ([§II.10.4](02_objects.md#ii-10-4)) | on-state duration = repair time |
| clustered distinct initiations (upsets, cascades) | Hawkes | $\varphi$ decaying over days |
| random operations | Beta–Bernoulli | $\pi_k$ |

!!! danger "The identifiability warning"
    With $n$ overpasses, $n_{\mathrm{clear}}$ clear, $n_{\mathrm{det}}$ detections at site $k$:

    $$
    \frac{n_{\mathrm{det}}}{n_{\mathrm{clear}}} \ \text{ estimates } \ \pi_k \cdot p(x_k, s_k),
    \quad \text{not } \pi_k
    $$

    Use the controlled-release detection curve to divide out $p$; otherwise report the product and say so.

???+ example "Running example — source 3"
    10 overpasses, 8 clear, 3 detections at $\approx 400\ \mathrm{kg\,h^{-1}}$. With $p(400\ \mathrm{kg\,h^{-1}}) \approx 0.9$ from the detection curve:

    $$
    \hat\pi_3 \approx \frac{3/8}{0.9} \approx 0.42
    $$

    against a truth of 0.3. The discrepancy is sampling noise on 8 scenes, and a $\mathrm{Beta}(1,1)$ prior gives a 90 % interval of roughly $(0.15, 0.7)$. Renewal structure would tighten this if the blowdowns are regular.

## III.3 Phase C — Estimation of rates and total {#iii-3}

**What is random**
:   The on-state rates $s_k$, the diffuse density $e(x)$, and their sums

**Object**
:   The full measure $\mu = \mu_{\mathrm{pt}} + \mu_{\mathrm{df}}$ with a heavy-tailed CRM prior on $\mu_{\mathrm{pt}}$ ([§II.4.3](02_objects.md#ii-4-3)), a log-GP prior on $e$ ([§II.6](02_objects.md#ii-6)), and the per-plume observation model ([§II.8.2](02_objects.md#ii-8-2), [§II.9.3](02_objects.md#ii-9-3))

!!! abstract "Decomposition of the basin total"
    Time-averaged, over the sources that exist:

    $$
    \bar\mu(S)
    = \underbrace{\sum_{k\ \mathrm{detected}} \pi_k s_k}_{\text{(i) monitored points}}
    + \underbrace{\sum_{k\ \mathrm{undetected}} \pi_k s_k}_{\text{(ii) undiscovered points}}
    + \underbrace{\int_S e(x)\,\mathrm{d}x}_{\text{(iii) diffuse}}
    $$

    This is an identity between **random** quantities: all three terms are uncertain, and the object to report is the posterior of the whole sum.

Term (ii) is a sum over sources nobody has seen, with the marks $(u, \pi)$ and campaign detection probability $p_{\mathrm{camp}}$ of [§III.1](#iii-1). By marked thinning, given $\lambda$ the undetected sources form a Poisson process with intensity $\bigl(1 - p_{\mathrm{camp}}\bigr)\,G(\mathrm{d}\pi \mid u)\,\rho(\mathrm{d}u)\,\lambda(\mathrm{d}x)$, independent of the detected ones. Campbell's theorem then gives its first two moments:

$$
\begin{aligned}
\mathbb{E}\bigl[\text{(ii)} \mid \lambda\bigr] &= s_0 \int_S \int_0^\infty\!\!\int_0^1 \pi u\,\bigl(1 - p_{\mathrm{camp}}(x,u,\pi)\bigr)\,G(\mathrm{d}\pi \mid u)\,\rho(\mathrm{d}u)\,\lambda(\mathrm{d}x),\\
\mathrm{Var}\bigl[\text{(ii)} \mid \lambda\bigr] &= s_0^2 \int_S \int_0^\infty\!\!\int_0^1 (\pi u)^2\,\bigl(1 - p_{\mathrm{camp}}(x,u,\pi)\bigr)\,G(\mathrm{d}\pi \mid u)\,\rho(\mathrm{d}u)\,\lambda(\mathrm{d}x).
\end{aligned}
$$

The posterior over $\lambda$ adds its own spread on top. Never report (i) + (iii) plus a point value for (ii): that silently drops the Poisson/CRM variance of everything below the detection limit.

Detection depends on both marks. Sources 2 and 4 have $p \approx 0$ and fall entirely into (ii). Source 3 is intermittent but bright: with $p_i \approx 0.9$ in each of 8 clear scenes, $1 - (1 - 0.3 \times 0.9)^{8} \approx 0.92$, so it is almost surely found, even though its time-averaged $123\ \mathrm{kg\,h^{-1}}$ is near the threshold. Collapsing it to one averaged rate would wrongly place it in (ii).

Each term has its own uncertainty:

=== "(i) Monitored points"

    Per-plume rate noise $\eta_k$ (log-normal, tens of percent) $\times$ persistence uncertainty from [§III.2](#iii-2).

=== "(ii) Undiscovered points"

    Spans **all** rates, each weighted by $1 - p_{\mathrm{camp}}$. In well-covered cells it is dominated by the tail of $\rho$ below $s_{\min}$. This is where the choice of Gamma versus generalised Gamma changes the answer by a large factor, and the only handle on it is bottom-up knowledge of small-source rates.

    Above the threshold, $1 - p_{\mathrm{camp}}$ stays positive wherever coverage is thin or persistence is low: an intermittent super-emitter in a rarely imaged cell can be missed and dominate the missing mass. Never truncate (ii) at $s_{\min}$.

=== "(iii) Diffuse"

    From the coarse-mapper stack; scales as $\sigma_r/\sqrt{n_{\mathrm{clear}}}$.

???+ example "Running example"
    The toy basin contains only the four listed sources, so its true (ii) is exactly sources 2 and 4. A real basin also has a tail of smaller sources, and the estimate of (ii) always includes one.

    | Term | Truth ($\mathrm{kg\,h^{-1}}$) | Estimate ($\mathrm{kg\,h^{-1}}$) | Note |
    |---|---|---|---|
    | (i) monitored | $0.9\cdot 120 + 0.3\cdot 410 = 231$ | $\approx 231$, $\pm\approx 30\,\%$ per source | sources 1, 3 |
    | (ii) undiscovered | $35 + 60 = 95$ | posterior of the missed-source sum over all rates, weighted by $1 - p_{\mathrm{camp}}$ (here mostly the tail below $100\ \mathrm{kg\,h^{-1}}$) | matches 95 only if the prior's missed mass happens to be right |
    | (iii) diffuse | $400$ | $\approx 400 \pm 60$ | 30-day mapper stack |
    | **total** | **726** | $\approx 631 + \text{(ii)}$ | **(ii) is set by the prior, not the data** |

    ```mermaid
    %%{init: {"themeVariables": {"pie1": "#00695c", "pie2": "#26a69a", "pie3": "#80cbc4", "pie4": "#b2dfdb", "pie5": "#ffb74d", "pieStrokeColor": "#ffffff", "pieOuterStrokeColor": "#9e9e9e", "pieTitleTextColor": "#8a8a8a", "pieLegendTextColor": "#8a8a8a", "pieSectionTextColor": "#ffffff"}}}%%
    pie showData title Where the true 726 kg/h comes from
        "(i) monitored points" : 231
        "(ii) undiscovered points" : 95
        "(iii) diffuse" : 400
    ```

**Attribution.** When two fine-imager plumes are near each other, DP/Pitman–Yor ([§II.5](02_objects.md#ii-5)) decides "one source or two", with assignment weights built from each candidate's observation rate $\pi_k\,p(x_k, s_k)$, not its emission fraction (see the warning in [§II.5](02_objects.md#ii-5)). Pitman–Yor is the safer choice given the heavy tail.

!!! tip "In `plumax`"
    The population-level machinery for this phase (TMTPP, POD-corrected totals, the missing-mass paradox) is laid out in [Tier V.D — Total emission estimation](../../design/06d_total_emission.md).

## III.4 Scale summary {#iii-4}

| | point (atom) | area (density) |
|---|---|---|
| **object** | $\mu_{\mathrm{pt}} = \sum_k s_k\delta_{x_k}$ | $\mu_{\mathrm{df}} = \int e\,\mathrm{d}x$ |
| **prior** | CRM (Gamma / GGP / Beta) | log-GP on $e$ |
| **seen by** | fine imager, 1 scene | coarse mapper, stack |
| **detection** | threshold on peak | averaging, $1/\sqrt{n}$ |
| **time** | sampled: $\pi_k s_k$ | averaged: persistent |
| **phase** | A, B, C | C |
| **fails when** | extent $\gtrsim \Delta x$ | extent $\lesssim \Delta x$ |

## III.5 The assumptions carrying the structure {#iii-5}

| # | Assumption | Fails when | Repaired by |
|:---:|---|---|---|
| 1 | *Independence across disjoint regions* ([§II.2](02_objects.md#ii-2)) | shared geology | Cox process ([§II.7](02_objects.md#ii-7)) |
| 2 | *Memorylessness in time* ([§II.10.2](02_objects.md#ii-10-2), Poisson) | persistent leaks, schedules, clustered upsets | on/off switching (leaks), renewal (schedules), Hawkes (clustered initiations) |
| 3 | *Point/area is fixed* | always: it is relative to $\Delta x$ ([§II.8.1](02_objects.md#ii-8-1)) | model at the fine scale and coarsen, never the reverse |
| 4 | *Overpasses are a fair sample of source state* ([§II.8.3](02_objects.md#ii-8-3)) | operations correlate with local overpass time | no fix from data alone |

!!! quote "The identifiability warning that runs through all three phases"
    The data see only products ($p\lambda$, $\pi_k p$, $\int u\,p(u)\,\rho(\mathrm{d}u)$), and **everything below the detection limit rests on calibration done outside the model.**
