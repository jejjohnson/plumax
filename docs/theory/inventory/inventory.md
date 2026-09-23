# From Satellite Detections to a Methane Inventory

**The real problem first, then the mathematics needed to describe it, with one running example carried to the end.**

A methane inventory has to answer three questions about a basin: *where* are the sources, *when* are they emitting, and *how much* do they emit in total. Satellites answer none of them directly. They return thinned, noisy, scale-dependent snapshots, and the gap between those snapshots and an inventory is where the mathematics lives.

This section is the theory behind [Tier V — Source population](../../design/06_tier5_population.md "Tier V design"). It builds, one idea at a time, the random objects (point processes, completely random measures, Gaussian processes, Cox processes, thinning) that describe the problem, then maps them back onto the three phases of inventory work.

## The task, in three phases

```mermaid
flowchart TD
    A["<b>Phase A — DISCOVERY</b><br/>Where are the sources we don't yet know about?<br/><i>random: LOCATIONS</i>"]
    B["<b>Phase B — MONITORING</b><br/>For sources we know, when are they emitting?<br/><i>random: STATE over TIME</i>"]
    C["<b>Phase C — ESTIMATION</b><br/>How much does each emit, and what is the total?<br/><i>random: RATES and their SUM</i>"]
    A --> B --> C
```

The three phases need different mathematical objects because **a different thing is uncertain in each**. Part II builds those objects; Part III maps them back onto the phases.

## How this section is organised

::::{grid} 1 1 2 2
:::{card} Part I — The real problem and the real data
:link: 01_inventory_problem.md
Point versus area sources, what the instruments measure, the data products, and the Permian running example used throughout.
:::
:::{card} Part II — Building the mathematical objects
:link: 02_inventory_objects.md
Measures, Poisson processes, completely random measures, fields, Cox processes, scale, thinning, and time, each introduced with the phase it serves.
:::
:::{card} Part III — Mapping onto the three phases
:link: 03_inventory_phases.md
Discovery, monitoring, and estimation of the basin total, with the running example worked to a number and the assumptions made explicit.
:::
:::{card} Scale summary
Point (atom) versus area (density), side by side: the prior, the instrument, the detection mechanism, and when each model fails. See [§III.4](#iii-4).
:::
::::

## Reading guide

Each object in Part II is tagged with the phase(s) it serves. If you care about one phase only, follow its thread:

::::{tab-set}
:::{tab-item} Phase A — Discovery
[Regions & measures](#ii-1) → [Poisson process](#ii-3) → [Fields](#ii-6) → [Cox process](#ii-7) → [Thinning](#ii-9) → [Phase A](#iii-1)
:::
:::{tab-item} Phase B — Monitoring
[Beta process](#ii-4-4) → [Temporal sampling](#ii-8-3) → [Observation operator](#ii-9-3) → [Time](#ii-10) → [Phase B](#iii-2)
:::
:::{tab-item} Phase C — Estimation
[CRMs](#ii-4) → [Generalised Gamma](#ii-4-3) → [Attribution](#ii-5) → [Area-source fields](#ii-6) → [Scale](#ii-8) → [Phase C](#iii-3)
:::
::::

::::{tip} Conventions
Notation and units are fixed once, in [§I.6](#i-6 "Notation and units"), and used unchanged on every page. Running-example boxes look like this:

:::{hint} Running example
:class: dropdown
The same $40 \times 40\ \mathrm{km}$ Permian region, four point sources and one diffuse field, from [§I.5](#i-5) to the basin total in [§III.3](#iii-3).
:::

and critical notes, where an assumption is physically wrong or an estimate is not what it looks like, are flagged as warnings.
::::
