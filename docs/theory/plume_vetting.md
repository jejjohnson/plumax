---
title: "Plume vetting — a spectral test for matched-filter candidates"
---

# Plume vetting

:::{important} Status: in progress
This page describes `plumax.vetting` ([#153](https://github.com/jejjohnson/plumax/issues/153)): [`select_seeds`](xref:api#plumax.vetting.select_seeds) and [`background_pool`](xref:api#plumax.vetting.background_pool) ([#154](https://github.com/jejjohnson/plumax/issues/154)), [`null_masks`](xref:api#plumax.vetting.null_masks) and [`null_pvalue`](xref:api#plumax.vetting.null_pvalue) ([#155](https://github.com/jejjohnson/plumax/issues/155)), [`vet_candidate`](xref:api#plumax.vetting.vet_candidate) and [`VettingResult`](xref:api#plumax.vetting.VettingResult) ([#156](https://github.com/jejjohnson/plumax/issues/156)). It runs radtrax's plume-agnostic [absorption-shape test](https://jejjohnson.github.io/radtrax/absorption-test/) (`radtrax.retrieval.absorption`) on plume candidates; the maths of that test (the pair ratio, the transmittance from the forward model, the profile fit and $D_{\text{norm}}$) is on radtrax's page and is not repeated here. The [plume vetting notebook](../notebooks/vetting/01_vetting_one_scene.ipynb) runs every function on one scene; the numbers below are that notebook's.
:::

A matched filter scores each pixel by one projection of its spectrum onto methane's target, so anything that removes light under methane's 2.3 μm band scores: carbonate and clay soils, salt pans, dark water, shadows, striping. A segmentation network on the matched-filter map learns plume shapes, which a plume-shaped confounder also has. **Plume vetting** (Xiang et al. [^xiang2024]; Růžička et al. [^ruzicka2026]) goes back to the radiances behind a candidate. It pairs the candidate's strongest pixels with pixels outside it that have the same surface, divides them, and asks whether the ratio has the shape of methane's transmittance. EMIT's daily methane digest alerts without review when the misfit score $D_{\text{norm}}$ is below 0.3.

The work splits along a package boundary, "radiative transfer does not know about plumes". **radtrax** answers a question about spectra: given two sets of pixels, how similar is the ratio of their paired mean radiances to a gas's transmittance? It returns the fitted column $\widehat{\Delta X}$ and $D_{\text{norm}}$, and knows nothing of candidates. **plumax** answers the questions about plume objects: which of a candidate's pixels are in (the seeds) and which are out (the background pool), how this scene's clutter scores in the candidate's shape (the null), and what to decide.

Vetting is a second test on the same data, not an independent one: its pixels are chosen by the matched filter and come from the same cube. It is more selective because it tests the shape of the absorption across a window of bands, where the matched filter tests one projection.

[^xiang2024]: Xiang, C., et al. (2024). A novel spectroscopic method for vetting methane point sources. AGU Fall Meeting, abstract A21K-1903.

[^ruzicka2026]: Růžička, V., Thompson, D. R., Fahlen, J. E., et al. (2026). Fully automatic trace gas plume detection. arXiv:2605.03372.

[^pv]: EMIT Science Data System (2025). plume-vetting, revision `47fc6ad`. https://github.com/emit-sds/plume-vetting (Apache-2.0).

[^phipson2010]: Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.

## 1. The test plumax runs

For in-pixels $I$ and out-pixels $O$ of one radiance cube, radtrax pairs them one to one on the bands the gas does not absorb, keeps the best-matched half, and divides the pairs' mean radiances into a ratio spectrum $R(\lambda) \approx \eta(\lambda) E(\lambda; \Delta X)$, with $\eta$ the ratio of the paired surfaces and $E$ the gas's band transmittance from radtrax's forward model. It fits $R$ with a Legendre continuum times $E$, profiling over a grid of $\Delta X$ refined between its nodes, and scores the fit with Xiang et al.'s eq. 11, $D_{\text{norm}}$: 0 for a perfect fit, about 1 when nothing gas-shaped is left. The derivations, the curve of growth and the reasons for each setting are on radtrax's [absorption-shape test](https://jejjohnson.github.io/radtrax/absorption-test/) page.

Everything below is how plumax chooses $I$ and $O$ from a candidate, and what it does with the scores.

## 2. Seeds and the background pool

Write $M$ for the candidate's pixels with a finite matched-filter score $\widehat{\Delta X}_{\text{MF}}$, and $q$ for the score's 99th percentile over $M$. The seeds are the $k = 40$ strongest pixels of $M$ at or below the cap, grown by one pixel inside $M$ ([`select_seeds`](xref:api#plumax.vetting.select_seeds)):

$$
I_0 = \operatorname{top}_k\{p \in M : \widehat{\Delta X}_{\text{MF}}(p) \le q\},
\qquad
I = \operatorname{grow}^{g}(I_0), \quad \operatorname{grow}(A) = (A \oplus 3{\times}3) \cap M ,
$$ (eq-vet-seeds)

with $\oplus\, 3{\times}3$ one binary dilation by the 3 × 3 square and $g = 1$. The background pool is every pixel within a box of $b$ pixels around the seeds' bounding box, outside every candidate $M_c$ of the scene, with a small matched-filter score ([`background_pool`](xref:api#plumax.vetting.background_pool)):

$$
O = \Big( \operatorname{box}_b(I) \setminus I \setminus \textstyle\bigcup_c M_c \Big) \cap \{p : |\widehat{\Delta X}_{\text{MF}}(p)| < \tau_{\text{bg}}\} ,
$$ (eq-vet-pool)

with $\tau_{\text{bg}}$ the window's `mf_background_mol_m2` (30 ppm·m for EMIT's CH₄ window) and $b = 200$. So $I \subseteq M$, $O \cap I = \emptyset$ and $O \cap M_c = \emptyset$ for every candidate. These are the reference package's settings [^pv], `pv/config/config.yaml`; the two sets then go to radtrax's pairing as flat pixel indices.

:::{dropdown} Why each setting
**Seeds come from the matched filter.** The depth of methane's absorption in $R$ grows with $\Delta X$, so the strongest pixels give the deepest and best-resolved transmittance.

**The seeds' noise is not innocent.** Choosing the $k$ largest scores selects pixels whose noise happens to point along methane's target, so the seeds' spectra are biased toward methane's shape by the selection itself. This is one reason vetting is not independent of the matched filter. Growing the seeds by one pixel adds neighbours that were not chosen for their own noise, and it keeps the in-set a connected region rather than a scatter of isolated pixels.

**The cap at the 99th percentile.** The reference code does not say why it caps there. The cap drops the mask's most extreme scores, which is where a single bad or saturated pixel would land.

**Growth is one dilation.** The reference code grows the seeds with a Gaussian filter of σ = 1 and keeps pixels above 0.05. On a binary mask that is exactly one 3 × 3 dilation, which plumax computes as a max-pool in JAX.

**The background pool.** A small matched-filter score keeps plume pixels out of $L_{\text{out}}$, where they would cancel the signal being tested. The box keeps the pool near the seeds, so that each pair shares its sun, elevation and water vapour, which is what lets them cancel in the ratio. Every candidate of the scene is kept out, not only this one, so a second plume or confounder never serves as background.
:::

## 3. A null from the same scene

The honest null is this scene's clutter with no plume. Vet $S$ copies $M_s$ of the mask, each rotated and translated to where there is no candidate ([`null_masks`](xref:api#plumax.vetting.null_masks)), exactly as the candidate was vetted, and report ([`null_pvalue`](xref:api#plumax.vetting.null_pvalue))

$$
\hat p = \frac{1}{S'} \sum_{s=1}^{S'} \mathbb{1}\left[ D_{\text{norm}}(M_s) \le D_{\text{norm}}(M) \right]
$$ (eq-vet-null)

over the $S'$ copies with a finite score. $D_{\text{norm}}$'s threshold depends on the band set, the continuum's degree and the ratio's noise, so $\hat p$ travels between sensors where 0.3 does not.

Each copy draws an angle $\theta \sim U[0°, 360°)$ and rotates $M$ about the image centre (nearest neighbour); a rotation that keeps fewer than 90 % of the pixels is discarded. Among the integer shifts $\Delta$ that keep the rotated mask $M_\theta$ inside the image, the free ones are those with

$$
(M_\theta + \Delta) \cap \Big( M \cup \textstyle\bigcup_c M_c \Big) = \emptyset ,
$$ (eq-vet-free)

found exactly as the zeros of the correlation of the candidates with $M_\theta$'s bounding box; one free shift is drawn uniformly. Randomness comes only from the caller's `jax.random` key, so the same key gives the same copies.

:::{dropdown} Reading $\hat p$
**What it estimates.** $\hat p$ is the fraction of null copies that score at least as well as the candidate. A small $\hat p$ says that a plume-free mask of the candidate's shape, placed elsewhere in this scene, rarely fits methane this well.

**Its resolution.** With $S$ copies, $\hat p$ moves in steps of $1/S$. $\hat p = 0$ says only that no copy did as well, not that the probability is zero. The usual Monte Carlo form, $(1 + S'\hat p) / (1 + S')$, is never zero and is a valid p-value when the candidate and the copies are exchangeable under the null [^phipson2010]. With $S = 100$, candidate A's $\hat p = 0.00$ is $1/101 \approx 0.01$ in that form. For a score where higher is better, such as a likelihood ratio, pass both scores negated.

**What the null samples.** Each copy runs the whole chain, seeds to $D_{\text{norm}}$, so the selection of seeds by the matched filter happens inside the null too. The selection of the mask itself does not. Copies sit wherever there is no candidate, so they sample this scene's background in the mask's shape, not background that the matched filter flagged. A confounder is unusual background by construction, which is why the null calibrates the score but cannot replace the fit's test of shape.

**The copies are not independent.** They overlap one another and draw on the same background pool, so the tail of the null is estimated less precisely than $S$ independent draws would estimate it.
:::

## 4. One candidate

[`vet_candidate`](xref:api#plumax.vetting.vet_candidate) runs the steps in their fixed order with the window's band selection: the seeds {eq}`eq-vet-seeds` and the pool {eq}`eq-vet-pool`, radtrax's pairing on the window's pairing bands, the ratio on its fitting bands, the fit and $D_{\text{norm}}$. Its [`VettingResult`](xref:api#plumax.vetting.VettingResult) carries $\widehat{\Delta X}$ in mol/m², $D_{\text{norm}}$, the number of pairs, the ratio and the fit (continuum $P$ and transmittance $E$, the spectra an analyst plots as $R/P$ against $E$), and the continuum-only residual at $\Delta X = 0$ beside the fit's residual at $\widehat{\Delta X}$.

$\widehat{\Delta X}$ is the column of the *mean* transmittance over the pairs, not the mean column: $E$ is convex in $\Delta X$, so it sits below the mean of the truth over the seeds. It is a score, not a retrieval; the column for an emission rate comes from a per-pixel retrieval of the vetted plume.

## 5. One scene

The [notebook](../notebooks/vetting/01_vetting_one_scene.ipynb) builds a 64 × 64 EMIT-like scene: 30 bands outside methane's band, 9 inside it (2296–2356 nm), three surface classes, noise at SNR 300. Candidate A is a methane plume; candidate B is a plume-shaped patch of a surface whose reflectance dips by 6 % at 2335 nm, as a carbonate's does.

| | Matched filter, mean | $\widehat{\Delta X}$ | $D_{\text{norm}}$ | $\hat p$ |
|---|---|---|---|---|
| A, plume | 593 ppm·m | 1,741 ppm·m | 0.14 | 0.00 |
| B, carbonate | 1,162 ppm·m | 0 | 1.00 | 0.45 |

The matched filter scores B at twice A; vetting reverses the order. A's column is 93 % of the truth over its kept pixels: about 3 points go to averaging transmittances, the rest to paired surfaces that do not match exactly. With a constant absorption coefficient A's column would be 1,691 ppm·m. B's $\hat p$ was 0.43 when the null rejected random shifts; exact placement {eq}`eq-vet-free` draws other copies and gives 0.45 (A's is 0 either way).

## 6. The decision

`plumax.vetting` returns scores, not verdicts. Three rules are on the table ([#158](https://github.com/jejjohnson/plumax/issues/158)): a fixed cut on $D_{\text{norm}}$ (EMIT's 0.3), which depends on the sensor's bands and noise; the scene null's $\hat p$ {eq}`eq-vet-null`, which calibrates the score against this scene's clutter; and a likelihood ratio between the fits with and without the gas, from `VettingResult`'s two residuals. Until #158 settles it, report $\hat p$ with every score.

## 7. What plumax computes

| Step | Equation | Function | Package | Issue |
|---|---|---|---|---|
| Seeds | {eq}`eq-vet-seeds` | [`select_seeds`](xref:api#plumax.vetting.select_seeds) | plumax | [#154](https://github.com/jejjohnson/plumax/issues/154) |
| Background pool | {eq}`eq-vet-pool` | [`background_pool`](xref:api#plumax.vetting.background_pool) | plumax | [#154](https://github.com/jejjohnson/plumax/issues/154) |
| A gas's fitting and pairing bands | | `AbsorptionWindow` | radtrax | [radtrax#155](https://github.com/jejjohnson/radtrax/issues/155) |
| Transmittance on a grid of $\Delta X$ | | `transmittance_table` | radtrax | [radtrax#154](https://github.com/jejjohnson/radtrax/issues/154) |
| One-to-one pairs and the ratio | | `pair_pixels`, `ratio_spectrum` | radtrax | [radtrax#149](https://github.com/jejjohnson/radtrax/issues/149) |
| Profile fit and score | | `fit_transmittance`, `TransmittanceFit`, `d_norm` | radtrax | [radtrax#150](https://github.com/jejjohnson/radtrax/issues/150) |
| One candidate | | [`vet_candidate`](xref:api#plumax.vetting.vet_candidate), [`VettingResult`](xref:api#plumax.vetting.VettingResult) | plumax | [#156](https://github.com/jejjohnson/plumax/issues/156) |
| The scene null | {eq}`eq-vet-free`, {eq}`eq-vet-null` | [`null_masks`](xref:api#plumax.vetting.null_masks), [`null_pvalue`](xref:api#plumax.vetting.null_pvalue) | plumax | [#155](https://github.com/jejjohnson/plumax/issues/155) |

The radtrax names are in `radtrax.retrieval.absorption` ([API](https://jejjohnson.github.io/radtrax/reference/retrieval/), [theory](https://jejjohnson.github.io/radtrax/absorption-test/)). `plumax.vetting` is an L1 module on arrays; the operator on carriers, `plumax.geo.PlumeVetting`, follows ([#157](https://github.com/jejjohnson/plumax/issues/157)).

## 8. Where plumax departs from the reference code

- **Exactly $k$ seeds.** The reference keeps every pixel at or above the $k$-th score, more than $k$ on ties; plumax takes the top $k$, breaking ties by the lower flat index.
- **Non-finite scores.** A pixel with a non-finite matched-filter score is never a seed and never in the pool, and growth stays inside $M$'s finite pixels.
- **The null avoids every candidate.** The reference package draws 100 rotated and translated copies that avoid only their own plume's boundary, and records their scores without a decision; plumax's copies avoid every candidate, the mask itself included, and $\hat p$ is returned with the scores.
- **Exact placement.** Rather than rejecting random shifts that hit a candidate (which needs about 90,000 attempts for candidate A's 100 copies), plumax finds each angle's free shifts exactly and draws one uniformly. Given its angle, a copy is uniform over the free places; rejection would weight angles by their fraction of free places.
