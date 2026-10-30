"""Bayesian inference for the Gaussian plume forward model (NumPyro).

Infers the emission rate Q from downwind concentration observations, and
optionally the atmospheric stability class as a discrete latent. The forward
model is :func:`plumax.gauss_plume.plume.plume_concentration`;
observations are modelled as noisy around the forward prediction plus a
half-normal background. The noise is a :class:`plumax.gauss_plume.likelihoods.Noise`
(Gaussian by default; Student-t or heteroscedastic on request).

Identifiability target (Tier I design): the inference is good enough when
three overpasses with varied wind direction give CV(Q) ≤ 50 % and a single
overpass gives CV(Q) ≤ 100 % under a tight met wind prior.

Public surface
--------------
- ``gaussian_plume_model`` : NumPyro probabilistic model
- ``infer_emission_rate``   : NUTS runner returning a dict of posterior arrays
- ``multi_source_plume_model`` : masked-K model, up to K_max sources with an
  on/off indicator each
- ``infer_multi_source``    : NUTS + Gibbs runner returning a
  :class:`~plumax.gauss_plume.multi_source.MultiSourcePosterior`
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpy.typing import NDArray
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

from plumax.gauss_plume.dispersion import (
    BRIGGS_DISPERSION_PARAMS,
    STABILITY_CLASSES,
    get_dispersion_params,
)
from plumax.gauss_plume.likelihoods import GaussianNoise, Noise
from plumax.gauss_plume.multi_source import MaskedSources, MultiSourcePosterior
from plumax.gauss_plume.plume import plume_concentration


def _lognormal_from_moments(mean: float, std: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert real-scale (mean, std) into the log-scale (μ, σ) of a LogNormal.

    Standard method-of-moments inversion: for ``X ~ LogNormal(μ, σ)`` with
    ``E[X] = mean`` and ``Var[X] = std**2``,

        σ² = ln(1 + (std / mean)²),   μ = ln(mean) − σ²/2.
    """
    cv_sq = (std / mean) ** 2
    sigma_log_sq = jnp.log1p(cv_sq)
    sigma_log = jnp.sqrt(sigma_log_sq)
    mu_log = jnp.log(mean) - 0.5 * sigma_log_sq
    return mu_log, sigma_log


def gaussian_plume_model(
    observations: jnp.ndarray | None = None,
    receptor_coords: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | None = None,
    source_location: tuple[float, float, float] | None = None,
    wind_u: float | None = None,
    wind_v: float | None = None,
    stability_class: str = "C",
    prior_emission_rate_mean: float = 0.1,
    prior_emission_rate_std: float = 0.05,
    infer_stability: bool = False,
    background_prior_std: float = 5e-7,
    obs_noise_std: float = 5e-7,
    noise: Noise | None = None,
) -> jnp.ndarray:
    """NumPyro model for the steady-state Gaussian plume.

    Priors:
        emission_rate ~ LogNormal(μ_log, σ_log)                    [kg/s]
        stability_idx ~ Categorical(1/6, ..., 1/6)  (optional)     [A..F]
        background    ~ HalfNormal(σ_bg)                            [kg/m³]

    where ``(μ_log, σ_log)`` are the log-space parameters derived from the
    requested real-scale ``(prior_emission_rate_mean, prior_emission_rate_std)``
    via the method-of-moments inversion in :func:`_lognormal_from_moments`.

    Likelihood:
        obs ~ noise(f_plume(emission_rate, ...) + background)

    with ``noise`` defaulting to ``GaussianNoise(obs_noise_std)``, i.e.
    ``obs ~ Normal(f_plume + background, σ_obs)``.

    Parameters
    ----------
    observations : jnp.ndarray, shape (N,), optional
        Observed concentrations [kg/m³]. If None, runs in prior-predictive mode.
    receptor_coords : tuple of jnp.ndarray, optional
        ``(x, y, z)``, each shape (N,), in the fixed frame [m]. Required
        whenever ``observations`` is not None.
    source_location : tuple of float, optional
        ``(x, y, z)`` source coordinates [m]. Required whenever ``observations``
        is not None.
    wind_u, wind_v : float, optional
        Wind velocity components [m/s]. Required whenever ``observations``
        is not None.
    stability_class : str
        Stability class used if ``infer_stability=False``. Default ``'C'``.
    prior_emission_rate_mean, prior_emission_rate_std : float
        **Real-scale** prior mean / std for the emission rate [kg/s]. The
        implementation converts these to log-space parameters internally, so
        the realised prior has the requested mean and variance.
    infer_stability : bool
        If True, sample a categorical ``stability_idx`` latent over A-F.
    background_prior_std : float
        HalfNormal scale for the background concentration [kg/m³].
    obs_noise_std : float
        Observation noise σ [kg/m³] of the default Gaussian noise. Ignored
        when ``noise`` is given.
    noise : Noise, optional
        Observation-noise model, e.g. ``StudentTNoise(scale, df=5.0)`` or
        ``HeteroscedasticNoise(std)`` from :mod:`plumax.gauss_plume.likelihoods`.
        Default ``None`` is ``GaussianNoise(obs_noise_std)``.

    Raises
    ------
    ValueError
        If ``observations`` is provided but any of ``receptor_coords``,
        ``source_location``, ``wind_u``, or ``wind_v`` is ``None`` — that
        combination disconnects the likelihood from the forward model and
        would produce a meaningless posterior. Prior-predictive runs
        (``observations=None``) are allowed with any subset of forward-model
        inputs.
    """
    forward_inputs_present = (
        receptor_coords is not None
        and source_location is not None
        and wind_u is not None
        and wind_v is not None
    )
    if observations is not None and not forward_inputs_present:
        missing = [
            name
            for name, value in (
                ("receptor_coords", receptor_coords),
                ("source_location", source_location),
                ("wind_u", wind_u),
                ("wind_v", wind_v),
            )
            if value is None
        ]
        raise ValueError(
            "gaussian_plume_model: `observations` were provided but the "
            "forward model cannot be evaluated — missing inputs: "
            f"{', '.join(missing)}. Either pass all of "
            "(receptor_coords, source_location, wind_u, wind_v) or set "
            "observations=None for prior-predictive mode."
        )

    mu_log, sigma_log = _lognormal_from_moments(
        prior_emission_rate_mean, prior_emission_rate_std
    )
    emission_rate = numpyro.sample("emission_rate", dist.LogNormal(mu_log, sigma_log))

    if infer_stability:
        stability_probs = jnp.ones(len(STABILITY_CLASSES)) / len(STABILITY_CLASSES)
        stability_idx = numpyro.sample(
            "stability_idx", dist.Categorical(probs=stability_probs)
        )
        all_params = jnp.stack([BRIGGS_DISPERSION_PARAMS[c] for c in STABILITY_CLASSES])
        dispersion_params = all_params[stability_idx]
    else:
        dispersion_params = get_dispersion_params(stability_class)

    background = numpyro.sample("background", dist.HalfNormal(background_prior_std))

    if forward_inputs_present:
        x_obs, y_obs, z_obs = receptor_coords
        src_x, src_y, src_z = source_location
        predicted = plume_concentration(
            x_obs,
            y_obs,
            z_obs,
            src_x,
            src_y,
            src_z,
            wind_u,
            wind_v,
            emission_rate,
            dispersion_params,
        )
        predicted = predicted + background
    else:
        # Prior-predictive mode (observations is None and no forward inputs).
        predicted = background

    noise_model = GaussianNoise(std=obs_noise_std) if noise is None else noise
    return noise_model("obs", predicted, observations)


def infer_emission_rate(
    observations: NDArray,
    observation_coords: tuple[NDArray, NDArray, NDArray],
    source_location: tuple[float, float, float],
    wind_speed: float,
    wind_direction: float,
    stability_class: str = "C",
    infer_stability: bool = False,
    prior_mean: float = 0.1,
    prior_std: float = 0.05,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
    print_summary: bool = False,
    obs_noise_std: float = 5e-7,
    background_prior_std: float = 5e-7,
    noise: Noise | None = None,
) -> dict[str, NDArray]:
    """Infer the plume emission rate via NUTS.

    Wraps :func:`gaussian_plume_model` with input validation, wind-from-angle
    to (u, v) conversion, and posterior unpacking into NumPy arrays.

    Identifiability target: three overpasses with varied wind direction
    should give a posterior CV(Q) ≤ 50 %, and a single overpass CV(Q) ≤ 100 %
    under a tight met wind prior (Tier I design, "Identifiability —
    quantitative target"). A wider posterior means the data do not constrain
    Q yet.

    Parameters
    ----------
    observations : ndarray, shape (N,)
        Observed concentrations [kg/m³].
    observation_coords : tuple of ndarray
        ``(x, y, z)``, each shape (N,), in the fixed frame [m].
    source_location : tuple of float, (3,)
        ``(x, y, z)`` source coordinates [m].
    wind_speed : float
        Wind speed magnitude [m/s]. Must be > 0.
    wind_direction : float
        Wind direction [degrees from North, meteorological "from" convention].
    stability_class : str
        Stability class used if ``infer_stability=False``. Default ``'C'``.
    infer_stability : bool
        If True, jointly sample a categorical stability-class latent.
    prior_mean, prior_std : float
        Prior parameters for the emission rate [kg/s].
    num_warmup, num_samples, num_chains : int
        NUTS sampler configuration.
    seed : int
        PRNG seed.
    progress_bar : bool
        If True, show the NumPyro progress bar. Default False.
    print_summary : bool
        If True, call ``mcmc.print_summary()`` after the run.
    obs_noise_std : float
        Observation-noise σ for the Gaussian likelihood [kg/m³]. Set this to
        the sensor noise so the posterior width reflects the instrument rather
        than the hard-coded default. Ignored when ``noise`` is given.
    background_prior_std : float
        HalfNormal scale for the background-concentration prior [kg/m³].
    noise : Noise, optional
        Observation-noise model (see :func:`gaussian_plume_model`). Default
        ``None`` is ``GaussianNoise(obs_noise_std)``, today's likelihood.

    Returns
    -------
    samples : dict[str, ndarray]
        Posterior draws keyed by site name: ``'emission_rate'``,
        ``'background'``, and (if ``infer_stability``) ``'stability_idx'``.

    Raises
    ------
    ValueError
        On invalid stability class, non-positive wind speed, shape mismatches
        between ``observations`` and ``observation_coords``, or empty data.
    """
    if not (wind_speed > 0.0):
        raise ValueError(
            f"infer_emission_rate: `wind_speed` must be > 0 (got {wind_speed!r})"
        )
    if not infer_stability and stability_class not in BRIGGS_DISPERSION_PARAMS:
        raise ValueError(
            f"infer_emission_rate: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    obs = np.asarray(observations)
    if obs.size == 0:
        raise ValueError("infer_emission_rate: `observations` must contain ≥ 1 point")
    if len(observation_coords) != 3:
        raise ValueError(
            "infer_emission_rate: `observation_coords` must be (x, y, z); "
            f"got length {len(observation_coords)}"
        )
    coords = tuple(np.asarray(c) for c in observation_coords)
    for axis_name, arr in zip("xyz", coords, strict=False):
        if arr.shape != obs.shape:
            raise ValueError(
                f"infer_emission_rate: `observation_coords` axis '{axis_name}' "
                f"has shape {arr.shape} ≠ observations shape {obs.shape}"
            )
    if not (prior_mean > 0.0):
        raise ValueError(
            f"infer_emission_rate: `prior_mean` must be > 0 (got {prior_mean!r})"
        )
    if not (prior_std > 0.0):
        raise ValueError(
            f"infer_emission_rate: `prior_std` must be > 0 (got {prior_std!r})"
        )
    if not (obs_noise_std > 0.0):
        raise ValueError(
            f"infer_emission_rate: `obs_noise_std` must be > 0 (got {obs_noise_std!r})"
        )
    if not (background_prior_std > 0.0):
        raise ValueError(
            "infer_emission_rate: `background_prior_std` must be > 0 "
            f"(got {background_prior_std!r})"
        )

    theta_rad = np.deg2rad(wind_direction)
    wind_u = float(-wind_speed * np.sin(theta_rad))
    wind_v = float(-wind_speed * np.cos(theta_rad))

    obs_jax = jnp.asarray(obs)
    coords_jax = tuple(jnp.asarray(c) for c in coords)

    inner = NUTS(gaussian_plume_model)
    # Discrete latents can't be handled by NUTS on their own without
    # `funsor` enumeration. `DiscreteHMCGibbs` wraps NUTS with a Gibbs
    # step over the categorical stability-class site, avoiding the extra
    # dependency while giving a valid joint sampler.
    kernel = DiscreteHMCGibbs(inner) if infer_stability else inner
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        observations=obs_jax,
        receptor_coords=coords_jax,
        source_location=source_location,
        wind_u=wind_u,
        wind_v=wind_v,
        stability_class=stability_class,
        prior_emission_rate_mean=prior_mean,
        prior_emission_rate_std=prior_std,
        infer_stability=infer_stability,
        obs_noise_std=obs_noise_std,
        background_prior_std=background_prior_std,
        noise=noise,
    )
    if print_summary:
        mcmc.print_summary()

    return {k: np.asarray(v) for k, v in mcmc.get_samples().items()}


# ── Masked-K multi-source ────────────────────────────────────────────────────


def _check_multi_source_layout(
    k_max: int,
    candidate_locations: jnp.ndarray | None,
    source_bounds: tuple[tuple[float, float], tuple[float, float]] | None,
    prior_active: float,
) -> None:
    if (candidate_locations is None) == (source_bounds is None):
        raise ValueError(
            "multi_source_plume_model: give exactly one of `candidate_locations` "
            "(fixed slots) or `source_bounds` (free, x-ordered slots)"
        )
    if k_max < 1:
        raise ValueError(f"multi_source_plume_model: `k_max` must be ≥ 1, got {k_max}")
    if candidate_locations is not None:
        shape = np.shape(candidate_locations)
        if shape != (k_max, 2):
            raise ValueError(
                "multi_source_plume_model: `candidate_locations` must have shape "
                f"(k_max, 2) = ({k_max}, 2), got {shape}"
            )
    if source_bounds is not None:
        (x_low, x_high), (y_low, y_high) = source_bounds
        if not (x_low < x_high and y_low < y_high):
            raise ValueError(
                "multi_source_plume_model: `source_bounds` needs low < high on "
                f"both axes, got {source_bounds!r}"
            )
    if not (0.0 < prior_active <= 1.0):
        raise ValueError(
            "multi_source_plume_model: `prior_active` must be in (0, 1], "
            f"got {prior_active!r}"
        )


def multi_source_plume_model(
    observations: jnp.ndarray | None = None,
    *,
    receptor_coords: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    wind_u: float,
    wind_v: float,
    k_max: int = 10,
    candidate_locations: jnp.ndarray | None = None,
    source_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    source_height: float = 0.0,
    stability_class: str = "C",
    prior_emission_rate_mean: float = 0.1,
    prior_emission_rate_std: float = 0.05,
    prior_active: float = 0.3,
    background_prior_std: float = 5e-7,
    obs_noise_std: float = 5e-7,
    noise: Noise | None = None,
) -> jnp.ndarray:
    """Masked-K NumPyro model: up to K_max plumes, each switched on or off.

    Priors, for each slot k = 1 … K_max:
        a_k ~ Bernoulli(prior_active)                               [on/off]
        Q_k ~ LogNormal(μ_log, σ_log)                               [kg/s]
        (x_k, y_k) = candidate k, or x ordered uniform and y uniform in
                     ``source_bounds``                               [m]
        background ~ HalfNormal(σ_bg)                               [kg/m³]

    Likelihood:
        obs ~ noise(Σ_k a_k Q_k c₁(x_k, y_k) + background)

    with ``noise`` defaulting to ``GaussianNoise(obs_noise_std)``. The active
    count K = Σ_k a_k is the posterior over the number of sources.

    Label switching: with free locations the slots are exchangeable, so their
    x positions are drawn as the order statistics of K_max uniforms on
    ``[x_low, x_high]`` (``x_low + (x_high − x_low) · cumsum(w)[:K_max]`` with
    ``w ~ Dirichlet(1, …, 1)`` of length K_max + 1), which pins slot k to the
    k-th source from the west. Fixed candidate locations (known facilities)
    have no symmetry to break.

    Promotion to RJMCMC (Tier I / IV design): masked-K caps K at K_max. Track
    :meth:`MultiSourcePosterior.saturated_fraction` per overpass; when more
    than 5 % of overpasses put their posterior mode on K = K_max, the cap is
    binding and the trans-dimensional (RJMCMC) model is due.

    Parameters
    ----------
    observations : jnp.ndarray, shape (N,), optional
        Observed concentrations [kg/m³]; ``None`` for predictive draws.
    receptor_coords : tuple of jnp.ndarray
        ``(x, y, z)``, each shape (N,), in the fixed frame [m].
    wind_u, wind_v : float
        Wind velocity components [m/s].
    k_max : int
        Number of slots K_max (default 10, the design's v1 cap).
    candidate_locations : jnp.ndarray, shape (k_max, 2), optional
        Fixed ``(x, y)`` per slot, e.g. inventory facilities.
    source_bounds : ((x_low, x_high), (y_low, y_high)), optional
        Box the free slot locations are drawn from. Give exactly one of this
        and ``candidate_locations``.
    source_height : float
        Release height shared by every slot [m].
    stability_class : str
        Pasquill–Gifford class ``'A'`` … ``'F'``.
    prior_emission_rate_mean, prior_emission_rate_std : float
        Real-scale prior moments of each Q_k [kg/s].
    prior_active : float
        Prior probability that a slot is on, in (0, 1]. ``1.0`` keeps every
        slot on (no discrete sites), which with K_max = 1 is the single-source
        model.
    background_prior_std : float
        HalfNormal scale of the background [kg/m³].
    obs_noise_std : float
        σ of the default Gaussian noise; ignored when ``noise`` is given.
    noise : Noise, optional
        Observation-noise model from :mod:`plumax.gauss_plume.likelihoods`.

    Raises
    ------
    ValueError
        On an invalid slot layout, ``k_max < 1`` or ``prior_active`` outside
        (0, 1].
    """
    _check_multi_source_layout(k_max, candidate_locations, source_bounds, prior_active)
    dispersion_params = get_dispersion_params(stability_class)

    if prior_active < 1.0:
        # a plate, not to_event: the Gibbs step enumerates each a_k's support
        with numpyro.plate("source", k_max):
            active = numpyro.sample("active", dist.Bernoulli(probs=prior_active))
    else:
        active = numpyro.deterministic("active", jnp.ones(k_max))

    mu_log, sigma_log = _lognormal_from_moments(
        prior_emission_rate_mean, prior_emission_rate_std
    )
    emission_rate = numpyro.sample(
        "emission_rate", dist.LogNormal(mu_log, sigma_log).expand([k_max]).to_event(1)
    )

    if candidate_locations is not None:
        locations = jnp.asarray(candidate_locations)
        source_x = numpyro.deterministic("source_x", locations[:, 0])
        source_y = numpyro.deterministic("source_y", locations[:, 1])
    else:
        (x_low, x_high), (y_low, y_high) = source_bounds
        spacing = numpyro.sample("x_spacing", dist.Dirichlet(jnp.ones(k_max + 1)))
        # x_k = x_low + (x_high − x_low) Σ_{j ≤ k} w_j: ordered west to east
        ordered = jnp.cumsum(spacing)[:k_max]
        source_x = numpyro.deterministic("source_x", x_low + (x_high - x_low) * ordered)
        source_y = numpyro.sample(
            "source_y", dist.Uniform(y_low, y_high).expand([k_max]).to_event(1)
        )

    background = numpyro.sample("background", dist.HalfNormal(background_prior_std))

    sources = MaskedSources(
        emission_rate=emission_rate,
        x=source_x,
        y=source_y,
        active=jnp.asarray(active, dtype=emission_rate.dtype),
    )
    predicted = sources.concentration(
        receptor_coords,
        source_z=source_height,
        wind_u=wind_u,
        wind_v=wind_v,
        dispersion_params=dispersion_params,
    )
    predicted = predicted + background

    noise_model = GaussianNoise(std=obs_noise_std) if noise is None else noise
    return noise_model("obs", predicted, observations)


def infer_multi_source(
    observations: NDArray,
    observation_coords: tuple[NDArray, NDArray, NDArray],
    *,
    wind_speed: float,
    wind_direction: float,
    k_max: int = 10,
    candidate_locations: NDArray | None = None,
    source_bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    source_height: float = 0.0,
    stability_class: str = "C",
    prior_mean: float = 0.1,
    prior_std: float = 0.05,
    prior_active: float = 0.3,
    obs_noise_std: float = 5e-7,
    background_prior_std: float = 5e-7,
    noise: Noise | None = None,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    seed: int = 0,
    progress_bar: bool = False,
) -> MultiSourcePosterior:
    """Infer up to K_max sources and their on/off indicators.

    Wraps :func:`multi_source_plume_model`: NUTS for the continuous sites, and
    a Gibbs step (``DiscreteHMCGibbs``) over the activity indicators, as for
    the stability class in :func:`infer_emission_rate`.

    Parameters
    ----------
    observations : ndarray, shape (N,)
        Observed concentrations [kg/m³].
    observation_coords : tuple of ndarray
        ``(x, y, z)``, each shape (N,), in the fixed frame [m].
    wind_speed : float
        Wind speed [m/s], > 0.
    wind_direction : float
        Wind direction [degrees from North, meteorological "from" convention].
    k_max, candidate_locations, source_bounds, source_height, stability_class
        Slot layout; see :func:`multi_source_plume_model`.
    prior_mean, prior_std : float
        Real-scale prior moments of each Q_k [kg/s].
    prior_active, obs_noise_std, background_prior_std, noise
        See :func:`multi_source_plume_model`.
    num_warmup, num_samples, num_chains, seed : int
        Sampler configuration.
    progress_bar : bool
        Show NumPyro's progress bar.

    Returns
    -------
    posterior : MultiSourcePosterior
        Draws of every slot; ``count_probabilities()`` is the posterior over
        the active count and ``per_source(k)`` the slot's summary.

    Raises
    ------
    ValueError
        On a non-positive wind speed, an unknown stability class, mismatched
        shapes, non-positive prior or noise scales, or an invalid slot layout.
    """
    if not (wind_speed > 0.0):
        raise ValueError(
            f"infer_multi_source: `wind_speed` must be > 0 (got {wind_speed!r})"
        )
    if stability_class not in BRIGGS_DISPERSION_PARAMS:
        raise ValueError(
            f"infer_multi_source: `stability_class` must be one of "
            f"{STABILITY_CLASSES}, got {stability_class!r}"
        )
    obs = np.asarray(observations)
    if obs.size == 0:
        raise ValueError("infer_multi_source: `observations` must contain ≥ 1 point")
    if len(observation_coords) != 3:
        raise ValueError(
            "infer_multi_source: `observation_coords` must be (x, y, z); "
            f"got length {len(observation_coords)}"
        )
    coords = tuple(np.asarray(c) for c in observation_coords)
    for axis_name, arr in zip("xyz", coords, strict=True):
        if arr.shape != obs.shape:
            raise ValueError(
                f"infer_multi_source: `observation_coords` axis '{axis_name}' "
                f"has shape {arr.shape} ≠ observations shape {obs.shape}"
            )
    for name, value in (
        ("prior_mean", prior_mean),
        ("prior_std", prior_std),
        ("obs_noise_std", obs_noise_std),
        ("background_prior_std", background_prior_std),
    ):
        if not (value > 0.0):
            raise ValueError(
                f"infer_multi_source: `{name}` must be > 0 (got {value!r})"
            )
    _check_multi_source_layout(k_max, candidate_locations, source_bounds, prior_active)

    theta_rad = np.deg2rad(wind_direction)
    wind_u = float(-wind_speed * np.sin(theta_rad))
    wind_v = float(-wind_speed * np.cos(theta_rad))

    inner = NUTS(multi_source_plume_model)
    kernel = DiscreteHMCGibbs(inner) if prior_active < 1.0 else inner
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    candidates = (
        None if candidate_locations is None else jnp.asarray(candidate_locations)
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        jnp.asarray(obs),
        receptor_coords=tuple(jnp.asarray(c) for c in coords),
        wind_u=wind_u,
        wind_v=wind_v,
        k_max=k_max,
        candidate_locations=candidates,
        source_bounds=source_bounds,
        source_height=source_height,
        stability_class=stability_class,
        prior_emission_rate_mean=prior_mean,
        prior_emission_rate_std=prior_std,
        prior_active=prior_active,
        background_prior_std=background_prior_std,
        obs_noise_std=obs_noise_std,
        noise=noise,
    )
    samples = mcmc.get_samples()
    num_draws = samples["emission_rate"].shape[0]
    if "active" in samples:
        active = np.asarray(samples["active"], dtype=bool)
    else:
        active = np.ones((num_draws, k_max), dtype=bool)
    if "source_x" in samples:
        source_x = np.asarray(samples["source_x"])
        source_y = np.asarray(samples["source_y"])
    else:
        # fixed candidates are deterministic sites, which get_samples omits
        locations = np.asarray(candidate_locations)
        source_x = np.broadcast_to(locations[:, 0], (num_draws, k_max))
        source_y = np.broadcast_to(locations[:, 1], (num_draws, k_max))
    return MultiSourcePosterior(
        emission_rate=np.asarray(samples["emission_rate"]),
        x=source_x,
        y=source_y,
        active=active,
        background=np.asarray(samples["background"]),
    )
