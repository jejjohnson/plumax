"""Mark registry: the population size distribution f(Q | θ) of emission rates.

One parameterised density per family, shared by the population fit, the
missing-mass simulator and the total-emission estimator. Every family has
the same four methods, with θ passed by keyword:

* ``log_prob(q, **θ)``, log f(q | θ), −∞ outside the support;
* ``sample(key, n, **θ)``, n draws of Q;
* ``truncated_mean(q_min, q_max, **θ)``, the partial first moment
  ∫ q f(q | θ) dq over [q_min, q_max], which the totals need;
* ``numpyro_prior(name)``, NumPyro sites for θ under the family's default
  priors, returned as the keyword dict the other three take.

Families, by registry name:

* ``"powerlaw"``: f(q) = (α − 1)/q_min · (q/q_min)^(−α) on q ≥ q_min, α > 1.
* ``"lognormal"``: log Q ~ N(μ, σ²).
* ``"lognormal_pareto"``: a lognormal body below Q_break spliced to a Pareto
  tail α Q_breakᵅ q^(−α−1) above it, the body weight set so the density is
  continuous at Q_break.
* ``"mixture_lognormal"``: Σₖ wₖ LN(q; μₖ, σₖ²).

Mark integrals against a lognormal per-event posterior (the Gaussian regime
of the importance correction): for ``lognormal`` and ``mixture_lognormal``
the integrand is smooth in log Q, so Gauss–Hermite quadrature in log space
converges quickly; ``powerlaw`` has the closed form
∫ q^(−α) LN(q; μ, σ²) dq = exp(−αμ + α²σ²/2) when the lognormal sits well
above q_min, and quadrature otherwise; ``lognormal_pareto`` has a kink at
Q_break and a heavy tail, so it needs quadrature or Pareto importance
sampling.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri
from jax.scipy.stats import norm
from jaxtyping import Array, ArrayLike, Float


@runtime_checkable
class Mark(Protocol):
    """A mark family f(Q | θ), with θ passed by keyword."""

    def log_prob(self, q: ArrayLike, **theta: Any) -> Array:
        """log f(q | θ)."""
        ...

    def sample(self, key: Array, n: int, **theta: Any) -> Array:
        """n draws of Q."""
        ...

    def truncated_mean(self, q_min: ArrayLike, q_max: ArrayLike, **theta: Any) -> Array:
        """∫ q f(q | θ) dq over [q_min, q_max]."""
        ...

    def numpyro_prior(self, name: str) -> dict[str, Array]:
        """NumPyro sites for θ under the default priors."""
        ...


def _lognormal_log_prob(
    q: Float[Array, "*q"], mu: Float[Array, "*mu"], sigma: Float[Array, "*sigma"]
) -> Float[Array, "*out"]:
    # log LN(q; μ, σ²) = log N(log q; μ, σ²) − log q, −∞ for q ≤ 0
    positive = q > 0.0
    log_q = jnp.log(jnp.where(positive, q, 1.0))
    density = norm.logpdf(log_q, mu, sigma) - log_q
    return jnp.where(positive, density, -jnp.inf)


def _log_lognormal_partial_mean(
    lower: Float[Array, "*lower"],
    upper: Float[Array, "*upper"],
    mu: Float[Array, "*mu"],
    sigma: Float[Array, "*sigma"],
) -> Float[Array, "*out"]:
    # log ∫ₐᵇ q LN(q) dq = μ + σ²/2 + log[Φ(z_b) − Φ(z_a)], z = (log q − μ − σ²)/σ
    shift = mu + sigma**2
    empty = upper <= lower
    open_upper = jnp.isinf(upper)
    open_lower = lower <= 0.0
    # finite stand-ins for the open ends, so that neither the value nor the
    # reverse-mode gradient meets ∞ · 0; the limits are selected below
    safe_upper = jnp.where(open_upper | empty, 1.0, upper)
    safe_lower = jnp.where(open_lower | empty, 0.5, lower)
    upper_z = (jnp.log(safe_upper) - shift) / sigma
    lower_z = (jnp.log(safe_lower) - shift) / sigma
    # log Φ and log(1 − Φ) at each end, with the limits at open ends
    log_cdf_upper = jnp.where(open_upper, 0.0, norm.logcdf(upper_z))
    log_cdf_lower = jnp.where(open_lower, -jnp.inf, norm.logcdf(lower_z))
    log_survival_upper = jnp.where(open_upper, -jnp.inf, norm.logcdf(-upper_z))
    log_survival_lower = jnp.where(open_lower, 0.0, norm.logcdf(-lower_z))
    # log[Φ(z_b) − Φ(z_a)] from the CDFs left of the centre and from the
    # survivals right of it, so neither difference cancels to zero; the
    # unused one gets a safe log-ratio, since its log1p(−1) would put a NaN
    # into the reverse-mode gradient through jnp.where
    right_of_centre = (~open_lower) & (lower_z > 0.0)
    cdf_ratio = jnp.where(right_of_centre, -1.0, log_cdf_lower - log_cdf_upper)
    survival_ratio = jnp.where(
        right_of_centre, log_survival_upper - log_survival_lower, -1.0
    )
    from_cdf = log_cdf_upper + jnp.log1p(-jnp.exp(cdf_ratio))
    from_survival = log_survival_lower + jnp.log1p(-jnp.exp(survival_ratio))
    log_mass = jnp.where(right_of_centre, from_survival, from_cdf)
    return jnp.where(empty, -jnp.inf, mu + 0.5 * sigma**2 + log_mass)


#: Newton steps of :func:`_normal_quantile_from_log`.
_QUANTILE_NEWTON_STEPS = 4


def _normal_quantile_from_log(log_p: Float[Array, "*p"]) -> Float[Array, "*p"]:
    # z with log Φ(z) = log p, also where p underflows float64
    probability = jnp.exp(log_p)
    representable = probability > jnp.finfo(log_p.dtype).tiny
    # start: Φ⁻¹(p) where p is representable, else the left-tail asymptote
    # z² ≈ t − log(2π t), t = −2 log p
    tail = -2.0 * jnp.where(representable, -1.0, log_p)
    asymptote = -jnp.sqrt(tail - jnp.log(2.0 * jnp.pi * tail))
    direct = ndtri(jnp.where(representable, probability, 0.5))
    quantile = jnp.where(representable, direct, asymptote)
    for _ in range(_QUANTILE_NEWTON_STEPS):
        # Newton on log Φ(z) − log p, with d/dz log Φ(z) = φ(z)/Φ(z)
        log_cdf = norm.logcdf(quantile)
        slope = jnp.exp(norm.logpdf(quantile) - log_cdf)
        quantile = quantile - (log_cdf - log_p) / slope
    return quantile


def _lognormal_partial_mean(
    lower: Float[Array, "*lower"],
    upper: Float[Array, "*upper"],
    mu: Float[Array, "*mu"],
    sigma: Float[Array, "*sigma"],
) -> Float[Array, "*out"]:
    # ∫ₐᵇ q LN(q) dq = exp(μ + σ²/2) [Φ(z_b) − Φ(z_a)]
    return jnp.exp(_log_lognormal_partial_mean(lower, upper, mu, sigma))


#: |y| below which expm1(y)/y is evaluated by its Taylor series.
_SERIES_THRESHOLD = 1e-4


def _expm1_over_x(y: Float[Array, "*y"]) -> Float[Array, "*y"]:
    # E(y) = (eʸ − 1)/y, smooth through y = 0 (E = 1 + y/2 + y²/6 + y³/24 + …)
    small = jnp.abs(y) < _SERIES_THRESHOLD
    safe = jnp.where(small, 1.0, y)
    series = 1.0 + y / 2.0 + y**2 / 6.0 + y**3 / 24.0
    return jnp.where(small, series, jnp.expm1(safe) / safe)


def _pareto_partial_mean(
    lower: Float[Array, "*lower"],
    upper: Float[Array, "*upper"],
    scale: Float[Array, "*scale"],
    alpha: Float[Array, "*alpha"],
) -> Float[Array, "*out"]:
    # ∫ q · α sᵅ q^(−α−1) dq over [a', b], a' = max(a, s), ε = α − 1:
    #   α sᵅ (a'^(−ε) − b^(−ε))/ε, which is α sᵅ log(b/a') at ε = 0 and
    #   +∞ for b = ∞ and ε ≤ 0, where the mean diverges
    lower = jnp.maximum(lower, scale)
    upper = jnp.maximum(upper, lower)
    excess = alpha - 1.0
    open_upper = jnp.isinf(upper)
    log_lower = jnp.log(lower)
    log_upper = jnp.log(jnp.where(open_upper, lower, upper))
    # (a'^(−ε) − b^(−ε))/ε = log b · E(−ε log b) − log a' · E(−ε log a'),
    # continuous and differentiable in ε through ε = 0
    finite = log_upper * _expm1_over_x(-excess * log_upper) - log_lower * (
        _expm1_over_x(-excess * log_lower)
    )
    # b = ∞: a'^(−ε)/ε for ε > 0, ∞ otherwise
    safe_excess = jnp.where(excess > 0.0, excess, 1.0)
    open_value = jnp.where(
        excess > 0.0, jnp.exp(-excess * log_lower) / safe_excess, jnp.inf
    )
    bracket = jnp.where(open_upper, open_value, finite)
    return alpha * jnp.exp(alpha * jnp.log(scale)) * bracket


def _bounds(q_min: ArrayLike, q_max: ArrayLike) -> tuple[Array, Array]:
    lower = jnp.asarray(q_min, dtype=float)
    upper = jnp.asarray(q_max, dtype=float)
    lower, upper = jnp.broadcast_arrays(lower, upper)
    return lower, upper


class PowerLawMark(eqx.Module):
    """Pareto power law f(q) = (α − 1)/q_min · (q/q_min)^(−α) on q ≥ q_min.

    θ = {α}, α > 1. E[Q] is finite for α > 2; for α ≤ 2 ``truncated_mean``
    over an unbounded interval is +∞ (logarithmic in the upper limit at
    α = 2).

    Attributes:
        q_min: Lower end of the support [kg/s].
        alpha_bounds: ``(low, high)`` of the uniform prior on α.

    Examples:
        >>> import jax.numpy as jnp
        >>> mark = PowerLawMark(q_min=1.0)
        >>> float(jnp.exp(mark.log_prob(jnp.asarray(1.0), alpha=3.0)))
        2.0
        >>> float(mark.truncated_mean(1.0, jnp.inf, alpha=3.0))
        2.0
    """

    q_min: float
    alpha_bounds: tuple[float, float] = (1.1, 4.5)

    def log_prob(self, q: ArrayLike, *, alpha: ArrayLike) -> Float[Array, "*q"]:
        q = jnp.asarray(q, dtype=float)
        alpha = jnp.asarray(alpha, dtype=float)
        inside = q >= self.q_min
        ratio = jnp.where(inside, q, self.q_min) / self.q_min
        # log f = log(α − 1) − log q_min − α log(q/q_min)
        density = jnp.log(alpha - 1.0) - jnp.log(self.q_min) - alpha * jnp.log(ratio)
        return jnp.where(inside, density, -jnp.inf)

    def sample(
        self, key: Array, n: int, *, alpha: ArrayLike
    ) -> Float[Array, " member"]:
        alpha = jnp.asarray(alpha, dtype=float)
        uniform = jax.random.uniform(key, (n,), minval=jnp.finfo(float).tiny)
        # inverse CDF: q = q_min u^(−1/(α − 1))
        return self.q_min * uniform ** (-1.0 / (alpha - 1.0))

    def truncated_mean(
        self, q_min: ArrayLike, q_max: ArrayLike, *, alpha: ArrayLike
    ) -> Float[Array, "*q"]:
        lower, upper = _bounds(q_min, q_max)
        alpha = jnp.asarray(alpha, dtype=float)
        # a Pareto with index α − 1 and scale q_min
        return _pareto_partial_mean(lower, upper, jnp.asarray(self.q_min), alpha - 1.0)

    def numpyro_prior(self, name: str) -> dict[str, Array]:
        import numpyro
        import numpyro.distributions as dist

        low, high = self.alpha_bounds
        return {"alpha": numpyro.sample(f"{name}.alpha", dist.Uniform(low, high))}


class LognormalMark(eqx.Module):
    """Lognormal f(q) = LN(q; μ, σ²).

    θ = {μ, σ}.

    Attributes:
        mu_prior_scale: Scale of the Normal(0, ·) prior on μ.
        sigma_prior_scale: Scale of the HalfNormal prior on σ.

    Examples:
        >>> import jax.numpy as jnp
        >>> mark = LognormalMark()
        >>> mean = mark.truncated_mean(0.0, jnp.inf, mu=0.0, sigma=1.0)
        >>> bool(jnp.allclose(mean, jnp.exp(0.5)))
        True
    """

    mu_prior_scale: float = 5.0
    sigma_prior_scale: float = 2.0

    def log_prob(
        self, q: ArrayLike, *, mu: ArrayLike, sigma: ArrayLike
    ) -> Float[Array, "*q"]:
        q = jnp.asarray(q, dtype=float)
        return _lognormal_log_prob(q, jnp.asarray(mu), jnp.asarray(sigma))

    def sample(
        self, key: Array, n: int, *, mu: ArrayLike, sigma: ArrayLike
    ) -> Float[Array, " member"]:
        noise = jax.random.normal(key, (n,))
        # Q = exp(μ + σ ε)
        return jnp.exp(jnp.asarray(mu) + jnp.asarray(sigma) * noise)

    def truncated_mean(
        self, q_min: ArrayLike, q_max: ArrayLike, *, mu: ArrayLike, sigma: ArrayLike
    ) -> Float[Array, "*q"]:
        lower, upper = _bounds(q_min, q_max)
        return _lognormal_partial_mean(
            lower, upper, jnp.asarray(mu, dtype=float), jnp.asarray(sigma, dtype=float)
        )

    def numpyro_prior(self, name: str) -> dict[str, Array]:
        import numpyro
        import numpyro.distributions as dist

        return {
            "mu": numpyro.sample(f"{name}.mu", dist.Normal(0.0, self.mu_prior_scale)),
            "sigma": numpyro.sample(
                f"{name}.sigma", dist.HalfNormal(self.sigma_prior_scale)
            ),
        }


def _splice_log_weights(
    mu: Float[Array, ""],
    sigma: Float[Array, ""],
    alpha: Float[Array, ""],
    q_break: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    # log F_LN(Q_b), the body's normaliser below the break
    log_body_mass = norm.logcdf((jnp.log(q_break) - mu) / sigma)
    # densities at the break of the normalised body and of the Pareto tail
    log_body_at_break = _lognormal_log_prob(q_break, mu, sigma) - log_body_mass
    log_tail_at_break = jnp.log(alpha) - jnp.log(q_break)
    # continuity w b(Q_b) = (1 − w) t(Q_b) ⇒ w = t / (t + b), 1 − w = b / (t + b)
    log_total = jnp.logaddexp(log_body_at_break, log_tail_at_break)
    log_body_weight = log_tail_at_break - log_total
    log_tail_weight = log_body_at_break - log_total
    return log_body_weight, log_tail_weight, log_body_mass


class LognormalParetoMark(eqx.Module):
    """Lognormal body spliced to a Pareto tail at Q_break, continuous there.

    f(q) = w LN(q; μ, σ²)/F_LN(Q_b) for q < Q_b and
    f(q) = (1 − w) α Q_bᵅ q^(−α−1) for q ≥ Q_b, with the body weight
    w = t/(t + b) set by continuity, where b = LN(Q_b; μ, σ²)/F_LN(Q_b) and
    t = α/Q_b are the two normalised pieces' densities at the break. One free
    tail index α; E[Q] is finite for α > 1 and Var[Q] for α > 2.

    θ = {μ, σ, α, Q_b}.

    Attributes:
        q_break_loc: Centre of the LogNormal(log ·, 0.5²) prior on Q_b, a
            published detection-floor value for the instrument class [kg/s].
        q_break_log_scale: Log-scale of that prior.
        mu_prior_scale: Scale of the Normal(0, ·) prior on μ.
        sigma_prior_scale: Scale of the HalfNormal prior on σ.
        alpha_bounds: ``(low, high)`` of the uniform prior on α.

    Examples:
        >>> import jax.numpy as jnp
        >>> mark = LognormalParetoMark(q_break_loc=2.0)
        >>> theta = dict(mu=0.0, sigma=1.0, alpha=2.5, q_break=2.0)
        >>> below = mark.log_prob(jnp.asarray(2.0 - 1e-9), **theta)
        >>> above = mark.log_prob(jnp.asarray(2.0), **theta)
        >>> bool(jnp.allclose(below, above))
        True
    """

    q_break_loc: float
    q_break_log_scale: float = 0.5
    mu_prior_scale: float = 5.0
    sigma_prior_scale: float = 2.0
    alpha_bounds: tuple[float, float] = (1.1, 4.5)

    def log_prob(
        self,
        q: ArrayLike,
        *,
        mu: ArrayLike,
        sigma: ArrayLike,
        alpha: ArrayLike,
        q_break: ArrayLike,
    ) -> Float[Array, "*q"]:
        q = jnp.asarray(q, dtype=float)
        mu, sigma, alpha, q_break = (
            jnp.asarray(value, dtype=float) for value in (mu, sigma, alpha, q_break)
        )
        log_body_weight, log_tail_weight, log_body_mass = _splice_log_weights(
            mu, sigma, alpha, q_break
        )
        # log w + log LN(q) − log F_LN(Q_b)
        body = log_body_weight + _lognormal_log_prob(q, mu, sigma) - log_body_mass
        # log(1 − w) + log α + α log Q_b − (α + 1) log q
        safe_q = jnp.where(q >= q_break, q, q_break)
        tail = (
            log_tail_weight
            + jnp.log(alpha)
            + alpha * jnp.log(q_break)
            - (alpha + 1.0) * jnp.log(safe_q)
        )
        return jnp.where(q < q_break, body, tail)

    def sample(
        self,
        key: Array,
        n: int,
        *,
        mu: ArrayLike,
        sigma: ArrayLike,
        alpha: ArrayLike,
        q_break: ArrayLike,
    ) -> Float[Array, " member"]:
        mu, sigma, alpha, q_break = (
            jnp.asarray(value, dtype=float) for value in (mu, sigma, alpha, q_break)
        )
        piece_key, body_key, tail_key = jax.random.split(key, 3)
        log_body_weight, _, _ = _splice_log_weights(mu, sigma, alpha, q_break)
        in_body = jax.random.uniform(piece_key, (n,)) < jnp.exp(log_body_weight)
        # body: inverse CDF of the lognormal truncated to (0, Q_b)
        # in log space: Φ(z_b) underflows when Q_b sits far below the body
        log_break_cdf = norm.logcdf((jnp.log(q_break) - mu) / sigma)
        body_uniform = jax.random.uniform(body_key, (n,), minval=jnp.finfo(float).tiny)
        log_level = jnp.log(body_uniform) + log_break_cdf
        body = jnp.exp(mu + sigma * _normal_quantile_from_log(log_level))
        # tail: inverse CDF of the Pareto, q = Q_b u^(−1/α)
        tail_uniform = jax.random.uniform(tail_key, (n,), minval=jnp.finfo(float).tiny)
        tail = q_break * tail_uniform ** (-1.0 / alpha)
        return jnp.where(in_body, body, tail)

    def truncated_mean(
        self,
        q_min: ArrayLike,
        q_max: ArrayLike,
        *,
        mu: ArrayLike,
        sigma: ArrayLike,
        alpha: ArrayLike,
        q_break: ArrayLike,
    ) -> Float[Array, "*q"]:
        lower, upper = _bounds(q_min, q_max)
        mu, sigma, alpha, q_break = (
            jnp.asarray(value, dtype=float) for value in (mu, sigma, alpha, q_break)
        )
        log_body_weight, log_tail_weight, log_body_mass = _splice_log_weights(
            mu, sigma, alpha, q_break
        )
        # body part on [a, min(b, Q_b)], renormalised by F_LN(Q_b) in log
        # space: far below the lognormal's centre both factors leave float64
        body_upper = jnp.minimum(upper, q_break)
        body_lower = jnp.minimum(lower, body_upper)
        log_body = _log_lognormal_partial_mean(body_lower, body_upper, mu, sigma)
        body = jnp.exp(log_body_weight - log_body_mass + log_body)
        # tail part on [max(a, Q_b), b]
        tail = _pareto_partial_mean(lower, upper, q_break, alpha)
        return body + jnp.exp(log_tail_weight) * tail

    def numpyro_prior(self, name: str) -> dict[str, Array]:
        import numpyro
        import numpyro.distributions as dist

        low, high = self.alpha_bounds
        log_loc = jnp.log(self.q_break_loc)
        return {
            "mu": numpyro.sample(f"{name}.mu", dist.Normal(0.0, self.mu_prior_scale)),
            "sigma": numpyro.sample(
                f"{name}.sigma", dist.HalfNormal(self.sigma_prior_scale)
            ),
            "alpha": numpyro.sample(f"{name}.alpha", dist.Uniform(low, high)),
            "q_break": numpyro.sample(
                f"{name}.q_break", dist.LogNormal(log_loc, self.q_break_log_scale)
            ),
        }


class MixtureLognormalMark(eqx.Module):
    """Mixture of K lognormals f(q) = Σₖ wₖ LN(q; μₖ, σₖ²).

    θ = {weights, locs, scales}, each of length K.

    Attributes:
        n_components: K.
        concentration: Dirichlet concentration of the prior on the weights
            (the same for every component).
        mu_prior_scale: Scale of the Normal(0, ·) prior on each μₖ; the
            sampled μₖ are sorted, which removes label switching.
        sigma_prior_scale: Scale of the HalfNormal prior on each σₖ.

    Examples:
        >>> import jax.numpy as jnp
        >>> mark = MixtureLognormalMark(n_components=2)
        >>> theta = dict(
        ...     weights=jnp.array([0.5, 0.5]),
        ...     locs=jnp.array([0.0, 0.0]),
        ...     scales=jnp.array([1.0, 1.0]),
        ... )
        >>> single = LognormalMark().log_prob(2.0, mu=0.0, sigma=1.0)
        >>> bool(jnp.allclose(mark.log_prob(2.0, **theta), single))
        True
    """

    n_components: int
    concentration: float = 1.0
    mu_prior_scale: float = 5.0
    sigma_prior_scale: float = 2.0

    def log_prob(
        self,
        q: ArrayLike,
        *,
        weights: ArrayLike,
        locs: ArrayLike,
        scales: ArrayLike,
    ) -> Float[Array, "*q"]:
        q = jnp.asarray(q, dtype=float)
        weights, locs, scales = (
            jnp.asarray(value, dtype=float) for value in (weights, locs, scales)
        )
        # log LN(q; μₖ, σₖ²) for every component
        component = _lognormal_log_prob(einx.rearrange("... -> ... 1", q), locs, scales)
        # log Σₖ wₖ LNₖ(q)
        return einx.logsumexp("... [k]", component + jnp.log(weights))

    def sample(
        self,
        key: Array,
        n: int,
        *,
        weights: ArrayLike,
        locs: ArrayLike,
        scales: ArrayLike,
    ) -> Float[Array, " member"]:
        weights, locs, scales = (
            jnp.asarray(value, dtype=float) for value in (weights, locs, scales)
        )
        label_key, noise_key = jax.random.split(key)
        labels = jax.random.categorical(label_key, jnp.log(weights), shape=(n,))
        noise = jax.random.normal(noise_key, (n,))
        # Q = exp(μ_k + σ_k ε), k the drawn component
        return jnp.exp(locs[labels] + scales[labels] * noise)

    def truncated_mean(
        self,
        q_min: ArrayLike,
        q_max: ArrayLike,
        *,
        weights: ArrayLike,
        locs: ArrayLike,
        scales: ArrayLike,
    ) -> Float[Array, "*q"]:
        lower, upper = _bounds(q_min, q_max)
        weights, locs, scales = (
            jnp.asarray(value, dtype=float) for value in (weights, locs, scales)
        )
        # Σₖ wₖ ∫ q LNₖ(q) dq
        per_component = _lognormal_partial_mean(
            einx.rearrange("... -> ... 1", lower),
            einx.rearrange("... -> ... 1", upper),
            locs,
            scales,
        )
        return einx.dot("... k, k -> ...", per_component, weights)

    def numpyro_prior(self, name: str) -> dict[str, Array]:
        import numpyro
        import numpyro.distributions as dist

        size = self.n_components
        weights = numpyro.sample(
            f"{name}.weights", dist.Dirichlet(jnp.full(size, self.concentration))
        )
        raw_locs = numpyro.sample(
            f"{name}.raw_locs",
            dist.Normal(0.0, self.mu_prior_scale).expand([size]).to_event(1),
        )
        scales = numpyro.sample(
            f"{name}.scales",
            dist.HalfNormal(self.sigma_prior_scale).expand([size]).to_event(1),
        )
        # sorted component centres: the mixture is symmetric under relabelling
        locs = numpyro.deterministic(f"{name}.locs", jnp.sort(raw_locs))
        return {"weights": weights, "locs": locs, "scales": scales}


#: Registry name → mark class, as for the intensity and POD registries.
MARKS: dict[str, type] = {
    "powerlaw": PowerLawMark,
    "lognormal": LognormalMark,
    "lognormal_pareto": LognormalParetoMark,
    "mixture_lognormal": MixtureLognormalMark,
}


def make_mark(name: str, **fields: Any) -> Mark:
    """Build a registry mark family by name.

    Args:
        name: A key of :data:`MARKS`.
        **fields: The family's fields (support bounds, prior settings).

    Returns:
        The mark family.

    Raises:
        KeyError: For an unknown name.
    """
    if name not in MARKS:
        raise KeyError(
            f"make_mark: unknown mark {name!r}; expected one of {sorted(MARKS)}."
        )
    return MARKS[name](**fields)
