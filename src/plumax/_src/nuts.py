"""One NUTS run for every NumPyro fit in plumax."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
from jaxtyping import Array


def run_nuts(
    model: Callable[..., Any],
    *model_args: Any,
    seed: int | Array,
    num_warmup: int,
    num_samples: int,
    num_chains: int = 1,
    progress_bar: bool = False,
    print_summary: bool = False,
    nuts_options: Mapping[str, Any] | None = None,
    discrete_gibbs: bool = False,
    **model_kwargs: Any,
) -> dict[str, Array]:
    """Sample a NumPyro model with NUTS and return its posterior draws.

    Builds ``MCMC(NUTS(model, **nuts_options), ...)``, optionally inside
    ``DiscreteHMCGibbs``, runs it on ``PRNGKey(seed)`` with the model's arguments, and returns
    ``mcmc.get_samples()``. Callers keep their own post-processing.

    Args:
        model: The NumPyro model.
        *model_args: Positional arguments of ``model``.
        seed: Seed of the sampler's ``PRNGKey``, or the key itself.
        num_warmup: Warm-up (adaptation) iterations per chain.
        num_samples: Kept draws per chain.
        num_chains: Number of chains.
        progress_bar: Show NumPyro's progress bar.
        print_summary: Print NumPyro's summary table after the run.
        nuts_options: Keyword arguments of ``NUTS`` (``target_accept_prob``,
            ``init_strategy``, ...).
        discrete_gibbs: Wrap NUTS in ``DiscreteHMCGibbs``, which adds a Gibbs
            step over the model's discrete sites (a categorical stability
            class, the on/off indicators of a masked-K model).
        **model_kwargs: Keyword arguments of ``model``.

    Returns:
        Draws keyed by site name, each with a leading axis of
        ``num_chains · num_samples``.
    """
    from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

    kernel = NUTS(model, **dict(nuts_options or {}))
    if discrete_gibbs:
        kernel = DiscreteHMCGibbs(kernel)
    key = jax.random.PRNGKey(seed) if isinstance(seed, int) else seed
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(key, *model_args, **model_kwargs)
    if print_summary:
        mcmc.print_summary()
    return mcmc.get_samples()
