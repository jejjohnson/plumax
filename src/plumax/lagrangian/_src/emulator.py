"""Footprint emulator: a network that maps met and receptor to a footprint row.

Tier II Step 3 replaces the backward particle run with a trained map

    F_emu(u, r) ≈ F(r, ·),

from a met context u (a feature vector, say the mean wind) and a receptor
position r to the receptor's footprint row over the surface cells. Stacking
the rows of every receptor gives the ``(obs, cell)`` matrix that
:func:`~plumax.lagrangian.linear_gaussian_inversion` takes, so Step 4
(emulator-based inference) is that inversion with F_emu in place of F.

The hard test of the design page is the emulator residual on held-out met
drawn from the operational distribution:

    ‖F_emu − F‖_F / ‖F‖_F < 5 %,

measured by :func:`emulator_relative_error`. Its probed form,
‖F_emu v − F v‖ / ‖F v‖ < 5 % for random source fields v, is
:func:`assert_emulator_calibrated`, which hands both maps to
``vardax.assert_adjoint_calibrated`` (the ``assim`` extra).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray

from plumax._src.errors import ParameterError
from plumax._vardax import require_vardax
from plumax.lagrangian.met_ensemble import ensemble_footprints


if TYPE_CHECKING:
    from plumax.lagrangian.particles import TurbulenceModel
    from plumax.lagrangian.wind_interp import GriddedWind


#: Maximum particle-model seed; each case draws its own from the key.
_MAX_SEED = 2**30


class FootprintTrainingSet(eqx.Module):
    """Particle-model footprints for sampled met contexts, the emulator's data.

    Attributes:
        met_context: Met context u of each case.
        receptors: Receptor positions (x, y, z) [m], shared by the cases.
        footprints: Footprint F of each case [s m² kg⁻¹], cells flattened
            row-major over ``(x, y)`` as in ``ensemble_footprints``.
    """

    met_context: Float[Array, "case feature"]
    receptors: Float[Array, "obs 3"]
    footprints: Float[Array, "case obs cell"]


class FootprintEmulator(eqx.Module):
    """Per-receptor network from met context and receptor to a footprint row.

    F_emu(u, r) = s · softplus(g(z; θ)),   z = ([u, r_xy] − m) / σ,

    with g an MLP and m, σ the mean and standard deviation of the training
    inputs. The softplus keeps every cell non-negative, as a residence
    time is, and s is the mean training footprint, so g works on order-one
    numbers.

    Attributes:
        network: The MLP g, from ``feature + 2`` inputs to ``cell`` outputs.
        input_mean: Mean m of the training inputs [u, r_xy].
        input_scale: Standard deviation σ of the training inputs.
        footprint_scale: Scale s [s m² kg⁻¹].

    Examples:
        An untrained emulator for a two-feature context on a 3-cell grid:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> network = eqx.nn.MLP(4, 3, 8, 1, key=jax.random.PRNGKey(0))
        >>> emulator = FootprintEmulator(
        ...     network, jnp.zeros(4), jnp.ones(4), jnp.asarray(0.5)
        ... )
        >>> row = emulator(jnp.array([3.0, 0.0]), jnp.array([0.0, 100.0]))
        >>> row.shape, bool((row >= 0.0).all())
        ((3,), True)
    """

    network: eqx.nn.MLP
    input_mean: Float[Array, " input"]
    input_scale: Float[Array, " input"]
    footprint_scale: Float[Array, ""]

    def __call__(
        self,
        met_context: Float[Array, " feature"],
        receptor_xy: Float[Array, " 2"],
    ) -> Float[Array, " cell"]:
        inputs = einx.rearrange(
            "feature, xy -> (feature + xy)", met_context, receptor_xy
        )
        # z = ([u, r_xy] − m) / σ
        standardised = (inputs - self.input_mean) / self.input_scale
        logits = self.network(standardised)
        # F_emu = s · softplus(g(z))
        footprint = self.footprint_scale * jax.nn.softplus(logits)
        return footprint


def generate_training_set(
    key: PRNGKeyArray,
    *,
    met_sampler: Callable[[PRNGKeyArray], Float[Array, " feature"]],
    wind_from_context: Callable[
        [Float[Array, " feature"]], GriddedWind | Callable[[jax.Array], jax.Array]
    ],
    receptors: Float[Array, "obs 3"],
    turbulence: TurbulenceModel,
    domain_x: tuple[float, float, int],
    domain_y: tuple[float, float, int],
    n_cases: int,
    n_particles: int = 300,
    t_back: float = 600.0,
    dt: float = 1.0,
    pbl_height: float = 1000.0,
    pbl_fraction: float = 0.5,
    air_density: float = 1.2,
) -> FootprintTrainingSet:
    """Sample met contexts and run the particle model for each of them.

    uᵢ ~ p(u), Fᵢ = compute_footprint(receptors, wind_from_context(uᵢ)).

    Draw u from the operational distribution at the facility, p(u | site,
    overpass time), not from a uniform climatology grid: the design page
    notes a coarse (hour, season, stability) grid covers under 10 % of the
    operational probability mass.

    Algorithm:
        1. Split the key into one context key and one seed key per case.
        2. Draw each context uᵢ with ``met_sampler``.
        3. Run :func:`~plumax.lagrangian.ensemble_footprints` for the case's
           wind with its own particle seed.

    Args:
        key: PRNG key for the contexts and the particle seeds.
        met_sampler: One draw u ~ p(u) from a key.
        wind_from_context: Mean wind for a context, a ``GriddedWind`` or a
            ``t -> (u, v, w)`` callable.
        receptors: Receptor positions (x, y, z) [m].
        turbulence: Turbulence model.
        domain_x: ``(start, stop, n_cells)`` along x for the surface grid.
        domain_y: ``(start, stop, n_cells)`` along y for the surface grid.
        n_cases: Number of met contexts N.
        n_particles: Particles per receptor and case.
        t_back: Backward integration time [s].
        dt: Time step [s].
        pbl_height: Boundary-layer height [m].
        pbl_fraction: Fraction of the PBL that sees the surface flux.
        air_density: Air density [kg m⁻³].

    Returns:
        The contexts, receptors and footprints.

    Raises:
        ParameterError: If ``n_cases`` is below one.
    """
    if n_cases < 1:
        raise ParameterError(f"n_cases must be ≥ 1, got {n_cases}")
    context_key, seed_key = jax.random.split(key)
    context_keys = jax.random.split(context_key, n_cases)
    contexts = jax.vmap(met_sampler)(context_keys)
    # receptor i of a case uses seed + i, so seeds are spaced by the receptor count
    seeds = jax.random.randint(seed_key, (n_cases,), 0, _MAX_SEED)
    footprints = []
    for context, seed in zip(contexts, seeds, strict=True):
        member = ensemble_footprints(
            [wind_from_context(context)],
            receptors,
            turbulence,
            domain_x,
            domain_y,
            n_particles=n_particles,
            t_back=t_back,
            dt=dt,
            pbl_height=pbl_height,
            pbl_fraction=pbl_fraction,
            air_density=air_density,
            seed=int(seed),
        )
        footprints.append(member[0])
    return FootprintTrainingSet(
        met_context=contexts,
        receptors=jnp.asarray(receptors, dtype=contexts.dtype),
        footprints=jnp.stack(footprints),
    )


def _case_inputs(
    met_context: Float[Array, "case feature"],
    receptors: Float[Array, "obs 3"],
) -> Float[Array, "case obs input"]:
    """Network inputs [u, r_xy] for every case and receptor."""
    receptor_xy = receptors[:, :2]
    tiled_context = einx.rearrange(
        "case feature -> case obs feature",
        met_context,
        obs=receptor_xy.shape[0],
    )
    tiled_receptor = einx.rearrange(
        "obs xy -> case obs xy", receptor_xy, case=met_context.shape[0]
    )
    return einx.rearrange(
        "case obs feature, case obs xy -> case obs (feature + xy)",
        tiled_context,
        tiled_receptor,
    )


def init_footprint_emulator(
    data: FootprintTrainingSet,
    *,
    width: int = 32,
    depth: int = 2,
    key: PRNGKeyArray,
) -> FootprintEmulator:
    """An untrained emulator, standardised to the training set.

    Args:
        data: The training set; fixes the input and output sizes, the input
            standardisation and the footprint scale s.
        width: Hidden width of the MLP.
        depth: Number of hidden layers.
        key: PRNG key for the MLP weights.

    Returns:
        The emulator, ready for :func:`train_footprint_emulator`.

    Raises:
        ParameterError: If ``width`` or ``depth`` is below one.
    """
    if width < 1 or depth < 1:
        raise ParameterError(f"width and depth must be ≥ 1, got {width}, {depth}")
    inputs = _case_inputs(data.met_context, data.receptors)
    input_mean = einx.mean("[case obs] input", inputs)
    input_std = einx.std("[case obs] input", inputs)
    # constant inputs (one case, one receptor) keep a unit scale
    input_scale = jnp.where(input_std > 0.0, input_std, 1.0)
    footprint_scale = jnp.mean(data.footprints)
    network = eqx.nn.MLP(
        inputs.shape[-1],
        data.footprints.shape[-1],
        width,
        depth,
        activation=jax.nn.tanh,
        key=key,
    )
    return FootprintEmulator(network, input_mean, input_scale, footprint_scale)


def emulated_footprint(
    emulator: FootprintEmulator,
    met_context: Float[Array, " feature"],
    receptors: Float[Array, "obs 3"],
) -> Float[Array, "obs cell"]:
    """The emulated footprint matrix F_emu of one met context.

    F_emu[i, s] = emulator(u, rᵢ)[s], in the layout of
    :func:`~plumax.lagrangian.ensemble_footprints` and of the inversions.

    Args:
        emulator: The trained emulator.
        met_context: Met context u.
        receptors: Receptor positions (x, y, z) [m]; z is not an input.

    Returns:
        The footprint rows [s m² kg⁻¹].

    Examples:
        Two receptors on a 3-cell grid:

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> network = eqx.nn.MLP(4, 3, 8, 1, key=jax.random.PRNGKey(0))
        >>> emulator = FootprintEmulator(
        ...     network, jnp.zeros(4), jnp.ones(4), jnp.asarray(0.5)
        ... )
        >>> receptors = jnp.array([[0.0, 0.0, 10.0], [0.0, 50.0, 10.0]])
        >>> emulated_footprint(emulator, jnp.array([3.0, 0.0]), receptors).shape
        (2, 3)
    """
    receptor_xy = receptors[:, :2]
    return jax.vmap(emulator, in_axes=(None, 0))(met_context, receptor_xy)


def train_footprint_emulator(
    emulator: FootprintEmulator,
    data: FootprintTrainingSet,
    *,
    max_steps: int = 500,
) -> FootprintEmulator:
    """Fit the network weights θ to the training footprints by BFGS.

    θ* = argmin_θ 𝓛(θ),  𝓛(θ) = mean over cases, receptors and cells of
    ((F_emu(uᵢ, r; θ) − Fᵢ) / s)².

    The standardisation and the scale s stay fixed. Training stops after
    ``max_steps`` BFGS steps or at convergence, whichever is first.

    Args:
        emulator: Initial emulator, e.g. from :func:`init_footprint_emulator`.
        data: The training set.
        max_steps: Maximum number of BFGS steps.

    Returns:
        The emulator with the fitted weights.
    """
    inputs = _case_inputs(data.met_context, data.receptors)
    weights, structure = eqx.partition(emulator.network, eqx.is_inexact_array)

    def loss(
        trainable: eqx.nn.MLP,
        args: tuple[Float[Array, "case obs input"], Float[Array, "case obs cell"]],
    ) -> Float[Array, ""]:
        case_inputs, footprints = args
        network = eqx.combine(trainable, structure)
        model = eqx.tree_at(lambda e: e.network, emulator, network)
        # F_emu(u, r) for every case and receptor
        standardised = (case_inputs - model.input_mean) / model.input_scale
        logits = jax.vmap(jax.vmap(network))(standardised)
        predicted = model.footprint_scale * jax.nn.softplus(logits)
        # (F_emu − F) / s
        scaled_residual = (predicted - footprints) / model.footprint_scale
        return jnp.mean(scaled_residual**2)

    solution = optx.minimise(
        loss,
        optx.BFGS(rtol=1e-8, atol=1e-8),
        weights,
        args=(inputs, data.footprints),
        max_steps=max_steps,
        throw=False,
    )
    network = eqx.combine(solution.value, structure)
    return eqx.tree_at(lambda e: e.network, emulator, network)


def emulator_relative_error(
    emulated: Float[Array, "*case obs cell"],
    reference: Float[Array, "*case obs cell"],
) -> Float[Array, ""]:
    """Relative Frobenius error ‖F_emu − F‖_F / ‖F‖_F over every case.

    Args:
        emulated: Emulated footprints F_emu.
        reference: Particle-model footprints F on the same met and receptors.

    Returns:
        The relative error; the design page's hard test asks for < 0.05.

    Examples:
        A uniform 2 % overestimate is a 2 % error:

        >>> import jax.numpy as jnp
        >>> reference = jnp.ones((2, 3, 4))
        >>> round(float(emulator_relative_error(1.02 * reference, reference)), 6)
        0.02
    """
    # ‖F_emu − F‖²_F
    squared_error = einx.sum("[...]", (emulated - reference) ** 2)
    # ‖F‖²_F
    squared_norm = einx.sum("[...]", reference**2)
    return jnp.sqrt(squared_error / squared_norm)


def assert_emulator_calibrated(
    emulator: FootprintEmulator,
    reference_footprints: Float[Array, "case obs cell"],
    *,
    met_context: Float[Array, "case feature"],
    receptors: Float[Array, "obs 3"],
    key: PRNGKeyArray,
    threshold: float = 0.05,
    n_probes: int = 10,
) -> None:
    """Check the emulated source–receptor maps against the particle model's.

    ‖F_emu v − F v‖ / ‖F v‖ < threshold for ``n_probes`` random source
    fields v, through ``vardax.assert_adjoint_calibrated``. The maps are
    q ↦ (F_emu,ᵢ q)ᵢ and q ↦ (Fᵢ q)ᵢ stacked over the held-out cases i, so
    this is the probed form of the Frobenius test over the same set, and it
    checks the sensitivity the inversion sees.

    Args:
        emulator: The trained emulator.
        reference_footprints: Particle-model footprints Fᵢ at ``met_context``.
        met_context: Held-out met contexts uᵢ.
        receptors: Receptor positions (x, y, z) [m].
        key: PRNG key for the probe vectors.
        threshold: Largest relative error allowed.
        n_probes: Number of probe vectors.

    Raises:
        AssertionError: If a probe exceeds ``threshold``.
        ImportError: Without the ``assim`` extra.
    """
    vdx = require_vardax("assert_emulator_calibrated")
    emulated = jax.vmap(emulated_footprint, in_axes=(None, 0, None))(
        emulator, met_context, receptors
    )
    source = jnp.ones(emulated.shape[-1], dtype=emulated.dtype)

    def emulated_map(q: Float[Array, " cell"]) -> Float[Array, "case obs"]:
        # F_emu,ᵢ q for every case i
        return einx.dot("case obs cell, cell -> case obs", emulated, q)

    def reference_map(q: Float[Array, " cell"]) -> Float[Array, "case obs"]:
        # Fᵢ q for every case i
        return einx.dot("case obs cell, cell -> case obs", reference_footprints, q)

    vdx.assert_adjoint_calibrated(
        emulated_map,
        reference_map,
        source,
        key=key,
        threshold=threshold,
        n_probes=n_probes,
    )
