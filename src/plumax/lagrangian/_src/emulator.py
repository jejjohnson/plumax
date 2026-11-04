"""Footprint emulator: a network that maps met and receptor to a footprint row.

Tier II Step 3 replaces the backward particle run with a trained map

    F_emu(u, r) ≈ F(r, ·),

from a met context u (a feature vector, say the mean wind) and a receptor
position r to the receptor's footprint row over the surface cells. Stacking
the rows of every receptor gives the ``(obs, cell)`` matrix that
:func:`~plumax.lagrangian.linear_gaussian_inversion` takes, so Step 4
(emulator-based inference) is that inversion with F_emu in place of F.

The training machinery is pipekit-train's (the ``emulate`` extra): the
particle model is a ``SimulationDataset`` forward model, cached once on disk
by ``CachedDataset``, and the network is fitted by a ``TrainingLoop`` on the
Equinox backend with its ``MSE`` loss and ``EarlyStopping`` callback. plumax
supplies the simulator, the met prior and the network.

The hard test of the design page is the emulator residual on held-out met
drawn from the operational distribution:

    ‖F_emu − F‖_F / ‖F‖_F < 5 %,

measured by :func:`emulator_relative_error`. Its probed form,
‖F_emu v − F v‖ / ‖F v‖ < 5 % for random source fields v, is
:func:`assert_emulator_calibrated`, which hands both maps to
``vardax.assert_adjoint_calibrated`` (the ``assim`` extra).
"""

from __future__ import annotations

import zlib
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pipekit
from jaxtyping import Array, Float, PRNGKeyArray

from plumax._src.errors import ParameterError
from plumax._vardax import require_vardax
from plumax.lagrangian.met_ensemble import ensemble_footprints


if TYPE_CHECKING:
    from plumax.lagrangian.particles import TurbulenceModel
    from plumax.lagrangian.wind_interp import GriddedWind


def _require_pipekit_train(caller: str) -> ModuleType:
    """Import pipekit-train, or raise an ``ImportError`` naming the extra."""
    try:
        import pipekit_train
    except ImportError as exc:
        raise ImportError(
            f"{caller} trains the emulator with pipekit-train; install it with "
            "`pip install 'plumax[emulate]'`."
        ) from exc
    return pipekit_train


class FootprintSimulator:
    """The particle model as a forward model from met context to footprint.

    F(u) = compute_footprint(receptors, wind_from_context(u)), one row per
    receptor, cells flattened row-major over ``(x, y)``.

    It satisfies ``pipekit_cycle.ForwardModel`` structurally (``step`` and
    ``dt``), so it is the ``forward_model`` of pipekit-train's
    ``SimulationDataset``. The particle seed of a case is a hash of its
    context, so a cached case is reproducible and two cases never share
    their particle noise.

    Args:
        wind_from_context: Mean wind for a context, a ``GriddedWind`` or a
            ``t -> (u, v, w)`` callable.
        receptors: Receptor positions (x, y, z) [m].
        turbulence: Turbulence model.
        domain_x: ``(start, stop, n_cells)`` along x for the surface grid.
        domain_y: ``(start, stop, n_cells)`` along y for the surface grid.
        n_particles: Particles per receptor and case.
        t_back: Backward integration time [s].
        dt: Particle time step [s]; the forward model's ``dt``.
        pbl_height: Boundary-layer height [m].
        pbl_fraction: Fraction of the PBL that sees the surface flux.
        air_density: Air density [kg m⁻³].

    Raises:
        ParameterError: If ``n_particles`` is below one.
    """

    def __init__(
        self,
        *,
        wind_from_context: Callable[
            [Float[Array, " feature"]],
            GriddedWind | Callable[[jax.Array], jax.Array],
        ],
        receptors: Float[Array, "obs 3"],
        turbulence: TurbulenceModel,
        domain_x: tuple[float, float, int],
        domain_y: tuple[float, float, int],
        n_particles: int = 300,
        t_back: float = 600.0,
        dt: float = 1.0,
        pbl_height: float = 1000.0,
        pbl_fraction: float = 0.5,
        air_density: float = 1.2,
    ) -> None:
        if n_particles < 1:
            raise ParameterError(f"n_particles must be ≥ 1, got {n_particles}")
        self.wind_from_context = wind_from_context
        self.receptors = jnp.asarray(receptors)
        self.turbulence = turbulence
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.n_particles = int(n_particles)
        self.t_back = float(t_back)
        self.dt = float(dt)
        self.pbl_height = float(pbl_height)
        self.pbl_fraction = float(pbl_fraction)
        self.air_density = float(air_density)

    @property
    def state_signature(self) -> tuple[Any, ...]:
        """Everything that changes the footprints, for the dataset cache key."""
        wind = self.wind_from_context
        return (
            f"{wind.__module__}.{wind.__qualname__}",
            np.asarray(self.receptors).tolist(),
            repr(self.turbulence),
            self.domain_x,
            self.domain_y,
            self.n_particles,
            self.t_back,
            self.pbl_height,
            self.pbl_fraction,
            self.air_density,
        )

    def step(
        self, state: Float[Array, " feature"], dt: float
    ) -> Float[Array, "obs cell"]:
        """Footprint F(u) of the met context u = ``state``."""
        context = np.asarray(state)
        # receptor i of a case uses seed + i, so a hash of u keeps cases apart
        seed = zlib.crc32(context.tobytes())
        member = ensemble_footprints(
            [self.wind_from_context(jnp.asarray(context))],
            self.receptors,
            self.turbulence,
            self.domain_x,
            self.domain_y,
            n_particles=self.n_particles,
            t_back=self.t_back,
            dt=dt,
            pbl_height=self.pbl_height,
            pbl_fraction=self.pbl_fraction,
            air_density=self.air_density,
            seed=seed,
        )
        return member[0]


class MetContextPrior(pipekit.Operator):
    """The operational met distribution p(u) as pipekit-train's prior operator.

    u ~ p(u | site, overpass time), one draw per integer seed. Draw u from the
    operational distribution at the facility, not from a uniform climatology
    grid: the design page notes a coarse (hour, season, stability) grid
    covers under 10 % of the operational probability mass.

    Args:
        met_sampler: One draw u ~ p(u) from a key.
    """

    __config_mixin_auto__: ClassVar[bool] = False
    forbid_in_yaml: ClassVar[bool] = True

    def __init__(
        self, met_sampler: Callable[[PRNGKeyArray], Float[Array, " feature"]]
    ) -> None:
        self.met_sampler = met_sampler

    def _apply(self, seed: int = 0) -> Float[Array, " feature"]:
        key = jax.random.PRNGKey(seed)
        return self.met_sampler(key)

    def get_config(self) -> dict[str, Any]:
        sampler = self.met_sampler
        return {"met_sampler": f"{sampler.__module__}.{sampler.__qualname__}"}


def footprint_dataset(
    simulator: FootprintSimulator,
    met_sampler: Callable[[PRNGKeyArray], Float[Array, " feature"]],
    *,
    n_cases: int,
    cache_dir: str,
    seed: int = 0,
    split: str = "train",
) -> Any:
    """Simulated ``(met context, footprint)`` pairs, cached once on disk.

    uᵢ ~ p(u), Fᵢ = simulator.step(uᵢ), as pipekit-train's
    ``SimulationDataset`` wrapped in its ``CachedDataset``: the particle
    model runs once per case on first use, and every later epoch reads the
    zarr cache under ``cache_dir``, keyed on the simulator, the sampler, the
    seed and the split.

    The cache is held in memory after its first read, because training reads
    it case by case and a zarr read costs about 1 ms per array.

    ``SimulationDataset`` partitions the ``n_cases`` seeds 80 / 10 / 10 into
    the ``"train"``, ``"val"`` and ``"test"`` splits, so a split holds a
    fraction of ``n_cases`` and the splits never share a case.

    Args:
        simulator: The particle model.
        met_sampler: One draw u ~ p(u) from a key.
        n_cases: Number of met contexts N over all three splits.
        cache_dir: Directory (or fsspec URI) of the cache.
        seed: Seed of the first case.
        split: ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        The cached pipekit-train dataset.

    Raises:
        ParameterError: If ``n_cases`` is below one.
        ImportError: Without the ``emulate`` extra.
    """
    if n_cases < 1:
        raise ParameterError(f"n_cases must be ≥ 1, got {n_cases}")
    pt = _require_pipekit_train("footprint_dataset")
    source = pt.SimulationDataset(
        forward_model=simulator,
        prior=MetContextPrior(met_sampler),
        n_samples=n_cases,
        seed=seed,
        split=split,
    )
    return pt.CachedDataset(source, cache_dir=cache_dir, in_memory=True)


def _stack_dataset(
    dataset: Any,
) -> tuple[Float[Array, "case feature"], Float[Array, "case obs cell"]]:
    """Every ``(u, F)`` pair of a dataset, stacked over cases."""
    pairs = list(dataset)
    contexts = jnp.stack([jnp.asarray(context) for context, _ in pairs])
    footprints = jnp.stack([jnp.asarray(footprint) for _, footprint in pairs])
    return contexts, footprints


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


class EmulatedFootprint(eqx.Module):
    """The emulator at fixed receptors, as the per-case map u ↦ F_emu(u).

    F_emu(u)[i, s] = emulator(u, rᵢ)[s]. This is the model pipekit-train's
    Equinox backend fits: it maps one met context to the ``(obs, cell)``
    target of a ``SimulationDataset`` pair. Gradients reach the network
    weights only; the standardisation, the scale s and the receptors are
    held fixed, as in :func:`train_footprint_emulator`.

    Attributes:
        emulator: The per-receptor network.
        receptors: Receptor positions (x, y, z) [m]; z is not an input.
    """

    emulator: FootprintEmulator
    receptors: Float[Array, "obs 3"]

    def __call__(
        self, met_context: Float[Array, " feature"]
    ) -> Float[Array, "obs cell"]:
        # only the network θ is trained: m, σ, s and the receptors stay fixed
        emulator = self.emulator
        fixed = jax.lax.stop_gradient(
            (
                emulator.input_mean,
                emulator.input_scale,
                emulator.footprint_scale,
                self.receptors,
            )
        )
        input_mean, input_scale, footprint_scale, receptors = fixed
        frozen = FootprintEmulator(
            emulator.network, input_mean, input_scale, footprint_scale
        )
        return emulated_footprint(frozen, met_context, receptors)


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
    dataset: Any,
    receptors: Float[Array, "obs 3"],
    *,
    width: int = 32,
    depth: int = 2,
    key: PRNGKeyArray,
) -> FootprintEmulator:
    """An untrained emulator, standardised to the training set.

    Args:
        dataset: The training ``(u, F)`` pairs, e.g. from
            :func:`footprint_dataset`; fixes the input and output sizes, the
            input standardisation and the footprint scale s.
        receptors: Receptor positions (x, y, z) [m] of the footprints.
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
    contexts, footprints = _stack_dataset(dataset)
    inputs = _case_inputs(contexts, jnp.asarray(receptors, dtype=contexts.dtype))
    input_mean = einx.mean("[case obs] input", inputs)
    input_std = einx.std("[case obs] input", inputs)
    # constant inputs (one case, one receptor) keep a unit scale
    input_scale = jnp.where(input_std > 0.0, input_std, 1.0)
    footprint_scale = jnp.mean(footprints)
    network = eqx.nn.MLP(
        inputs.shape[-1],
        footprints.shape[-1],
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
    dataset: Any,
    receptors: Float[Array, "obs 3"],
    *,
    val_dataset: Any | None = None,
    max_steps: int = 2000,
    batch_size: int = 16,
    learning_rate: float = 1e-2,
    patience: int = 10,
    eval_every_n_steps: int = 100,
    seed: int = 0,
) -> FootprintEmulator:
    """Fit the network weights θ with pipekit-train's Equinox backend.

    θ* = argmin_θ 𝓛(θ),  𝓛(θ) = mean over cases, receptors and cells of
    (F_emu(uᵢ, r; θ) − Fᵢ)²,

    by Adam on mini-batches of cases. The standardisation and the scale s
    stay fixed.

    Algorithm:
        1. Wrap the emulator at ``receptors`` as an ``EmulatedFootprint``,
           the model u ↦ F_emu(u) that matches the dataset's pairs.
        2. Run a pipekit-train ``TrainingLoop`` on the ``"equinox"`` backend
           with its ``MSE`` loss for at most ``max_steps`` steps.
        3. With ``val_dataset``, evaluate every ``eval_every_n_steps`` steps
           and stop through pipekit-train's ``EarlyStopping`` once the
           validation MSE has not improved for ``patience`` evaluations.

    Args:
        emulator: Initial emulator, e.g. from :func:`init_footprint_emulator`.
        dataset: The training pairs, e.g. from :func:`footprint_dataset`.
        receptors: Receptor positions (x, y, z) [m] of the footprints.
        val_dataset: Optional held-out pairs for early stopping.
        max_steps: Maximum number of optimiser steps.
        batch_size: Cases per step; at most the number of training cases.
        learning_rate: Adam learning rate.
        patience: Evaluations without improvement before stopping.
        eval_every_n_steps: Validation cadence in steps.
        seed: Seed of the batch shuffle.

    Returns:
        The emulator with the fitted weights.

    Raises:
        ImportError: Without the ``emulate`` extra.
    """
    pt = _require_pipekit_train("train_footprint_emulator")
    from pipekit_train.adapters.equinox import EquinoxModelOp

    model = EmulatedFootprint(emulator, jnp.asarray(receptors))
    callbacks = ()
    if val_dataset is not None:
        callbacks = (pt.EarlyStopping(metric="mse", patience=patience),)
    # the loss is in the footprint's units, so Adam's ε must sit below s²
    epsilon = 1e-8 * float(emulator.footprint_scale) ** 2
    loop = pt.TrainingLoop(
        model_op=EquinoxModelOp(model),
        dataset=dataset,
        val_dataset=val_dataset,
        loss=pt.MSE(),
        optimizer_config={"name": "adam", "lr": learning_rate, "eps": epsilon},
        max_steps=max_steps,
        batch_size=batch_size,
        backend="equinox",
        callbacks=callbacks,
        eval_every_n_steps=eval_every_n_steps,
        log_every_n_steps=max_steps,
        seed=seed,
    )
    trained_op, _ = loop.run()
    return trained_op.module.emulator


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
