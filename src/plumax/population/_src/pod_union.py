"""Multi-satellite POD: the union over instruments and its calibration prior.

A catalog merged from K instruments is thinned by "any instrument saw it",

    P_d^∪(Q) = 1 − ∏_k (1 − P_d^k(Q)),

which assumes detections are conditionally independent given Q. The likelihood
replaces P_d with P_d^∪ and stays the TMTPP of 06b. This is the v1 decision of the
design pages (06, 06d): union, not a categorical mark for which instrument detected
the event. The categorical mark gives more information per event but doubles the POD
parameters, and is v2.

Calibration is the v1 hierarchical prior of 06b § POD calibration uncertainty:

    Q₅₀,k ~ LogNormal(log Q₅₀,pub, σ²_Q₅₀,pub),  k_k ~ LogNormal(log k_pub, σ²_k,pub).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float


#: Below this η, log softplus(η) = η and log(1 − e^{−a}) = log a to double precision.
_TAIL = -20.0

#: The registry curves that depend on Q alone, so a union over instruments is defined.
_FLUX_ONLY_MODELS = ("log_logistic", "logistic")

Calibration = Mapping[str, Mapping[str, ArrayLike]]


@dataclass(frozen=True)
class InstrumentPOD:
    """One instrument's detection curve with its published calibration.

    Args:
        name: Instrument name, unique within a union; names the prior's sites.
        model: A registry curve in Q alone. ``"log_logistic"`` is the Hill curve of
            06b, P_d(Q) = 1 / (1 + (Q₅₀ / Q)^k) = σ(k log(Q / Q₅₀));
            ``"logistic"`` is σ(k (Q − Q₅₀)).
        q50: Published Q₅₀,pub > 0, in the marks' units.
        q50_std: σ_Q₅₀,pub ≥ 0, on the log scale; 0 fixes Q₅₀ at its published value.
        slope: Published k_pub > 0.
        slope_std: σ_k,pub ≥ 0, on the log scale; 0 fixes k at its published value.

    Raises:
        ValueError: For another model, or a parameter out of range.
    """

    name: str
    model: str
    q50: float
    q50_std: float
    slope: float
    slope_std: float = 0.0

    def __post_init__(self) -> None:
        if self.model not in _FLUX_ONLY_MODELS:
            raise ValueError(
                f"InstrumentPOD: model must be one of {_FLUX_ONLY_MODELS}, the "
                f"registry curves in Q alone, got {self.model!r}."
            )
        if not (self.q50 > 0.0 and self.slope > 0.0):
            raise ValueError(
                f"InstrumentPOD: q50 and slope must be > 0, got "
                f"({self.q50}, {self.slope})."
            )
        if self.q50_std < 0.0 or self.slope_std < 0.0:
            raise ValueError(
                f"InstrumentPOD: q50_std and slope_std must be ≥ 0, got "
                f"({self.q50_std}, {self.slope_std})."
            )


def _logits(
    pods: Sequence[InstrumentPOD],
    q: Float[Array, "*q"],
    calibration: Calibration | None,
) -> Float[Array, "*q instrument"]:
    per_instrument = []
    for pod in pods:
        values = calibration[pod.name] if calibration else {}
        q50 = jnp.asarray(values.get("q50", pod.q50), dtype=q.dtype)
        slope = jnp.asarray(values.get("slope", pod.slope), dtype=q.dtype)
        if pod.model == "log_logistic":
            # η_k = k log(Q / Q₅₀)
            logit = slope * (jnp.log(q) - jnp.log(q50))
        else:
            # η_k = k (Q − Q₅₀)
            logit = slope * (q - q50)
        per_instrument.append(logit)
    return jnp.stack(per_instrument, axis=-1)


def _log_softplus(logit: Float[Array, "*q"]) -> Float[Array, "*q"]:
    # log log(1 + e^η): softplus underflows in the tail, where it equals e^η. The
    # stand-in keeps the unused branch's gradient finite.
    in_tail = logit < _TAIL
    safe_logit = jnp.where(in_tail, 0.0, logit)
    return jnp.where(in_tail, logit, jnp.log(jax.nn.softplus(safe_logit)))


def _log_union(
    logits: Float[Array, "*q instrument"], mask: Bool[Array, "*mask instrument"]
) -> Float[Array, "*q"]:
    # a = −log(1 − P^∪) = Σ_k c_k log(1 + e^η_k), held as its log for the tail
    log_miss_scale = logsumexp(_log_softplus(logits), b=mask.astype(float), axis=-1)
    in_tail = log_miss_scale < _TAIL
    safe_scale = jnp.where(in_tail, 0.0, log_miss_scale)
    # log P^∪ = log(1 − e^{−a}), which is log a when a ≪ 1
    log_detected = jnp.log(-jnp.expm1(-jnp.exp(safe_scale)))
    return jnp.where(in_tail, log_miss_scale, log_detected)


def _resolve_coverage(
    pods: Sequence[InstrumentPOD], coverage: ArrayLike | None
) -> Bool[Array, "*event instrument"]:
    if not pods:
        raise ValueError("pod_union: no instruments given.")
    if coverage is None:
        return jnp.ones(len(pods), dtype=bool)
    mask = jnp.asarray(coverage, dtype=bool)
    if mask.shape[-1] != len(pods):
        raise ValueError(
            f"pod_union: coverage needs one column per instrument ({len(pods)}), "
            f"got shape {mask.shape}."
        )
    return mask


def pod_union(
    pods: Sequence[InstrumentPOD],
    q: ArrayLike,
    *,
    coverage: Bool[Array, "event instrument"] | None = None,
) -> Float[Array, "*q"]:
    """Probability that any covering instrument detects a rate Q, at published values.

    P_d^∪(Q) = 1 − ∏_k (1 − P_d^k(Q))^{c_k}, with c_k = 1 where instrument k overpassed.

    Args:
        pods: The K instruments.
        q: Rates Q > 0, in the marks' units.
        coverage: The mask c_k, broadcast against ``q`` with instruments last; default
            every instrument. A row with no instrument gives P_d^∪ = 0.

    Returns:
        P_d^∪ at each rate.

    Raises:
        ValueError: With no instruments, or a mask of the wrong width.

    Examples:
        One instrument is its own curve; at Q₅₀ that is one half:

        >>> import jax.numpy as jnp
        >>> tropomi = InstrumentPOD("tropomi", "log_logistic", 25.0, 0.3, 2.0)
        >>> bool(jnp.allclose(pod_union([tropomi], jnp.array(25.0)), 0.5))
        True

        Two instruments at their midpoints miss one time in four:

        >>> ghgsat = InstrumentPOD("ghgsat", "log_logistic", 25.0, 0.3, 4.0)
        >>> bool(jnp.allclose(pod_union([tropomi, ghgsat], jnp.array(25.0)), 0.75))
        True

        An uncovered instrument contributes nothing:

        >>> mask = jnp.array([[True, False]])
        >>> rate = jnp.array([25.0])
        >>> bool(jnp.allclose(pod_union([tropomi, ghgsat], rate, coverage=mask), 0.5))
        True
    """
    mask = _resolve_coverage(pods, coverage)
    rates = jnp.asarray(q, dtype=float)
    logits = _logits(pods, rates, None)
    # a = Σ_k c_k log(1 + e^η_k)
    miss_scale = jnp.sum(jnp.where(mask, jax.nn.softplus(logits), 0.0), axis=-1)
    # P^∪ = 1 − e^{−a}
    return -jnp.expm1(-miss_scale)


def log_pod_union(
    pods: Sequence[InstrumentPOD],
    *,
    coverage: Bool[Array, "*event instrument"] | None = None,
    calibration: Calibration | None = None,
) -> Callable[[Array], Array]:
    """log P_d^∪ as the callable the TMTPP model and ``total_emission`` take.

    Accurate in both tails: log P_d^∪ stays finite with a finite gradient far below
    every Q₅₀, where 1 − ∏_k (1 − P_d^k) would round to zero.

    Args:
        pods: The K instruments.
        coverage: The mask c_k, broadcast against the rates with instruments last;
            default every instrument. Every row needs at least one instrument.
        calibration: Q₅₀ and k per instrument (``{name: {"q50": …, "slope": …}}``), as
            :func:`calibration_prior` returns; default the published values.

    Returns:
        log P_d^∪ on rates in the marks' units.

    Raises:
        ValueError: With no instruments, or a mask of the wrong width.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> pod = InstrumentPOD("tanager", "log_logistic", 10.0, 0.2, 2.0)
        >>> log_pod = log_pod_union([pod])
        >>> bool(jnp.allclose(log_pod(jnp.array(10.0)), math.log(0.5)))
        True
    """
    mask = _resolve_coverage(pods, coverage)

    def log_pod(q: Array) -> Array:
        return _log_union(_logits(pods, jnp.asarray(q, dtype=float), calibration), mask)

    return log_pod


def calibration_prior(pods: Sequence[InstrumentPOD]) -> dict[str, dict[str, Array]]:
    """Sample each instrument's (Q₅₀, k) under its published calibration.

    Q₅₀,k ~ LogNormal(log Q₅₀,pub, σ²_Q₅₀,pub), k_k ~ LogNormal(log k_pub, σ²_k,pub),
    at sites ``pod.<name>.q50`` and ``pod.<name>.slope``. A parameter with zero spread
    stays at its published value and gets no site.

    Args:
        pods: The K instruments.

    Returns:
        ``{name: {"q50": …, "slope": …}}`` for :func:`log_pod_union`.

    Raises:
        ValueError: If two instruments share a name.
    """
    import numpyro
    import numpyro.distributions as dist

    names = [pod.name for pod in pods]
    if len(set(names)) != len(names):
        raise ValueError(f"calibration_prior: instrument names repeat in {names}.")

    def draw(site: str, published: float, spread: float) -> Array:
        if spread == 0.0:
            return jnp.asarray(published, dtype=float)
        location = jnp.log(jnp.asarray(published, dtype=float))
        return numpyro.sample(site, dist.LogNormal(location, spread))

    return {
        pod.name: {
            "q50": draw(f"pod.{pod.name}.q50", pod.q50, pod.q50_std),
            "slope": draw(f"pod.{pod.name}.slope", pod.slope, pod.slope_std),
        }
        for pod in pods
    }


def calibration_from_draw(
    pods: Sequence[InstrumentPOD], draw: Mapping[str, Array]
) -> dict[str, dict[str, Array]]:
    """One posterior draw's (Q₅₀, k) per instrument, published where it has no site.

    Args:
        pods: The K instruments.
        draw: One draw with the sites of :func:`calibration_prior`.

    Returns:
        ``{name: {"q50": …, "slope": …}}`` for :func:`log_pod_union`.
    """
    return {
        pod.name: {
            "q50": draw.get(f"pod.{pod.name}.q50", jnp.asarray(pod.q50, dtype=float)),
            "slope": draw.get(
                f"pod.{pod.name}.slope", jnp.asarray(pod.slope, dtype=float)
            ),
        }
        for pod in pods
    }


def _from_record(record: Mapping[str, Any]) -> InstrumentPOD:
    return InstrumentPOD(
        name=str(record["name"]),
        model=str(record["model"]),
        q50=float(record["q50"]),
        q50_std=float(record["q50_std"]),
        slope=float(record["slope"]),
        slope_std=float(record.get("slope_std") or 0.0),
    )


def load_pod_calibrations(path: str | Path) -> list[InstrumentPOD]:
    """Read per-instrument calibrations from a CSV or YAML table.

    A CSV has a header naming ``name, model, q50, q50_std, slope`` and optionally
    ``slope_std``; a YAML file is a list of mappings with the same keys. plumax ships
    no values: the table is the caller's record of the campaign it trusts.

    Args:
        path: The ``.csv``, ``.yaml`` or ``.yml`` file.

    Returns:
        One :class:`InstrumentPOD` per row.

    Raises:
        ValueError: For another extension, a missing key, or a value out of range.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".csv":
        with source.open(newline="") as handle:
            records: list[dict[str, Any]] = list(csv.DictReader(handle))
    elif suffix in {".yaml", ".yml"}:
        import yaml

        with source.open() as handle:
            records = yaml.safe_load(handle)
    else:
        raise ValueError(
            f"load_pod_calibrations: expected .csv, .yaml or .yml, got {source.name}."
        )
    try:
        return [_from_record(record) for record in records]
    except KeyError as exc:
        raise ValueError(
            f"load_pod_calibrations: {source.name} has no column {exc}."
        ) from exc
