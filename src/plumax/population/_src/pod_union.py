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

Time of day is the v1 time-varying option, as a per-bin shift of Q₅₀ on the log-Q scale:

    log Q₅₀,k(b) = log Q₅₀,k + δ_k,b,  for each hour-of-day bin b.

A shift, not a factor on P_d: poor illumination raises the detection threshold while
large plumes stay detectable, whereas a factor would cap P_d below 1 at every size.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, ArrayLike, Bool, Float, Int


#: Below this η, log softplus(η) = η and log(1 − e^{−a}) = log a to double precision.
_TAIL = -20.0

#: Hour-of-day bins of an instrument's ``tod_bins`` table.
TOD_BINS = 24

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
        tod_bins: The shift δ_b of log Q₅₀ per hour-of-day bin, ``TOD_BINS`` values,
            so that Q₅₀(b) = Q₅₀ exp(δ_b); ``None`` for no time-of-day dependence. The
            shifts are fixed, not sampled.

    Raises:
        ValueError: For another model, a parameter out of range, or a table of the
            wrong length or with a non-finite shift.
    """

    name: str
    model: str
    q50: float
    q50_std: float
    slope: float
    slope_std: float = 0.0
    tod_bins: tuple[float, ...] | None = None

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
        if self.tod_bins is not None:
            shifts = tuple(float(shift) for shift in self.tod_bins)
            if len(shifts) != TOD_BINS or not all(map(math.isfinite, shifts)):
                raise ValueError(
                    f"InstrumentPOD: tod_bins needs {TOD_BINS} finite shifts, got "
                    f"{self.tod_bins}."
                )
            # a tuple, so the record stays hashable and comparable
            object.__setattr__(self, "tod_bins", shifts)


def tod_bin_index(times: ArrayLike, *, tod_origin: float = 0.0) -> Int[Array, "*time"]:
    """The hour-of-day bin of each time.

    b(t) = ⌊24 (t + h₀ / 24)⌋ mod 24, with t in days and h₀ the hour of day at t = 0.

    Args:
        times: Times t [days].
        tod_origin: Local hour of day h₀ at t = 0 [hours].

    Returns:
        The bin b, in 0, …, 23.

    Examples:
        >>> import jax.numpy as jnp
        >>> tod_bin_index(jnp.array([0.0, 0.5, 1.25])).tolist()
        [0, 12, 6]
        >>> tod_bin_index(jnp.array([0.0]), tod_origin=23.5).tolist()
        [23]
    """
    # ⌊24 t + h₀⌋ mod 24
    hours = TOD_BINS * jnp.asarray(times, dtype=float) + tod_origin
    return jnp.floor(hours).astype(int) % TOD_BINS


def is_tod_binned(pods: Sequence[InstrumentPOD]) -> bool:
    """Whether any instrument has a time-of-day table."""
    return any(pod.tod_bins is not None for pod in pods)


def _require_bin(pods: Sequence[InstrumentPOD], tod_bin: ArrayLike | None) -> None:
    if tod_bin is None and is_tod_binned(pods):
        raise ValueError(
            "pod_union: an instrument has tod_bins, so P_d^∪ needs a tod_bin; "
            "see tod_bin_index."
        )


def _logits(
    pods: Sequence[InstrumentPOD],
    q: Float[Array, "*q"],
    calibration: Calibration | None,
    tod_bin: ArrayLike | None = None,
) -> Float[Array, "... instrument"]:
    per_instrument = []
    for pod in pods:
        values = calibration[pod.name] if calibration else {}
        q50 = jnp.asarray(values.get("q50", pod.q50), dtype=q.dtype)
        slope = jnp.asarray(values.get("slope", pod.slope), dtype=q.dtype)
        if pod.tod_bins is not None:
            # log Q₅₀(b) = log Q₅₀ + δ_b
            shift = jnp.asarray(pod.tod_bins, dtype=q.dtype)[tod_bin]
            q50 = q50 * jnp.exp(shift)
        if pod.model == "log_logistic":
            # η_k = k log(Q / Q₅₀)
            logit = slope * (jnp.log(q) - jnp.log(q50))
        else:
            # η_k = k (Q − Q₅₀)
            logit = slope * (q - q50)
        per_instrument.append(logit)
    # one rate over many bins: only the binned instruments carry the bins' shape
    return jnp.stack(jnp.broadcast_arrays(*per_instrument), axis=-1)


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
    tod_bin: Int[Array, "*bin"] | None = None,
) -> Float[Array, ...]:
    """Probability that any covering instrument detects a rate Q, at published values.

    P_d^∪(Q) = 1 − ∏_k (1 − P_d^k(Q))^{c_k}, with c_k = 1 where instrument k overpassed.

    Args:
        pods: The K instruments.
        q: Rates Q > 0, in the marks' units.
        coverage: The mask c_k, broadcast against ``q`` with instruments last; default
            every instrument. A row with no instrument gives P_d^∪ = 0.
        tod_bin: The hour-of-day bin b, broadcast against ``q``; needed when an
            instrument has ``tod_bins``, and ignored by those without.

    Returns:
        P_d^∪ at each rate.

    Raises:
        ValueError: With no instruments, a mask of the wrong width, or an instrument
            with ``tod_bins`` and no ``tod_bin``.

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
    _require_bin(pods, tod_bin)
    rates = jnp.asarray(q, dtype=float)
    logits = _logits(pods, rates, None, tod_bin)
    # a = Σ_k c_k log(1 + e^η_k)
    miss_scale = jnp.sum(jnp.where(mask, jax.nn.softplus(logits), 0.0), axis=-1)
    # P^∪ = 1 − e^{−a}
    return -jnp.expm1(-miss_scale)


def log_pod_union(
    pods: Sequence[InstrumentPOD],
    *,
    coverage: Bool[Array, "*event instrument"] | None = None,
    calibration: Calibration | None = None,
    tod_bin: Int[Array, "*bin"] | None = None,
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
        tod_bin: The hour-of-day bin b, broadcast against the rates; needed when an
            instrument has ``tod_bins``, and ignored by those without.

    Returns:
        log P_d^∪ on rates in the marks' units.

    Raises:
        ValueError: With no instruments, a mask of the wrong width, or an instrument
            with ``tod_bins`` and no ``tod_bin``.

    Examples:
        >>> import math
        >>> import jax.numpy as jnp
        >>> pod = InstrumentPOD("tanager", "log_logistic", 10.0, 0.2, 2.0)
        >>> log_pod = log_pod_union([pod])
        >>> bool(jnp.allclose(log_pod(jnp.array(10.0)), math.log(0.5)))
        True
    """
    mask = _resolve_coverage(pods, coverage)
    _require_bin(pods, tod_bin)

    def log_pod(q: Array) -> Array:
        rates = jnp.asarray(q, dtype=float)
        return _log_union(_logits(pods, rates, calibration, tod_bin), mask)

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


def _tod_table(cell: Any) -> tuple[float, ...] | None:
    # a YAML list, or a CSV cell of space-separated values
    if cell is None or cell == "":
        return None
    values = cell.split() if isinstance(cell, str) else cell
    return tuple(float(value) for value in values)


def _from_record(record: Mapping[str, Any]) -> InstrumentPOD:
    return InstrumentPOD(
        name=str(record["name"]),
        model=str(record["model"]),
        q50=float(record["q50"]),
        q50_std=float(record["q50_std"]),
        slope=float(record["slope"]),
        slope_std=float(record.get("slope_std") or 0.0),
        tod_bins=_tod_table(record.get("tod_bins")),
    )


def load_pod_calibrations(path: str | Path) -> list[InstrumentPOD]:
    """Read per-instrument calibrations from a CSV or YAML table.

    A CSV has a header naming ``name, model, q50, q50_std, slope`` and optionally
    ``slope_std`` and ``tod_bins`` (24 space-separated shifts); a YAML file is a list of
    mappings with the same keys, ``tod_bins`` a list. plumax ships no values: the table
    is the caller's record of the campaign it trusts.

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
