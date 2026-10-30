"""TimeAxis: one clock for every tier, seconds since ``t0`` ↔ UTC timestamps.

Models count time as float seconds since a reference ``t0``; real inputs carry
UTC timestamps. Timestamps are naive ``datetime64`` read as UTC, the convention
xarray and the satellite products use.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from jaxtyping import Float, Shaped

from plumax._src.errors import TimeAxisError


#: Resolution of every timestamp a :class:`TimeAxis` hands out.
TIMESTAMP_UNIT = "datetime64[ms]"


def _as_datetime64(when: object) -> Shaped[np.ndarray, " ..."]:
    return np.asarray(when, dtype=TIMESTAMP_UNIT)


@dataclasses.dataclass(frozen=True)
class TimeAxis:
    """Seconds since ``t0`` for a set of UTC sample times.

    s_k = (τ_k − t₀) / 1 s,

    with ``τ_k`` the sample timestamps and ``t₀`` the reference the model counts
    from (by default the first sample).

    Args:
        t0: Reference time t₀, ``datetime64`` (naive, UTC).
        seconds: Sample times s_k as seconds since ``t0``.

    Examples:
        Hourly samples, and back from seconds to timestamps:

        >>> axis = TimeAxis.from_datetimes(
        ...     ["2024-06-01T00:00", "2024-06-01T01:00", "2024-06-01T03:30"]
        ... )
        >>> axis.seconds.tolist()
        [0.0, 3600.0, 12600.0]
        >>> str(axis.to_datetime(5400.0))
        '2024-06-01T01:30:00.000'

        An explicit origin, and the stamp ``MetField.t0`` stores:

        >>> late = TimeAxis.from_datetimes(["2024-06-01T01:00"], t0="2024-06-01")
        >>> late.seconds.tolist(), late.t0_iso
        ([3600.0], '2024-06-01_00:00:00')

        Unsorted timestamps are rejected:

        >>> TimeAxis.from_datetimes(["2024-06-02", "2024-06-01"])
        ... # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        TimeAxisError: TimeAxis timestamps must be strictly increasing
    """

    t0: np.datetime64
    seconds: Float[np.ndarray, " time"]

    @classmethod
    def from_datetimes(cls, stamps: object, *, t0: object | None = None) -> TimeAxis:
        """Build the axis from timestamps τ_k (anything ``datetime64`` accepts).

        Raises:
            TimeAxisError: If ``stamps`` is empty or not strictly increasing.
        """
        values = np.atleast_1d(_as_datetime64(stamps))
        if values.size == 0:
            raise TimeAxisError("TimeAxis needs at least one timestamp")
        # Δτ_k = τ_{k+1} − τ_k > 0
        steps = np.diff(values)
        if np.any(steps <= np.timedelta64(0, "ms")):
            raise TimeAxisError("TimeAxis timestamps must be strictly increasing")
        reference = values[0] if t0 is None else _as_datetime64(t0)
        # s_k = (τ_k − t₀) / 1 s
        seconds = (values - reference) / np.timedelta64(1, "s")
        return cls(t0=reference.astype(TIMESTAMP_UNIT)[()], seconds=seconds)

    def to_seconds(self, when: object) -> Float[np.ndarray, " ..."]:
        """Seconds since ``t0`` of timestamps: s = (τ − t₀) / 1 s."""
        # s = (τ − t₀) / 1 s
        seconds = (_as_datetime64(when) - self.t0) / np.timedelta64(1, "s")
        return np.asarray(seconds)

    def to_datetime(self, seconds: object) -> Shaped[np.ndarray, " ..."]:
        """Timestamps τ = t₀ + s, rounded to the millisecond."""
        # n_ms = round(1000 s)
        milliseconds = np.round(np.asarray(seconds, dtype=float) * 1e3)
        offsets = milliseconds.astype(np.int64).astype("timedelta64[ms]")
        # τ = t₀ + n_ms · 1 ms
        return np.asarray(self.t0 + offsets)

    @property
    def stamps(self) -> Shaped[np.ndarray, " time"]:
        """The sample times τ_k as ``datetime64[ms]``."""
        return self.to_datetime(self.seconds)

    @property
    def t0_iso(self) -> str:
        """``t0`` as ``YYYY-MM-DD_HH:MM:SS[.fff]``, the stamp ``MetField.t0`` stores.

        Milliseconds are kept when ``t0`` has them, so ``t0_iso`` plus
        ``seconds`` reproduces every sample time exactly.
        """
        milliseconds = np.asarray(self.t0).astype(TIMESTAMP_UNIT)
        whole_seconds = milliseconds.astype("datetime64[s]")
        has_fraction = milliseconds != whole_seconds
        text = str(milliseconds if has_fraction else whole_seconds)
        return text.replace("T", "_")
