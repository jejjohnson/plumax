"""One clock for every tier: seconds since ``t0`` ↔ UTC timestamps.

Models count time as float seconds since a reference ``t0`` (``MetField``,
the puff and Lagrangian integrators); real inputs carry UTC timestamps.
A :class:`TimeAxis` holds both, so every loader converts the same way.
Timestamps are naive ``datetime64`` read as UTC, the convention xarray
and the satellite products use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: Resolution of every timestamp a :class:`TimeAxis` hands out.
_UNIT = "datetime64[ms]"


def _as_datetime64(when: object) -> np.ndarray:
    return np.asarray(when, dtype=_UNIT)


@dataclass(frozen=True)
class TimeAxis:
    """Seconds since ``t0`` for a set of UTC sample times.

    ``seconds[k] = (stamp[k] − t0) / 1 s``, with ``t0`` the reference the
    model counts from (by default the first sample).

    Attributes:
        t0: Reference time, ``datetime64`` (naive, UTC).
        seconds: Sample times as float seconds since ``t0``, shape ``(T,)``.
    """

    t0: np.datetime64
    seconds: np.ndarray

    @classmethod
    def from_datetimes(cls, stamps: object, *, t0: object | None = None) -> TimeAxis:
        """Build the axis from timestamps.

        Args:
            stamps: Sample times, anything ``numpy.datetime64`` accepts
                (strings, ``datetime64`` arrays, pandas ``DatetimeIndex``).
            t0: Reference time; the first stamp when ``None``.

        Returns:
            The axis.

        Raises:
            ValueError: If ``stamps`` is empty or not strictly increasing.
        """
        values = np.atleast_1d(_as_datetime64(stamps))
        if values.size == 0:
            raise ValueError("TimeAxis needs at least one timestamp")
        if np.any(np.diff(values) <= np.timedelta64(0, "ms")):
            raise ValueError("TimeAxis timestamps must be strictly increasing")
        ref = values[0] if t0 is None else _as_datetime64(t0)
        seconds = (values - ref) / np.timedelta64(1, "s")
        return cls(t0=ref.astype("datetime64[ms]"), seconds=seconds.astype(np.float64))

    def to_seconds(self, when: object) -> np.ndarray:
        """Seconds since ``t0`` of timestamps.

        Args:
            when: One timestamp or an array of them.

        Returns:
            Float seconds, of ``when``'s shape.
        """
        return (_as_datetime64(when) - self.t0) / np.timedelta64(1, "s")

    def to_datetime(self, seconds: object) -> np.ndarray:
        """Timestamps of seconds since ``t0``, rounded to the millisecond.

        Args:
            seconds: One value or an array of them.

        Returns:
            ``datetime64[ms]`` of ``seconds``' shape.
        """
        ms = np.round(np.asarray(seconds, dtype=np.float64) * 1e3).astype(np.int64)
        return self.t0 + ms.astype("timedelta64[ms]")

    @property
    def stamps(self) -> np.ndarray:
        """The sample times as ``datetime64[ms]``."""
        return self.to_datetime(self.seconds)

    @property
    def t0_iso(self) -> str:
        """``t0`` as ``YYYY-MM-DD_HH:MM:SS[.fff]``, the stamp ``MetField.t0`` stores.

        Milliseconds are kept when ``t0`` has them, so ``t0_iso`` plus
        ``seconds`` reproduces every sample time exactly.
        """
        ms = np.asarray(self.t0).astype("datetime64[ms]")
        whole = ms.astype("datetime64[s]")
        text = str(ms if ms != whole else whole)
        return text.replace("T", "_")
