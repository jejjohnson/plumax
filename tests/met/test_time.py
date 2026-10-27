"""Tests for ``plumax.met.time``."""

from __future__ import annotations

import numpy as np
import pytest

from plumax.met import TimeAxis


STAMPS = ["2024-06-01T00:00:00", "2024-06-01T01:00:00", "2024-06-01T03:30:00"]


def test_seconds_roundtrip():
    axis = TimeAxis.from_datetimes(STAMPS)
    np.testing.assert_array_equal(axis.seconds, [0.0, 3600.0, 12600.0])
    np.testing.assert_array_equal(axis.stamps, np.array(STAMPS, dtype="datetime64[ms]"))
    when = np.datetime64("2024-06-01T02:15:30.250")
    assert axis.to_datetime(axis.to_seconds(when)) == when


def test_explicit_t0_and_iso_stamp():
    axis = TimeAxis.from_datetimes(STAMPS[1:], t0="2024-06-01T00:00:00")
    np.testing.assert_array_equal(axis.seconds, [3600.0, 12600.0])
    assert axis.t0_iso == "2024-06-01_00:00:00"


def test_rejects_empty_and_unsorted():
    with pytest.raises(ValueError, match="at least one"):
        TimeAxis.from_datetimes([])
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeAxis.from_datetimes(STAMPS[::-1])


def test_iso_stamp_keeps_milliseconds():
    axis = TimeAxis.from_datetimes(["2024-06-01T00:00:00.250", "2024-06-01T00:00:01"])
    assert axis.t0_iso == "2024-06-01_00:00:00.250"
    # the stamp and the offsets reproduce the samples exactly
    t0 = np.datetime64(axis.t0_iso.replace("_", "T"), "ms")
    np.testing.assert_array_equal(
        t0 + (axis.seconds * 1e3).astype("timedelta64[ms]"), axis.stamps
    )
