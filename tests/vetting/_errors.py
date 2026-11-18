"""A shape check that either plumax or the test suite's type-check hook may raise.

``tests/conftest.py`` type-checks ``plumax.vetting._src`` with beartype, which
rejects an argument of the wrong rank before plumax's own check runs. The
helper accepts plumax's ``ShapeError`` with its message, or the hook's
``TypeCheckError`` on the same parameter.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator

import pytest
from jaxtyping import TypeCheckError
from radtrax import ShapeError


@contextlib.contextmanager
def raises_shape_error(match: str, *, parameter: str) -> Iterator[None]:
    """``ShapeError`` matching ``match``, or ``TypeCheckError`` on ``parameter``."""
    with pytest.raises((ShapeError, TypeCheckError)) as caught:
        yield
    if isinstance(caught.value, TypeCheckError):
        assert f"parameter '{parameter}'" in str(caught.value)
    else:
        assert re.search(match, str(caught.value))
