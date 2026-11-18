"""Input checks that either plumax or the test suite's type-check hook may raise.

``tests/conftest.py`` type-checks plumax's ``_src`` packages with beartype,
which rejects an argument of the wrong shape or type before plumax's own
check runs. :func:`raises_checked` accepts plumax's error with its message,
or the hook's ``TypeCheckError`` on the same parameter.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator

import pytest
from jaxtyping import TypeCheckError


@contextlib.contextmanager
def raises_checked(
    error: type[Exception], match: str, *, parameter: str
) -> Iterator[None]:
    """``error`` matching ``match``, or ``TypeCheckError`` on ``parameter``."""
    with pytest.raises((error, TypeCheckError)) as caught:
        yield
    if isinstance(caught.value, TypeCheckError):
        assert f"parameter '{parameter}'" in str(caught.value)
    else:
        assert re.search(match, str(caught.value))
