"""Input checks that either plumax or the test suite's type-check hook may raise.

``tests/conftest.py`` type-checks plumax's ``_src`` packages with beartype,
which rejects an argument of the wrong shape or type before plumax's own
check runs. :func:`raises_checked` accepts plumax's error with its message,
or the hook's ``TypeCheckError`` on the same parameter.

:data:`pep484_beartype` is the checker the hook runs: beartype with PEP 484's
implicit numeric tower, so an ``int`` passes where ``float`` is annotated, as
the annotations mean and as ty reads them.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator

import pytest
from beartype import BeartypeConf, beartype
from jaxtyping import TypeCheckError


#: beartype with the implicit numeric tower: int where float, float where complex
pep484_beartype = beartype(conf=BeartypeConf(is_pep484_tower=True))


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
