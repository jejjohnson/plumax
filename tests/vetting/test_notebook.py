"""Gate VT9: the plume-vetting notebook is a pure user of the two packages (#160).

The committed notebook (``docs/notebooks/vetting/01_vetting_one_scene.ipynb``)
defines none of the vetting names, imports each from ``plumax.vetting`` or
``radtrax.retrieval.absorption``, and its stored outputs print the numbers
that the theory page (``docs/theory/plume_vetting.md``) quotes. The notebook
is not re-executed here: mystmd renders the stored outputs, so they are what
a reader sees.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


DOCS = Path(__file__).resolve().parents[2] / "docs"
NOTEBOOK = DOCS / "notebooks" / "vetting" / "01_vetting_one_scene.ipynb"
PAGE = DOCS / "theory" / "plume_vetting.md"

PLUMAX = {"select_seeds", "background_pool", "null_masks", "null_pvalue"}
PLUMAX_ALL = PLUMAX | {"vet_candidate", "VettingResult"}
RADTRAX = {
    "AbsorptionWindow",
    "transmittance_table",
    "pair_pixels",
    "ratio_spectrum",
    "TransmittanceFit",
    "fit_transmittance",
    "d_norm",
}


def _cells(kind: str) -> list[dict]:
    notebook = json.loads(NOTEBOOK.read_text())
    return [c for c in notebook["cells"] if c["cell_type"] == kind]


def _code() -> ast.Module:
    source = "\n".join("".join(c["source"]) for c in _cells("code"))
    return ast.parse(source)


def _printed() -> str:
    return "".join(
        "".join(o.get("text", ""))
        for c in _cells("code")
        for o in c.get("outputs", [])
        if o["output_type"] == "stream"
    )


def test_the_notebook_defines_no_vetting_name() -> None:
    defined = {
        node.name
        for node in ast.walk(_code())
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    assert not defined & (PLUMAX_ALL | RADTRAX | {"_grow", "VettingWindow"})


def test_the_notebook_imports_from_plumax_vetting_and_radtrax_absorption() -> None:
    imported: dict[str, set[str]] = {}
    for node in ast.walk(_code()):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.setdefault(node.module, set()).update(a.name for a in node.names)
    used_plumax = imported.get("plumax.vetting", set())
    used_radtrax = imported.get("radtrax.retrieval.absorption", set())
    assert used_plumax and used_plumax <= PLUMAX_ALL
    assert used_radtrax and used_radtrax <= RADTRAX
    assert "vet_candidate" in used_plumax
    # No vetting name comes from anywhere else.
    for module, names in imported.items():
        if module not in {"plumax.vetting", "radtrax.retrieval.absorption"}:
            assert not names & (PLUMAX_ALL | RADTRAX), module


def test_no_stored_output_is_an_error() -> None:
    outputs = [o for c in _cells("code") for o in c.get("outputs", [])]
    assert outputs
    assert all(o["output_type"] != "error" for o in outputs)


@pytest.mark.parametrize(
    ("printed", "on_page"),
    [
        ("A: ΔX̂ = 1,741 ppm·m, D_norm = 0.14", ("1,741 ppm·m", "0.14")),
        ("B: ΔX̂ = 0 ppm·m, D_norm = 1.00", ("1.00",)),
        ("mean MF 593 ppm·m", ("593 ppm·m",)),
        ("mean MF 1,162 ppm·m", ("1,162 ppm·m",)),
        ("the fit recovers 93 %", ("93 %",)),
        ("A with a constant coefficient: 1,691 ppm·m", ("1,691 ppm·m",)),
        ("A: p̂ = 0.00", ("0.00",)),
        ("B: p̂ = 0.45", ("0.45",)),
    ],
)
def test_the_notebook_prints_the_pages_numbers(
    printed: str, on_page: tuple[str, ...]
) -> None:
    assert printed in _printed()
    page = PAGE.read_text()
    for number in on_page:
        assert number in page
