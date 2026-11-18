"""Array operations go through einx, never ``axis=``, ``.T``, reshapes or ``None`` (#312).

The house style writes reductions, contractions, broadcasts and rearranges
with einx, so every axis has a name. This test walks the AST of every module
under ``src/plumax`` and counts the sites that do it another way: a call
with ``axis=`` or ``axes=``, the attributes ``.T`` and ``.mT``, a call named
after a reshape (``reshape``, ``ravel``, ``transpose``, ``repeat``, …), a
subscript holding ``None`` or ``newaxis``, and any call to the deprecated
``einx.rearrange``.

A site is allowed when ``KEPT`` lists it with a reason, or while its package
still has an allowance in ``PENDING``. Each package's issue under #311
lowers its allowance to zero and deletes it. The counts are exact: the test
also fails when a package has fewer sites than it is allowed, or a KEPT
entry has fewer sites than it lists, so neither table goes stale.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "src" / "plumax"

#: calls that are a reshape, a transpose or a broadcast under another name
RESHAPES = frozenset(
    {
        "reshape",
        "ravel",
        "flatten",
        "transpose",
        "moveaxis",
        "swapaxes",
        "expand_dims",
        "squeeze",
        "repeat",
        "tile",
        "column_stack",
        "vstack",
        "hstack",
        "dstack",
        "atleast_1d",
        "atleast_2d",
        "atleast_3d",
    }
)

_XARRAY = "xarray `transpose` by dim names, which already names every axis"
_PYTREE = "pytree `flatten()` of a prior, not an array operation"
_VARIADIC = "a list of variable length; einx takes a fixed number of arrays"
_TRAPEZOID = "the trapezoid sum moves to integrix in #247"
_PLAN_244 = "rewritten on functax plans in #244"
_PLAN_245 = "rewritten on functax plans in #245"

#: (module, enclosing function, kind) -> (count, why the sites stay)
KEPT: dict[tuple[str, str, str], tuple[int, str]] = {
    ("coupled/_src/diagnostics.py", "fit_diagnostics", "flatten"): (1, _PYTREE),
    ("coupled/_src/fusion.py", "_stack_design", "axis="): (2, _VARIADIC),
    ("coupled/_src/inversion.py", "size", "flatten"): (1, _PYTREE),
    ("coupled/_src/inversion.py", "solve_coupled", "flatten"): (1, _PYTREE),
    ("coupled/_src/rtm.py", "predict", "axis="): (1, _VARIADIC),
    ("gauss_plume/_src/plume.py", "simulate_plume", "axis="): (1, _TRAPEZOID),
    ("gauss_puff/_src/puff.py", "simulate_puff", "axis="): (1, _TRAPEZOID),
    ("geo/_src/plume.py", "_masked_mass", "axis="): (
        1,
        "`integrix.rule.integrate` takes the axis as its own API",
    ),
    ("lagrangian/_src/concentration.py", "bin_positions", "reshape"): (1, _PLAN_244),
    ("lagrangian/_src/footprint.py", "_bin_surface", "reshape"): (1, _PLAN_244),
    ("les_fvm/_src/fourdvar.py", "incremental_analysis", ".T"): (
        1,
        "`.T` of a lineax operator, which keeps its own methods",
    ),
    ("met/_src/era5.py", "load_era5", "transpose"): (7, _XARRAY),
    ("met/_src/pbl.py", "pbl_height_bulk_richardson", "transpose"): (1, _XARRAY),
    ("met/_src/time.py", "from_datetimes", "atleast_1d"): (
        1,
        "coerces a scalar or a sequence of timestamps on the host",
    ),
    ("met/_src/wrf.py", "_bilinear", "None-index"): (4, _PLAN_245),
    ("met/_src/wrf.py", "_grid_relative_wind", "transpose"): (2, _XARRAY),
    ("met/_src/wrf.py", "_interpolate_vertical", "None-index"): (1, _PLAN_245),
    ("met/_src/wrf.py", "_interpolate_vertical", "axis="): (5, _PLAN_245),
    ("obs/_src/ingest.py", "load_tropomi_l2", "transpose"): (1, _XARRAY),
    ("obs/_src/ingest.py", "pixels_of", "transpose"): (1, _XARRAY),
    ("obs/_src/plume_scene.py", "_band_first", "transpose"): (1, _XARRAY),
    ("obs/_src/plume_scene.py", "_raster", "transpose"): (1, _XARRAY),
    ("population/_src/pod_union.py", "_log_union", "axis="): (
        1,
        "`logsumexp` with weights `b=`, which `einx.logsumexp` does not take",
    ),
    ("population/_src/pod_union.py", "_logits", "axis="): (1, _VARIADIC),
    ("priors/_src/inventory.py", "load_inventory", "transpose"): (1, _XARRAY),
}

#: package -> sites still to rewrite; each issue under #311 deletes its entry
PENDING: dict[str, int] = {
    "geo": 12,
    "lagrangian": 13,
    "les_fvm": 17,
    "obs": 10,
    "population": 17,
}


def _kind(node: ast.AST) -> list[str]:
    """The ways ``node`` breaks the einx rule, one entry per site."""
    kinds = []
    if isinstance(node, ast.Call):
        kinds += [f"{kw.arg}=" for kw in node.keywords if kw.arg in ("axis", "axes")]
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name in RESHAPES:
            kinds.append(name)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "rearrange"
            and getattr(func.value, "id", "") == "einx"
        ):
            kinds.append("einx.rearrange")
    elif isinstance(node, ast.Attribute) and node.attr in ("T", "mT"):
        kinds.append(".T")
    elif isinstance(node, ast.Subscript):
        index = node.slice
        parts = index.elts if isinstance(index, ast.Tuple) else [index]
        for part in parts:
            is_none = isinstance(part, ast.Constant) and part.value is None
            is_newaxis = isinstance(part, ast.Attribute) and part.attr == "newaxis"
            if is_none or is_newaxis:
                kinds.append("None-index")
                break
    return kinds


def _sites(path: Path) -> Counter[tuple[str, str, str]]:
    """Sites in one module, keyed by (module, enclosing function, kind)."""
    module = path.relative_to(PACKAGE).as_posix()
    sites: Counter[tuple[str, str, str]] = Counter()

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = function
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = child.name
            for kind in _kind(child):
                sites[(module, inner, kind)] += 1
            visit(child, inner)

    visit(ast.parse(path.read_text()), "<module>")
    return sites


def _all_sites() -> Counter[tuple[str, str, str]]:
    sites: Counter[tuple[str, str, str]] = Counter()
    for path in sorted(PACKAGE.rglob("*.py")):
        if "__pycache__" not in path.parts:
            sites += _sites(path)
    return sites


def _package(module: str) -> str:
    return module.split("/")[0]


SITES = _all_sites()


def test_no_einx_rearrange() -> None:
    rearranges = sorted(key for key in SITES if key[2] == "einx.rearrange")
    assert rearranges == [], "einx.rearrange is deprecated; call einx.id instead"


@pytest.mark.parametrize("key", sorted(KEPT), ids=lambda key: ":".join(key))
def test_kept_sites_are_exact(key: tuple[str, str, str]) -> None:
    count, _ = KEPT[key]
    # sites beyond the count are pending, and test_pending_sites_are_exact
    # counts them against the package
    assert SITES[key] >= count, (
        f"{':'.join(key)} has {SITES[key]} sites, KEPT says {count}; "
        "lower KEPT when a kept site is rewritten or removed"
    )


@pytest.mark.parametrize(
    "package",
    sorted({_package(module) for module, _, _ in SITES} | set(PENDING)),
)
def test_pending_sites_are_exact(package: str) -> None:
    pending = sum(
        count
        for key, count in SITES.items()
        if _package(key[0]) == package and key not in KEPT
    )
    pending += sum(
        max(SITES[key] - KEPT[key][0], 0) for key in KEPT if _package(key[0]) == package
    )
    allowed = PENDING.get(package, 0)
    assert pending == allowed, (
        f"plumax.{package} has {pending} array operations outside einx and "
        f"{allowed} allowed; rewrite new ones with einx (or list them in KEPT "
        "with a reason), and lower PENDING as sites are rewritten"
    )
