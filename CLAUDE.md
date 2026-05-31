# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`plumax` is a differentiable, probabilistic library of atmospheric plume
dispersion forward models and methane-retrieval operators, built on JAX
(JIT / vmap / autodiff) with NumPyro for Bayesian inference. It follows
the carrier-agnostic `Operator` primitives from
[`pipekit`](https://github.com/jejjohnson/pipekit) and plugs into data
assimilation via the `pipekit_cycle` protocols. Built with Python 3.12+,
uv, pytest, and MkDocs.

The architecture is organised as a five-tier **data-driven modeling
cycle** (simple model → model-based inference → emulator → emulator-based
inference → amortized predictor). The full roadmap lives in
[`docs/design/`](docs/design/).

## Common Commands

```bash
make install              # Install all deps (uv sync --all-groups) + pre-commit hooks
make test                 # Run tests: uv run pytest -v -o addopts=
make test-cov             # Tests with coverage gate
make format               # Auto-fix: ruff format . && ruff check --fix .
make lint                 # Lint code: ruff check .
make typecheck            # Type check: ty check src/plumax
make precommit            # Run pre-commit on all files
make docs-serve           # Local docs server
```

### Running a single test

```bash
uv run pytest tests/gauss_plume/test_plume.py::test_name -v
```

Skip the slow MCMC tests with `-m 'not slow'`.

### Pre-commit checklist (all four must pass)

```bash
uv run pytest -v                              # Tests
uv run --group lint ruff check .              # Lint — ENTIRE repo, not just src/plumax/
uv run --group lint ruff format --check .     # Format — ENTIRE repo
uv run --group typecheck ty check src/plumax  # Typecheck — package only
```

**Critical**: Always lint/format with `.` (repo root), not `src/plumax/`. CI runs `ruff check .` which includes `tests/` and `scripts/`.

## Architecture

### Package structure

All implementation lives in `src/plumax/`. Each dispersion / retrieval
model is its own sub-package so new ports can be added as siblings. The
public API is re-exported through `src/plumax/__init__.py`.

| Sub-package        | Purpose |
|--------------------|---------|
| `gauss_plume`      | Steady-state Gaussian plume (Briggs dispersion) + NumPyro emission-rate inference. **Tier I.** |
| `gauss_puff`       | Time-resolved Gaussian puff (Pasquill–Gifford), diffrax wind integration, OU sub-grid turbulence. **Tier I.** |
| `les_fvm`          | Eulerian 3-D advection–diffusion on an Arakawa C-grid via `finitevolX`. **Tier III.** |
| `hapi_lut`         | HITRAN line-by-line Voigt LUTs + Beer–Lambert forward model. **RTM stack.** |
| `radtran`          | Band-integrated radiative transfer + matched-filter retrieval. **RTM stack.** |
| `matched_filter`   | Hyperspectral matched-filter detection pipeline. **RTM stack.** |
| `assimilation`     | 3D/4D-Var cost / control / solve scaffolding (optimistix). |
| `operators`        | `pipekit.Operator` wrappers over the forward models. |
| `adapters`         | `pipekit_cycle` `ForwardModel` / `ObservationOperator` adapters. |

### pipekit integration

- Forward models are pure JAX functions; `plumax.operators` wraps them as
  `pipekit.Operator` subclasses (config round-trip via `ConfigMixin`).
- `plumax.adapters` provides classes that **structurally** satisfy
  `pipekit_cycle.protocols.ForwardModel` / `ObservationOperator` —
  `plumax` does not import `pipekit_cycle` at runtime.

### Optional / lazy dependencies

- NumPyro inference submodules are imported lazily (PEP 562) — importing a
  forward model never pulls in NumPyro.
- `hapi_lut` imports HAPI lazily; install the `hapi` extra to build LUTs.
- `gaussx` and `pipekit` resolve from git (`[tool.uv.sources]`); they are
  not yet on PyPI.

## Documentation

- Design / roadmap pages live in `docs/design/` and render under the
  **Design** nav section (MkDocs Material).
- Example notebooks live in `docs/notebooks/` as jupytext percent-format
  `.py` files; see `.github/instructions/docs-examples.instructions.md`.

## Coding Conventions

- `from __future__ import annotations` at the top of every module.
- Google-style docstrings; type hints on all public functions and methods.
- `dataclasses` for plain data carriers; keep scientific computation pure
  and isolate IO / CLI side effects.
- Surgical changes only — don't refactor adjacent code or add docstrings to
  unchanged code.

## Plans

Plans and scratch implementation docs go in `.plans/` (gitignored, never
committed). Track work via GitHub issues instead. (The committed
`docs/design/` roadmap is product design, distinct from scratch plans.)

## PR Review Comments

When addressing PR review comments, always resolve each review thread after fixing it via the GitHub GraphQL API (`resolveReviewThread` mutation). Do not leave addressed comments unresolved. To obtain the required `threadId`, first list the pull request's review threads via the GitHub GraphQL API (see the "Pull Request Review Comments" section in `AGENTS.md` for a minimal query and end-to-end workflow).

## Code Review

Follow the guidance in `/CODE_REVIEW.md` for all code review tasks.
