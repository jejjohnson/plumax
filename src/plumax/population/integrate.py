"""Log-space integration for the Tier V mark likelihood (implementation in ``plumax.population._src.integrate``)."""

from __future__ import annotations

from plumax.population._src.integrate import (
    log_gauss_hermite,
    log_simpson,
    pareto_smoothed_log_weights,
)


__all__ = ["log_gauss_hermite", "log_simpson", "pareto_smoothed_log_weights"]
