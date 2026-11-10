"""Deprecated aliases of integrix's log-space rules (``integrix.expect``, plumax#253); they go after one release."""

from __future__ import annotations

from plumax.population._src.integrate import (
    log_gauss_hermite,
    log_simpson,
    pareto_smoothed_log_weights,
)


__all__ = ["log_gauss_hermite", "log_simpson", "pareto_smoothed_log_weights"]
