"""Legendre polynomial helpers shared across VRR calculations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.special as sp


def _to_output(value: np.ndarray | float) -> float | np.ndarray:
    """Return a Python float when ``value`` is scalar, otherwise an array."""

    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr


def legendre_series_ells(l_max: int, start: int = 0) -> np.ndarray:
    """Return even Legendre degrees from ``start`` up to ``l_max`` (inclusive)."""

    if l_max < start:
        return np.array([], dtype=int)
    first = start + (start % 2)
    return np.arange(first, l_max + 1, 2, dtype=int)


def even_ells(ells: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return the doubled multipole indices ``L = 2ℓ`` for the provided ``ℓ``."""

    return 2 * np.asarray(ells, dtype=int)


def legendre_P(l: int | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    """Evaluate the Legendre polynomial ``P_l(x)`` for scalar or array input."""

    l_arr = np.asarray(l)
    x_arr = np.asarray(x)
    l_broadcast, x_broadcast = np.broadcast_arrays(l_arr, x_arr)
    l_broadcast = l_broadcast.astype(int, copy=False)
    x_broadcast = x_broadcast.astype(float, copy=False)
    values = np.real_if_close(sp.lpmv(0, l_broadcast, x_broadcast), tol=1.0e4)
    return _to_output(values)


def legendre_P_derivative(
    l: int | np.ndarray, x: float | np.ndarray
) -> float | np.ndarray:
    """Derivative of the Legendre polynomial ``d/dx P_l(x)`` (vectorised)."""

    l_arr = np.asarray(l)
    x_arr = np.asarray(x)
    l_broadcast, x_broadcast = np.broadcast_arrays(l_arr, x_arr)
    l_broadcast = l_broadcast.astype(int, copy=False)
    x_broadcast = x_broadcast.astype(float, copy=False)

    result = np.zeros_like(x_broadcast, dtype=float)
    mask = np.isclose(np.abs(x_broadcast), 1.0)
    if np.any(~mask):
        l_non = l_broadcast[~mask]
        x_non = x_broadcast[~mask]
        denom = np.sqrt(np.maximum(1.0 - x_non * x_non, 1.0e-300))
        values = -sp.lpmv(1, l_non, x_non) / denom
        result[~mask] = np.real_if_close(values, tol=1.0e4)
    return _to_output(result)


def legendre_P_zero(l: int | np.ndarray) -> float | np.ndarray:
    """Value of ``P_l(0)`` computed in log-space for numerical stability."""

    ell = np.asarray(l, dtype=int)
    if ell.size == 0:
        return np.asarray(ell, dtype=float)
    result = np.zeros_like(ell, dtype=float)
    mask_even = (ell % 2) == 0
    if np.any(mask_even):
        k = (ell[mask_even] // 2).astype(int)
        log_gamma_2k = sp.gammaln(2 * k + 1)
        log_gamma_k = sp.gammaln(k + 1)
        log_val = log_gamma_2k - 2 * k * np.log(2.0) - 2 * log_gamma_k
        result[mask_even] = ((-1.0) ** k) * np.exp(log_val)
    return _to_output(result)


__all__ = [
    "even_ells",
    "legendre_series_ells",
    "legendre_P",
    "legendre_P_derivative",
    "legendre_P_zero",
]
