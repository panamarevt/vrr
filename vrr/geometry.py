"""Geometry-focused helpers for VRR couplings and Legendre series.

Logging
-------
This module emits debug-level logs using :mod:`loguru` when available.
Key functions such as :func:`s_ijl`, :func:`J_exact`, and :func:`J_series`
log the ℓ values used, chosen computational paths, and basic result
summaries. A convenience wrapper :func:`timed_J_ijl` measures and logs
the execution time of :func:`J_exact`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.special as sp

from .legendre import even_ells, legendre_P, legendre_P_derivative, legendre_P_zero
from .orbits import OrbitPair, TorqueValue

# Prefer loguru's logger; fall back to stdlib logging if unavailable
try:  # pragma: no cover - trivial import guard
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback path
    import logging as _logging

    logger = _logging.getLogger("vrr.geometry")
    if not logger.handlers:
        _handler = _logging.StreamHandler()
        _formatter = _logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(_logging.DEBUG)


def _to_output(value: np.ndarray | float) -> float | np.ndarray:
    """Return a scalar when ``value`` is zero-dimensional."""

    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr


def _s_non_overlapping(pair: OrbitPair, ell: np.ndarray) -> np.ndarray:
    """Return ``s_{ijℓ}`` using the non-overlapping closed form."""

    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    result = np.ones_like(ell_arr, dtype=float)
    mask = ell_arr > 0
    if not np.any(mask):
        return result

    chi_in = pair.inner.chi
    chi_out = pair.outer.chi
    ell_pos = ell_arr[mask]

    prefactor = (chi_out ** ell_pos) / (chi_in ** (ell_pos + 1))
    P_lp1_in = sp.eval_legendre(ell_pos + 1, chi_in)
    P_lm1_out = sp.eval_legendre(ell_pos - 1, chi_out)
    values = np.real_if_close(prefactor * P_lp1_in * P_lm1_out, tol=1.0e4)
    result[mask] = values.astype(float, copy=False)
    return result


def _s_overlapping(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Evaluate ``s_{ijℓ}`` for overlapping annuli via Gauss-Legendre quadrature."""
    t0_glob = perf_counter()
    logger.debug(
        f"s_ijl/overlap: start | nodes={int(nodes)} | ecc_tol={float(ecc_tol):.2e} | ell_size={np.size(ell)}"
    )
    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        out = np.asarray(ell_arr, dtype=float)
        dt = perf_counter() - t0_glob
        logger.debug(f"s_ijl/overlap: done | empty ℓ | {dt*1e3:.2f} ms")
        return out

    inner, outer = pair.inner, pair.outer
    alpha = inner.a / outer.a
    if alpha <= 0.0 or not np.isfinite(alpha):
        return _to_output(np.zeros_like(ell_arr, dtype=float))
    if abs(inner.e) < ecc_tol and abs(outer.e) < ecc_tol:
        return _to_output(np.ones_like(ell_arr, dtype=float))

    nodes_in, weights_in = np.polynomial.legendre.leggauss(nodes)
    nodes_out, weights_out = np.polynomial.legendre.leggauss(nodes)

    phi_in = 0.5 * (nodes_in + 1.0) * np.pi
    phi_out = 0.5 * (nodes_out + 1.0) * np.pi
    w_in = 0.5 * np.pi * weights_in
    w_out = 0.5 * np.pi * weights_out

    A = 1.0 + inner.e * np.cos(phi_in)
    B = 1.0 + outer.e * np.cos(phi_out)

    A2d = A.reshape(nodes, 1)
    B2d = B.reshape(1, nodes)
    weights = w_in.reshape(nodes, 1) * w_out.reshape(1, nodes)

    ell_flat = ell_arr.reshape(-1)
    A_L = A[:, None] ** ell_flat[None, :]
    A_Lp1 = A[:, None] ** (ell_flat[None, :] + 1)
    B_L = B[:, None] ** ell_flat[None, :]
    B_Lp1 = B[:, None] ** (ell_flat[None, :] + 1)

    reg1_mask = (alpha * A2d <= B2d)[..., None]
    denom_B = np.maximum(B_L[None, :, :], 1.0e-300)
    reg1 = A_Lp1[:, None, :] / denom_B

    alpha_pow = alpha ** (-(2 * ell_flat + 1))
    denom_A = np.maximum(A_L[:, None, :], 1.0e-300)
    reg2 = alpha_pow[None, None, :] * (B_Lp1[None, :, :] / denom_A)

    integrand = np.where(reg1_mask, reg1, reg2)
    total = np.tensordot(weights, integrand, axes=([0, 1], [0, 1]))
    result = (total / (np.pi ** 2)).reshape(ell_arr.shape)
    out = _to_output(result.astype(float, copy=False))
    dt = perf_counter() - t0_glob
    logger.debug(
        f"s_ijl/overlap: done | nodes={int(nodes)} | ell_size={ell_arr.size} | out.shape={np.shape(out)} | {dt*1e3:.2f} ms"
    )
    return out


def s_ijl(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    method: str = "auto",
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Return the coefficient ``s_{ijℓ}`` for the supplied pair and indices.

    Debug logging:
    - Logs input ℓ values (size/min/max and first few entries) and ``method``.
    - Logs which computational path was taken (closed form vs overlapping
      quadrature vs auto path resolution) along with key parameters.
    """

    t0_total = perf_counter()
    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    ell_flat = ell_arr.reshape(-1)
    result_flat = np.ones_like(ell_flat, dtype=float)

    # Summarise ℓ input for logs without spamming large arrays
    def _ell_summary(vals: np.ndarray) -> str:
        vals = vals.reshape(-1)
        head = ", ".join(map(str, vals[:5]))
        return f"size={vals.size}, min={vals.min()}, max={vals.max()}, head=[{head}{'' if vals.size<=5 else ', …'}]"

    logger.debug(
        f"s_ijl: start | {_ell_summary(ell_flat)} | non_overlap={bool(getattr(pair, 'non_overlapping', False))} "
        f"| nodes={int(nodes)} | ecc_tol={float(ecc_tol):.2e} | method={str(method)}"
    )

    if np.all(ell_flat == 0):
        logger.debug("s_ijl: all ℓ are zero; returning ones")
        return _to_output(result_flat.reshape(ell_arr.shape))

    inner, outer = pair.inner, pair.outer
    if abs(inner.e) < ecc_tol and abs(outer.e) < ecc_tol:
        logger.debug("s_ijl: both orbits ~circular (|e|<tol); returning ones")
        return _to_output(result_flat.reshape(ell_arr.shape))

    mask_nonzero = ell_flat != 0
    if not np.any(mask_nonzero):
        logger.debug("s_ijl: no non-zero ℓ entries after mask; returning ones")
        return _to_output(result_flat.reshape(ell_arr.shape))

    mode = (method or "auto").lower()
    if mode == "simplified":
        mode = "auto"

    if mode not in {"auto", "exact", "closed_form"}:
        raise ValueError(f"Unknown s_ijl method '{method}'.")

    if mode == "closed_form":
        logger.debug("s_ijl: using closed-form non-overlapping expression")
        if not pair.non_overlapping:
            raise ValueError("Closed-form s_ijl is only valid for non-overlapping orbits.")
        values = _s_non_overlapping(pair, ell_flat[mask_nonzero])
    elif mode == "exact":
        logger.debug(f"s_ijl: using exact overlapping quadrature (nodes={int(nodes)})")
        values = _s_overlapping(
            pair,
            ell_flat[mask_nonzero],
            nodes=nodes,
            ecc_tol=ecc_tol,
        )
    else:
        if pair.non_overlapping:
            logger.debug("s_ijl: auto mode resolved to non-overlapping closed form")
            values = _s_non_overlapping(pair, ell_flat[mask_nonzero])
        else:
            logger.debug(f"s_ijl: auto mode resolved to overlapping quadrature (nodes={int(nodes)})")
            values = _s_overlapping(
                pair,
                ell_flat[mask_nonzero],
                nodes=nodes,
                ecc_tol=ecc_tol,
            )

    result_flat[mask_nonzero] = np.asarray(values, dtype=float)
    out = _to_output(result_flat.reshape(ell_arr.shape))
    dt_total = perf_counter() - t0_total
    logger.debug(f"s_ijl: done | output.shape={np.shape(out)} | {dt_total*1e3:.2f} ms")
    return out


def J_exact(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    method: str = "auto",
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Return the geometry-only coupling ``J_{ijℓ}`` for ``pair``.

    Debug logging:
    - Logs ℓ summary and the chosen ``s_ijl`` configuration.
    - Logs basic geometry scale factors used in the closed form.
    """

    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    s_val = np.asarray(
        s_ijl(pair, ell_arr, method=method, nodes=nodes, ecc_tol=ecc_tol), dtype=float
    )
    leg0 = np.asarray(legendre_P_zero(ell_arr), dtype=float)

    a_inner = pair.inner.a
    a_outer = max(pair.outer.a, 1.0e-300)

    def _ell_summary(vals: np.ndarray) -> str:
        vals = np.asarray(vals).reshape(-1)
        head = ", ".join(map(str, vals[:5]))
        return f"size={vals.size}, min={vals.min()}, max={vals.max()}, head=[{head}{'' if vals.size<=5 else ', …'}]"

    logger.debug(
        (
            f"J_exact: ℓ {_ell_summary(ell_arr)} | s_method={str(method)} | nodes={int(nodes)} "
            f"| ecc_tol={float(ecc_tol):.2e} | a_in={float(a_inner):.6g} | a_out={float(a_outer):.6g}"
        )
    )

    values = s_val * (leg0 ** 2) * (a_inner ** ell_arr) / (a_outer ** (ell_arr + 1))
    out = _to_output(values.reshape(ell_arr.shape))
    logger.debug(f"J_exact: done | output.shape={np.shape(out)}")
    return out


def J_series(
    pair: OrbitPair,
    ells: Sequence[int],
    *,
    use_sijl: bool = True,
    s_method: str = "auto",
    N_overlap: int = 200,
    ecc_tol: float = 1.0e-3,
) -> np.ndarray:
    """Return geometry-only ``J_{ijℓ}`` for multipole indices ``ℓ``.

    Debug logging:
    - Logs requested ℓ entries and whether ``s_ijl`` is used or not.
    - Logs derived even-degree indices ``L = 2ℓ`` and output shape.
    """

    ell_arr = np.asarray(ells, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    L_arr = even_ells(ell_arr)

    def _ell_summary(vals: np.ndarray) -> str:
        vals = np.asarray(vals).reshape(-1)
        head = ", ".join(map(str, vals[:5]))
        return f"size={vals.size}, min={vals.min()}, max={vals.max()}, head=[{head}{'' if vals.size<=5 else ', …'}]"

    logger.debug(
        (
            f"J_series: ℓ {_ell_summary(ell_arr)} | use_sijl={bool(use_sijl)} | s_method={str(s_method)} "
            f"| N_overlap={int(N_overlap)} | ecc_tol={float(ecc_tol):.2e} | "
            f"L_range=[{int(L_arr.min()) if L_arr.size else '-'}..{int(L_arr.max()) if L_arr.size else '-'}]"
        )
    )

    if not use_sijl:
        s_vals = np.ones_like(ell_arr, dtype=float)
        leg0 = np.asarray(legendre_P_zero(L_arr), dtype=float)
        a_in = pair.inner.a
        a_out = pair.outer.a
        numer = a_in ** L_arr
        denom = np.maximum(a_out, 1.0e-300) ** (L_arr + 1)
        geom = s_vals * (leg0 ** 2) * numer / denom
    else:
        geom = np.asarray(
            J_exact(
                pair,
                L_arr,
                method=s_method,
                nodes=N_overlap,
                ecc_tol=ecc_tol,
            ),
            dtype=float,
        )

    geom = np.asarray(geom, dtype=float).reshape(ell_arr.shape)
    logger.debug(f"J_series: done | output.shape={np.shape(geom)}")
    return geom


def timed_J_series(
    pair: OrbitPair,
    ells: Sequence[int],
    *,
    use_sijl: bool = True,
    s_method: str = "auto",
    N_overlap: int = 200,
    ecc_tol: float = 1.0e-3,
) -> np.ndarray:
    """Timed wrapper for :func:`J_series` that logs elapsed time at DEBUG level.

    Returns the same geometry-only array as :func:`J_series` while emitting a
    timing summary that includes key configuration parameters.
    """

    t0 = perf_counter()
    out = J_series(
        pair,
        ells,
        use_sijl=use_sijl,
        s_method=s_method,
        N_overlap=N_overlap,
        ecc_tol=ecc_tol,
    )
    dt = perf_counter() - t0

    ell_arr = np.asarray(ells, dtype=int)
    def _ell_summary(vals: np.ndarray) -> str:
        vals = np.asarray(vals).reshape(-1)
        head = ", ".join(map(str, vals[:5]))
        return f"size={vals.size}, min={vals.min()}, max={vals.max()}, head=[{head}{'' if vals.size<=5 else ', …'}]"

    logger.debug(
        (
            f"timed_J_series: {dt*1e3:.3f} ms | ℓ {_ell_summary(ell_arr)} | "
            f"use_sijl={bool(use_sijl)} | s_method={str(s_method)} | N_overlap={int(N_overlap)} | "
            f"ecc_tol={float(ecc_tol):.2e} | out.shape={np.shape(out)}"
        )
    )
    return out

def timed_J_ijl(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    method: str = "auto",
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Timed wrapper for :func:`J_exact` that logs elapsed evaluation time.

    Parameters
    ----------
    pair
        The interacting :class:`~vrr.orbits.OrbitPair`.
    ell
        Multipole index/indices for which to compute ``J_{ijℓ}``.
    method, nodes, ecc_tol
        Passed directly to :func:`J_exact` / :func:`s_ijl`.

    Returns
    -------
    float | np.ndarray
        The same output as :func:`J_exact`.

    Notes
    -----
    Emits a debug log with the total wall-clock time using
    :func:`time.perf_counter`.
    """

    t0 = perf_counter()
    out = J_exact(pair, ell, method=method, nodes=nodes, ecc_tol=ecc_tol)
    dt = perf_counter() - t0

    # Avoid large array printing
    shape = np.shape(out)
    logger.debug(
        f"timed_J_ijl: J_exact took {dt * 1e3:.3f} ms | output.shape={shape} | method={str(method)} | nodes={int(nodes)}"
    )
    return out


def H_terms(ells: Sequence[int], J_ell: np.ndarray, cos_theta: float) -> np.ndarray:
    """Return ``H_{ijℓ}`` contributions (Legendre degree ``L = 2ℓ``)."""

    ell_arr = np.asarray(ells, dtype=int)
    L_arr = even_ells(ell_arr)
    J_arr = np.asarray(J_ell, dtype=float)
    P_vals = np.asarray(legendre_P(L_arr, cos_theta), dtype=float)
    return J_arr * P_vals


def Omega_terms(
    ells: Sequence[int],
    J_ell: np.ndarray,
    cos_theta: float,
    L_i: float,
) -> np.ndarray:
    """Return ``Ω_{ijℓ}`` contributions (Legendre degree ``L = 2ℓ``)."""

    L_safe = max(float(L_i), 1.0e-300)
    ell_arr = np.asarray(ells, dtype=int)
    J_arr = np.asarray(J_ell, dtype=float).reshape(ell_arr.shape)
    result = np.zeros_like(J_arr, dtype=float)

    mask = ell_arr >= 2
    if np.any(mask):
        L_arr = even_ells(ell_arr[mask])
        P_prime_vals = np.asarray(legendre_P_derivative(L_arr, cos_theta), dtype=float)
        result[mask] = J_arr[mask] * P_prime_vals / L_safe

    return result


def H_partial_sums(ells: Sequence[int], J_ell: np.ndarray, cos_theta: float) -> np.ndarray:
    """Return cumulative sums of ``H_{ijℓ}`` contributions."""

    return np.cumsum(H_terms(ells, J_ell, cos_theta))


def Omega_partial_sums(
    ells: Sequence[int],
    J_ell: np.ndarray,
    cos_theta: float,
    L_i: float,
) -> np.ndarray:
    """Return cumulative sums of ``Ω_{ijℓ}`` contributions."""

    return np.cumsum(Omega_terms(ells, J_ell, cos_theta, L_i))


@dataclass
class SeriesEvaluationResult:
    """Container with geometry, physical and dynamical series outputs."""

    ells: np.ndarray
    geometry_coefficients: np.ndarray
    physical_coefficients: np.ndarray
    hamiltonian: float
    omega: float
    torque: TorqueValue
    hamiltonian_terms: Optional[np.ndarray] = None
    hamiltonian_partial_sums: Optional[np.ndarray] = None
    omega_terms: Optional[np.ndarray] = None
    omega_partial_sums: Optional[np.ndarray] = None


def _evaluate_legendre_series(
    pair: OrbitPair,
    ells: Sequence[int],
    geometry_coefficients: np.ndarray,
    *,
    include_terms: bool = False,
    include_partials: bool = False,
) -> SeriesEvaluationResult:
    """Return Hamiltonian, frequency and torque from geometry-only series."""

    ell_arr = np.asarray(ells, dtype=int)
    geom_arr = np.asarray(geometry_coefficients, dtype=float).reshape(ell_arr.shape)
    mass = pair.mass_prefactor()
    physical = mass * geom_arr

    if ell_arr.size == 0:
        torque = pair.torque_from_omega(0.0)
        return SeriesEvaluationResult(
            ell_arr,
            geom_arr,
            physical,
            0.0,
            0.0,
            torque,
            np.zeros(0, dtype=float) if include_terms else None,
            np.zeros(0, dtype=float) if include_partials else None,
            np.zeros(0, dtype=float) if include_terms else None,
            np.zeros(0, dtype=float) if include_partials else None,
        )

    cos_inc = float(pair.cos_inclination)
    P_vals = np.asarray(legendre_P(ell_arr, cos_inc), dtype=float)
    h_terms_geom = geom_arr * P_vals
    hamiltonian_geom = float(np.sum(h_terms_geom))

    omega_mask = ell_arr >= 2
    omega_terms_geom = np.zeros_like(geom_arr, dtype=float)
    omega_partial_geom = None

    if np.any(omega_mask):
        P_prime_vals = np.asarray(
            legendre_P_derivative(ell_arr[omega_mask], cos_inc), dtype=float
        )
        gradient_geom = geom_arr[omega_mask] * P_prime_vals
        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        omega_terms_geom[omega_mask] = -gradient_geom / L_i
        if include_partials:
            omega_partial_geom = np.zeros_like(geom_arr, dtype=float)
            omega_partial_geom[omega_mask] = np.cumsum(omega_terms_geom[omega_mask])
    elif include_partials:
        omega_partial_geom = np.zeros_like(geom_arr, dtype=float)

    omega_geom = float(np.sum(omega_terms_geom))

    hamiltonian = mass * hamiltonian_geom
    omega_scalar = mass * omega_geom
    torque = pair.torque_from_omega(omega_scalar)

    h_terms = mass * h_terms_geom if include_terms else None
    omega_terms = mass * omega_terms_geom if include_terms else None

    h_partial = mass * np.cumsum(h_terms_geom) if include_partials else None
    if include_partials:
        if omega_partial_geom is None:
            omega_partial = mass * np.zeros_like(geom_arr, dtype=float)
        else:
            omega_partial = mass * omega_partial_geom
    else:
        omega_partial = None

    return SeriesEvaluationResult(
        ell_arr,
        geom_arr,
        physical,
        hamiltonian,
        omega_scalar,
        torque,
        h_terms,
        h_partial,
        omega_terms,
        omega_partial,
    )


__all__ = [
    "s_ijl",
    "J_exact",
    "J_series",
    "timed_J_ijl",
    "H_terms",
    "Omega_terms",
    "H_partial_sums",
    "Omega_partial_sums",
    "SeriesEvaluationResult",
    "_evaluate_legendre_series",
]
