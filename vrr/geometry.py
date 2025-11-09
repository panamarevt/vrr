"""Geometry-focused helpers for VRR couplings and Legendre series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.special as sp

from .legendre import even_ells, legendre_P, legendre_P_derivative, legendre_P_zero
from .orbits import OrbitPair, TorqueValue


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

    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

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
    return _to_output(result.astype(float, copy=False))


def s_ijl(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    method: str = "auto",
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Return the coefficient ``s_{ijℓ}`` for the supplied pair and indices."""

    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    ell_flat = ell_arr.reshape(-1)
    result_flat = np.ones_like(ell_flat, dtype=float)

    if np.all(ell_flat == 0):
        return _to_output(result_flat.reshape(ell_arr.shape))

    inner, outer = pair.inner, pair.outer
    if abs(inner.e) < ecc_tol and abs(outer.e) < ecc_tol:
        return _to_output(result_flat.reshape(ell_arr.shape))

    mask_nonzero = ell_flat != 0
    if not np.any(mask_nonzero):
        return _to_output(result_flat.reshape(ell_arr.shape))

    mode = (method or "auto").lower()
    if mode == "simplified":
        mode = "auto"

    if mode not in {"auto", "exact", "closed_form"}:
        raise ValueError(f"Unknown s_ijl method '{method}'.")

    if mode == "closed_form":
        if not pair.non_overlapping:
            raise ValueError("Closed-form s_ijl is only valid for non-overlapping orbits.")
        values = _s_non_overlapping(pair, ell_flat[mask_nonzero])
    elif mode == "exact":
        values = _s_overlapping(
            pair,
            ell_flat[mask_nonzero],
            nodes=nodes,
            ecc_tol=ecc_tol,
        )
    else:
        if pair.non_overlapping:
            values = _s_non_overlapping(pair, ell_flat[mask_nonzero])
        else:
            values = _s_overlapping(
                pair,
                ell_flat[mask_nonzero],
                nodes=nodes,
                ecc_tol=ecc_tol,
            )

    result_flat[mask_nonzero] = np.asarray(values, dtype=float)
    return _to_output(result_flat.reshape(ell_arr.shape))


def J_exact(
    pair: OrbitPair,
    ell: int | np.ndarray,
    *,
    method: str = "auto",
    nodes: int = 200,
    ecc_tol: float = 1.0e-3,
) -> float | np.ndarray:
    """Return the geometry-only coupling ``J_{ijℓ}`` for ``pair``."""

    ell_arr = np.asarray(ell, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    s_val = np.asarray(
        s_ijl(pair, ell_arr, method=method, nodes=nodes, ecc_tol=ecc_tol), dtype=float
    )
    leg0 = np.asarray(legendre_P_zero(ell_arr), dtype=float)

    a_inner = pair.inner.a
    a_outer = max(pair.outer.a, 1.0e-300)

    values = s_val * (leg0 ** 2) * (a_inner ** ell_arr) / (a_outer ** (ell_arr + 1))
    return _to_output(values.reshape(ell_arr.shape))


def J_series(
    pair: OrbitPair,
    ells: Sequence[int],
    *,
    use_sijl: bool = True,
    s_method: str = "auto",
    N_overlap: int = 200,
    ecc_tol: float = 1.0e-3,
) -> np.ndarray:
    """Return geometry-only ``J_{ijℓ}`` for multipole indices ``ℓ``."""

    ell_arr = np.asarray(ells, dtype=int)
    if ell_arr.size == 0:
        return np.asarray(ell_arr, dtype=float)

    L_arr = even_ells(ell_arr)

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
    return geom


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
    "H_terms",
    "Omega_terms",
    "H_partial_sums",
    "Omega_partial_sums",
    "SeriesEvaluationResult",
    "_evaluate_legendre_series",
]
