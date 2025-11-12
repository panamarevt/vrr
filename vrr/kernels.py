"""Asymptotic kernel helpers used by the VRR evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.integrate import quad

from .legendre import even_ells, legendre_P_derivative
from .orbits import OrbitPair, _build_pair, _radii_sorted
from .geometry import Omega_partial_sums

# Prefer loguru's logger; fall back gracefully to stdlib logging
try:  # pragma: no cover - trivial import guard
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback path
    import logging as _logging

    logger = _logging.getLogger("vrr.kernels")
    if not logger.handlers:
        _handler = _logging.StreamHandler()
        _formatter = _logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(_logging.DEBUG)


@dataclass
class AsymptoticMetadata:
    """Book-keeping structure for asymptotic coupling/Omega reconstructions."""

    case: str
    regime: str
    prefactor: Optional[float]
    z: Optional[float]
    I2: Optional[float] = None
    ell_power: float = 2.0
    kernel_supported: bool = False


def _series_from_meta(ell_arr: np.ndarray, meta: AsymptoticMetadata) -> Optional[np.ndarray]:
    """Return asymptotic series coefficients implied by ``meta``."""

    if meta.prefactor is None:
        return None

    ell_arr = np.asarray(ell_arr, dtype=int)
    if ell_arr.size == 0:
        return np.zeros(0, dtype=float)

    L = even_ells(ell_arr).astype(float)
    safe_L = np.where(L == 0.0, np.inf, L)
    if meta.regime == "overlap":
        result = meta.prefactor / (safe_L ** meta.ell_power)
    else:
        if meta.z is None:
            return None
        powers = np.asarray(meta.z ** ell_arr, dtype=float)
        result = meta.prefactor * powers / (safe_L ** meta.ell_power)
    return np.asarray(result, dtype=float)


def _kernel_roots(x: float, z: float) -> tuple[float, float]:
    """Return helper square-root terms used across the kernels."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    z_val = float(z)
    X = np.sqrt(max(1.0 - 2.0 * x_clipped * z_val + z_val * z_val, 0.0))
    Y = np.sqrt(max(1.0 + 2.0 * x_clipped * z_val + z_val * z_val, 0.0))
    return X, Y


def Sprime_ecc_kernel(x: float, z: float) -> float:
    """Return the eccentric--eccentric asymptotic kernel ``S'(x; z)``."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    z_val = float(z)
    X, Y = _kernel_roots(x_clipped, z_val)
    eps = 1.0e-15
    numerator1 = ((1.0 + X + z_val) * (1.0 + Y - z_val)) / 4.0
    numerator2 = ((1.0 + X - z_val) * (1.0 + Y + z_val)) / 4.0
    log1 = np.log(max(numerator1, eps))
    log2 = np.log(max(numerator2, eps))
    denom1 = max(1.0 - x_clipped, eps)
    denom2 = max(1.0 + x_clipped, eps)
    return 0.5 * (log1 / denom1 - log2 / denom2)


def Sprime_circ_log_kernel(x: float, alpha: float) -> float:
    """Return the circular--circular logarithmic kernel."""

    X, Y = _kernel_roots(x, alpha)
    c2 = (1.0 / np.pi) - 0.25
    c4 = (1.0 / (2.0 * np.pi)) - (9.0 / 64.0)
    z = alpha
    value = (
        c2 * legendre_P_derivative(2, x) * z * z
        - c4 * legendre_P_derivative(4, x) * (z ** 4)
    )
    term1 = z * (1.0 + X) / (np.pi * (1.0 - x * z + X) * max(X, 1.0e-15))
    term2 = z * (1.0 + Y) / (np.pi * (1.0 + x * z + Y) * max(Y, 1.0e-15))
    return value + term1 - term2


def _integrate_sprime(
    func: Callable[[float], float],
    x: float,
    *,
    epsabs: float,
    epsrel: float,
) -> float:
    """Return the integral of ``func`` from alignment to ``x``."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    if abs(x_clipped - 1.0) < 1.0e-12:
        return 0.0

    def integrand(u: float) -> float:
        return float(func(float(u)))

    val, _ = quad(
        integrand,
        1.0,
        x_clipped,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=200,
    )
    return -float(val)


def S_circ_log_kernel(x: float, alpha: float, *, epsabs: float, epsrel: float) -> float:
    """Return the circular--circular Hamiltonian kernel ``S(x; alpha)``."""

    return _integrate_sprime(
        lambda u: Sprime_circ_log_kernel(u, alpha),
        x,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def S_ecc_kernel(x: float, z: float, *, epsabs: float, epsrel: float) -> float:
    """Return the eccentric--eccentric Hamiltonian kernel ``S(x; z)``."""

    return _integrate_sprime(
        lambda u: Sprime_ecc_kernel(u, z),
        x,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def S2_even_integral_kernel(x: float, z: float) -> float:
    """Return the ``S_2`` kernel for one-circular mixed configurations."""

    def integrand(t: float) -> float:
        """Integrand for the ``S_2`` kernel quadrature."""

        q = z * t
        A = (1.0 - 2.0 * x * q + q * q) ** (-1.5)
        B = (1.0 + 2.0 * x * q + q * q) ** (-1.5)
        return np.sqrt(-np.log(max(t, 1.0e-300))) * z * (A - B)

    val, _ = quad(integrand, 0.0, 1.0, epsabs=1.0e-10, epsrel=1.0e-10, limit=200)
    return float(val / np.sqrt(np.pi))


def S2_even_kernel(x: float, z: float, *, epsabs: float, epsrel: float) -> float:
    """Return the mixed one-circular Hamiltonian kernel ``S_2(x; z)``."""

    return _integrate_sprime(
        lambda u: S2_even_integral_kernel(u, z),
        x,
        epsabs=epsabs,
        epsrel=epsrel,
    )


def S_overlap_kernel(x: float) -> float:
    """Return the overlapping-orbit Hamiltonian kernel ``S(x)``."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    return -np.sqrt(max(1.0 - x_clipped * x_clipped, 0.0))


def _kernel_series_RHS(x: float, z: float) -> float:
    """Closed-form RHS for the eccentric kernel (no extra ``π²`` factor)."""

    return float(Sprime_ecc_kernel(x, z))


def _kernel_z1_full_RHS(x: float) -> float:
    """Return the ``z→1`` eccentric kernel in its exact form."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    s = np.sqrt(max(0.0, 0.5 * (1.0 - x_clipped)))
    c = np.sqrt(max(0.0, 0.5 * (1.0 + x_clipped)))
    eps = 1.0e-15
    s2 = max(s * s, eps)
    c2 = max(c * c, eps)
    term1 = np.log(max((1.0 + s) * c, eps)) / (4.0 * s2)
    term2 = np.log(max((1.0 + c) * s, eps)) / (4.0 * c2)
    return term1 - term2


def _kernel_z1_cot_RHS(x: float) -> float:
    """Return the small-angle ``cot`` approximation to the ``z→1`` kernel."""

    theta = np.arccos(np.clip(x, -1.0, 1.0))
    return -0.5 / max(np.tan(theta), 1.0e-15)


def omega_kernel_normalized(
    meta: AsymptoticMetadata,
    cos_theta: float,
    *,
    a_outer: float = 1.0,
    use_cot_approx: bool = False,
    signed: bool = False,
) -> float:
    """Return analytic Ω-kernel (no ℓ-sum) in the notebook normalisation."""

    if not meta.kernel_supported or meta.prefactor is None:
        return np.nan

    x = float(np.clip(cos_theta, -1.0, 1.0))
    scale = a_outer * meta.prefactor

    if meta.case == "both_circular":
        return np.nan

    if meta.regime == "non_overlap":
        if meta.z is None:
            return np.nan
        kernel = _kernel_series_RHS(x, meta.z)
        value = abs(kernel) if not signed else kernel
        return value * scale

    kernel = _kernel_z1_cot_RHS(x) if use_cot_approx else _kernel_z1_full_RHS(x)
    value = abs(kernel) if not signed else kernel
    return value * scale


def omega_asymptotic_from_meta(
    ells: Sequence[int],
    cos_theta: float,
    L_i: float,
    meta: AsymptoticMetadata,
) -> Optional[np.ndarray]:
    """Return asymptotic Ω partial sums for the given metadata."""

    series = _series_from_meta(ells, meta)
    if series is None:
        return None
    return Omega_partial_sums(ells, series, cos_theta, L_i)


def omega_hybrid_from_meta(
    ells: Sequence[int],
    J_exact_vals: np.ndarray,
    meta: AsymptoticMetadata,
    cos_theta: float,
    L_i: float,
    *,
    a_outer: float,
    use_cot_approx: bool = False,
) -> Optional[tuple[float, np.ndarray, np.ndarray, float]]:
    """Return hybrid Ω value (exact + analytic − asymptotic) and components."""

    asympt_series = _series_from_meta(ells, meta)
    if asympt_series is None:
        return None

    exact_ps = Omega_partial_sums(ells, J_exact_vals, cos_theta, L_i)
    asympt_ps = Omega_partial_sums(ells, asympt_series, cos_theta, L_i)
    kernel_val = omega_kernel_normalized(
        meta,
        cos_theta,
        a_outer=a_outer,
        use_cot_approx=use_cot_approx,
        signed=True,
    )

    if not np.isfinite(kernel_val):
        return None

    hybrid_val = exact_ps[-1] + kernel_val - asympt_ps[-1]
    return hybrid_val, exact_ps, asympt_ps, kernel_val


def I2_numeric(a: float, b: float, c: float, d: float) -> float:
    """Evaluate the integral ``I_2`` for overlapping eccentric annuli.

    Emits DEBUG logs with inputs and elapsed time. Uses fixed tolerances
    (epsabs=1e-9, epsrel=1e-9) currently.
    """

    from time import perf_counter

    logger.debug(
        (
            f"I2_numeric: start | a={float(a):.6g}, b={float(b):.6g}, c={float(c):.6g}, d={float(d):.6g}"
        )
    )
    t0 = perf_counter()
    # Handle touching annuli (b≈c) with analytic limit to avoid 0/0 roundoff
    if abs(b - c) <= 1.0e-12 * max(d, 1.0):
        denom = np.sqrt(max((b - a) * (d - b), 1.0e-300))
        val = float(np.pi * (b * b) / denom)
        dt = perf_counter() - t0
        logger.debug(
            f"I2_numeric: touching limit (b≈c) | value={val:.6g} | {dt*1e3:.2f} ms"
        )
        return val

    if not (a < b <= c < d):
        raise ValueError(f"Bad ordering in I2 integral: {a}, {b}, {c}, {d}")

    midpoint = 0.5 * (b + c)
    half_range = 0.5 * (c - b)

    def integrand(t: float) -> float:
        """Integrand for the ``I_2`` overlap quadrature."""

        r = midpoint + half_range * np.sin(t)
        dr = half_range * np.cos(t)
        denominator = np.sqrt(
            max(r - a, 1.0e-300)
            * max(r - b, 1.0e-300)
            * max(c - r, 1.0e-300)
            * max(d - r, 1.0e-300)
        )
        return (r * r) * dr / denominator

    val, _ = quad(integrand, -np.pi / 2.0, np.pi / 2.0, epsabs=1.0e-9, epsrel=1.0e-9, limit=200)
    dt = perf_counter() - t0
    logger.debug(f"I2_numeric: done | value={float(val):.6g} | {dt*1e3:.2f} ms")
    return float(val)


def I2_from_pair(pair: OrbitPair, *, nodes: int = 400) -> float:
    """Convenience wrapper computing ``I_2`` from an :class:`OrbitPair`."""

    del nodes
    a, b, c, d = _radii_sorted(pair)
    return I2_numeric(a, b, c, d)


def I2_from_orbits(a_i: float, e_i: float, a_j: float, e_j: float, *, nodes: int = 400) -> float:
    """Convenience wrapper computing ``I_2`` from two orbital configurations."""

    pair = _build_pair(a_i, e_i, 1.0, a_j, e_j, 1.0)
    return I2_from_pair(pair, nodes=nodes)


def Jbar_ecc_nonoverlap(pair: OrbitPair) -> float:
    """Return the geometry-only non-overlap asymptotic coupling ``\bar{J}``.

    Cached per-pair in the orbit's geometry cache to avoid recomputation.
    Logs inputs and value at DEBUG only on first compute (cache miss).
    """

    cache_key = ("Jbar_non_overlap",)
    if hasattr(pair, "_geometry_cache") and cache_key in pair._geometry_cache:  # type: ignore[attr-defined]
        return float(pair._geometry_cache[cache_key][0])  # type: ignore[index]

    from time import perf_counter
    t0 = perf_counter()

    inner, outer = pair.inner, pair.outer
    ratio = ((1.0 + inner.e) * (1.0 - outer.e)) ** 1.5
    denom = np.sqrt(max(inner.e * outer.e, 1.0e-300))
    value = float(ratio / (np.pi ** 2 * denom * outer.periapsis))
    if hasattr(pair, "_geometry_cache"):
        pair._geometry_cache[cache_key] = np.array([value], dtype=float)  # type: ignore[attr-defined]
    dt = perf_counter() - t0
    logger.debug(
        (
            f"Jbar_ecc_nonoverlap: a_in={inner.a:.6g}, e_in={inner.e:.3g} | a_out={outer.a:.6g}, e_out={outer.e:.3g} | "
            f"Jbar={value:.6g} | {dt*1e3:.2f} ms"
        )
    )
    return value


def Jbar_ecc_overlap(pair: OrbitPair) -> float:
    """Return the geometry-only overlapping asymptotic coupling ``\bar{J}``.

    Cached per-pair in the orbit's geometry cache to avoid recomputation.
    Logs the I2 value and Jbar only on first compute (cache miss).
    """

    cache_key = ("Jbar_overlap",)
    if hasattr(pair, "_geometry_cache") and cache_key in pair._geometry_cache:  # type: ignore[attr-defined]
        return float(pair._geometry_cache[cache_key][0])  # type: ignore[index]

    from time import perf_counter
    t0 = perf_counter()

    inner, outer = pair.inner, pair.outer
    a = min(inner.periapsis, outer.periapsis)
    b = max(inner.periapsis, outer.periapsis)
    c = min(inner.apoapsis, outer.apoapsis)
    d = max(inner.apoapsis, outer.apoapsis)
    integral = I2_numeric(a, b, c, d)
    value = float(4.0 * integral / (np.pi ** 3 * inner.a * outer.a))
    if hasattr(pair, "_geometry_cache"):
        pair._geometry_cache[cache_key] = np.array([value], dtype=float)  # type: ignore[attr-defined]
    dt = perf_counter() - t0
    logger.debug(
        (
            f"Jbar_ecc_overlap: a_in={inner.a:.6g}, e_in={inner.e:.3g} | a_out={outer.a:.6g}, e_out={outer.e:.3g} | "
            f"I2={integral:.6g} | Jbar={value:.6g} | {dt*1e3:.2f} ms"
        )
    )
    return value


def _omega_kernel_both_circular(
    pair: OrbitPair, x: float, tag: Optional[str] = None
) -> float:
    """Return the analytic ``Ω`` kernel for two circular orbits."""

    alpha = max(pair.inner.a / max(pair.outer.a, 1.0e-300), 0.0)
    kernel = Sprime_circ_log_kernel(float(x), alpha)
    radius_ratio = pair.primary.a / max(pair.outer.a, 1.0e-300)
    omega_orb = pair.orbital_frequency_primary
    return (
        2.0
        * np.pi
        * omega_orb
        * (pair.secondary.m / pair.M_central)
        * radius_ratio
        * kernel
    )


def _omega_kernel_one_circular_non_overlap(
    pair: OrbitPair, x: float, tag: Optional[str]
) -> float:
    """Return the analytic ``Ω`` kernel for mixed non-overlap configurations."""

    if tag not in {"inner", "outer"}:
        raise ValueError("Circular configuration tag required for mixed regime.")

    z = float(pair.z_parameter)
    ecc_orbit = pair.outer if tag == "inner" else pair.inner
    a_out = max(pair.outer.a, 1.0e-300)
    prefactor = (
        (1.0 / a_out)
        * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
        * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
    )
    dH_dx_geom = prefactor * S2_even_integral_kernel(float(x), z)
    omega_orb = pair.orbital_frequency_primary
    return -pair.mass_prefactor() * dH_dx_geom * omega_orb


def _omega_kernel_non_overlap(
    pair: OrbitPair, x: float, tag: Optional[str] = None
) -> float:
    """Return the analytic ``Ω`` kernel for eccentric non-overlapping orbits."""

    z = float(pair.z_parameter)
    # Cache non-overlap Jbar to avoid recomputation per-ℓ
    cache_key = ("Jbar_non_overlap",)
    if not hasattr(pair, "_geometry_cache") or cache_key not in pair._geometry_cache:  # type: ignore[attr-defined]
        J_bar_geom = Jbar_ecc_nonoverlap(pair)
        if hasattr(pair, "_geometry_cache"):
            pair._geometry_cache[cache_key] = np.array([J_bar_geom], dtype=float)  # type: ignore[attr-defined]
    else:  # pragma: no cover - cache hit
        J_bar_geom = float(pair._geometry_cache[cache_key][0])  # type: ignore[index]
    omega_orb = pair.orbital_frequency_primary
    L_i = max(pair.angular_momentum_primary, 1.0e-300)
    denom = max(L_i * omega_orb, 1.0e-300)
    kappa_geom = J_bar_geom / denom
    return -pair.mass_prefactor() * kappa_geom * omega_orb * Sprime_ecc_kernel(float(x), z)


def _omega_kernel_overlap(
    pair: OrbitPair, x: float, tag: Optional[str] = None
) -> float:
    """Return the analytic ``Ω`` kernel for overlapping or embedded orbits."""
    # Reuse cached Jbar if available to avoid recomputing I2 multiple times
    cache_key = ("Jbar_overlap",)
    if not hasattr(pair, "_geometry_cache") or cache_key not in pair._geometry_cache:  # type: ignore[attr-defined]
        J_bar_geom = Jbar_ecc_overlap(pair)
        if hasattr(pair, "_geometry_cache"):
            pair._geometry_cache[cache_key] = np.array([J_bar_geom], dtype=float)  # type: ignore[attr-defined]
    else:  # pragma: no cover - simple cache hit path
        J_bar_geom = float(pair._geometry_cache[cache_key][0])  # type: ignore[index]
    omega_orb = pair.orbital_frequency_primary
    L_i = max(pair.angular_momentum_primary, 1.0e-300)
    denom = max(L_i * omega_orb, 1.0e-300)
    kappa_geom = J_bar_geom / denom
    theta = np.arccos(np.clip(float(x), -1.0, 1.0))
    sin_theta = max(np.sin(theta), 1.0e-15)
    factor = -0.5 * pair.mass_prefactor() * kappa_geom * omega_orb
    return factor * (np.cos(theta) / sin_theta)


def _hamiltonian_kernel_both_circular(
    pair: OrbitPair,
    x: float,
    tag: Optional[str],
    quad_epsabs: float,
    quad_epsrel: float,
) -> float:
    """Return the Hamiltonian kernel for two circular orbits."""

    alpha = max(pair.inner.a / max(pair.outer.a, 1.0e-300), 0.0)
    radius_ratio = pair.primary.a / max(pair.outer.a, 1.0e-300)
    omega_orb = pair.orbital_frequency_primary
    prefactor = (
        2.0
        * np.pi
        * omega_orb
        * (pair.secondary.m / pair.M_central)
        * radius_ratio
    )
    S_val = S_circ_log_kernel(float(x), alpha, epsabs=quad_epsabs, epsrel=quad_epsrel)
    return -pair.angular_momentum_primary * prefactor * S_val


def _hamiltonian_kernel_one_circular_non_overlap(
    pair: OrbitPair,
    x: float,
    tag: Optional[str],
    quad_epsabs: float,
    quad_epsrel: float,
) -> float:
    """Return the Hamiltonian kernel for mixed non-overlapping configurations."""

    if tag not in {"inner", "outer"}:
        raise ValueError("Circular configuration tag required for mixed regime.")

    z = float(pair.z_parameter)
    ecc_orbit = pair.outer if tag == "inner" else pair.inner
    a_out = max(pair.outer.a, 1.0e-300)
    prefactor = (
        (1.0 / a_out)
        * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
        * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
    )
    S_geom = S2_even_kernel(float(x), z, epsabs=quad_epsabs, epsrel=quad_epsrel)
    return pair.mass_prefactor() * prefactor * S_geom


def _hamiltonian_kernel_non_overlap(
    pair: OrbitPair,
    x: float,
    tag: Optional[str],
    quad_epsabs: float,
    quad_epsrel: float,
) -> float:
    """Return the Hamiltonian kernel for eccentric non-overlapping orbits."""

    del tag
    z = float(pair.z_parameter)
    cache_key = ("Jbar_non_overlap",)
    if not hasattr(pair, "_geometry_cache") or cache_key not in pair._geometry_cache:  # type: ignore[attr-defined]
        J_bar_geom = Jbar_ecc_nonoverlap(pair)
        if hasattr(pair, "_geometry_cache"):
            pair._geometry_cache[cache_key] = np.array([J_bar_geom], dtype=float)  # type: ignore[attr-defined]
    else:  # pragma: no cover - cache hit
        J_bar_geom = float(pair._geometry_cache[cache_key][0])  # type: ignore[index]
    S_geom = S_ecc_kernel(float(x), z, epsabs=quad_epsabs, epsrel=quad_epsrel)
    return pair.mass_prefactor() * J_bar_geom * S_geom


def _hamiltonian_kernel_overlap(
    pair: OrbitPair,
    x: float,
    tag: Optional[str],
    quad_epsabs: float,
    quad_epsrel: float,
) -> float:
    """Return the Hamiltonian kernel for overlapping or embedded orbits."""

    del quad_epsabs, quad_epsrel
    del tag
    cache_key = ("Jbar_overlap",)
    if not hasattr(pair, "_geometry_cache") or cache_key not in pair._geometry_cache:  # type: ignore[attr-defined]
        J_bar_geom = Jbar_ecc_overlap(pair)
        if hasattr(pair, "_geometry_cache"):
            pair._geometry_cache[cache_key] = np.array([J_bar_geom], dtype=float)  # type: ignore[attr-defined]
    else:  # pragma: no cover - cache hit
        J_bar_geom = float(pair._geometry_cache[cache_key][0])  # type: ignore[index]
    S_geom = S_overlap_kernel(float(x))
    return 0.5 * pair.mass_prefactor() * J_bar_geom * S_geom


__all__ = [
    "AsymptoticMetadata",
    "_kernel_roots",
    "_kernel_series_RHS",
    "_kernel_z1_full_RHS",
    "_kernel_z1_cot_RHS",
    "Sprime_ecc_kernel",
    "Sprime_circ_log_kernel",
    "S_circ_log_kernel",
    "S_ecc_kernel",
    "S2_even_integral_kernel",
    "S2_even_kernel",
    "S_overlap_kernel",
    "I2_numeric",
    "I2_from_pair",
    "I2_from_orbits",
    "Jbar_ecc_nonoverlap",
    "Jbar_ecc_overlap",
    "_omega_kernel_both_circular",
    "_omega_kernel_one_circular_non_overlap",
    "_omega_kernel_non_overlap",
    "_omega_kernel_overlap",
    "_hamiltonian_kernel_both_circular",
    "_hamiltonian_kernel_one_circular_non_overlap",
    "_hamiltonian_kernel_non_overlap",
    "_hamiltonian_kernel_overlap",
    "omega_kernel_normalized",
    "omega_asymptotic_from_meta",
    "omega_hybrid_from_meta",
]
