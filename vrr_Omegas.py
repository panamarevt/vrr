"""Utilities for evaluating vector resonant relaxation interaction terms.

The module provides three complementary evaluators that cover the most
commonly encountered orbital configurations (circular/circular,
non-overlapping eccentric, overlapping eccentric and mixed
configurations).  Each evaluator can return the interaction Hamiltonian,
its associated precession frequency ``Omega`` and the torque acting on
an orbit.  The evaluators operate on :class:`OrbitPair` instances, but
they also accept batches of pairs so that large ensembles can be handled
efficiently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import scipy.special as sp
from scipy.integrate import quad


# ---------------------------------------------------------------------------
# Orbit and pair helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Orbit:
    """Simple container for Keplerian orbital elements."""

    a: float
    e: float
    m: float

    def __post_init__(self) -> None:
        """Validate the provided orbital elements."""

        if self.a <= 0:
            raise ValueError("The semi-major axis must be positive.")
        if not (0.0 <= self.e < 1.0):
            raise ValueError("The eccentricity must lie within [0, 1).")
        if self.m <= 0:
            raise ValueError("The mass must be positive.")

    @property
    def b(self) -> float:
        """Semi-minor axis ``b = a sqrt(1 - e^2)``."""

        return self.a * np.sqrt(max(1.0 - self.e * self.e, 0.0))

    @property
    def periapsis(self) -> float:
        """Periapsis distance ``r_p = a (1 - e)``."""

        return self.a * (1.0 - self.e)

    @property
    def apoapsis(self) -> float:
        """Apoapsis distance ``r_a = a (1 + e)``."""

        return self.a * (1.0 + self.e)

    @property
    def chi(self) -> float:
        """Axis ratio ``chi = a / b`` used by the asymptotic kernels."""

        # Guard against b -> 0 in the very eccentric limit.
        b = max(self.b, 1.0e-300)
        return self.a / b


@dataclass
class OrbitPair:
    """Container describing two interacting stellar orbits."""

    primary: Orbit
    secondary: Orbit
    cos_inclination: float
    G: float = 1.0
    M_central: float = 1.0

    def __post_init__(self) -> None:
        """Validate inputs and pre-compute helper state."""

        self.cos_inclination = float(np.clip(self.cos_inclination, -1.0, 1.0))
        if self.G <= 0:
            raise ValueError("The gravitational constant must be positive.")
        if self.M_central <= 0:
            raise ValueError("The central mass must be positive.")

        if self.primary.a <= self.secondary.a:
            self._inner = self.primary
            self._outer = self.secondary
            self._primary_is_inner = True
        else:
            self._inner = self.secondary
            self._outer = self.primary
            self._primary_is_inner = False

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def inner(self) -> Orbit:
        """Return the orbit with the smaller semi-major axis."""

        return self._inner

    @property
    def outer(self) -> Orbit:
        """Return the orbit with the larger semi-major axis."""

        return self._outer

    @property
    def primary_is_inner(self) -> bool:
        """Indicate whether the primary orbit is the inner orbit."""

        return self._primary_is_inner

    @property
    def mutual_inclination(self) -> float:
        """Return the mutual inclination in radians."""

        return float(np.arccos(self.cos_inclination))

    @property
    def sin_inclination(self) -> float:
        """Return ``sin(i)`` for the mutual inclination angle ``i``."""

        return float(np.sqrt(max(1.0 - self.cos_inclination ** 2, 0.0)))

    @property
    def non_overlapping(self) -> bool:
        """Return ``True`` if the orbital annuli do not overlap."""

        inner, outer = self.inner, self.outer
        return inner.apoapsis < outer.periapsis

    @property
    def z_parameter(self) -> float:
        """Return the ratio ``z = r_a,in / r_p,out`` for non-overlapping orbits."""

        if self.non_overlapping:
            inner, outer = self.inner, self.outer
            return inner.apoapsis / outer.periapsis
        return 1.0

    @property
    def semi_major_ratio(self) -> float:
        """Return the semi-major-axis ratio ``alpha = a_in / a_out``."""

        return self.inner.a / self.outer.a

    @property
    def circular_configuration(self) -> Optional[str]:
        """Return a tag describing whether any orbit is circular."""

        inner, outer = self.inner, self.outer
        if inner.e == 0.0 and outer.e == 0.0:
            return "both"
        if inner.e == 0.0 and outer.e > 0.0:
            return "inner"
        if outer.e == 0.0 and inner.e > 0.0:
            return "outer"
        return None

    @property
    def angular_momentum_primary(self) -> float:
        """Return the magnitude of the primary orbit's angular momentum."""

        orb = self.primary
        return orb.m * np.sqrt(
            self.G * self.M_central * orb.a * max(1.0 - orb.e ** 2, 0.0)
        )

    @property
    def orbital_frequency_primary(self) -> float:
        """Return the Keplerian orbital frequency of the primary orbit."""

        orb = self.primary
        return np.sqrt(self.G * self.M_central / (orb.a ** 3))


# ---------------------------------------------------------------------------
# Legendre helpers
# ---------------------------------------------------------------------------


def even_ells(l_max: int, start: int = 2) -> np.ndarray:
    """Return even Legendre indices from ``start`` up to ``l_max`` (inclusive)."""

    if l_max < start:
        return np.array([], dtype=int)
    first = start + (start % 2)
    return np.arange(first, l_max + 1, 2, dtype=int)


def legendre_P(l: int, x: float) -> float:
    """Evaluate the Legendre polynomial ``P_l(x)``."""

    return float(sp.lpmv(0, l, x))


def legendre_P_derivative(l: int, x: float) -> float:
    """Derivative of the Legendre polynomial ``d/dx P_l(x)``."""

    if abs(x) == 1.0:
        return 0.0
    return float(-sp.lpmv(1, l, x) / np.sqrt(max(1.0 - x * x, 1.0e-300)))


def legendre_P_zero(l: int) -> float:
    """Value of ``P_l(0)`` computed in log-space for numerical stability."""

    if l % 2 == 1:
        return 0.0
    k = l // 2
    log_gamma_2k = sp.gammaln(2 * k + 1)
    log_gamma_k = sp.gammaln(k + 1)
    log_val = log_gamma_2k - 2 * k * np.log(2.0) - 2 * log_gamma_k
    return ((-1) ** k) * np.exp(log_val)


# ---------------------------------------------------------------------------
# Exact s_ijl and J_{ijl}
# ---------------------------------------------------------------------------


def _s_ijl_non_overlapping(pair: OrbitPair, ell: int) -> float:
    """Return ``s_{ijℓ}`` for non-overlapping orbits using the closed form."""

    inner, outer = pair.inner, pair.outer
    chi_in = inner.chi
    chi_out = outer.chi
    prefactor = (chi_out ** ell) / (chi_in ** (ell + 1))
    value = prefactor * legendre_P(ell + 1, chi_in) * legendre_P(ell - 1, chi_out)
    return float(value)


def _s_ijl_overlapping(pair: OrbitPair, ell: int, nodes: int = 80) -> float:
    """Return ``s_{ijℓ}`` for overlapping orbits via Gauss-Legendre quadrature."""

    inner, outer = pair.inner, pair.outer
    alpha = inner.a / outer.a
    c_val = 1.0 / max(alpha, 1.0e-300)

    nodes_in, weights_in = np.polynomial.legendre.leggauss(nodes)
    phi_in = 0.5 * (nodes_in + 1.0) * np.pi
    w_in = 0.5 * np.pi * weights_in

    nodes_out, weights_out = np.polynomial.legendre.leggauss(nodes)
    phi_out = 0.5 * (nodes_out + 1.0) * np.pi
    w_out = 0.5 * np.pi * weights_out

    x_in = (1.0 + inner.e * np.cos(phi_in)) ** (ell + 1)
    y_out = (1.0 + outer.e * np.cos(phi_out)) ** ell

    x2d = x_in.reshape(nodes, 1)
    y2d = y_out.reshape(1, nodes)
    weights = w_in.reshape(nodes, 1) * w_out.reshape(1, nodes)

    mask = x2d < c_val * y2d
    region1 = x2d / np.maximum(y2d, 1.0e-300)
    region2 = (1.0 / (alpha ** 2)) * (y2d / np.maximum(x2d, 1.0e-300))
    integrand = np.where(mask, region1, region2)

    total = np.sum(integrand * weights)
    return float(total / (np.pi ** 2))


def s_ijl(pair: OrbitPair, ell: int) -> float:
    """Return the coefficient ``s_{ijℓ}`` for the supplied pair and index."""

    if ell == 0:
        return 1.0
    if pair.inner.e == 0.0 and pair.outer.e == 0.0:
        return 1.0
    if pair.non_overlapping:
        return _s_ijl_non_overlapping(pair, ell)
    return _s_ijl_overlapping(pair, ell)


def J_exact(pair: OrbitPair, ell: int) -> float:
    """Return the exact coupling coefficient ``J_{ijℓ}``."""

    alpha = pair.semi_major_ratio
    s_val = s_ijl(pair, ell)
    r_outer = pair.outer.a
    prefactor = pair.G * pair.primary.m * pair.secondary.m / r_outer
    return float(prefactor * (legendre_P_zero(ell) ** 2) * s_val * (alpha ** ell))


# ---------------------------------------------------------------------------
# Asymptotic kernels and auxiliary integrals
# ---------------------------------------------------------------------------


def Sprime_ecc_kernel(x: float, z: float) -> float:
    """Return the eccentric--eccentric asymptotic kernel ``S'(x; z)``."""

    X = np.sqrt(max(1.0 - 2.0 * x * z + z * z, 0.0))
    Y = np.sqrt(1.0 + 2.0 * x * z + z * z)
    log1 = np.log(((1.0 + X + z) * (1.0 + Y - z)) / 4.0)
    log2 = np.log(((1.0 + X - z) * (1.0 + Y + z)) / 4.0)
    return 0.5 * (log1 / (1.0 - x) - log2 / (1.0 + x))


def Sprime_circ_log_kernel(x: float, alpha: float) -> float:
    """Return the circular--circular logarithmic kernel."""

    z = alpha
    X = np.sqrt(max(1.0 - 2.0 * x * z + z * z, 0.0))
    Y = np.sqrt(1.0 + 2.0 * x * z + z * z)
    c2 = (1.0 / np.pi) - 0.25
    c4 = (1.0 / (2.0 * np.pi)) - (9.0 / 64.0)
    value = (
        c2 * legendre_P_derivative(2, x) * z * z
        - c4 * legendre_P_derivative(4, x) * (z ** 4)
    )
    term1 = z * (1.0 + X) / (np.pi * (1.0 - x * z + X) * max(X, 1.0e-15))
    term2 = z * (1.0 + Y) / (np.pi * (1.0 + x * z + Y) * max(Y, 1.0e-15))
    return value + term1 - term2


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


def I2_numeric(a: float, b: float, c: float, d: float) -> float:
    """Evaluate the integral ``I_2`` for overlapping eccentric annuli."""

    if not (a < b <= c < d):
        if abs(b - c) <= 1.0e-12 * max(d, 1.0):
            denom = np.sqrt(max((b - a) * (d - b), 1.0e-300))
            return float(np.pi * (b * b) / denom)
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
    return float(val)


def Jbar_ecc_nonoverlap(pair: OrbitPair) -> float:
    """Return the non-overlapping eccentric asymptotic coupling ``\bar{J}``."""

    inner, outer = pair.inner, pair.outer
    prefactor = pair.G * pair.primary.m * pair.secondary.m / (np.pi * np.pi)
    ratio = ((1.0 + inner.e) * (1.0 - outer.e)) ** 1.5
    denom = np.sqrt(max(inner.e * outer.e, 1.0e-300))
    return float(prefactor * ratio / denom / outer.periapsis)


def Jbar_ecc_overlap(pair: OrbitPair) -> float:
    """Return the overlapping eccentric asymptotic coupling ``\bar{J}``."""

    inner, outer = pair.inner, pair.outer
    a = min(inner.periapsis, outer.periapsis)
    b = max(inner.periapsis, outer.periapsis)
    c = min(inner.apoapsis, outer.apoapsis)
    d = max(inner.apoapsis, outer.apoapsis)
    integral = I2_numeric(a, b, c, d)
    prefactor = 4.0 * pair.G * pair.primary.m * pair.secondary.m
    return float(prefactor * integral / (np.pi ** 3 * inner.a * outer.a))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class InteractionResult:
    """Container bundling the interaction Hamiltonian, frequency and torque."""

    hamiltonian: float
    omega: float
    torque: float
    series_ell: Optional[np.ndarray] = None
    series_coefficients: Optional[np.ndarray] = None
    method: str = ""


# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------


class BaseEvaluator:
    """Abstract base class shared by the interaction evaluators."""

    method_name: str = "base"

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Return the interaction result for a single orbit pair."""

        raise NotImplementedError

    def evaluate_many(self, pairs: Sequence[OrbitPair]) -> List[InteractionResult]:
        """Evaluate a sequence of orbit pairs."""

        return [self.evaluate_pair(pair) for pair in pairs]

    def evaluate_from_arrays(
        self,
        a_primary: Sequence[float],
        e_primary: Sequence[float],
        m_primary: Sequence[float],
        a_secondary: Sequence[float],
        e_secondary: Sequence[float],
        m_secondary: Sequence[float],
        cos_inclinations: Sequence[float],
        G: float = 1.0,
        M_central: float = 1.0,
    ) -> List[InteractionResult]:
        """Instantiate pairs from arrays and evaluate them sequentially."""

        pairs = [
            OrbitPair(
                Orbit(a_p, e_p, m_p),
                Orbit(a_s, e_s, m_s),
                cos_inc,
                G=G,
                M_central=M_central,
            )
            for a_p, e_p, m_p, a_s, e_s, m_s, cos_inc in zip(
                a_primary,
                e_primary,
                m_primary,
                a_secondary,
                e_secondary,
                m_secondary,
                cos_inclinations,
            )
        ]
        return self.evaluate_many(pairs)


# ---------------------------------------------------------------------------
# Exact evaluator
# ---------------------------------------------------------------------------


class ExactSeriesEvaluator(BaseEvaluator):
    """Compute the interaction via a truncated exact Legendre series."""

    method_name = "exact"

    def __init__(self, ell_max: int = 20) -> None:
        """Initialise the evaluator with the maximum multipole order."""

        if ell_max < 2:
            raise ValueError("ell_max must be at least 2 for the quadrupole term.")
        self.ell_max = ell_max

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Evaluate the Hamiltonian, frequency and torque for ``pair``."""

        ells_h = even_ells(self.ell_max, start=0)
        if ells_h.size == 0:
            return InteractionResult(0.0, 0.0, 0.0, method=self.method_name)

        J_vals = np.array([J_exact(pair, int(ell)) for ell in ells_h], dtype=float)
        P_vals = sp.lpmv(0, ells_h, pair.cos_inclination)
        hamiltonian = float(np.dot(J_vals, P_vals))

        mask = ells_h >= 2
        if np.any(mask):
            ells_omega = ells_h[mask]
            J_omega = J_vals[mask]
            P_prime_vals = np.array(
                [
                    legendre_P_derivative(int(ell), pair.cos_inclination)
                    for ell in ells_omega
                ],
                dtype=float,
            )
            dH_dx = float(np.dot(J_omega, P_prime_vals))
        else:
            dH_dx = 0.0

        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        omega = -dH_dx / L_i
        torque = dH_dx * pair.sin_inclination

        return InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=ells_h,
            series_coefficients=J_vals,
            method=self.method_name,
        )


# ---------------------------------------------------------------------------
# Asymptotic evaluator
# ---------------------------------------------------------------------------


class AsymptoticEvaluator(BaseEvaluator):
    """Evaluate interactions using asymptotic kernels for large ``ℓ``."""

    method_name = "asymptotic"

    def __init__(self, quad_epsabs: float = 1.0e-9, quad_epsrel: float = 1.0e-9) -> None:
        """Initialise the evaluator with quadrature tolerances."""

        self.quad_epsabs = quad_epsabs
        self.quad_epsrel = quad_epsrel

    def _omega_scalar(self, pair: OrbitPair, x: float) -> float:
        """Return the asymptotic precession frequency for cosine ``x``."""

        z = pair.z_parameter
        omega_orb = pair.orbital_frequency_primary
        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        tag = pair.circular_configuration

        if tag == "both":
            kernel = Sprime_circ_log_kernel(x, pair.inner.a / pair.outer.a)
            return -2.0 * np.pi * omega_orb * (pair.secondary.m / pair.M_central) * (
                pair.inner.a / pair.outer.a
            ) * kernel

        if tag in {"inner", "outer"}:
            if pair.non_overlapping:
                ecc_orbit = pair.outer if tag == "inner" else pair.inner
                a_out = pair.outer.a
                prefactor = (
                    pair.G
                    * pair.primary.m
                    * pair.secondary.m
                    / a_out
                    * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
                    * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
                )
                dH_dx = prefactor * S2_even_integral_kernel(x, z)
                return -dH_dx / L_i

            J_bar = Jbar_ecc_overlap(pair)
            kappa = J_bar / max(L_i * omega_orb, 1.0e-300)
            theta = np.arccos(np.clip(x, -1.0, 1.0))
            sin_theta = max(np.sin(theta), 1.0e-15)
            return -0.5 * kappa * omega_orb * (np.cos(theta) / sin_theta)

        if pair.non_overlapping:
            J_bar = Jbar_ecc_nonoverlap(pair)
            kappa = J_bar / max(L_i * omega_orb, 1.0e-300)
            return -kappa * omega_orb * Sprime_ecc_kernel(x, z)

        J_bar = Jbar_ecc_overlap(pair)
        kappa = J_bar / max(L_i * omega_orb, 1.0e-300)
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        sin_theta = max(np.sin(theta), 1.0e-15)
        return -0.5 * kappa * omega_orb * (np.cos(theta) / sin_theta)

    def _hamiltonian_from_omega(self, pair: OrbitPair) -> float:
        """Recover the Hamiltonian by integrating the asymptotic frequency."""

        x = pair.cos_inclination
        if abs(x - 1.0) < 1.0e-12:
            return 0.0

        def integrand(u: float) -> float:
            """Integrand for reconstructing the Hamiltonian from ``Omega``."""

            return -pair.angular_momentum_primary * self._omega_scalar(pair, u)

        val, _ = quad(
            integrand,
            1.0,
            x,
            epsabs=self.quad_epsabs,
            epsrel=self.quad_epsrel,
            limit=200,
        )
        return float(val)

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Return the asymptotic interaction result for ``pair``."""

        omega = self._omega_scalar(pair, pair.cos_inclination)
        hamiltonian = self._hamiltonian_from_omega(pair)
        torque = -pair.angular_momentum_primary * omega * pair.sin_inclination
        return InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            method=self.method_name,
        )


# ---------------------------------------------------------------------------
# Hybrid evaluator (asymptotic with low-order corrections)
# ---------------------------------------------------------------------------


class AsymptoticWithCorrectionsEvaluator(AsymptoticEvaluator):
    """Augment the asymptotic result with low-order exact corrections."""

    method_name = "asymptotic_with_corrections"

    def __init__(self, lmax_correction: int = 4, **kwargs) -> None:
        """Initialise the hybrid evaluator and select correction order."""

        super().__init__(**kwargs)
        if lmax_correction < 2:
            raise ValueError("The correction order must be at least 2.")
        self.lmax_correction = lmax_correction

    def _asymptotic_J(self, pair: OrbitPair, ell: int) -> float:
        """Return the asymptotic approximation of ``J_{ijℓ}``."""

        tag = pair.circular_configuration
        if tag == "both":
            return 0.0

        if tag in {"inner", "outer"} and pair.non_overlapping:
            z = pair.z_parameter
            ecc_orbit = pair.outer if tag == "inner" else pair.inner
            a_out = pair.outer.a
            prefactor = (
                pair.G
                * pair.primary.m
                * pair.secondary.m
                / a_out
                * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
                * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
            )
            return float(prefactor * (z ** ell) / (ell ** 1.5))

        if pair.non_overlapping:
            J_bar = Jbar_ecc_nonoverlap(pair)
            return float(J_bar / (ell * ell))

        J_bar = Jbar_ecc_overlap(pair)
        return float(J_bar / (ell * ell))

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Return the hybrid interaction result for ``pair``."""

        base_result = super().evaluate_pair(pair)
        ells = even_ells(self.lmax_correction)
        if ells.size == 0:
            return base_result

        delta_J = []
        for ell in ells:
            exact_val = J_exact(pair, int(ell))
            asym_val = self._asymptotic_J(pair, int(ell))
            delta_J.append(exact_val - asym_val)

        delta_J = np.array(delta_J, dtype=float)
        P_vals = sp.lpmv(0, ells, pair.cos_inclination)
        P_prime_vals = np.array(
            [legendre_P_derivative(int(ell), pair.cos_inclination) for ell in ells],
            dtype=float,
        )

        delta_H = float(np.dot(delta_J, P_vals))
        delta_dH_dx = float(np.dot(delta_J, P_prime_vals))

        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        omega = base_result.omega - delta_dH_dx / L_i
        torque = base_result.torque + delta_dH_dx * pair.sin_inclination
        hamiltonian = base_result.hamiltonian + delta_H

        return InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=ells,
            series_coefficients=delta_J,
            method=self.method_name,
        )


__all__ = [
    "Orbit",
    "OrbitPair",
    "InteractionResult",
    "ExactSeriesEvaluator",
    "AsymptoticEvaluator",
    "AsymptoticWithCorrectionsEvaluator",
]
