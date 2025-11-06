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
from typing import Any, Callable, List, Optional, Sequence

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
        """Validate inputs and initialise lazy caches."""

        self.cos_inclination = float(np.clip(self.cos_inclination, -1.0, 1.0))
        if self.G <= 0:
            raise ValueError("The gravitational constant must be positive.")
        if self.M_central <= 0:
            raise ValueError("The central mass must be positive.")

        self._update_orbit_ordering()

        # Lazy caches populated on demand.  The geometry cache stores
        # geometry-only quantities such as J_{ijℓ}, while the dynamics cache
        # keeps track of Hamiltonian, Omega and torque values keyed by the
        # evaluator/mode that produced them.
        self._geometry_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dynamics_cache: dict[str, dict[str, Any]] = {}
        self._mass_prefactor_cache: Optional[float] = None

        # Signatures used to invalidate the caches when the orbital elements
        # or the orientation of the pair change.  They are initialised with the
        # current state so that the first cache access operates on a clean
        # slate.
        self._geometry_signature: Optional[tuple[float, ...]] = None
        self._dynamics_signature: Optional[tuple[float, ...]] = None
        self._geometry_signature = self._compute_geometry_signature()
        self._dynamics_signature = self._compute_dynamics_signature()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_orbit_ordering(self) -> None:
        """Assign cached inner/outer orbits for fast access."""

        if self.primary.a <= self.secondary.a:
            self._inner = self.primary
            self._outer = self.secondary
            self._primary_is_inner = True
        else:
            self._inner = self.secondary
            self._outer = self.primary
            self._primary_is_inner = False

    def _compute_geometry_signature(self) -> tuple[float, ...]:
        """Return a hashable signature for geometry-dependent caches."""

        return (
            float(self.primary.a),
            float(self.primary.e),
            float(self.primary.m),
            float(self.secondary.a),
            float(self.secondary.e),
            float(self.secondary.m),
            float(self.G),
            float(self.M_central),
        )

    def _compute_dynamics_signature(self) -> tuple[float, ...]:
        """Return a signature for orientation dependent caches."""

        geom_signature = self._compute_geometry_signature()
        return (*geom_signature, float(self.cos_inclination))

    def _clear_geometry_cache(self) -> None:
        """Remove geometry-only cached data."""

        self._geometry_cache.clear()
        self._mass_prefactor_cache = None

    def _clear_dynamics_cache(self) -> None:
        """Remove cached Hamiltonian, Omega and torque values."""

        self._dynamics_cache.clear()

    def _ensure_geometry_signature(self) -> None:
        """Refresh geometry caches when the orbital elements change."""

        signature = self._compute_geometry_signature()
        if signature != self._geometry_signature:
            self._geometry_signature = signature
            self._update_orbit_ordering()
            self._clear_geometry_cache()
            self._clear_dynamics_cache()

    def _ensure_dynamics_signature(self) -> None:
        """Refresh orientation caches when the inclination changes."""

        self._ensure_geometry_signature()
        signature = self._compute_dynamics_signature()
        if signature != self._dynamics_signature:
            self._dynamics_signature = signature
            self._clear_dynamics_cache()

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

    # ------------------------------------------------------------------
    # Cached accessors
    # ------------------------------------------------------------------
    @staticmethod
    def _to_hashable(value: Any) -> Any:
        """Convert ``value`` into a hashable representation for cache keys."""

        if isinstance(value, np.ndarray):
            return tuple(np.asarray(value).reshape(-1).tolist())
        if isinstance(value, (list, tuple)):
            return tuple(OrbitPair._to_hashable(item) for item in value)
        return value

    def mass_prefactor(self) -> float:
        """Return the interaction mass scaling ``G m_i m_j`` (cached)."""

        self._ensure_geometry_signature()
        if self._mass_prefactor_cache is None:
            self._mass_prefactor_cache = float(
                self.G * self.primary.m * self.secondary.m
            )
        return self._mass_prefactor_cache

    def torque_prefactor(self) -> float:
        """Return the factor multiplying ``Omega`` to obtain the torque."""

        self._ensure_dynamics_signature()
        return -self.angular_momentum_primary * self.sin_inclination

    def torque_from_omega(self, omega: float) -> float:
        """Return the torque corresponding to the supplied ``Omega``."""

        return self.torque_prefactor() * float(omega)

    def get_couplings(
        self,
        mode: str,
        ells: Sequence[int],
        compute: Optional[
            Callable[["OrbitPair", np.ndarray], np.ndarray | float]
        ] = None,
        **compute_kwargs: Any,
    ) -> np.ndarray:
        """Return geometry-only couplings for ``ells`` using lazy caching."""

        self._ensure_geometry_signature()
        ell_arr = np.asarray(ells, dtype=int)
        ell_tuple = tuple(int(x) for x in ell_arr.reshape(-1))
        extra = tuple(
            sorted((key, OrbitPair._to_hashable(val)) for key, val in compute_kwargs.items())
        )
        cache_key = (mode, ell_tuple, extra)

        if cache_key not in self._geometry_cache:
            if compute is None:
                raise ValueError(
                    "No cached couplings available and no compute function provided."
                )
            computed = compute(self, ell_arr, **compute_kwargs)
            values = np.asarray(computed, dtype=float).reshape(-1)
            self._geometry_cache[cache_key] = values.copy()

        cached = self._geometry_cache[cache_key]
        return np.array(cached, copy=True).reshape(ell_arr.shape)

    def cache_dynamics(self, mode: str, **values: Any) -> None:
        """Store Hamiltonian, Omega and torque results for ``mode``."""

        self._ensure_dynamics_signature()
        entry = self._dynamics_cache.setdefault(mode, {})
        for key, value in values.items():
            if isinstance(value, np.ndarray):
                entry[key] = np.array(value, copy=True)
            else:
                entry[key] = value

    def get_cached_dynamics(self, mode: str) -> Optional[dict[str, Any]]:
        """Return cached dynamical quantities for ``mode`` if available."""

        self._ensure_dynamics_signature()
        entry = self._dynamics_cache.get(mode)
        if entry is None:
            return None

        result: dict[str, Any] = {}
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                result[key] = np.array(value, copy=True)
            else:
                result[key] = value
        return result

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
# Geometry utilities
# ---------------------------------------------------------------------------


def _build_pair(
    a_i: float,
    e_i: float,
    m_i: float,
    a_j: float,
    e_j: float,
    m_j: float,
    *,
    cos_inc: float = 1.0,
    G: float = 1.0,
    M_central: float = 1.0,
) -> OrbitPair:
    """Instantiate an :class:`OrbitPair` for convenience wrappers."""

    return OrbitPair(
        Orbit(a_i, e_i, m_i),
        Orbit(a_j, e_j, m_j),
        cos_inc,
        G=G,
        M_central=M_central,
    )


def _order_by_apocenter(pair: OrbitPair) -> tuple[Orbit, Orbit]:
    """Return orbits sorted by apocentre (inner first)."""

    first, second = pair.primary, pair.secondary
    if first.apoapsis <= second.apoapsis:
        return first, second
    return second, first


def _radii_sorted(pair: OrbitPair) -> tuple[float, float, float, float]:
    """Return peri/apo distances of both orbits sorted ascending."""

    radii = np.array(
        [
            pair.primary.periapsis,
            pair.primary.apoapsis,
            pair.secondary.periapsis,
            pair.secondary.apoapsis,
        ],
        dtype=float,
    )
    a, b, c, d = np.sort(radii)
    return float(a), float(b), float(c), float(d)


def _regime_tag(pair: OrbitPair) -> str:
    """Classify the geometric regime of ``pair``."""

    rp_p, ra_p = pair.primary.periapsis, pair.primary.apoapsis
    rp_s, ra_s = pair.secondary.periapsis, pair.secondary.apoapsis

    if ra_p < rp_s or ra_s < rp_p:
        return "non_overlap"
    if rp_p >= rp_s and ra_p <= ra_s:
        return "embedded_primary"
    if rp_s >= rp_p and ra_s <= ra_p:
        return "embedded_secondary"
    return "overlap"


def _regime_flags(pair: OrbitPair) -> tuple[bool, bool, bool]:
    """Return (non-overlap, overlap, embedded) booleans for ``pair``."""

    tag = _regime_tag(pair)
    if tag == "non_overlap":
        return True, False, False
    if tag.startswith("embedded"):
        return False, False, True
    return False, True, False


# ---------------------------------------------------------------------------
# Legendre helpers
# ---------------------------------------------------------------------------


def _to_output(value: np.ndarray | float) -> float | np.ndarray:
    """Return a Python float when ``value`` is scalar, otherwise an array."""

    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return float(arr)
    return arr


def even_ells(l_max: int, start: int = 2) -> np.ndarray:
    """Return even Legendre indices from ``start`` up to ``l_max`` (inclusive)."""

    if l_max < start:
        return np.array([], dtype=int)
    first = start + (start % 2)
    return np.arange(first, l_max + 1, 2, dtype=int)


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


# ---------------------------------------------------------------------------
# Exact s_ijl and J_{ijl}
# ---------------------------------------------------------------------------


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
    else:  # auto
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

    L_arr = 2 * ell_arr

    if not use_sijl:
        # fall back to s_ℓ = 1 while keeping the geometry consistent
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
    L_arr = 2 * ell_arr
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
    L_arr = 2 * ell_arr
    P_prime_vals = np.asarray(legendre_P_derivative(L_arr, cos_theta), dtype=float)
    return np.asarray(J_ell, dtype=float) * P_prime_vals / L_safe


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


# ---------------------------------------------------------------------------
# Asymptotic kernels and auxiliary integrals
# ---------------------------------------------------------------------------


def _kernel_roots(x: float, z: float) -> tuple[float, float]:
    """Return helper square-root terms used across the kernels."""

    x_clipped = float(np.clip(x, -1.0, 1.0))
    z_val = float(z)
    X = np.sqrt(max(1.0 - 2.0 * x_clipped * z_val + z_val * z_val, 0.0))
    Y = np.sqrt(max(1.0 + 2.0 * x_clipped * z_val + z_val * z_val, 0.0))
    return X, Y


def Sprime_ecc_kernel(x: float, z: float) -> float:
    """Return the eccentric--eccentric asymptotic kernel ``S'(x; z)``."""

    X, Y = _kernel_roots(x, z)
    log1 = np.log(((1.0 + X + z) * (1.0 + Y - z)) / 4.0)
    log2 = np.log(((1.0 + X - z) * (1.0 + Y + z)) / 4.0)
    return 0.5 * (log1 / (1.0 - x) - log2 / (1.0 + x))


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


def I2_from_pair(pair: OrbitPair, *, nodes: int = 400) -> float:
    """Convenience wrapper computing ``I_2`` from an :class:`OrbitPair`."""

    # ``nodes`` retained for API compatibility though the integrator uses ``quad``.
    del nodes
    a, b, c, d = _radii_sorted(pair)
    return I2_numeric(a, b, c, d)


def I2_from_orbits(a_i: float, e_i: float, a_j: float, e_j: float, *, nodes: int = 400) -> float:
    """Convenience wrapper computing ``I_2`` from two orbital configurations."""

    pair = _build_pair(a_i, e_i, 1.0, a_j, e_j, 1.0)
    return I2_from_pair(pair, nodes=nodes)


def regime_masks(
    eta_grid: Sequence[float],
    e_i: float,
    e_j: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify ratio ``η=a_i/a_j`` into non-overlap, overlap, and embedded regimes."""

    eta = np.asarray(eta_grid, dtype=float)
    a_j = 1.0

    non_overlap = np.zeros_like(eta, dtype=bool)
    overlap = np.zeros_like(eta, dtype=bool)
    embedded = np.zeros_like(eta, dtype=bool)

    for idx, eta_val in enumerate(eta):
        pair = _build_pair(eta_val * a_j, e_i, 1.0, a_j, e_j, 1.0)
        non_overlap[idx], overlap[idx], embedded[idx] = _regime_flags(pair)

    return non_overlap, overlap, embedded


def contiguous_segments(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index pairs for contiguous True segments."""

    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def Jbar_ecc_nonoverlap(pair: OrbitPair) -> float:
    """Return the geometry-only non-overlap asymptotic coupling ``\bar{J}``."""

    # Multiply the result by :meth:`OrbitPair.mass_prefactor` for the physical value.

    inner, outer = pair.inner, pair.outer
    ratio = ((1.0 + inner.e) * (1.0 - outer.e)) ** 1.5
    denom = np.sqrt(max(inner.e * outer.e, 1.0e-300))
    return float(ratio / (np.pi ** 2 * denom * outer.periapsis))


def Jbar_ecc_overlap(pair: OrbitPair) -> float:
    """Return the geometry-only overlapping asymptotic coupling ``\bar{J}``."""

    # Multiply the result by :meth:`OrbitPair.mass_prefactor` for the physical value.

    inner, outer = pair.inner, pair.outer
    a = min(inner.periapsis, outer.periapsis)
    b = max(inner.periapsis, outer.periapsis)
    c = min(inner.apoapsis, outer.apoapsis)
    d = max(inner.apoapsis, outer.apoapsis)
    integral = I2_numeric(a, b, c, d)
    return float(4.0 * integral / (np.pi ** 3 * inner.a * outer.a))


# ---------------------------------------------------------------------------
# Asymptotic metadata helpers for series reconstruction
# ---------------------------------------------------------------------------


@dataclass
class AsymptoticMetadata:
    """Book-keeping structure for asymptotic Coupling/Omega reconstructions."""

    case: str
    regime: str
    prefactor: Optional[float]
    z: Optional[float]
    I2: Optional[float] = None
    ell_power: float = 2.0
    kernel_supported: bool = False


def _classify_circular(e_i: float, e_j: float, tol: float) -> str:
    e_i_abs = abs(float(e_i))
    e_j_abs = abs(float(e_j))
    if e_i_abs <= tol and e_j_abs <= tol:
        return "both_circular"
    if (e_i_abs <= tol) ^ (e_j_abs <= tol):
        return "one_circular"
    return "eccentric"


def _L_from_ells(ells: np.ndarray) -> np.ndarray:
    return 2 * np.asarray(ells, dtype=int)


def _series_from_meta(ell_arr: np.ndarray, meta: AsymptoticMetadata) -> Optional[np.ndarray]:
    if meta.prefactor is None:
        return None

    ell_arr = np.asarray(ell_arr, dtype=int)
    if ell_arr.size == 0:
        return np.zeros(0, dtype=float)

    L = _L_from_ells(ell_arr).astype(float)
    safe_L = np.where(L == 0.0, np.inf, L)
    if meta.regime == "overlap":
        base = meta.prefactor * np.ones_like(safe_L)
    else:
        if meta.z is None:
            return None
        base = meta.prefactor * (meta.z ** L)

    with np.errstate(divide="ignore", over="ignore"):
        series = base / (safe_L ** meta.ell_power)
    return series.astype(float, copy=False)


def asymp_J_for_eta(
    ells,
    pair: OrbitPair,
    *,
    use_overlap=False,
    I2_func=None,
    I2_value=None,
    return_meta=False,
    eccentricity_tol: float = 1.0e-12,
):
    """Return asymptotic ``J_{ijℓ}`` values (optionally with metadata)."""

    ell_arr = np.asarray([] if ells is None else ells, dtype=int)
    ai = pair.primary.a
    aj = pair.secondary.a
    e_i = pair.primary.e
    e_j = pair.secondary.e
    case = _classify_circular(pair.primary.e, pair.secondary.e, eccentricity_tol)
    regime = "overlap" if use_overlap else "non_overlap"

    I2_val: Optional[float] = None
    prefactor: Optional[float] = None
    z_value: Optional[float] = None
    ell_power = 2.0
    kernel_supported = False

    if case == "both_circular":
        outer = pair.outer
        inner = pair.inner
        if outer.a > 0.0:
            prefactor = 1.0 / outer.a
            z_value = inner.a / outer.a
            ell_power = 1.0

    elif case == "one_circular":
        if not use_overlap:
            outer = pair.outer
            if outer.a > 0.0:
                circ_primary = abs(pair.primary.e) <= eccentricity_tol
                ecc_orbit = pair.secondary if circ_primary else pair.primary
                ecc = ecc_orbit.e
                prefactor = (
                    (1.0 / outer.a)
                    * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
                    * np.sqrt(max(1.0 - ecc, 0.0) / max(abs(ecc), 1.0e-300))
                )
                inner_by_ap, outer_by_ap = _order_by_apocenter(pair)
                rp_in = inner_by_ap.periapsis
                ra_out = outer_by_ap.apoapsis
                if ra_out > 0.0:
                    z_value = min(1.0, rp_in / ra_out)
                    ell_power = 1.5
                else:
                    prefactor = None
                    z_value = None

    else:  # both eccentric
        inner_ap, outer_ap = _order_by_apocenter(pair)
        if use_overlap:
            if I2_value is not None:
                I2_val = float(I2_value)
            elif I2_func is not None:
                I2_val = float(I2_func(ai, e_i, aj, e_j))
            if I2_val is not None:
                prefactor = (
                    (4.0 / (np.pi ** 3))
                    * I2_val
                    / (pair.primary.a * pair.secondary.a)
                )
                z_value = 1.0
                kernel_supported = True
        else:
            rp_out = max(outer_ap.periapsis, 1.0e-300)
            ra_in = inner_ap.apoapsis
            prefactor = (
                (1.0 / (np.pi ** 2))
                * ((1.0 + inner_ap.e) * (1.0 - outer_ap.e)) ** 1.5
                / np.sqrt(max(inner_ap.e, 1.0e-300) * max(outer_ap.e, 1.0e-300))
                * (1.0 / rp_out)
            )
            z_value = ra_in / rp_out
            kernel_supported = True

    meta = AsymptoticMetadata(
        case=case,
        regime=regime,
        prefactor=None if prefactor is None else float(prefactor),
        z=None if z_value is None else float(z_value),
        I2=None if I2_val is None else float(I2_val),
        ell_power=float(ell_power),
        kernel_supported=bool(kernel_supported),
    )

    series = _series_from_meta(ell_arr, meta)

    if return_meta:
        return series, meta
    return series


def _kernel_series_RHS(x: float, z: float) -> float:
    """Closed-form RHS for the eccentric kernel (no extra ``π²`` factor)."""

    x = float(np.clip(x, -1.0, 1.0))
    z = float(z)
    X, Y = _kernel_roots(x, z)
    eps = 1.0e-15
    t1 = max(((1.0 + X + z) * (1.0 + Y - z)) / 4.0, eps)
    t2 = max(((1.0 + X - z) * (1.0 + Y + z)) / 4.0, eps)
    return 0.5 / (1.0 - x) * np.log(t1) - 0.5 / (1.0 + x) * np.log(t2)


def _kernel_z1_full_RHS(x: float) -> float:
    x = float(np.clip(x, -1.0, 1.0))
    s = np.sqrt(max(0.0, 0.5 * (1.0 - x)))
    c = np.sqrt(max(0.0, 0.5 * (1.0 + x)))
    eps = 1.0e-15
    s2 = max(s * s, eps)
    c2 = max(c * c, eps)
    term1 = np.log(max((1.0 + s) * c, eps)) / (4.0 * s2)
    term2 = np.log(max((1.0 + c) * s, eps)) / (4.0 * c2)
    return term1 - term2


def _kernel_z1_cot_RHS(x: float) -> float:
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

    # overlap / embedded cases
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


def kernel_line_Omega_normalized(
    eta_grid: Sequence[float],
    e_i: float,
    e_j: float,
    cos_theta: float,
    *,
    I2_func=None,
    use_cot_approx: bool = False,
    return_signed: bool = False,
) -> np.ndarray:
    """Analytic Ω (no ℓ-sum) normalised for plotting against η."""

    eta = np.asarray(eta_grid, dtype=float)
    out = np.full_like(eta, np.nan, dtype=float)
    non_mask, over_mask, emb_mask = regime_masks(eta, e_i, e_j)

    a_j = 1.0
    for idx, eta_val in enumerate(eta):
        if not (non_mask[idx] or over_mask[idx] or emb_mask[idx]):
            continue

        ai = eta_val * a_j
        use_overlap = bool(over_mask[idx] or emb_mask[idx])
        pair = _build_pair(ai, e_i, 1.0, a_j, e_j, 1.0)
        _, meta = asymp_J_for_eta(
            None,
            pair,
            use_overlap=use_overlap,
            I2_func=I2_func if use_overlap else None,
            return_meta=True,
        )
        out[idx] = omega_kernel_normalized(
            meta,
            cos_theta,
            a_outer=a_j,
            use_cot_approx=use_cot_approx,
            signed=return_signed,
        )

    return out


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
    hamiltonian_terms: Optional[np.ndarray] = None
    hamiltonian_partial_sums: Optional[np.ndarray] = None
    omega_terms: Optional[np.ndarray] = None
    omega_partial_sums: Optional[np.ndarray] = None
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

    def __init__(
        self,
        ell_max: int = 20,
        *,
        s_method: str = "auto",
        overlap_nodes: int = 200,
        eccentricity_tol: float = 1.0e-3,
    ) -> None:
        """Initialise the evaluator with its truncation and s_ijl configuration."""

        if ell_max < 2:
            raise ValueError("ell_max must be at least 2 for the quadrupole term.")
        self.ell_max = ell_max
        self.s_method = s_method
        self.overlap_nodes = overlap_nodes
        self.eccentricity_tol = eccentricity_tol

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Evaluate the Hamiltonian, frequency and torque for ``pair``."""

        cached = pair.get_cached_dynamics(self.method_name)
        if cached is not None and {
            "hamiltonian",
            "omega",
            "torque",
        }.issubset(cached):
            return InteractionResult(
                hamiltonian=float(cached["hamiltonian"]),
                omega=float(cached["omega"]),
                torque=float(cached["torque"]),
                series_ell=cached.get("series_ell"),
                series_coefficients=cached.get("series_coefficients"),
                hamiltonian_terms=cached.get("hamiltonian_terms"),
                hamiltonian_partial_sums=cached.get("hamiltonian_partial_sums"),
                omega_terms=cached.get("omega_terms"),
                omega_partial_sums=cached.get("omega_partial_sums"),
                method=self.method_name,
            )

        series_ell = even_ells(self.ell_max, start=0)
        if series_ell.size == 0:
            return InteractionResult(0.0, 0.0, 0.0, method=self.method_name)

        geom_vals = pair.get_couplings(
            "exact:J_exact",
            series_ell,
            J_exact,
            method=self.s_method,
            nodes=self.overlap_nodes,
            ecc_tol=self.eccentricity_tol,
        )

        mass = pair.mass_prefactor()

        cos_inc = pair.cos_inclination
        P_vals = np.asarray(legendre_P(series_ell, cos_inc), dtype=float)
        h_terms_geom = geom_vals * P_vals
        h_partial_geom = np.cumsum(h_terms_geom)
        hamiltonian = float(mass * h_partial_geom[-1])

        P_prime_vals = np.asarray(legendre_P_derivative(series_ell, cos_inc), dtype=float)
        gradient_geom = geom_vals * P_prime_vals
        dH_dx_geom = float(np.sum(gradient_geom))
        dH_dx = mass * dH_dx_geom

        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        omega_terms_geom = -gradient_geom / L_i
        omega_partial_geom = np.cumsum(omega_terms_geom)
        omega = float(mass * omega_partial_geom[-1])
        torque = pair.torque_from_omega(omega)

        J_vals = mass * geom_vals
        h_terms = mass * h_terms_geom
        h_partial = mass * h_partial_geom
        omega_terms = mass * omega_terms_geom
        omega_partial = mass * omega_partial_geom

        result = InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=series_ell,
            series_coefficients=J_vals,
            hamiltonian_terms=h_terms,
            hamiltonian_partial_sums=h_partial,
            omega_terms=omega_terms,
            omega_partial_sums=omega_partial,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=series_ell,
            series_coefficients=J_vals,
            hamiltonian_terms=h_terms,
            hamiltonian_partial_sums=h_partial,
            omega_terms=omega_terms,
            omega_partial_sums=omega_partial,
        )

        return result


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
            radius_ratio = pair.primary.a / pair.outer.a
            return 2.0 * np.pi * omega_orb * (pair.secondary.m / pair.M_central) * (
                radius_ratio
            ) * kernel

        if tag in {"inner", "outer"}:
            if pair.non_overlapping:
                ecc_orbit = pair.outer if tag == "inner" else pair.inner
                a_out = pair.outer.a
                geom_prefactor = (
                    (1.0 / a_out)
                    * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
                    * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
                )
                dH_dx_geom = geom_prefactor * S2_even_integral_kernel(x, z)
                return -(pair.mass_prefactor() * dH_dx_geom) / L_i

            J_bar_geom = Jbar_ecc_overlap(pair)
            kappa_geom = J_bar_geom / max(L_i * omega_orb, 1.0e-300)
            theta = np.arccos(np.clip(x, -1.0, 1.0))
            sin_theta = max(np.sin(theta), 1.0e-15)
            return (
                -0.5
                * pair.mass_prefactor()
                * kappa_geom
                * omega_orb
                * (np.cos(theta) / sin_theta)
            )

        if pair.non_overlapping:
            J_bar_geom = Jbar_ecc_nonoverlap(pair)
            kappa_geom = J_bar_geom / max(L_i * omega_orb, 1.0e-300)
            return (
                -pair.mass_prefactor()
                * kappa_geom
                * omega_orb
                * Sprime_ecc_kernel(x, z)
            )

        J_bar_geom = Jbar_ecc_overlap(pair)
        kappa_geom = J_bar_geom / max(L_i * omega_orb, 1.0e-300)
        theta = np.arccos(np.clip(x, -1.0, 1.0))
        sin_theta = max(np.sin(theta), 1.0e-15)
        return (
            -0.5
            * pair.mass_prefactor()
            * kappa_geom
            * omega_orb
            * (np.cos(theta) / sin_theta)
        )

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

        cached = pair.get_cached_dynamics(self.method_name)
        if cached is not None and {
            "hamiltonian",
            "omega",
            "torque",
        }.issubset(cached):
            return InteractionResult(
                hamiltonian=float(cached["hamiltonian"]),
                omega=float(cached["omega"]),
                torque=float(cached["torque"]),
                method=self.method_name,
            )

        omega = self._omega_scalar(pair, pair.cos_inclination)
        hamiltonian = self._hamiltonian_from_omega(pair)
        torque = pair.torque_from_omega(omega)

        result = InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
        )

        return result


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

    def _asymptotic_J(self, pair: OrbitPair, ell: int | np.ndarray) -> float | np.ndarray:
        """Return the geometry-only asymptotic approximation of ``J_{ijℓ}``."""

        ell_arr = np.asarray(ell, dtype=int)
        if ell_arr.size == 0:
            return np.asarray(ell_arr, dtype=float)

        ell_flat = ell_arr.reshape(-1)
        result_flat = np.zeros_like(ell_flat, dtype=float)
        safe_ell = np.where(ell_flat == 0, np.inf, ell_flat.astype(float))

        tag = pair.circular_configuration
        if tag == "both":
            return _to_output(result_flat.reshape(ell_arr.shape))

        if tag in {"inner", "outer"} and pair.non_overlapping:
            z = pair.z_parameter
            ecc_orbit = pair.outer if tag == "inner" else pair.inner
            a_out = pair.outer.a
            geom_prefactor = (
                (1.0 / a_out)
                * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
                * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
            )
            result_flat = geom_prefactor * (z ** ell_flat) / (safe_ell ** 1.5)
            return _to_output(result_flat.reshape(ell_arr.shape))

        if pair.non_overlapping:
            J_bar = Jbar_ecc_nonoverlap(pair)
            result_flat = J_bar / (safe_ell ** 2)
        else:
            J_bar = Jbar_ecc_overlap(pair)
            result_flat = J_bar / (safe_ell ** 2)

        return _to_output(result_flat.reshape(ell_arr.shape))

    def evaluate_pair(self, pair: OrbitPair) -> InteractionResult:
        """Return the hybrid interaction result for ``pair``."""

        cached = pair.get_cached_dynamics(self.method_name)
        if cached is not None and {
            "hamiltonian",
            "omega",
            "torque",
        }.issubset(cached):
            return InteractionResult(
                hamiltonian=float(cached["hamiltonian"]),
                omega=float(cached["omega"]),
                torque=float(cached["torque"]),
                series_ell=cached.get("series_ell"),
                series_coefficients=cached.get("series_coefficients"),
                method=self.method_name,
            )

        base_result = super().evaluate_pair(pair)
        ells = even_ells(self.lmax_correction)
        if ells.size == 0:
            return base_result

        geom_exact = pair.get_couplings("exact:J_exact", ells, J_exact)
        geom_asym = pair.get_couplings(
            "asymptotic:approximation",
            ells,
            self._asymptotic_J,
        )
        geom_delta = geom_exact - geom_asym
        mass = pair.mass_prefactor()
        P_vals = np.asarray(legendre_P(ells, pair.cos_inclination), dtype=float)
        P_prime_vals = np.asarray(
            legendre_P_derivative(ells, pair.cos_inclination), dtype=float
        )

        delta_H_geom = float(np.dot(geom_delta, P_vals))
        delta_dH_dx_geom = float(np.dot(geom_delta, P_prime_vals))
        delta_H = mass * delta_H_geom
        delta_dH_dx = mass * delta_dH_dx_geom

        L_i = max(pair.angular_momentum_primary, 1.0e-300)
        omega = base_result.omega - delta_dH_dx / L_i
        torque = pair.torque_from_omega(omega)
        hamiltonian = base_result.hamiltonian + delta_H

        result = InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=ells,
            series_coefficients=mass * geom_delta,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=hamiltonian,
            omega=omega,
            torque=torque,
            series_ell=ells,
            series_coefficients=mass * geom_delta,
        )

        return result


__all__ = [
    "Orbit",
    "OrbitPair",
    "InteractionResult",
    "J_series",
    "H_partial_sums",
    "Omega_partial_sums",
    "asymp_J_for_eta",
    "omega_kernel_normalized",
    "omega_asymptotic_from_meta",
    "omega_hybrid_from_meta",
    "regime_masks",
    "contiguous_segments",
    "I2_from_orbits",
    "kernel_line_Omega_normalized",
    "ExactSeriesEvaluator",
    "AsymptoticEvaluator",
    "AsymptoticWithCorrectionsEvaluator",
]
