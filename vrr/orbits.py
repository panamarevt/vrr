"""Orbit data structures and helpers for vector resonant relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import math
import numpy as np


OmegaValue = Union[float, np.ndarray]
TorqueValue = Union[float, np.ndarray]


@dataclass(frozen=True)
class Orbit:
    """Simple container for Keplerian orbital elements."""

    a: float
    e: float
    m: float
    G: float = 1.0
    M_central: float = 1.0
    Lx: Optional[float] = None
    Ly: Optional[float] = None
    Lz: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate the provided orbital elements."""

        if self.a <= 0:
            raise ValueError("The semi-major axis must be positive.")
        if not (0.0 <= self.e < 1.0):
            raise ValueError("The eccentricity must lie within [0, 1).")
        if self.m <= 0:
            raise ValueError("The mass must be positive.")
        if self.G <= 0:
            raise ValueError("The gravitational constant must be positive.")
        if self.M_central <= 0:
            raise ValueError("The central mass must be positive.")

        components = (self.Lx, self.Ly, self.Lz)
        has_components = [component is not None for component in components]
        if any(has_components) and not all(has_components):
            raise ValueError(
                "All angular momentum vector components must be provided together."
            )

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

        b = max(self.b, 1.0e-300)
        return self.a / b

    @property
    def angular_momentum_vector(self) -> Optional[np.ndarray]:
        """Return the supplied angular-momentum vector, if available."""

        if self.Lx is None and self.Ly is None and self.Lz is None:
            return None
        return np.array([float(self.Lx), float(self.Ly), float(self.Lz)], dtype=float)

    @property
    def angular_momentum_magnitude(self) -> float:
        """Return ``|\vec{L}|`` either from the vector or Keplerian elements."""

        vector = self.angular_momentum_vector
        if vector is not None:
            return float(np.linalg.norm(vector))

        return self.m * np.sqrt(
            self.G * self.M_central * self.a * max(1.0 - self.e ** 2, 0.0)
        )


@dataclass
class OrbitPair:
    """Container describing two interacting stellar orbits."""

    primary: Orbit
    secondary: Orbit
    cos_inclination: float

    def __post_init__(self) -> None:
        """Validate inputs and initialise lazy caches."""

        self.cos_inclination = float(np.clip(self.cos_inclination, -1.0, 1.0))

        primary_constants = (self.primary.G, self.primary.M_central)
        secondary_constants = (self.secondary.G, self.secondary.M_central)

        if not math.isclose(primary_constants[0], secondary_constants[0], rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Both orbits must use the same gravitational constant.")
        if not math.isclose(primary_constants[1], secondary_constants[1], rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("Both orbits must use the same central mass.")

        self._G = float(primary_constants[0])
        self._M_central = float(primary_constants[1])

        self._update_orbit_ordering()

        self._geometry_cache: dict[tuple[Any, ...], np.ndarray] = {}
        self._dynamics_cache: dict[str, dict[str, Any]] = {}
        self._mass_prefactor_cache: Optional[float] = None

        self._geometry_signature: Optional[tuple[float, ...]] = None
        self._dynamics_signature: Optional[tuple[float, ...]] = None
        self._geometry_signature = self._compute_geometry_signature()
        self._dynamics_signature = self._compute_dynamics_signature()

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

    @property
    def inner(self) -> Orbit:
        """Return the orbit with the smaller semi-major axis."""

        return self._inner

    @property
    def outer(self) -> Orbit:
        """Return the orbit with the larger semi-major axis."""

        return self._outer

    @property
    def G(self) -> float:
        """Return the gravitational constant shared by the orbits."""

        return self._G

    @property
    def M_central(self) -> float:
        """Return the central mass shared by the orbits."""

        return self._M_central

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

    def has_angular_momentum_vectors(self) -> bool:
        """Return ``True`` when both orbits supply ``\vec{L}`` components."""

        return (
            self.primary.angular_momentum_vector is not None
            and self.secondary.angular_momentum_vector is not None
        )

    def _omega_vector_from_scalar(self, omega: float) -> np.ndarray:
        """Return the precession vector corresponding to ``omega``."""

        secondary_vector = self.secondary.angular_momentum_vector
        if secondary_vector is None:
            return np.zeros(3, dtype=float)

        magnitude = float(np.linalg.norm(secondary_vector))
        if magnitude <= 0.0:
            return np.zeros(3, dtype=float)

        return float(omega) * secondary_vector / magnitude

    def torque_from_omega(self, omega: OmegaValue) -> TorqueValue:
        """Return the torque acting on the primary orbit."""

        if self.has_angular_momentum_vectors():
            primary_vector = self.primary.angular_momentum_vector
            if primary_vector is None:
                return np.zeros(3, dtype=float)

            if isinstance(omega, np.ndarray):
                omega_vector = np.asarray(omega, dtype=float)
            else:
                omega_vector = self._omega_vector_from_scalar(float(omega))

            return np.cross(omega_vector, primary_vector)

        omega_scalar = float(np.asarray(omega, dtype=float))
        return self.torque_prefactor() * omega_scalar

    def omega_from_scalar(self, omega: float) -> OmegaValue:
        """Return ``omega`` converted to vector form when ``\vec{L}`` is known."""

        if self.has_angular_momentum_vectors():
            return self._omega_vector_from_scalar(float(omega))
        return float(omega)

    def get_couplings(
        self,
        mode: str,
        ells: Sequence[int],
        compute: Optional[
            Callable[["OrbitPair", np.ndarray], np.ndarray | float]
        ] = None,
        **compute_kwargs: Any,
    ) -> np.ndarray:
        """Return cached geometry-only ``J_{ij\ell}`` values."""

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

        return self.primary.angular_momentum_magnitude

    @property
    def orbital_frequency_primary(self) -> float:
        """Return the Keplerian orbital frequency of the primary orbit."""

        orb = self.primary
        return np.sqrt(self.G * self.M_central / (orb.a ** 3))


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
        Orbit(a_i, e_i, m_i, G=G, M_central=M_central),
        Orbit(a_j, e_j, m_j, G=G, M_central=M_central),
        cos_inc,
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


__all__ = [
    "Orbit",
    "OrbitPair",
    "OmegaValue",
    "TorqueValue",
    "_build_pair",
    "_order_by_apocenter",
    "_radii_sorted",
    "_regime_tag",
    "_regime_flags",
]
