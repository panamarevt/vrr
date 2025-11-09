"""Evaluators and asymptotic helpers for vector resonant relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .geometry import J_exact, J_series, Omega_partial_sums, _evaluate_legendre_series
from .legendre import legendre_series_ells
from .kernels import (
    AsymptoticMetadata,
    Jbar_ecc_nonoverlap,
    Jbar_ecc_overlap,
    _hamiltonian_kernel_both_circular,
    _hamiltonian_kernel_non_overlap,
    _hamiltonian_kernel_one_circular_non_overlap,
    _hamiltonian_kernel_overlap,
    _omega_kernel_both_circular,
    _omega_kernel_non_overlap,
    _omega_kernel_one_circular_non_overlap,
    _omega_kernel_overlap,
    _series_from_meta,
    omega_asymptotic_from_meta,
    omega_hybrid_from_meta,
    omega_kernel_normalized,
)
from .orbits import (
    Orbit,
    OrbitPair,
    OmegaValue,
    TorqueValue,
    _build_pair,
    _order_by_apocenter,
    _regime_flags,
)


@dataclass
class InteractionResult:
    """Container bundling the interaction Hamiltonian, frequency and torque."""

    hamiltonian: float
    omega: OmegaValue
    torque: TorqueValue
    series_ell: Optional[np.ndarray] = None
    series_coefficients: Optional[np.ndarray] = None
    hamiltonian_terms: Optional[np.ndarray] = None
    hamiltonian_partial_sums: Optional[np.ndarray] = None
    omega_terms: Optional[np.ndarray] = None
    omega_partial_sums: Optional[np.ndarray] = None
    method: str = ""


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
                Orbit(a_p, e_p, m_p, G=G, M_central=M_central),
                Orbit(a_s, e_s, m_s, G=G, M_central=M_central),
                cos_inc,
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
                omega=cached["omega"],
                torque=cached["torque"],
                series_ell=cached.get("series_ell"),
                series_coefficients=cached.get("series_coefficients"),
                hamiltonian_terms=cached.get("hamiltonian_terms"),
                hamiltonian_partial_sums=cached.get("hamiltonian_partial_sums"),
                omega_terms=cached.get("omega_terms"),
                omega_partial_sums=cached.get("omega_partial_sums"),
                method=self.method_name,
            )

        ells = legendre_series_ells(self.ell_max, start=0)
        geometry = pair.get_couplings(
            "exact:J_series",
            ells,
            lambda p, e: J_series(
                p,
                e,
                use_sijl=True,
                s_method=self.s_method,
                N_overlap=self.overlap_nodes,
                ecc_tol=self.eccentricity_tol,
            ),
        )

        series_data = _evaluate_legendre_series(
            pair,
            ells,
            geometry,
            include_terms=True,
            include_partials=True,
        )
        omega_value = pair.omega_from_scalar(series_data.omega)

        result = InteractionResult(
            hamiltonian=series_data.hamiltonian,
            omega=omega_value,
            torque=series_data.torque,
            series_ell=series_data.ells,
            series_coefficients=series_data.physical_coefficients,
            hamiltonian_terms=series_data.hamiltonian_terms,
            hamiltonian_partial_sums=series_data.hamiltonian_partial_sums,
            omega_terms=series_data.omega_terms,
            omega_partial_sums=series_data.omega_partial_sums,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=series_data.hamiltonian,
            omega=omega_value,
            torque=series_data.torque,
            series_ell=series_data.ells,
            series_coefficients=series_data.physical_coefficients,
            hamiltonian_terms=series_data.hamiltonian_terms,
            hamiltonian_partial_sums=series_data.hamiltonian_partial_sums,
            omega_terms=series_data.omega_terms,
            omega_partial_sums=series_data.omega_partial_sums,
        )

        return result


# Additional asymptotic helpers and evaluators will be appended below.

def _asymptotic_ell_grid_default(ell_max: int) -> np.ndarray:
    """Return the default ℓ-grid used by the asymptotic evaluators."""

    return legendre_series_ells(ell_max, start=2)


def _J_asymptotic_both_circular(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str] = None
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` values for two circular orbits."""

    ell_arr = np.asarray(ells, dtype=int)
    return np.zeros_like(ell_arr, dtype=float)


def _J_asymptotic_one_circular_non_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str]
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for mixed (one circular) non-overlap pairs."""

    if tag not in {"inner", "outer"}:
        raise ValueError("Circular configuration tag required for mixed regime.")

    ell_arr = np.asarray(ells, dtype=int)
    if ell_arr.size == 0:
        return np.zeros_like(ell_arr, dtype=float)

    safe_ell = np.where(ell_arr == 0, np.inf, ell_arr.astype(float))
    z = float(pair.z_parameter)
    ecc_orbit = pair.outer if tag == "inner" else pair.inner
    a_out = max(pair.outer.a, 1.0e-300)
    prefactor = (
        (1.0 / a_out)
        * (2.0 / (np.pi * np.sqrt(2.0 * np.pi)))
        * np.sqrt(max(1.0 - ecc_orbit.e, 0.0) / max(ecc_orbit.e, 1.0e-300))
    )
    powers = np.asarray(z ** ell_arr, dtype=float)
    result = prefactor * powers / (safe_ell ** 1.5)
    return result.astype(float, copy=False)


def _J_asymptotic_non_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str] = None
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for eccentric non-overlapping orbits."""

    ell_arr = np.asarray(ells, dtype=int)
    if ell_arr.size == 0:
        return np.zeros_like(ell_arr, dtype=float)

    safe_ell = np.where(ell_arr == 0, np.inf, ell_arr.astype(float))
    J_bar = Jbar_ecc_nonoverlap(pair)
    result = J_bar / (safe_ell ** 2)
    return np.asarray(result, dtype=float)


def _J_asymptotic_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str] = None
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for overlapping or embedded orbits."""

    ell_arr = np.asarray(ells, dtype=int)
    if ell_arr.size == 0:
        return np.zeros_like(ell_arr, dtype=float)

    safe_ell = np.where(ell_arr == 0, np.inf, ell_arr.astype(float))
    J_bar = Jbar_ecc_overlap(pair)
    result = J_bar / (safe_ell ** 2)
    return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class AsymptoticRegimeEntry:
    """Describe the ℓ-grid, couplings and kernels for an orbital regime."""

    ell_factory: Callable[[int], np.ndarray]
    j_function: Callable[[OrbitPair, Sequence[int], Optional[str]], np.ndarray]
    hamiltonian_kernel: Callable[[OrbitPair, float, Optional[str], float, float], float]
    omega_kernel: Callable[[OrbitPair, float, Optional[str]], float]

    def ell_grid(self, ell_max: int) -> np.ndarray:
        return np.asarray(self.ell_factory(ell_max), dtype=int)

    def couplings(
        self, pair: OrbitPair, ells: Sequence[int], tag: Optional[str]
    ) -> np.ndarray:
        return np.asarray(self.j_function(pair, ells, tag), dtype=float)

    def kernel_values(
        self,
        pair: OrbitPair,
        quad_epsabs: float,
        quad_epsrel: float,
        tag: Optional[str],
    ) -> tuple[float, float]:
        h_val = float(
            self.hamiltonian_kernel(pair, pair.cos_inclination, tag, quad_epsabs, quad_epsrel)
        )
        omega_val = float(self.omega_kernel(pair, pair.cos_inclination, tag))
        return h_val, omega_val


ASYMPTOTIC_REGISTRY: dict[str, AsymptoticRegimeEntry] = {
    "both_circular": AsymptoticRegimeEntry(
        _asymptotic_ell_grid_default,
        _J_asymptotic_both_circular,
        _hamiltonian_kernel_both_circular,
        _omega_kernel_both_circular,
    ),
    "one_circular_non_overlap": AsymptoticRegimeEntry(
        _asymptotic_ell_grid_default,
        _J_asymptotic_one_circular_non_overlap,
        _hamiltonian_kernel_one_circular_non_overlap,
        _omega_kernel_one_circular_non_overlap,
    ),
    "eccentric_non_overlap": AsymptoticRegimeEntry(
        _asymptotic_ell_grid_default,
        _J_asymptotic_non_overlap,
        _hamiltonian_kernel_non_overlap,
        _omega_kernel_non_overlap,
    ),
    "overlap": AsymptoticRegimeEntry(
        _asymptotic_ell_grid_default,
        _J_asymptotic_overlap,
        _hamiltonian_kernel_overlap,
        _omega_kernel_overlap,
    ),
}


@dataclass(frozen=True)
class AsymptoticComponents:
    """Bundle asymptotic series data with its analytic kernel values."""

    regime: str
    tag: Optional[str]
    ells: np.ndarray
    geometry: np.ndarray
    h_kernel: float
    omega_kernel: float


def _classify_asymptotic_regime(pair: OrbitPair) -> tuple[str, Optional[str]]:
    """Return the registry key and circular tag for ``pair``."""

    tag = pair.circular_configuration
    if tag == "both":
        return "both_circular", tag
    if tag in {"inner", "outer"}:
        if pair.non_overlapping:
            return "one_circular_non_overlap", tag
        return "overlap", tag
    if pair.non_overlapping:
        return "eccentric_non_overlap", None
    return "overlap", None


def _compute_asymptotic_couplings(
    pair: OrbitPair, ell_arr: np.ndarray, *, regime: str, tag: Optional[str]
) -> np.ndarray:
    """Helper passed to :meth:`OrbitPair.get_couplings` for asymptotics."""

    entry = ASYMPTOTIC_REGISTRY[regime]
    return entry.couplings(pair, ell_arr, tag)


def asymptotic_components(
    pair: OrbitPair,
    ell_max: int,
    quad_epsabs: float,
    quad_epsrel: float,
) -> AsymptoticComponents:
    """Return ℓ-grid, couplings and kernels for the asymptotic regime."""

    regime, tag = _classify_asymptotic_regime(pair)
    entry = ASYMPTOTIC_REGISTRY[regime]
    ells = entry.ell_grid(ell_max)
    geometry = pair.get_couplings(
        f"asymptotic:{regime}",
        ells,
        _compute_asymptotic_couplings,
        regime=regime,
        tag=tag,
    )
    h_kernel, omega_kernel = entry.kernel_values(pair, quad_epsabs, quad_epsrel, tag)
    return AsymptoticComponents(
        regime=regime,
        tag=tag,
        ells=ells,
        geometry=np.asarray(geometry, dtype=float),
        h_kernel=h_kernel,
        omega_kernel=omega_kernel,
    )

class AsymptoticEvaluator(BaseEvaluator):
    """Evaluate interactions using asymptotic kernels for large ``ℓ``."""

    method_name = "asymptotic"

    def __init__(
        self,
        ell_max: int = 40,
        *,
        quad_epsabs: float = 1.0e-9,
        quad_epsrel: float = 1.0e-9,
    ) -> None:
        """Initialise the evaluator with its ℓ-grid and quadrature tolerances."""

        if ell_max < 2:
            raise ValueError("ell_max must be at least 2 for the quadrupole term.")
        self.ell_max = int(ell_max)
        self.quad_epsabs = quad_epsabs
        self.quad_epsrel = quad_epsrel

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
                omega=cached["omega"],
                torque=cached["torque"],
                series_ell=cached.get("series_ell"),
                series_coefficients=cached.get("series_coefficients"),
                method=self.method_name,
            )

        components = asymptotic_components(
            pair,
            self.ell_max,
            self.quad_epsabs,
            self.quad_epsrel,
        )
        series_data = _evaluate_legendre_series(pair, components.ells, components.geometry)

        omega_value = pair.omega_from_scalar(series_data.omega)

        result = InteractionResult(
            hamiltonian=series_data.hamiltonian,
            omega=omega_value,
            torque=series_data.torque,
            series_ell=series_data.ells,
            series_coefficients=series_data.physical_coefficients,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=series_data.hamiltonian,
            omega=omega_value,
            torque=series_data.torque,
            series_ell=series_data.ells,
            series_coefficients=series_data.physical_coefficients,
            asymptotic_kernel_H=components.h_kernel,
            asymptotic_kernel_Omega=components.omega_kernel,
        )

        return result


class AsymptoticWithCorrectionsEvaluator(AsymptoticEvaluator):
    """Augment the asymptotic result with low-order exact corrections."""

    method_name = "asymptotic_with_corrections"

    def __init__(self, lmax_correction: int = 4, **kwargs: Any) -> None:
        """Initialise the hybrid evaluator and select correction order."""

        super().__init__(**kwargs)
        if lmax_correction < 2:
            raise ValueError("The correction order must be at least 2.")
        self.lmax_correction = lmax_correction

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
                omega=cached["omega"],
                torque=cached["torque"],
                series_ell=cached.get("series_ell"),
                series_coefficients=cached.get("series_coefficients"),
                method=self.method_name,
            )

        components = asymptotic_components(
            pair,
            self.ell_max,
            self.quad_epsabs,
            self.quad_epsrel,
        )

        ells = legendre_series_ells(self.lmax_correction, start=2)
        if ells.size == 0:
            omega_scalar = components.omega_kernel
            omega_value = pair.omega_from_scalar(omega_scalar)
            torque = pair.torque_from_omega(omega_scalar)
            empty = np.zeros(0, dtype=float)
            result = InteractionResult(
                hamiltonian=components.h_kernel,
                omega=omega_value,
                torque=torque,
                series_ell=ells,
                series_coefficients=empty,
                method=self.method_name,
            )
            pair.cache_dynamics(
                self.method_name,
                hamiltonian=components.h_kernel,
                omega=omega_value,
                torque=torque,
                series_ell=ells,
                series_coefficients=empty,
            )
            return result

        geom_exact = pair.get_couplings("exact:J_exact", ells, J_exact)
        geom_asym = pair.get_couplings(
            f"asymptotic:{components.regime}",
            ells,
            _compute_asymptotic_couplings,
            regime=components.regime,
            tag=components.tag,
        )

        exact_series = _evaluate_legendre_series(pair, ells, geom_exact)
        asymp_series = _evaluate_legendre_series(pair, ells, geom_asym)

        hamiltonian = (
            components.h_kernel
            + exact_series.hamiltonian
            - asymp_series.hamiltonian
        )
        omega_scalar = (
            components.omega_kernel
            + exact_series.omega
            - asymp_series.omega
        )
        omega_value = pair.omega_from_scalar(omega_scalar)
        torque = pair.torque_from_omega(omega_scalar)

        mass = pair.mass_prefactor()
        series_coefficients = mass * (geom_exact - geom_asym)

        result = InteractionResult(
            hamiltonian=hamiltonian,
            omega=omega_value,
            torque=torque,
            series_ell=ells,
            series_coefficients=series_coefficients,
            method=self.method_name,
        )

        pair.cache_dynamics(
            self.method_name,
            hamiltonian=hamiltonian,
            omega=omega_value,
            torque=torque,
            series_ell=ells,
            series_coefficients=series_coefficients,
        )

        return result

def _classify_circular(e_i: float, e_j: float, tol: float) -> str:
    e_i_abs = abs(float(e_i))
    e_j_abs = abs(float(e_j))
    if e_i_abs <= tol and e_j_abs <= tol:
        return "both_circular"
    if (e_i_abs <= tol) ^ (e_j_abs <= tol):
        return "one_circular"
    return "eccentric"


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


def regime_masks(eta_grid: Sequence[float], e_i: float, e_j: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for non-overlap, overlap and embedded regimes."""

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


__all__ = [
    "InteractionResult",
    "BaseEvaluator",
    "ExactSeriesEvaluator",
    "AsymptoticEvaluator",
    "AsymptoticWithCorrectionsEvaluator",
    "AsymptoticComponents",
    "AsymptoticMetadata",
    "asymptotic_components",
    "asymp_J_for_eta",
    "omega_kernel_normalized",
    "omega_asymptotic_from_meta",
    "omega_hybrid_from_meta",
    "regime_masks",
    "contiguous_segments",
    "kernel_line_Omega_normalized",
]
