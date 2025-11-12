"""Evaluators and asymptotic helpers for vector resonant relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
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

# Prefer loguru's logger; fall back gracefully to stdlib logging
try:  # pragma: no cover - trivial import guard
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover - fallback path
    import logging as _logging

    logger = _logging.getLogger("vrr.evaluators")
    if not logger.handlers:
        _handler = _logging.StreamHandler()
        _formatter = _logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(_logging.DEBUG)


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
    from time import perf_counter

    ell_arr = np.asarray(ells, dtype=int)
    t0 = perf_counter()
    out = np.zeros_like(ell_arr, dtype=float)
    dt = perf_counter() - t0
    if ell_arr.size:
        head = ", ".join(map(str, ell_arr.reshape(-1)[:5]))
    else:
        head = ""
    logger.debug(
        (
            f"J_asymptotic_both_circular: ℓ size={ell_arr.size} head=[{head}{'' if ell_arr.size<=5 else ', …'}] | "
            f"out.shape={out.shape} | {dt*1e3:.2f} ms"
        )
    )
    return out


def _J_asymptotic_one_circular_non_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str]
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for mixed (one circular) non-overlap pairs."""
    from time import perf_counter

    if tag not in {"inner", "outer"}:
        raise ValueError("Circular configuration tag required for mixed regime.")

    ell_arr = np.asarray(ells, dtype=int)
    t0 = perf_counter()
    if ell_arr.size == 0:
        out = np.zeros_like(ell_arr, dtype=float)
    else:
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
        out = (prefactor * powers / (safe_ell ** 1.5)).astype(float, copy=False)
    dt = perf_counter() - t0
    if ell_arr.size:
        head = ", ".join(map(str, ell_arr.reshape(-1)[:5]))
    else:
        head = ""
    logger.debug(
        (
            f"J_asymptotic_one_circ_non_overlap: tag={tag} | z={float(pair.z_parameter):.6g} | "
            f"ℓ size={ell_arr.size} head=[{head}{'' if ell_arr.size<=5 else ', …'}] | out.shape={out.shape} | {dt*1e3:.2f} ms"
        )
    )
    return out


def _J_asymptotic_non_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str] = None
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for eccentric non-overlapping orbits."""
    from time import perf_counter

    ell_arr = np.asarray(ells, dtype=int)
    t0 = perf_counter()
    if ell_arr.size == 0:
        out = np.zeros_like(ell_arr, dtype=float)
        J_bar = Jbar_ecc_nonoverlap(pair)
    else:
        safe_ell = np.where(ell_arr == 0, np.inf, ell_arr.astype(float))
        J_bar = Jbar_ecc_nonoverlap(pair)
        out = np.asarray(J_bar / (safe_ell ** 2), dtype=float)
    dt = perf_counter() - t0
    if ell_arr.size:
        head = ", ".join(map(str, ell_arr.reshape(-1)[:5]))
    else:
        head = ""
    logger.debug(
        (
            f"J_asymptotic_non_overlap: Jbar={float(J_bar):.6g} | ℓ size={ell_arr.size} head=[{head}{'' if ell_arr.size<=5 else ', …'}] "
            f"| out.shape={out.shape} | {dt*1e3:.2f} ms"
        )
    )
    return out


def _J_asymptotic_overlap(
    pair: OrbitPair, ells: Sequence[int], tag: Optional[str] = None
) -> np.ndarray:
    """Return asymptotic ``J_{ijℓ}`` for overlapping or embedded orbits."""
    from time import perf_counter

    ell_arr = np.asarray(ells, dtype=int)
    t0 = perf_counter()
    if ell_arr.size == 0:
        out = np.zeros_like(ell_arr, dtype=float)
        J_bar = Jbar_ecc_overlap(pair)
    else:
        safe_ell = np.where(ell_arr == 0, np.inf, ell_arr.astype(float))
        J_bar = Jbar_ecc_overlap(pair)
        out = np.asarray(J_bar / (safe_ell ** 2), dtype=float)
    dt = perf_counter() - t0
    if ell_arr.size:
        head = ", ".join(map(str, ell_arr.reshape(-1)[:5]))
    else:
        head = ""
    logger.debug(
        (
            f"J_asymptotic_overlap: Jbar={float(J_bar):.6g} | ℓ size={ell_arr.size} head=[{head}{'' if ell_arr.size<=5 else ', …'}] "
            f"| out.shape={out.shape} | {dt*1e3:.2f} ms"
        )
    )
    return out


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
        logger.debug(
            (
                f"asymptotic: regime={components.regime} tag={str(components.tag)} | "
                f"ell_max={int(self.ell_max)} | h_kernel={float(components.h_kernel):.6g} | "
                f"omega_kernel={float(components.omega_kernel):.6g}"
            )
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
    """Augment the asymptotic result with low-order exact corrections.

    The hybrid estimate = asymptotic kernels (large-ℓ tail) + exact low-order
    Legendre contributions − asymptotic approximation to those same low-order
    terms. This class now exposes extra knobs to tune the geometry-only
    evaluation of the correction band to improve performance for overlapping
    orbits where :func:`s_ijl` quadrature dominates runtime.

    Parameters
    ----------
    ell_max : int
        Maximum ℓ used for the asymptotic tail (inherited from parent).
    lmax_correction : int, default 4
        Highest ℓ included in the low-order exact correction band (ℓ ≥ 2).
    correction_s_method : {"auto", "exact", "closed_form"}, default "auto"
        Mode passed to :func:`J_exact` / :func:`s_ijl` for the correction band.
        Use "closed_form" to force the non-overlap expression when regimes
        are guaranteed non-overlapping. Use "exact" to force quadrature.
    correction_overlap_nodes : int, default 120
        Gauss-Legendre node count for overlapping quadrature inside the
        correction band (smaller than the default 200 to reduce cost).
    correction_ecc_tol : float, default 1e-3
        Eccentricity threshold forwarded to :func:`s_ijl`.
    **kwargs
        Forwarded to :class:`AsymptoticEvaluator` (ell_max, quad tolerances).
    """

    method_name = "asymptotic_with_corrections"

    def __init__(
        self,
        lmax_correction: int = 4,
        *,
        correction_s_method: str = "auto",
        correction_overlap_nodes: int = 120,
        correction_ecc_tol: float = 1.0e-3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if lmax_correction < 2:
            raise ValueError("The correction order must be at least 2.")
        self.lmax_correction = int(lmax_correction)
        # Enforce a single ℓ_max across asymptotic and correction band
        self.ell_max = int(self.lmax_correction)
        self.correction_s_method = str(correction_s_method)
        self.correction_overlap_nodes = int(correction_overlap_nodes)
        self.correction_ecc_tol = float(correction_ecc_tol)

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

        logger.debug(
            (
                f"hybrid: start | lmax_corr={int(self.lmax_correction)} | ell_max={int(self.ell_max)} | "
                f"quad=({float(self.quad_epsabs):.1e}, {float(self.quad_epsrel):.1e}) | "
                f"a=({float(pair.primary.a):.6g}, {float(pair.secondary.a):.6g}) | "
                f"e=({float(pair.primary.e):.3g}, {float(pair.secondary.e):.3g}) | "
                f"cos_inc={float(pair.cos_inclination):.4f}"
            )
        )
        t0 = perf_counter()
        components = asymptotic_components(
            pair,
            self.ell_max,
            self.quad_epsabs,
            self.quad_epsrel,
        )
        t_components = perf_counter() - t0
        logger.debug(
            (
                f"hybrid: asymptotic components | regime={components.regime} tag={str(components.tag)} | "
                f"n_ell={int(components.ells.size)} | h_kernel={float(components.h_kernel):.6g} | "
                f"omega_kernel={float(components.omega_kernel):.6g}"
            )
        )

        # Reuse the asymptotic ℓ-grid; avoids recomputing Jbar/I2 for a different set
        ells = components.ells
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

        logger.debug(
            (
                f"hybrid: corrections over ℓ size={int(ells.size)} min={str(int(ells.min())) if ells.size else '-'} "
                f"max={str(int(ells.max())) if ells.size else '-'} | computing geom_exact (reuse asymptotic geom)"
            )
        )

        t1 = perf_counter()
        # Use a lightweight wrapper so we can tune s_ijl parameters separately
        # for the correction band without affecting other parts of the code.
        def _exact_wrapper(p: OrbitPair, ell_arr: np.ndarray) -> np.ndarray:
            return np.asarray(
                J_exact(
                    p,
                    ell_arr,
                    method=self.correction_s_method,
                    nodes=self.correction_overlap_nodes,
                    ecc_tol=self.correction_ecc_tol,
                ),
                dtype=float,
            )

        geom_exact = pair.get_couplings("exact:J_exact", ells, _exact_wrapper)
        # Reuse asymptotic geometry computed above; avoid recomputation of Jbar/I2
        geom_asym = np.asarray(components.geometry, dtype=float)
        t_couplings = perf_counter() - t1

        t2 = perf_counter()
        exact_series = _evaluate_legendre_series(pair, ells, geom_exact)
        asymp_series = _evaluate_legendre_series(pair, ells, geom_asym)
        t_series = perf_counter() - t2

        # Basic diagnostics on the corrections
        try:
            diff = np.asarray(geom_exact) - np.asarray(geom_asym)
            d_abs = float(np.linalg.norm(diff, ord=1))
            d_rel = float(d_abs / (np.linalg.norm(geom_exact, ord=1) + 1.0e-300))
            logger.debug(f"hybrid: correction geom | L1_abs={d_abs:.3e} | L1_rel={d_rel:.3e}")
        except Exception:
            pass

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

        # Build robust summaries for omega and torque (scalar or vector)
        try:
            if isinstance(omega_value, np.ndarray):
                omega_summary = f"|Omega|={float(np.linalg.norm(omega_value)):.6g}"
            else:
                omega_summary = f"Omega={float(omega_value):.6g}"
        except Exception:
            omega_summary = f"Omega={omega_value}"

        try:
            if isinstance(result.torque, np.ndarray):
                t = np.asarray(result.torque, dtype=float).reshape(-1)
                if t.size >= 3:
                    torque_summary = f"torque=({t[0]:.6g}, {t[1]:.6g}, {t[2]:.6g})"
                else:
                    torque_summary = f"torque={t.tolist()}"
            else:
                torque_summary = f"torque={float(result.torque):.6g}"
        except Exception:
            torque_summary = f"torque={result.torque}"

        t_total = perf_counter() - t0
        logger.debug(
            (
                f"hybrid: done | H={float(hamiltonian):.6g} | {omega_summary} | {torque_summary} | "
                f"t_components={t_components*1e3:.2f} ms | t_couplings={t_couplings*1e3:.2f} ms | "
                f"t_series={t_series*1e3:.2f} ms | t_total={t_total*1e3:.2f} ms"
            )
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
