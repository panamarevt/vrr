"""Vector resonant relaxation toolkit organised into focused submodules.

The :mod:`vrr` package splits the original :mod:`vrr_Omegas` helpers into
cohesive building blocks:

``vrr.orbits``
    Data structures describing single orbits and interacting orbit pairs.
``vrr.geometry``
    Geometry-only series, Legendre helpers and utilities for rebuilding
    Hamiltonian/Omega contributions from cached coefficients.
``vrr.kernels``
    Asymptotic kernels, integral evaluations and mass-pre-factor helpers.
``vrr.evaluators``
    High-level evaluators that combine the geometry and kernel pieces to
    produce interaction Hamiltonians, frequencies and torques.

The package re-exports the most commonly used symbols so existing code can
transition gradually while benefiting from the clearer layout."""

from __future__ import annotations

from .orbits import Orbit, OrbitPair
from .geometry import H_partial_sums, J_exact, J_series, Omega_partial_sums, s_ijl
from .kernels import (
    I2_from_orbits,
    I2_from_pair,
    I2_numeric,
    Jbar_ecc_nonoverlap,
    Jbar_ecc_overlap,
    S2_even_integral_kernel,
    S2_even_kernel,
    S_circ_log_kernel,
    S_ecc_kernel,
    S_overlap_kernel,
    Sprime_circ_log_kernel,
    Sprime_ecc_kernel,
    omega_asymptotic_from_meta,
    omega_hybrid_from_meta,
    omega_kernel_normalized,
)
from .evaluators import (
    AsymptoticEvaluator,
    AsymptoticWithCorrectionsEvaluator,
    ExactSeriesEvaluator,
    InteractionResult,
    asymp_J_for_eta,
    asymptotic_components,
    kernel_line_Omega_normalized,
)
from .legendre import (
    even_ells,
    legendre_P,
    legendre_P_derivative,
    legendre_P_zero,
    legendre_series_ells,
)

__all__ = [
    "Orbit",
    "OrbitPair",
    "InteractionResult",
    "ExactSeriesEvaluator",
    "AsymptoticEvaluator",
    "AsymptoticWithCorrectionsEvaluator",
    "J_exact",
    "J_series",
    "s_ijl",
    "H_partial_sums",
    "Omega_partial_sums",
    "even_ells",
    "legendre_series_ells",
    "legendre_P",
    "legendre_P_derivative",
    "legendre_P_zero",
    "I2_numeric",
    "I2_from_pair",
    "I2_from_orbits",
    "Jbar_ecc_nonoverlap",
    "Jbar_ecc_overlap",
    "Sprime_ecc_kernel",
    "Sprime_circ_log_kernel",
    "S_ecc_kernel",
    "S_circ_log_kernel",
    "S2_even_kernel",
    "S2_even_integral_kernel",
    "S_overlap_kernel",
    "asymp_J_for_eta",
    "asymptotic_components",
    "omega_kernel_normalized",
    "omega_asymptotic_from_meta",
    "omega_hybrid_from_meta",
    "kernel_line_Omega_normalized",
]
