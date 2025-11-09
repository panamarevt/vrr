"""Utilities for evaluating vector resonant relaxation interaction terms."""

from __future__ import annotations

from vrr.evaluators import (
    AsymptoticEvaluator,
    AsymptoticWithCorrectionsEvaluator,
    ExactSeriesEvaluator,
    InteractionResult,
    asymp_J_for_eta,
    contiguous_segments,
    kernel_line_Omega_normalized,
    omega_asymptotic_from_meta,
    omega_hybrid_from_meta,
    omega_kernel_normalized,
    regime_masks,
)
from vrr.geometry import H_partial_sums, J_series, Omega_partial_sums
from vrr.kernels import I2_from_orbits
from vrr.orbits import Orbit, OrbitPair

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
