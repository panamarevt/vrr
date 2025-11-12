#!/usr/bin/env python3
"""Benchmark hybrid vs exact on overlapping orbits with timing.

This script compares runtime and outputs for:
- Hybrid (AsymptoticWithCorrectionsEvaluator) with lmax_correction=4
- Exact series (ExactSeriesEvaluator) truncated to a moderate ell_max

It focuses on overlapping regimes and prints step timing summaries.
It also times geometry builds via timed_J_series for the correction band.
"""
from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from loguru import logger

# Local import when running from source tree
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vrr import (
    AsymptoticWithCorrectionsEvaluator,
    ExactSeriesEvaluator,
    Orbit,
    OrbitPair,
    legendre_series_ells,
    timed_J_series,
)


def make_overlapping_pairs():
    G = 1.0
    M = 1.0
    m_i = 1.0
    m_j = 1.0

    pairs = []

    # Overlap examples
    pairs.append(
        (
            "overlap_moderate",
            OrbitPair(
                Orbit(1.0, 0.6, m_i, G=G, M_central=M),
                Orbit(1.4, 0.4, m_j, G=G, M_central=M),
                cos_inclination=0.1,
            ),
        )
    )
    pairs.append(
        (
            "overlap_strong",
            OrbitPair(
                Orbit(1.0, 0.8, m_i, G=G, M_central=M),
                Orbit(1.2, 0.6, m_j, G=G, M_central=M),
                cos_inclination=-0.3,
            ),
        )
    )
    return pairs


def run_case(name: str, pair: OrbitPair) -> None:
    logger.info(f"\nCase: {name}")

    # Configure evaluators
    lmax_correction=4
    hybrid = AsymptoticWithCorrectionsEvaluator(ell_max=lmax_correction, quad_epsabs=1e-9, quad_epsrel=1e-9, lmax_correction=lmax_correction)
    exact = ExactSeriesEvaluator(ell_max=48, s_method="simplified", overlap_nodes=200, eccentricity_tol=1e-3)

    # Hybrid timing
    t0 = perf_counter()
    hr = hybrid.evaluate_pair(pair)
    t_hybrid = perf_counter() - t0

    # Exact timing
    t1 = perf_counter()
    er = exact.evaluate_pair(pair)
    t_exact = perf_counter() - t1

    # Time J_series over correction band for diagnostics
    corr_ells = legendre_series_ells(4, start=2)
    _ = timed_J_series(pair, corr_ells, use_sijl=True, s_method="auto", N_overlap=200, ecc_tol=1e-3)

    # Print summary
    def omega_desc(val):
        if isinstance(val, np.ndarray):
            return f"|Omega|={np.linalg.norm(val):.6e}"
        return f"Omega={float(val):.6e}"

    print(
        f"hybrid: H={hr.hamiltonian:.6e} | {omega_desc(hr.omega)} | t={t_hybrid*1e3:.2f} ms"
    )
    print(
        f" exact: H={er.hamiltonian:.6e} | {omega_desc(er.omega)} | t={t_exact*1e3:.2f} ms"
    )

    speedup = (t_exact / t_hybrid) if t_hybrid > 0 else np.inf
    print(f" speedup (exact/hybrid): {speedup:.2f}x")


def main() -> None:
    logger.remove()
    logger.add(lambda m: print(m, end=""), level="DEBUG", format="<lvl>{level}</lvl> | {name}:{function}:{line} | {message}")

    for name, pair in make_overlapping_pairs():
        run_case(name, pair)


if __name__ == "__main__":
    main()
