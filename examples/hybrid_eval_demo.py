#!/usr/bin/env python3
"""Hybrid evaluator demo with logging

This script evaluates several orbit pairs using the hybrid asymptotic-with-
corrections evaluator. It configures the loguru logger at DEBUG level so that
internal debug logs from vrr.geometry and vrr.evaluators are visible.

It also exercises the timed_J_ijl wrapper to demonstrate timing of geometry-only
couplings for a small set of multipoles.

Usage
-----
Run directly:
    python examples/hybrid_eval_demo.py

You can tweak parameters below (ell_max, lmax_correction, etc.).
"""
from __future__ import annotations

from time import perf_counter
from typing import List, Tuple

import numpy as np
from loguru import logger

import sys
from pathlib import Path

# Ensure local repo import when running from source tree
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from vrr import (
    AsymptoticWithCorrectionsEvaluator,
    Orbit,
    OrbitPair,
    legendre_series_ells,
    timed_J_ijl,
)


def make_pairs() -> List[Tuple[str, OrbitPair]]:
    G = 1.0
    M = 1.0
    m_i = 1.0
    m_j = 1.0

    pairs: List[Tuple[str, OrbitPair]] = []

    # Both circular (non-overlap)
    pairs.append(
        (
            "both_circular_non_overlap",
            OrbitPair(
                Orbit(1.0, 0.0, m_i, G=G, M_central=M),
                Orbit(2.0, 0.0, m_j, G=G, M_central=M),
                cos_inclination=0.3,
            ),
        )
    )

    # One circular (inner circular)
    pairs.append(
        (
            "one_circular_inner",
            OrbitPair(
                Orbit(1.0, 0.0, m_i, G=G, M_central=M),
                Orbit(1.8, 0.7, m_j, G=G, M_central=M),
                cos_inclination=-0.5,
            ),
        )
    )

    # One circular (outer circular)
    pairs.append(
        (
            "one_circular_outer",
            OrbitPair(
                Orbit(1.2, 0.8, m_i, G=G, M_central=M),
                Orbit(2.2, 0.0, m_j, G=G, M_central=M),
                cos_inclination=0.0,
            ),
        )
    )

    # Both eccentric, non-overlap: r_a,in < r_p,out
    # Example: a_in=1, e_in=0.7 => r_a,in=1.7; a_out=2, e_out=0.1 => r_p,out=1.8
    pairs.append(
        (
            "both_ecc_non_overlap",
            OrbitPair(
                Orbit(1.0, 0.7, m_i, G=G, M_central=M),
                Orbit(2.0, 0.1, m_j, G=G, M_central=M),
                cos_inclination=0.8,
            ),
        )
    )

    # Both eccentric, overlap
    pairs.append(
        (
            "both_ecc_overlap",
            OrbitPair(
                Orbit(1.0, 0.7, m_i, G=G, M_central=M),
                Orbit(1.3, 0.4, m_j, G=G, M_central=M),
                cos_inclination=-0.2,
            ),
        )
    )

    return pairs


def main() -> None:
    # Configure loguru at DEBUG level with a simple format
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG", format="<lvl>{level}</lvl> | {name}:{function}:{line} | {message}")

    pairs = make_pairs()

    # Hybrid evaluator settings
    evaluator = AsymptoticWithCorrectionsEvaluator(
        ell_max=40,         # asymptotic ℓ grid upper bound (start=2 internally for asymptotic grid)
        quad_epsabs=1e-9,
        quad_epsrel=1e-9,
        lmax_correction=4,  # exact corrections up to this order (start=2)
    )

    print("\n== Hybrid evaluator demo ==\n")
    for name, pair in pairs:
        t0 = perf_counter()
        result = evaluator.evaluate_pair(pair)
        dt = perf_counter() - t0

        # Omega may be scalar or vector depending on L-vector availability
        omega_val = result.omega
        if isinstance(omega_val, np.ndarray):
            omega_desc = f"|Omega|={np.linalg.norm(omega_val):.6e}, vec={omega_val}"
        else:
            omega_desc = f"Omega={float(omega_val):.6e}"

        print(
            f"case={name:>24s} | H={result.hamiltonian:.6e} | {omega_desc} | torque={result.torque} | time={dt*1e3:.2f} ms"
        )

    # Optional: exercise timed_J_ijl on one of the pairs to demonstrate geometry timing
    # sample_name, sample_pair = pairs[-1]
    # ells = legendre_series_ells(10, start=2)
    # _ = timed_J_ijl(sample_pair, ells, method="auto", nodes=200, ecc_tol=1e-3)
    # print(f"\nTimed J_ijl demo completed on case={sample_name} for {ells.size} ℓ values.\n")


if __name__ == "__main__":
    main()
