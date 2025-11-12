"""Quick benchmark: hybrid (ℓ_max=lmax_corr=4) vs exact (ℓ_max=20) on overlapping pairs.

This script exercises the evaluators on a small suite of clearly overlapping
orbit pairs and prints wall-clock timings. It relies on the DEBUG logs already
instrumented in geometry (s_ijl, J_exact/J_series) and kernels to help identify
which parts dominate runtime.

Run with:
    python -m scripts.overlap_hybrid_vs_exact
"""

from __future__ import annotations

from time import perf_counter
from typing import List, Tuple

import numpy as np

try:  # Prefer loguru if available
    from loguru import logger

    logger.remove()
    logger.add(lambda m: print(m, end=""), level="DEBUG")
except Exception:  # Fallback to stdlib logging
    import logging as _logging

    logger = _logging.getLogger("scripts.benchmark_overlap")
    if not logger.handlers:
        _handler = _logging.StreamHandler()
        _formatter = _logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    logger.setLevel(_logging.DEBUG)

from vrr.evaluators import AsymptoticWithCorrectionsEvaluator, ExactSeriesEvaluator
from vrr.orbits import Orbit, OrbitPair


def overlapping_pairs() -> List[OrbitPair]:
    """Return a small set of clearly overlapping pairs with varied eccentricities."""

    cases: List[Tuple[float, float, float, float]] = [
        # (a_i, e_i, a_j, e_j)  — ensure overlap by choosing close semi-major axes
        (1.0, 0.6, 1.5, 0.3),
        (1.1, 0.5, 1.6, 0.4),
        (0.9, 0.7, 1.4, 0.2),
        (1.2, 0.4, 1.7, 0.5),
    ]

    pairs: List[OrbitPair] = []
    for a_i, e_i, a_j, e_j in cases:
        p = Orbit(a=a_i, e=e_i, m=1.0, G=1.0)
        s = Orbit(a=a_j, e=e_j, m=1.0, G=1.0)
        pair = OrbitPair(p, s, cos_inclination=0.3)
        # Keep only genuinely overlapping pairs
        inner, outer = pair.inner, pair.outer
        if inner.apoapsis >= outer.periapsis:
            pairs.append(pair)
        else:
            logger.warning(
                f"Skipping non-overlapping case: ai={a_i}, ei={e_i}, aj={a_j}, ej={e_j}"
            )
    return pairs


def bench_pair(idx: int, pair: OrbitPair, nodes: int) -> None:
    """Benchmark a single pair using a shared Gauss-Legendre node count.

    The same ``nodes`` value is applied to the exact series (via ``overlap_nodes``)
    and to the hybrid correction band (``correction_overlap_nodes``) to ensure a
    fair comparison of algorithmic differences rather than quadrature accuracy.
    """

    exact = ExactSeriesEvaluator(ell_max=20, s_method="auto", overlap_nodes=nodes)
    hybrid = AsymptoticWithCorrectionsEvaluator(
        lmax_correction=4,
        correction_s_method="exact",
        correction_overlap_nodes=nodes,
    )

    logger.info(
        f"\nCase #{idx}: a=({pair.primary.a:.3g},{pair.secondary.a:.3g}) e=({pair.primary.e:.3g},{pair.secondary.e:.3g}) \n"
    )

    t0 = perf_counter()
    r_exact = exact.evaluate_pair(pair)
    t_exact = (perf_counter() - t0) * 1e3

    t0 = perf_counter()
    r_hyb = hybrid.evaluate_pair(pair)
    t_hyb = (perf_counter() - t0) * 1e3

    # Basic consistency check: neither should be NaN or inf
    def _finite(x: float) -> bool:
        return np.isfinite(float(x))

    ok = all(
        [
            _finite(r_exact.hamiltonian),
            _finite(r_hyb.hamiltonian),
            _finite(r_exact.omega if isinstance(r_exact.omega, float) else np.linalg.norm(r_exact.omega)),
            _finite(r_hyb.omega if isinstance(r_hyb.omega, float) else np.linalg.norm(r_hyb.omega)),
        ]
    )

    speedup = t_exact / max(t_hyb, 1e-12)
    logger.info(
        f"exact: {t_exact:8.2f} ms | hybrid: {t_hyb:8.2f} ms | speedup x{speedup:5.2f} | finite={ok}"
    )


def main() -> None:
    pairs = overlapping_pairs()
    if not pairs:
        logger.error("No overlapping pairs were constructed; adjust cases list.")
        return
    # Use a unified node count; adjust here to explore accuracy/performance trade-offs
    nodes = 160
    logger.info(f"Using unified Gauss-Legendre node count: nodes={nodes}\n")
    for i, pair in enumerate(pairs, 1):
        bench_pair(i, pair, nodes)


if __name__ == "__main__":
    main()
