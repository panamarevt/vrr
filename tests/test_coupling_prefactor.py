"""Tests for separating the mass prefactor from geometric couplings."""

from __future__ import annotations

import math
import unittest

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

if np is not None:  # pragma: no branch - guard import for optional dependency
    from vrr_Omegas import (
        I2_numeric,
        J_exact,
        Jbar_ecc_nonoverlap,
        Jbar_ecc_overlap,
        Orbit,
        OrbitPair,
    )
else:  # pragma: no cover - keep type checkers satisfied when NumPy missing
    I2_numeric = J_exact = Jbar_ecc_nonoverlap = Jbar_ecc_overlap = None  # type: ignore[assignment]
    Orbit = OrbitPair = None  # type: ignore[assignment]


@unittest.skipUnless(np is not None, "NumPy is required for coupling tests")
class CouplingPrefactorTests(unittest.TestCase):
    """Validate that geometry-only helpers recover the full couplings."""

    def test_exact_coupling_is_geometry_only(self) -> None:
        """``J_exact`` does not depend on ``G`` or the stellar masses."""

        primary = Orbit(a=1.3, e=0.2, m=2.0, G=0.8)
        secondary = Orbit(a=2.1, e=0.4, m=1.5, G=0.8)
        pair = OrbitPair(primary, secondary, cos_inclination=0.3)

        ells = np.array([2, 4, 6], dtype=int)
        geom = np.asarray(J_exact(pair, ells), dtype=float)

        scaled_pair = OrbitPair(
            Orbit(a=primary.a, e=primary.e, m=5.0, G=2.3),
            Orbit(a=secondary.a, e=secondary.e, m=3.2, G=2.3),
            cos_inclination=pair.cos_inclination,
        )
        geom_scaled = np.asarray(J_exact(scaled_pair, ells), dtype=float)

        np.testing.assert_allclose(geom_scaled, geom, rtol=1.0e-12, atol=1.0e-12)

        physical = pair.mass_prefactor() * geom
        physical_scaled = scaled_pair.mass_prefactor() * geom_scaled

        nonzero = np.abs(physical) > 1.0e-14
        self.assertTrue(np.any(nonzero))

        ratio_expected = scaled_pair.mass_prefactor() / pair.mass_prefactor()
        ratio_observed = physical_scaled[nonzero] / physical[nonzero]

        np.testing.assert_allclose(
            ratio_observed,
            np.full_like(ratio_observed, ratio_expected),
        )

    def test_non_overlap_jbar_scaling(self) -> None:
        """The non-overlap asymptotic coupling scales with ``G m_i m_j``."""

        primary = Orbit(a=1.0, e=0.2, m=1.1, G=0.6)
        secondary = Orbit(a=3.0, e=0.5, m=0.9, G=0.6)
        pair = OrbitPair(primary, secondary, cos_inclination=0.1)

        inner, outer = pair.inner, pair.outer
        geom_val = Jbar_ecc_nonoverlap(pair)

        expected = (
            pair.G
            * pair.primary.m
            * pair.secondary.m
            / (math.pi ** 2)
            * ((1.0 + inner.e) * (1.0 - outer.e)) ** 1.5
            / math.sqrt(max(inner.e * outer.e, 1.0e-300))
            / outer.periapsis
        )

        self.assertAlmostEqual(pair.mass_prefactor() * geom_val, expected, places=12)

    def test_overlap_jbar_scaling(self) -> None:
        """The overlapping asymptotic coupling rescales with the mass prefactor."""

        primary = Orbit(a=1.0, e=0.6, m=1.0, G=1.2)
        secondary = Orbit(a=1.6, e=0.3, m=0.7, G=1.2)
        pair = OrbitPair(primary, secondary, cos_inclination=-0.4)

        inner, outer = pair.inner, pair.outer
        a = min(inner.periapsis, outer.periapsis)
        b = max(inner.periapsis, outer.periapsis)
        c = min(inner.apoapsis, outer.apoapsis)
        d = max(inner.apoapsis, outer.apoapsis)
        integral = I2_numeric(a, b, c, d)

        geom_val = Jbar_ecc_overlap(pair)
        expected = (
            4.0
            * pair.G
            * pair.primary.m
            * pair.secondary.m
            * integral
            / (math.pi ** 3 * inner.a * outer.a)
        )

        self.assertAlmostEqual(pair.mass_prefactor() * geom_val, expected, places=12)


if __name__ == "__main__":
    unittest.main()
