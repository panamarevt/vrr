# vrr: Vector Resonant Relaxation

This repository contains a Python package `vrr` for calculating the Hamiltonian and angular precession frequencies (Ω) associated with Vector Resonant Relaxation (VRR).

Vector Resonant Relaxation is the process by which mutual gravitational torques between orbits cause them to precess (change their orbital plane orientation) while conserving their semi-major axes and eccentricities. This package provides tools to compute these interaction rates for pairs of Keplerian orbits.

## Core Concepts

The interaction between two orbits, `i` and `j`, is calculated by expanding the gravitational potential in a Legendre series. This library provides two primary ways to compute the resulting Hamiltonian and precession rates:

1.  **Exact Series Summation**: Computes the interaction by directly summing the Legendre series up to a specified multipole `ell_max`. This is handled by the `ExactSeriesEvaluator`.
2.  **Asymptotic Approximation**: Uses analytical formulas (kernels) that approximate the full series sum. This is useful for specific orbital regimes (e.g., marginally non-overlapping, overlapping, circular with the same redii) and is handled by the `AsymptoticEvaluator`.
3.  **Hybrid Method**: A combined approach that uses the asymptotic kernel and adds low-order exact terms for correction, providing a balance of speed and accuracy (`AsymptoticWithCorrectionsEvaluator`).

## Module Structure

The `vrr` package is organized into several core modules:

* **`vrr.orbits`**: Defines the fundamental data structures, `Orbit` (for a single orbit) and `OrbitPair` (for two interacting orbits).
* **`vrr.legendre`**: Provides mathematical helpers for Legendre polynomials and their derivatives.
* **`vrr.geometry`**: Contains functions to compute the geometry-only coefficients (`J_exact`, `s_ijl`) of the exact series expansion.
* **`vrr.kernels`**: Implements the analytical asymptotic kernels for different orbital configurations (e.g., `Sprime_ecc_kernel`, `S_overlap_kernel`).
* **`vrr.evaluators`**: This is the main user-facing module. It contains the evaluator classes (`ExactSeriesEvaluator`, `AsymptoticEvaluator`, etc.) that tie all the components together to compute the final physical Hamiltonian, precession frequency (omega), and torque.

## Basic Usage

To compute the interaction between two orbits, create an `OrbitPair` and pass it to an evaluator.

```python
from vrr_Omegas import Orbit, OrbitPair, ExactSeriesEvaluator, AsymptoticEvaluator

# Define two orbits (a, e, m)
orbit_i = Orbit(a=1.0, e=0.1, m=1.0)
orbit_j = Orbit(a=2.0, e=0.3, m=1.0)

# Create an interacting pair with a mutual inclination of 60 degrees (cos(i)=0.5)
pair = OrbitPair(orbit_i, orbit_j, cos_inclination=0.5)

# 1. Use the Exact Series method (sum up to l=10)
exact_eval = ExactSeriesEvaluator(ell_max=10)
exact_result = exact_eval.evaluate_pair(pair)
print(f"Exact Omega: {exact_result.omega}")

# 2. Use the Asymptotic method
asymp_eval = AsymptoticEvaluator(ell_max=10)
asymp_result = asymp_eval.evaluate_pair(pair)
print(f"Asymptotic Omega: {asymp_result.omega}")
