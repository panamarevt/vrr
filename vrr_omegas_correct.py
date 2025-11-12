import numpy as np
import matplotlib.pyplot as plt

#import phiGtools as phiG # Relaxartion time, orbital parameters
import pandas as pd

import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

import seaborn as sns
import scipy.special as sp
#from scipy.integrate import dblquad, nquad, odeint

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from dataclasses import dataclass
from typing import Optional

from loguru import logger

import warnings


# ================== Vectorized Legendre polynomials ==================
# --- Vectorized P'_{2ℓ}(x) using SciPy broadcasting  ---
def P_prime(ell, x):
    """
    Vectorized derivative d/dx P_{2ℓ}(x).
    Uses d/dx P_L(x) = - P_L^1(x) / sqrt(1-x^2), with L = 2ℓ.
    For |x|≈1 returns 0 (matches your scalar behavior).
    """
    ell = np.asarray(ell, dtype=int)
    x   = np.asarray(x, dtype=float)
    Lb, Xb = np.broadcast_arrays(2*ell, np.clip(x, -1.0, 1.0))
    denom = np.sqrt(np.maximum(1.0 - Xb*Xb, 1e-300))
    deriv = -sp.lpmv(1, Lb, Xb) / denom
    mask_edge = (np.abs(Xb) >= 1.0 - 1e-12)
    return np.where(mask_edge, 0.0, np.real_if_close(deriv, tol=1e4)).astype(float)

# --- Vectorized P_{2ℓ}(0) via your closed form (works on arrays) ---
def P_2ell_0_vec(ells):
    ells = np.asarray(ells, dtype=int)
    log_num = sp.gammaln(2*ells + 1)
    log_den = (2*ells)*np.log(2.0) + 2*sp.gammaln(ells + 1)
    return ((-1.0)**ells) * np.exp(log_num - log_den)

def compute_integral_gauss_vec(e_in, e_out, L_arr, a_in, a_out, N=200, tol=1e-3):
    """
    Vectorized counterpart for many ACTUAL Legendre degrees L at once (L = 2ℓ).
    Implements exactly:
      s = (1/π^2) ∬ [ min(A, α^{-1}B)^(L+1) / max(αA, B)^L ] dφ dφ'
    with A=1+e_in cosφ, B=1+e_out cosφ'.
    """
    L_arr = np.asarray(L_arr, dtype=int)
    alpha = a_in / a_out
    if alpha <= 0 or not np.isfinite(alpha):
        return np.zeros_like(L_arr, dtype=float)
    if (abs(e_in) < tol) and (abs(e_out) < tol):
        return np.ones_like(L_arr, dtype=float)

    # Gauss–Legendre on [0, π]
    nodes_in,  w_in_  = np.polynomial.legendre.leggauss(N)
    nodes_out, w_out_ = np.polynomial.legendre.leggauss(N)
    phi_in  = 0.5*(nodes_in  + 1.0)*np.pi
    phi_out = 0.5*(nodes_out + 1.0)*np.pi
    w_in    = 0.5*np.pi*w_in_
    w_out   = 0.5*np.pi*w_out_
    W2d     = w_in[:, None] * w_out[None, :]

    # Bases
    A = 1.0 + e_in*np.cos(phi_in)     # (N,)
    B = 1.0 + e_out*np.cos(phi_out)   # (N,)

    # Powers for all L
    A_L    = A[:, None] ** L_arr[None, :]
    A_Lp1  = A[:, None] ** (L_arr[None, :] + 1)
    B_L    = B[:, None] ** L_arr[None, :]
    B_Lp1  = B[:, None] ** (L_arr[None, :] + 1)

    # Region split: region 1 if αA <= B
    reg1_mask = (alpha * A[:, None] <= B[None, :])[:, :, None]  # (N,N,1)

    # Region 1: A^{L+1}/B^{L}
    reg1 = A_Lp1[:, None, :] / np.maximum(B_L[None, :, :], 1e-300)

    # Region 2: α^{-(2L+1)} * B^{L+1} / A^{L}   <-- FIXED FACTOR
    alpha_pow = alpha ** (-(2 * L_arr + 1))      # shape (L,)
    reg2 = alpha_pow[None, None, :] * (B_Lp1[None, :, :] / np.maximum(A_L[:, None, :], 1e-300))

    integrand = np.where(reg1_mask, reg1, reg2)  # (N,N,L)
    # Integrate over φ, φ'
    tot = np.tensordot(W2d, integrand, axes=([0,1],[0,1]))  # (L,)
    return tot / (np.pi**2)


# --- Vectorized closed-form s_L for non-overlap (all L at once) ---
def s_nonoverlap_closed_form_vec(e_in, e_out, L_arr):
    """
    Vectorized version of your non-overlap closed form:
      s^L = (χ_out^L / χ_in^{L+1}) * P_{L+1}(χ_in) * P_L(-χ_out)
    where χ = 1/sqrt(1-e^2). Uses eval_legendre (stable for χ>1).
    """
    L_arr = np.asarray(L_arr, dtype=int)
    chi_in  = 1.0 / np.sqrt(max(1.0 - e_in*e_in, 1e-300))
    chi_out = 1.0 / np.sqrt(max(1.0 - e_out*e_out, 1e-300))
    P_lp1_in = sp.eval_legendre(L_arr + 1, chi_in)
    P_l_out  = sp.eval_legendre(L_arr - 1, chi_out)
    return (chi_out**L_arr) / (chi_in**(L_arr + 1)) * P_lp1_in * P_l_out

# --- Vectorized s_ℓ driver (uses non-overlap closed form or overlap 2D integral) ---
def s_series(a_i, a_j, e_i, e_j, ells, method="exact", N_overlap=200, e_tol=1.05e-3):
    """
    Return s_{ij}^{(2ℓ)} for an array of ℓ, vectorized:
      - Determine inner/outer by semimajor axes.
      - If both nearly circular: s=1 for all ℓ.
      - If NON-overlap and method!='exact': use closed form (fast).
      - Else (overlap or method='exact'): use vectorized 2D Gauss integral across all L=2ℓ.
    """
    ells = np.asarray(ells, dtype=int)
    L_arr = 2 * ells  # ACTUAL Legendre degrees

    # inner/outer by a
    if a_i < a_j:
        a_in, a_out = a_i, a_j
        e_in, e_out = e_i, e_j
    else:
        a_in, a_out = a_j, a_i
        e_in, e_out = e_j, e_i

    # circular-circular → s=1
    if (abs(e_in) < e_tol) and (abs(e_out) < e_tol):
        return np.ones_like(ells, dtype=float)

    # overlap test: r_a,in >= r_p,out ?
    r_p_in = a_in*(1.0 - e_in); r_a_in = a_in*(1.0 + e_in)
    r_p_out = a_out*(1.0 - e_out); r_a_out = a_out*(1.0 + e_out)
    non_overlap = (r_a_in < r_p_out)

    if non_overlap and (method != "exact"):
        # closed-form vectorized (fast path)
        return s_nonoverlap_closed_form_vec(e_in, e_out, L_arr)
    else:
        # overlap or forced exact → vectorized Gauss integral (robust)
        return compute_integral_gauss_vec(e_in, e_out, L_arr, a_in, a_out, N=N_overlap)

# --- Vectorized J over ℓ (uses your exact J_ijl math) ---
def J_series(m_i, m_j, a_i, a_j, e_i, e_j, ells, G=1.0, use_sijl=True, s_method="exact", N_overlap=200):
    """
    Return array J_{ijℓ} for all ℓ in `ells` (vectorized).
    If use_sijl=False, sets s_ℓ = 1 (useful for circular tests).
    use s_method=''exact'' to evaluate the integral in s_ijl  even for non-overlapping orbits.
    """
    ells = np.asarray(ells, dtype=int)
    if use_sijl:
        s_arr = s_series(a_i, a_j, e_i, e_j, ells, method=s_method, N_overlap=N_overlap)
    else:
        s_arr = np.ones_like(ells, dtype=float)

    r_less = min(a_i, a_j)
    r_more = max(a_i, a_j)
    P0 = P_2ell_0_vec(ells)  # vectorized P_{2ℓ}(0)
    return (G * m_i * m_j) * s_arr * (P0**2) * (r_less**(2*ells)) / (r_more**(2*ells + 1))

# --- Optional per-ℓ contributions (handy for diagnostics) ---
def H_terms(ells, J_ell, cos_theta):
    return J_ell * sp.lpmv(0, 2*np.asarray(ells, int), float(cos_theta))

def Omega_terms(ells, J_ell, cos_theta, L_i):
    return (J_ell / float(L_i)) * P_prime(ells, cos_theta)

# --- Partial sums explicitly take `ells` (as requested) ---
def H_partial_sums(ells, J_ell, cos_theta):
    return np.cumsum(H_terms(ells, J_ell, cos_theta))

def Omega_partial_sums(ells, J_ell, cos_theta, L_i):
    return np.cumsum(Omega_terms(ells, J_ell, cos_theta, L_i))




# -------------------- Small geometry helpers ----------------------
def radial_extrema(a, e) :
    """Return (r_peri, r_apo)."""
    return a * (1.0 - e), a * (1.0 + e)


def in_out_by_apocenter(ai, ei, aj , ej) :
    """Return ((a_in, e_in), (a_out, e_out)) by comparing apocenters."""
    rpi, rai = radial_extrema(ai, ei)
    rpj, raj = radial_extrema(aj, ej)
    if rai <= raj:
        return (ai, ei), (aj, ej)
    else:
        return (aj, ej), (ai, ei)


def overlap_flag(ai: float, ei: float, aj: float, ej: float) -> bool:
    """True if orbits overlap: r_a,in >= r_p,out."""
    (a_in, e_in), (a_out, e_out) = in_out_by_apocenter(ai, ei, aj, ej)
    r_p_in, r_a_in = radial_extrema(a_in, e_in)
    r_p_out, r_a_out = radial_extrema(a_out, e_out)
    return r_a_in >= r_p_out


def alpha_ratio(ai: float, aj: float) -> float:
    """alpha = a_in / a_out by semimajor axis."""
    return min(ai, aj) / max(ai, aj)


def r_out_for_circular(ai: float, aj: float) -> float:
    """For circular-circular asymptotics use r_out = a_out."""
    return max(ai, aj)



def _L_from_ells(ells):
    ells = np.asarray(ells, dtype=int)
    return 2 * ells

def J_asymp_circ_circ(ells, G, mi, mj, ai, aj):
    """
    Circular–circular (even L >= 2):
      J_L ≈ (G m_i m_j / r_out) * z^L / L,   z = a_in/a_out,  L = 2ℓ.
    """
    L = _L_from_ells(ells)
    a_out = max(ai, aj)
    z = min(ai, aj) / a_out
    pref = (G * mi * mj) / a_out
    return pref * (z ** L) / L

def _prefactor_ecc_ecc_nonoverlap(G, mi, mj, a_in, e_in, a_out, e_out, r_p_out):
    """
    Non-overlap prefactor from your Eq. (B68):
      (G m_i m_j / π^2) * [(1+e_in)(1-e_out)]^{3/2} / sqrt(e_in e_out) * (1 / r_p_out)
    """
    return ((G * mi * mj) / (np.pi**2)
            * ((1.0 + e_in) * (1.0 - e_out))**1.5
            / np.sqrt(max(e_in, 1e-300) * max(e_out, 1e-300))
            * (1.0 / r_p_out))

def _prefactor_ecc_ecc_overlap(G, mi, mj, ai, aj, I2):
    """
    Overlap/embedded prefactor (requires I2):
      (4/π^3) * (G m_i m_j / (a_i a_j)) * I2
    """
    return (4.0 / (np.pi**3)) * (G * mi * mj) / (ai * aj) * I2

def J_asymp_ecc_ecc(ells, G, mi, mj, ai, ei, aj, ej, I2=None, use_overlap=None):
    """
    Both eccentric (even L >= 2). Uses L=2ℓ throughout.

    Non-overlap:
      J_L ≈ [ (G m_i m_j / π^2) * ((1+e_in)(1-e_out))^{3/2} / sqrt(e_in e_out) * 1/r_{p,out} ] * (z^L / L^2),
      with z = r_{a,in} / r_{p,out}  (< 1 here).

    Overlap/embedded (requires I2):
      J_L ≈ [ (4/π^3) * (G m_i m_j / (a_i a_j)) * I2 ] * (1 / L^2), with z = 1.
    """
    L = _L_from_ells(ells)

    # inner/outer by apocentre (as in your exact code)
    (a_in, e_in), (a_out, e_out) = in_out_by_apocenter(ai, ei, aj, ej)
    r_p_in, r_a_in   = radial_extrema(a_in, e_in)
    r_p_out, r_a_out = radial_extrema(a_out, e_out)

    # regime detection (same criterion you use elsewhere)
    is_overlap = overlap_flag(ai, ei, aj, ej) if use_overlap is None else bool(use_overlap)

    if not is_overlap:
        z = r_a_in / r_p_out                 # < 1 in true non-overlap
        pref = _prefactor_ecc_ecc_nonoverlap(G, mi, mj, a_in, e_in, a_out, e_out, r_p_out)
        return pref * (z ** L) / (L ** 2), z
    else:
        z = 1.0
        if I2 is None:
            # cannot evaluate asymptotic overlap without I2 — return NaNs so the caller can skip plotting
            return np.full_like(L, np.nan, dtype=float), 1.0
        pref = _prefactor_ecc_ecc_overlap(G, mi, mj, ai, aj, I2)
        return pref / (L ** 2), 1.0

def J_asymp_one_circular(ells, prefactor, ai, ei, aj, ej):
    """
    One circular (even L >= 2):
      J_L ≈ Prefactor * z^L / L^{3/2},   z = r_{p,in}/r_{a,out},  L = 2ℓ.
    """
    L = _L_from_ells(ells)
    (a_in, e_in), (a_out, e_out) = in_out_by_apocenter(ai, ei, aj, ej)
    r_p_in, _   = radial_extrema(a_in, e_in)
    _, r_a_out  = radial_extrema(a_out, e_out)
    z = r_p_in / r_a_out
    z = min(1.0, z)  # enforce non-overlap definition for the base ratio
    return prefactor * (z ** L) / (L ** 1.5), z




# ==================== FIXED analytic Ω-kernel (no ℓ-sum) helpers ====================
def _kernel_series_RHS(x, z):
    """
    Return the RHS of your identity (NO extra π²):
      sum_{ℓ=2,4,...} P'_ℓ(x) z^ℓ / (π² ℓ²)  =  0.5/(1-x) ln(((1+X+z)(1+Y-z))/4)
                                              - 0.5/(1+x) ln(((1+X-z)(1+Y+z))/4),
    with X = sqrt(1 - 2 x z + z²),  Y = sqrt(1 + 2 x z + z²).
    """
    x = float(x); z = float(z)
    X = np.sqrt(max(0.0, 1.0 - 2.0*x*z + z*z))
    Y = np.sqrt(max(0.0, 1.0 + 2.0*x*z + z*z))
    eps = 1e-15
    t1 = max(((1.0 + X + z)*(1.0 + Y - z))/4.0, eps)
    t2 = max(((1.0 + X - z)*(1.0 + Y + z))/4.0, eps)
    return 0.5/(1.0 - x)*np.log(t1) - 0.5/(1.0 + x)*np.log(t2)

def _kernel_z1_full_RHS(x):
    """
    z = 1 closed-form **RHS** (NO extra π²):
      RHS = { ln[(1+s)c]/(4 s²)  -  ln[(1+c)s]/(4 c²) },  with
      s = sqrt((1-x)/2),  c = sqrt((1+x)/2).
    """
    x = float(x)
    s = np.sqrt(max(0.0, 0.5*(1.0 - x)))
    c = np.sqrt(max(0.0, 0.5*(1.0 + x)))
    eps = 1e-15
    s2 = max(s*s, eps); c2 = max(c*c, eps)
    term1 = np.log(max((1.0 + s)*c, eps)) / (4.0*s2)
    term2 = np.log(max((1.0 + c)*s, eps)) / (4.0*c2)
    return term1 - term2



def _kernel_z1_cot_RHS(x):
    """
    z = 1 cotangent **RHS** approximation (NO extra π²):
      RHS ≈ -1/2 * cot(θ),  where x = cos θ.
    """
    theta = np.arccos(x)
    return -0.5 / np.tan(theta)

# ---- Simple aliases to match previously used internal names in your code ----
def _kernel_sum_no_pi(x, z):   # used by your Ω-line builder
    return _kernel_series_RHS(x, z)

def _kernel_z1_full(x):        # used by your Ω-line builder
    return _kernel_z1_full_RHS(x)

def _kernel_z1_cot_half(x):    # used by your Ω-line builder
    return _kernel_z1_cot_RHS(x)
# ============================================
# ================================================================================

# ---------------- radii helpers ----------------
def peri_apo(a, e):
    return a*(1.0 - e), a*(1.0 + e)

def abcd_from_orbits(ai, ei, aj, ej):
    """Return (a,b,c,d) from {rp_i, rp_j, ra_i, ra_j} sorted ascending: a<b<c<d."""
    rpi, rai = peri_apo(ai, ei)
    rpj, raj = peri_apo(aj, ej)
    vals = np.array([rpi, rpj, rai, raj], dtype=float)
    a, b, c, d = np.sort(vals)
    return a, b, c, d

# ---------- I2: corrected Jacobian + embedded-circular limit ----------
def I2_from_sorted_abcd(a, b, c, d, N=400):
    # Embedded circular case (b == c): closed form
    if np.isclose(b, c, rtol=0.0, atol=1e-14*(abs(a)+abs(b)+abs(c)+abs(d)+1.0)):
        return np.pi * b*b / np.sqrt(max(b-a, 0.0)*max(d-b, 0.0))

    # φ ∈ [0, π/2], r = b + (c-b) sin^2 φ
    x, w = np.polynomial.legendre.leggauss(N)
    phi  = 0.25*np.pi*(x + 1.0)     # [-1,1]→[0,π/2]
    dphi = 0.25*np.pi*w

    s2 = np.sin(phi)**2
    r  = b + (c - b)*s2

    denom = np.sqrt(np.maximum((r - a)*(d - r), 0.0))
    integrand = 2.0 * (r*r) / np.maximum(denom, 1e-300)
    return float(np.sum(integrand * dphi))

def I2_from_orbits(ai, ei, aj, ej, N=400):
    rpi, rai = ai*(1-ei), ai*(1+ei)
    rpj, raj = aj*(1-ej), aj*(1+ej)
    a, b, c, d = np.sort([rpi, rpj, rai, raj])
    return I2_from_sorted_abcd(a, b, c, d, N=N)


# ---------- small wrapper for asymptotic J (ecc-ecc) on one η ----------
def asymp_J_for_eta(ells, G, mi, mj, ai, ei, aj, ej, use_overlap, I2_func=None):
    """
    Return asymptotic J_ell (array) for the ecc-ecc case on a single (ai,aj),
    using your J_asymp_ecc_ecc. If use_overlap is True, requires I2_func.
    """
    if use_overlap:
        if I2_func is None:
            return None  # no asymptotic shown in overlap if I2 is not provided
        I2 = I2_func(ai, ei, aj, ej)
        J_as, _ = J_asymp_ecc_ecc(ells, G, mi, mj, ai, ei, aj, ej, I2=I2, use_overlap=True)
    else:
        J_as, _ = J_asymp_ecc_ecc(ells, G, mi, mj, ai, ei, aj, ej, I2=None, use_overlap=False)
    return J_as

def _kernel_line_Omega_normalized(
    eta_grid, e_i, e_j, cos_theta, G, mi, mj,
    I2_func=None, use_cot_approx=False, return_signed=False
):
    """
    Build the analytic-Ω curve (no ℓ-sum) in the Ω panel's normalization:
      Y(η) = |Ω| / (G mi mj / (a_j L_i)).
    Uses the RHS kernels above (NO extra π²) for both non-overlap and overlap.
    """
    a_j = 1.0
    x = float(cos_theta)
    out = np.full_like(eta_grid, np.nan, dtype=float)

    non_mask, over_mask, emb_mask = regime_masks(eta_grid, e_i, e_j)

    for k, eta in enumerate(eta_grid):
        ai = eta * a_j
        aj = a_j

        if non_mask[k]:
            (a_in, e_in), (a_out, e_out) = in_out_by_apocenter(ai, e_i, aj, e_j)
            r_p_in, r_a_in   = radial_extrema(a_in, e_in)
            r_p_out, r_a_out = radial_extrema(a_out, e_out)
            z = r_a_in / r_p_out  # < 1
            # prefactor for non-overlap (your Eq. B68):
            calJ = _prefactor_ecc_ecc_nonoverlap(G, mi, mj, a_in, e_in, a_out, e_out, r_p_out)
            K_RHS = _kernel_series_RHS(x, z)     # <<< NO π² factor

        elif over_mask[k] or emb_mask[k]:
            if I2_func is None:
                continue
            I2 = I2_func(ai, e_i, aj, e_j)
            calJ = _prefactor_ecc_ecc_overlap(G, mi, mj, ai, aj, I2)
            # choose exact z=1 kernel or cot(θ)/2 approximation — BOTH are RHS (NO π²):
            K_RHS = _kernel_z1_cot_RHS(x) if use_cot_approx else _kernel_z1_full_RHS(x)

        else:
            continue

        # Normalization matches Ω panel; magnitude for overlay line
        out[k] = abs(K_RHS) * (aj * calJ) / (G * mi * mj)

    return out


# ---------- main plotter: H & Omega vs eta with exact vs asympt ----------
def plot_H_Omega_vs_eta_with_asymptotics(
    e_i=0.2, e_j=0.8,
    cos_theta=0.5,               # argument x = cos(theta) used in H and Omega terms
    L_i=1.0,                      # (kept for signature/back-compat; not used for Ω now)
    L_max_list=(4, 10, 20, 40),   # compare several ℓ_max (even degrees are 2ℓ with ℓ=1..ℓ_max)
    ells=None,                    # OPTIONAL: explicit array of ℓ (e.g. np.arange(10,41))
    eta_grid=np.geomspace(0.005, 20.0, 400),
    N_overlap=150,                # quadrature nodes for your overlap integrals in J_series
    G=1.0, mi=1.0, mj=1.0,
    s_method="simplified",        # closed-form in non-overlap; exact 2D in overlap (per your s_series)
    I2_func=None,                 # callable(ai, ei, aj, ej) -> I2 for overlap/embedded asymptotics
    xlim=(0.005, 20.0), ylim_H=(1e-6, 1.0), ylim_Om=(1e-6, 1.0),
    # --- NEW options (backward-compatible defaults) ---
    draw_kernel=False,            # add analytic Ω-kernel (no ℓ-sum) as an extra overlay (Ω panel only)
    kernel_use_cot_approx=False,  # if True, use ~ -½ cotθ approximation in overlap; else use full z=1 kernel
    kernel_color="k", kernel_lw=1.4, kernel_alpha=0.9, kernel_ls="-",
    # --- Back-compat aliases (do not document in the legend) ---
    show_kernel_line=None,        # old name for draw_kernel
    use_cot_approx=None,          # old name for kernel_use_cot_approx
    # --- Optional hybrid Ω (exact + analytic − asympt up to ℓ_max) ---
    show_hybrid_omega=False,
    hybrid_ell_max=4,
    hybrid_color="k", hybrid_lw=1.5, hybrid_alpha=0.9, hybrid_ls=":"
):
    """
    Two-panel figure: top = |H|/(G m_i m_j / a_j), bottom = |Ω|/(G m_i m_j / (a_j L_i(η)))
    as functions of η = a_i/a_j. Thick lines = exact (partial sums up to ℓ_max); thin dashed = asymptotic.
    Colors encode ℓ_max (or 'ells' if provided). Line styles show regimes (non-overlap, overlap, embedded).

    Optional (Ω panel only):
      - draw_kernel / show_kernel_line: overlays the analytic Ω-kernel (no ℓ-sum) using your prefactors:
          non-overlap:  K_no_pi(x,z) with z = r_a,in / r_p,out  and J_prefactor = _prefactor_ecc_ecc_nonoverlap
          overlap:      K_no_pi(x,1)  and J_prefactor = _prefactor_ecc_ecc_overlap (needs I2_func)
        The overlay is normalized as: |Ω| / (G m_i m_j / (a_j L_i)) = |K_no_pi| * [ a_j * J_prefactor / (G m_i m_j) ].
      - kernel_use_cot_approx / use_cot_approx: in overlap, use the ≈ -½ cotθ form instead of the full z=1 kernel.

    Optional hybrid Ω:
      - show_hybrid_omega: plots a single black dotted curve
            Ω_hybrid(η) = Ω_exact^{(ℓ≤hybrid_ell_max)} + Ω_kernel − Ω_asymp^{(ℓ≤hybrid_ell_max)}
        (all in the same normalization as the Ω panel; computed with signed sums, |·| plotted).
    """
    # -------- back-compat flag mapping --------
    effective_draw_kernel = bool(draw_kernel or (show_kernel_line is True))
    effective_kernel_use_cot = kernel_use_cot_approx if (use_cot_approx is None) else bool(use_cot_approx)

    # choose ℓ sequences to compare (each produces one color family)
    if ells is not None:
        ell_sets = [np.asarray(ells, dtype=int)]
        ell_labels = [f"ℓ={ell_sets[0][0]}..{ell_sets[0][-1]}"]
        L_max_for_title = int(2*ell_sets[0][-1])
    else:
        ell_sets = [np.arange(1, m+1, dtype=int) for m in L_max_list]
        ell_labels = [rf"ℓ≤{m}" for m in L_max_list]
        L_max_for_title = int(2*max(L_max_list))

    # regime masks
    non_mask, over_mask, emb_mask = regime_masks(eta_grid, e_i, e_j)

    # prepare storage: for each ell_set, we store curves over eta_grid
    a_j = 1.0
    H_exact_list, H_asymp_list = [], []
    Om_exact_list, Om_asymp_list = [], []

    # loop ell-sets
    for ells_vec in ell_sets:
        # exact totals on the grid
        H_ex_curve = np.empty_like(eta_grid, dtype=float)
        Om_ex_curve = np.empty_like(eta_grid, dtype=float)

        # asymptotic totals on the grid
        H_as_curve = np.full_like(eta_grid, np.nan, dtype=float)
        Om_as_curve = np.full_like(eta_grid, np.nan, dtype=float)

        for k, eta in enumerate(eta_grid):
            ai = eta * a_j
            aj = a_j

            # ----- exact: get all J for these ells at this eta in one go -----
            J_ex = J_series(mi, mj, ai, aj, e_i, e_j, ells_vec,
                            G=G, use_sijl=True, s_method=s_method, N_overlap=N_overlap)

            # partial sums up to the last ℓ in this set
            H_tot = H_partial_sums(ells_vec, J_ex, cos_theta)[-1]

            # dynamic L_i(η) = m_i * a_i * sqrt(1 - e_i^2)
            L_i_k = mi * ai * np.sqrt(max(0.0, 1.0 - e_i*e_i))
            Om_tot = Omega_partial_sums(ells_vec, J_ex, cos_theta, L_i_k)[-1]

            # normalize
            H_ex_curve[k] = np.abs(H_tot) / ((G * mi * mj) / a_j)
            Om_ex_curve[k] = np.abs(Om_tot) / ((G * mi * mj) / (a_j * L_i_k))

            # ----- asymptotic: pick branch by regime; sum the same ells -----
            if non_mask[k]:
                J_as = asymp_J_for_eta(ells_vec, G, mi, mj, ai, e_i, aj, e_j, use_overlap=False, I2_func=None)
            elif (over_mask[k] or emb_mask[k]) and (I2_func is not None):
                J_as = asymp_J_for_eta(ells_vec, G, mi, mj, ai, e_i, aj, e_j, use_overlap=True, I2_func=I2_func)
            else:
                J_as = None

            if J_as is not None:
                H_as_curve[k] = np.abs(H_partial_sums(ells_vec, J_as, cos_theta)[-1]) / ((G * mi * mj) / a_j)
                Om_as_curve[k] = np.abs(Omega_partial_sums(ells_vec, J_as, cos_theta, L_i_k)[-1]) / ((G * mi * mj) / (a_j * L_i_k))

        H_exact_list.append(H_ex_curve); H_asymp_list.append(H_as_curve)
        Om_exact_list.append(Om_ex_curve); Om_asymp_list.append(Om_as_curve)

    # ---------- draw figure (two rows: H (top), Ω (bottom)) ----------
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i/(len(ell_sets)-1 + 1e-12)) for i in range(len(ell_sets))]

    # styles per regime for exact curves
    style_map = {"non": "-.", "ovr": ":", "emb": "-"}

    # panel A: H
    ax = axs[0]
    asymp_handles = []
    for i_set, (H_ex, H_as, label) in enumerate(zip(H_exact_list, H_asymp_list, ell_labels)):
        c = colors[i_set]

        # exact, split by regime with different linestyles
        for mask, ls in [(non_mask, style_map["non"]),
                         (over_mask, style_map["ovr"]),
                         (emb_mask, style_map["emb"])]:
            for s, e in contiguous_segments(mask):
                ax.plot(eta_grid[s:e+1], H_ex[s:e+1], color=c, lw=2.0, ls=ls, alpha=0.30)

        # asymptotic (thin dashed) wherever available — use as legend carriers for ℓmax
        h_as = None
        if np.any(np.isfinite(H_as)):
            h_as, = ax.plot(eta_grid, H_as, color=c, lw=1.0, ls="--", label=label)
        asymp_handles.append(h_as)

    ax.set_yscale("log"); ax.set_xscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim_H is not None: ax.set_ylim(*ylim_H)

    # --- boundaries A–D (after limits are set) ---
    bounds = np.array([
        (1.0 - e_j)/(1.0 - e_i),
        (1.0 - e_j)/(1.0 + e_i),
        (1.0 + e_j)/(1.0 - e_i),
        (1.0 + e_j)/(1.0 + e_i),
    ], dtype=float)
    bounds = bounds[np.isfinite(bounds) & (bounds > 0)]
    bounds.sort()
    labels = ["A", "B", "C", "D"][:len(bounds)]

    x_left, x_right = ax.get_xlim()
    _, y_top = ax.get_ylim()  # for text placement

    for xb, lab in zip(bounds, labels):
        if x_left <= xb <= x_right:
            ax.axvline(xb, color="r", lw=1.0, alpha=0.35)
            ax.text(xb, 0.93*y_top, lab, ha="center", va="bottom", fontsize=8, color='r')

    ax.set_ylabel(r"$|H_{ij}| / (G m_i m_j / a_j)$")
    title = (
        rf"$e_i={e_i:g},\ e_j={e_j:g}$;  "
        rf"$x=\cos\theta={float(cos_theta):g}$;  "
        rf"$L=2\ell_{{\max}}\leq {L_max_for_title}$"
    )
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    # Legend: label by ℓ_max (colors); no regime labels
    leg_handles = [h for h in asymp_handles if h is not None]
    if leg_handles:
        ax.legend(handles=leg_handles, title=r"$\ell_{\max}$", fontsize='small', framealpha=0.9)

    # panel B: Omega
    ax = axs[1]
    for i_set, (Om_ex, Om_as, label) in enumerate(zip(Om_exact_list, Om_asymp_list, ell_labels)):
        c = colors[i_set]

        for mask, ls in [(non_mask, style_map["non"]),
                         (over_mask, style_map["ovr"]),
                         (emb_mask, style_map["emb"])]:
            for s, e in contiguous_segments(mask):
                ax.plot(eta_grid[s:e+1], Om_ex[s:e+1], color=c, lw=2.0, ls=ls, alpha=0.30)

        if np.any(np.isfinite(Om_as)):
            ax.plot(eta_grid, Om_as, color=c, lw=1.0, ls="--")

    # --- Analytic Ω-kernel overlay (no ℓ-sum), optional (back-compat with show_kernel_line) ---
    if effective_draw_kernel:
        Om_kernel = _kernel_line_Omega_normalized(
            eta_grid, e_i, e_j, cos_theta, G, mi, mj,
            I2_func=I2_func, use_cot_approx=effective_kernel_use_cot
        )
        ax.plot(eta_grid, Om_kernel, color=kernel_color, lw=kernel_lw,
                alpha=kernel_alpha, ls=kernel_ls)

    # --- Optional: HYBRID Ω (black dotted), using ℓ≤hybrid_ell_max ---
    if show_hybrid_omega:
        ells_hyb = np.arange(1, int(hybrid_ell_max)+1, dtype=int)
        Om_hyb = np.full_like(eta_grid, np.nan, dtype=float)

        # signed components for a proper "exact + analytic − asympt" combination; |·| plotted
        Om_kernel_signed = _kernel_line_Omega_normalized(
            eta_grid, e_i, e_j, cos_theta, G, mi, mj,
            I2_func=I2_func, use_cot_approx=effective_kernel_use_cot, return_signed=True
        )

        non_mask_h, over_mask_h, emb_mask_h = non_mask, over_mask, emb_mask  # names for clarity

        for k, eta in enumerate(eta_grid):
            ai = eta * a_j; aj = a_j
            L_i_k = mi * ai * np.sqrt(max(0.0, 1.0 - e_i*e_i))

            # exact (signed)
            J_ex_h = J_series(mi, mj, ai, aj, e_i, e_j, ells_hyb,
                              G=G, use_sijl=True, s_method=s_method, N_overlap=N_overlap)
            Om_ex_signed = Omega_partial_sums(ells_hyb, J_ex_h, cos_theta, L_i_k)[-1] / ((G * mi * mj) / (a_j * L_i_k))

            # asymp (signed): choose branch by regime; if unavailable, leave NaN
            if non_mask_h[k]:
                J_as_h = asymp_J_for_eta(ells_hyb, G, mi, mj, ai, e_i, aj, e_j, use_overlap=False, I2_func=None)
            elif (over_mask_h[k] or emb_mask_h[k]) and (I2_func is not None):
                J_as_h = asymp_J_for_eta(ells_hyb, G, mi, mj, ai, e_i, aj, e_j, use_overlap=True, I2_func=I2_func)
            else:
                J_as_h = None

            if J_as_h is not None:
                Om_as_signed = Omega_partial_sums(ells_hyb, J_as_h, cos_theta, L_i_k)[-1] / ((G * mi * mj) / (a_j * L_i_k))
                # hybrid (abs for plotting)
                Om_hyb[k] = abs(Om_ex_signed + Om_kernel_signed[k] - Om_as_signed)

        ax.plot(eta_grid, Om_hyb, color=hybrid_color, lw=hybrid_lw, alpha=hybrid_alpha, ls=hybrid_ls)

    ax.set_yscale("log"); ax.set_xscale("log")
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim_Om is not None: ax.set_ylim(*ylim_Om)

    # --- boundaries A–D (after limits are set) ---
    bounds = np.array([
        (1.0 - e_j)/(1.0 - e_i),
        (1.0 - e_j)/(1.0 + e_i),
        (1.0 + e_j)/(1.0 - e_i),
        (1.0 + e_j)/(1.0 + e_i),
    ], dtype=float)
    bounds = bounds[np.isfinite(bounds) & (bounds > 0)]
    bounds.sort()
    labels = ["A", "B", "C", "D"][:len(bounds)]

    x_left, x_right = ax.get_xlim()
    _, y_top = ax.get_ylim()

    for xb, lab in zip(bounds, labels):
        if x_left <= xb <= x_right:
            ax.axvline(xb, color="r", lw=1.0, alpha=0.35)
            ax.text(xb, 0.93*y_top, lab, ha="center", va="bottom", fontsize=8, color='r')

    ax.set_xlabel(r"$a_i/a_j$")
    ax.set_ylabel(r"$|\Omega_{ij}| / (G m_i m_j / (a_j L_i))$")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()