import numpy as np
import scipy.special as sp
import warnings
from mpmath import quad, legendre 

from scipy.integrate import dblquad, nquad, odeint

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from loguru import logger


def P_l(l, x):
    """
    Return P_l(x), the standard Legendre polynomial of degree l.
    Assumes l is an integer ≥ 0.  If l is odd and you know it should vanish,
    you will get exactly zero (within numerical precision).
    """
    return sp.lpmv(0, l, x)


def P_l_prime(l, x):
    """
    Return d/dx [P_l(x)].  Formula:
      P_l'(x) = - (l/(sqrt(1 - x^2))) P_l^1(x).
    We use the associated Legendre function P_l^1(x) = l x P_l(x)/(sqrt(1-x^2)) - (l(l-1) P_{l-2}(x))/(sqrt(1-x^2)), 
    but the simplest is sp.lpmv(1,l,x)/sqrt(1-x^2) with the correct sign.
    """
    # Handle x = ±1 safely:
    if abs(x) == 1.0:
        return 0.0
    else:
        return -sp.lpmv(1, l, x) / np.sqrt(1 - x**2)


def P_l_at_zero(l):
    """
    Compute P_l(0).  For even l=2k, P_{2k}(0) = (-1)^k * ( (2k)! / [2^(2k) (k!)^2 ] ).  
    If l is odd, this returns 0 automatically.
    We do everything in log‐space to avoid overflow.
    """
    if (l % 2) == 1:
        return 0.0

    k = l // 2
    log_gamma_2k = sp.gammaln(2*k + 1)
    log_gamma_k  = sp.gammaln(k + 1)
    log_val = log_gamma_2k - (2*k)*np.log(2) - 2*log_gamma_k
    return ((-1)**k) * np.exp(log_val)


def s_ijl(a_i, a_j, e_i, e_j, l):
    """
    Compute s_{ij}^l for orbits i,j at eccentricities e_i,e_j 
    and semimajor axes a_i,a_j.  Here l is the *actual* Legendre index.
    We assume l is even (if l is odd, s_{ij}^l should vanish).
    """
    # If both e=0, s_{ij}^l = 1 for any l:
    if (e_i == 0) and (e_j == 0):
        return 1.0

    # Determine which orbit is inner vs outer:
    if a_i < a_j:
        a_in,  a_out  = a_i,  a_j
        e_in,  e_out  = e_i,  e_j
    else:
        a_in,  a_out  = a_j,  a_i
        e_in,  e_out  = e_j,  e_i

    chi_in  = 1.0 / np.sqrt(1 - e_in**2)
    chi_out = 1.0 / np.sqrt(1 - e_out**2)

    # NON‐OVERLAPPING CASE:  a_in*(1 + e_in) < a_out*(1 - e_out)
    if a_in * (1 + e_in) < a_out * (1 - e_out):
        # Exactly: s_{ij}^l = (chi_out^l / chi_in^{l+1}) * P_{l+1}(chi_in) * P_l(-chi_out)
        P_lp1_in = P_l(l+1,  chi_in)    # P_{l+1}(chi_in)
        P_l_out  = P_l(l,    -chi_out)  # P_{l} (-chi_out)
        return (chi_out**l / chi_in**(l+1)) * P_lp1_in * P_l_out

    # OVERLAPPING CASE: do the 2D integral via Gauss–Legendre
    else:
        return compute_integral_gauss(e_in, e_out, l, a_in, a_out)


def compute_integral_gauss(e_in, e_out, l, a_in, a_out, N=100):
    """
    Double integral for s_{ij}^l when orbits overlap. l is the *actual* Legendre index.
    """
    alpha = a_in / a_out
    c     = 1.0 / alpha  # = a_out/a_in

    # 1) Gauss–Legendre nodes & weights on [0, π] for phi_in
    nodes_in, weights_in = np.polynomial.legendre.leggauss(N)
    phi_in  = 0.5*(nodes_in + 1)*np.pi
    w_in    = 0.5*np.pi*weights_in

    # 2) Gauss–Legendre nodes & weights on [0, π] for phi_out
    nodes_out, weights_out = np.polynomial.legendre.leggauss(N)
    phi_out = 0.5*(nodes_out + 1)*np.pi
    w_out   = 0.5*np.pi*weights_out

    # 3) Build X = (1 + e_in cos(phi_in))^(l+1),  Y = (1 + e_out cos(phi_out))^l
    X_in  = (1 + e_in*np.cos(phi_in))**(l+1)   # shape (N,)
    Y_out = (1 + e_out*np.cos(phi_out))**(l)   # shape (N,)

    # 4) Broadcast to 2-D arrays
    X2d = X_in.reshape(N,1)   # shape (N,1)
    Y2d = Y_out.reshape(1,N)  # shape (1,N)

    # 5) Weight matrix
    W2d = w_in.reshape(N,1) * w_out.reshape(1,N)

    # 6) Piecewise integrand
    #    region1 where X2d < c * Y2d ⇒ integrand = X2d / Y2d
    #    region2 otherwise             ⇒ integrand = (1/alpha^2) * (Y2d / X2d)
    mask        = (X2d < c * Y2d)
    region1_vals = X2d / Y2d
    region2_vals = (1.0/alpha**2) * (Y2d / X2d)

    integrand = np.where(mask, region1_vals, region2_vals)  # (N,N)
    total     = np.sum(integrand * W2d)
    return total / (np.pi**2)


def J_ijl(m_i, m_j, r_i, r_j, l, G=1.0, s_ijl=1.0):
    """
    Compute J_{ijl} = G m_i m_j s_{ij}^l [P_l(0)]^2 r_<^l / r_>^{l+1}.
    Here l is the *actual* Legendre index (even).  
    """
    r_less    = np.minimum(r_i, r_j)
    r_greater = np.maximum(r_i, r_j)
    P_l0      = P_l_at_zero(l)  # = P_l(0)

    return G * m_i * m_j * s_ijl * (P_l0**2) * (r_less**l) / (r_greater**(l+1))


def X_func(x, z):
    """
    Compute X = sqrt(1 - 2 x z + z^2)
    """
    return np.sqrt(1 - 2*x*z + z**2)

def Y_func(x, z):
    """
    Compute Y = sqrt(1 + 2 x z + z^2)
    """
    return np.sqrt(1 + 2*x*z + z**2)

# def Omega(m_i, m_j, a_i, a_j, cos_theta, e_i=0, e_j=0,  G=1, Mbh=1, l_max = 2):

#     L_i = m_i*np.sqrt(G*Mbh*a_i*(1-e_i**2))
#     L_j = m_j*np.sqrt(G*Mbh*a_j*(1-e_j**2))

#     l = 2
#     Omega = 0
#     while (l < l_max+1):

#         s_ijl_ = s_ijl(a_i, a_j, e_i, e_j, l)
#         Jijl_ = J_ijl(m_i, m_j, a_i, a_j, l,s_ijl = s_ijl_ )
#         Omega += Jijl_/(L_i)*P_l_prime(l, cos_theta)
        
#         l += 2
    
#     return Omega


import concurrent.futures as _cf

def Omega(m_i, m_j, a_i, a_j, cos_theta,
          e_i=0, e_j=0, G=1, Mbh=1, l_max=2,
          parallel=False, max_workers=None):
    """
    Compute Ω_ij up to multipole order l_max (even l ≥ 2).

    Parameters
    ----------
    parallel : bool, default False
        If True, each ℓ-term is evaluated concurrently with
        `concurrent.futures.ProcessPoolExecutor`.
    max_workers : int or None
        Passed to the executor; None lets Python pick a sensible value.

    All other parameters are identical to the original routine.
    """
    # pre-compute the two angular momenta
    L_i = m_i * (G * Mbh * a_i * (1 - e_i**2))**0.5
    L_j = m_j * (G * Mbh * a_j * (1 - e_j**2))**0.5

    # -----------------------------------------------------------------
    # helper that returns one term  J_ijl / L_i * P'_l(cos θ)
    # -----------------------------------------------------------------
    def _one_term(l):
        s   = s_ijl(a_i, a_j, e_i, e_j, l)
        Jij = J_ijl(m_i, m_j, a_i, a_j, l, s_ijl=s)
        return Jij / L_i * P_l_prime(l, cos_theta)

    # -----------------------------------------------------------------
    # serial or parallel evaluation
    # -----------------------------------------------------------------
    l_vals = list(range(2, l_max + 1, 2))
    if not parallel or len(l_vals) == 1:
        # original behaviour (serial)
        Omega_val = sum(_one_term(l) for l in l_vals)
    else:
        # parallel execution
        #with _cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:    
            Omega_val = sum(ex.map(_one_term, l_vals))

    return Omega_val


def omega_orb(a, G = 1, Mbh = 1):
    '''Keplerian orbital frequency'''
    t_orb = 2*np.pi*np.sqrt(a**3/(G*Mbh))
    return 1.0/t_orb

def Omega_c_asymp_z1(a_i, m_j, cos_theta, G=1, Mbh=1):
    '''Small-angle (theta << 1) approximation for z = 1'''
    theta = np.arccos(cos_theta)
    return 2*omega_orb(a_i, G = 1, Mbh = 1)*m_j/Mbh *1/theta**2 

def Omega_c_asymp(a_i, a_j, m_j, cos_theta, G=1, Mbh=1):
    """
    Asymptotic + correction approximation of Omega_{ij} for circular orbits.
    Determines inner/outer orbit from a_i, a_j.
    """
    # Determine inner and outer semimajor axes
    if a_i < a_j:
        a_in, a_out = a_i, a_j
    else:
        a_in, a_out = a_j, a_i

    x = cos_theta
    z = a_in / a_out

    X = X_func(x, z)
    Y = Y_func(x, z)

    # Precompute P'_2(x) and P'_4(x)
    P2p = P_l_prime(2, x)
    P4p = P_l_prime(4, x)

    # Coefficients
    c2 = (1.0/np.pi) - 0.25
    c4 = (1.0/(2*np.pi)) - (9.0/64)

    # Terms from S'_{ij}
    term_quad  = c2 * P2p * z**2
    term_oct   = -c4 * P4p * z**4
    term_corr1 = z * (1 + X) / (np.pi * (1 - x*z + X) * X)
    term_corr2 = z * (1 + Y) / (np.pi * (1 + x*z + Y) * Y)

    S_prime = term_quad + term_oct + term_corr1 - term_corr2

    # Omega_{ij} = 2π ω_i (m_j / Mbh) (a_in / a_out) S_prime
    Omega = 2 * np.pi * omega_orb(a_in, G, Mbh) * (m_j / Mbh) * (a_in / a_out) * S_prime
    return Omega

def dOmega_c_dtheta(a_i, a_j, m_j, cos_theta, G=1, Mbh=1, h=1e-6):
    theta = np.arccos(cos_theta)
    return (Omega_c_asymp(a_i, a_j, m_j, np.cos(theta + h), G, Mbh)
          - Omega_c_asymp(a_i, a_j, m_j, np.cos(theta - h), G, Mbh)) / (2*h)


def dOmega_dtheta(m_i, m_j, a_i, a_j, cos_theta,
                  e_i=0, e_j=0, G=1, Mbh=1, l_max=2, h=1e-6):
    """
    Numerical θ-derivative of the general Ω function.
    
    Uses central difference in θ:
      θ = arccos(cos_theta)
      dΩ/dθ ≈ [Ω(cos(θ+h)) – Ω(cos(θ–h))] / (2h)
    
    Parameters match those of Omega(...), plus:
      h     : small step in radians for finite difference
      l_max : maximum even harmonic (passed to Omega)
    """
    # Recover θ from cosθ
    theta = np.arccos(cos_theta)

    # Evaluate Ω at θ + h and θ - h
    cos_plus  = np.cos(theta + h)
    cos_minus = np.cos(theta - h)

    Omega_plus  = Omega(m_i, m_j, a_i, a_j, cos_plus,
                        e_i, e_j, G, Mbh, l_max)
    Omega_minus = Omega(m_i, m_j, a_i, a_j, cos_minus,
                        e_i, e_j, G, Mbh, l_max)

    return (Omega_plus - Omega_minus) / (2 * h)
