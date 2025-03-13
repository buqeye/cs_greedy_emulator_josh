# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)

###   ###   ###   imports   ###   ###   ###
# import numpy as np
from scipy.special import spherical_jn, spherical_yn
from modules.Constants import *


###   ###   ###   spherical bessels   ###   ###   ###
def F(r, 
      k, 
      l: int):
    r"""Riccati Bessel F function.
    
    Parameters
    ----------
    r : number or array-like
        Either the value or the coordinate-space mesh for the function.
    k : number
        The value for the center-of-mass momentum, p = k = sqrt(2 mu E).
    l : int
        The relative orbital angular momentum quantum number.

    Returns
    -------
    riccati_bessel_F : number or array-like
        The Riccati Bessel F function.
    """
    rho = r * k
    riccati_bessel_F = rho * spherical_jn(l, rho)
    
    return riccati_bessel_F


def G(r, 
      k, 
      l: int):
    r"""Riccati Bessel G function.
    
    Parameters
    ----------
    r : number or array-like
        Either the value or the coordinate-space mesh for the function.
    k : number
        The value for the center-of-mass momentum, p = k = sqrt(2 mu E).
    l : int
        The relative orbital angular momentum quantum number.
    
    Returns
    -------
    riccati_bessel_G : number or array-like
        The Riccati Bessel G function.
    """
    rho = r * k
    riccati_bessel_G = -rho * spherical_yn(l, rho)
    
    return riccati_bessel_G


def F_prime(r, 
            k, 
            l: int):
    r"""Analytic derivative of Riccati Bessel F function.
    
    Parameters
    ----------
    r : number or array-like
        Either the value or the coordinate-space mesh for the function.
    k : number
        The value for the center-of-mass momentum, p = k = sqrt(2 mu E).
    l : int
        The relative orbital angular momentum quantum number.
    
    Returns
    -------
    riccati_bessel_F_prime : number or array-like
        The Riccati Bessel F function.
    """
    rho = r * k
    riccati_bessel_F_prime = (spherical_jn(l, rho, derivative=False) + rho * spherical_jn(l, rho, derivative=True)) * k
    
    return riccati_bessel_F_prime


def G_prime(r, 
            k, 
            l: int):
    r"""Analytic derivative of Riccati Bessel G function.
    
    Parameters
    ----------
    r : number or array-like
        Either the value or the coordinate-space mesh for the function.
    k : number
        The value for the center-of-mass momentum, p = k = sqrt(2 mu E).
    l : int
        The relative orbital angular momentum quantum number.
    
    Returns
    -------
    riccati_bessel_G_prime : number or array-like
        The Riccati Bessel G function.
    """
    rho = r * k
    riccati_bessel_G_prime = (-spherical_yn(l, rho, derivative=False) - rho * spherical_yn(l, rho, derivative=True)) * k
    
    return riccati_bessel_G_prime


def H(r, 
      k, 
      l: int, 
      plus: bool = True, 
      derivative: bool = False):
    r"""Coulomb Hankle function and its analytic derivative.
    
    This function encompasses the implementation of H+, H'+, H-, H'-. These different functions can
    be accessed using the `plus` and `derivative` boolean args.
    
    Parameters
    ----------
    r : number or array-like
        Either the value or the coordinate-space mesh for the function.
    k : number
        The value for the center-of-mass momentum, p = k = sqrt(2 mu E).
    l : int
        The relative orbital angular momentum quantum number.
    plus : bool (optional)
        When `True`, the plus variant of the Coulomb Hankle function will be returned. Otherwise, 
        when `False` the minus variant will be used.
    derivative : bool (optional)
        When `True`, the derivative of the Coulomb Hankle function will be returned. Otherwise, 
        when `False`, non-derivative will be used.
    
    Returns
    -------
    coulomb_hankle_function_H
    _or_
    coulomb_hankle_function_H_prime : number or array-like
        Depending on the values of plus and derivative, the plus or minus, variants of the Coulomb 
        Hankle function will be used.
    """
    plusQ = (+1 if plus else -1)
    
    if not derivative:
        coulomb_hankle_function_H = G(r, k, l) + plusQ * 1j * F(r, k, l)
        return coulomb_hankle_function_H
    else:
        coulomb_hankle_function_H_prime = G_prime(r, k, l) + plusQ * 1j * F_prime(r, k, l)
        return coulomb_hankle_function_H_prime


###   ###   ###   free wave   ###   ###   ###
def analytic_phi(r, 
                 l : int, 
                 energy=50.,
                 mass=neutron_mass / 2):
    r"""Analytic implementation of the free wave, phi.
    
    Phi from its analytic form from Thompson & Nunes. This is also scaled by 1/k, which gives the 
    function units of fm^(-1). This is referred to as "analytic" because it is calling scipy special 
    functions to calculate phi, rather than obtaining it in other ways.
    
    Parameters
    ----------
    r : array-like
        The coordinate space mesh for the problem.
    l : int
        The orbital angular momentum, or partial wave, for which the problem is being solved.
    energy : number (optional)
        The value for the energy in the scattering problem.
    mass : number (optional)
        The value for the reduced mass, mu, as given in the particles in the scattering problem.
    
    Returns
    -------
    phi : numpy array
        The free wave for the particle. This must be the result that one gets for the full wave, 
        psi, when there is no potential. This is in units of fm^(-1).
    """
    p2 = (2 * mass * energy) / hbarc ** 2  # fm ^(-2)
    k = p2 ** 0.5  # p = hbar k but hbar here is 1
    
    phi = F(r, k, l) / k
    
    return phi
