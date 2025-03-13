# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)

###   ###   ###   imports   ###   ###   ###
import numpy as np
from scipy.sparse import diags
from numba import njit
from modules.SpecialFunctions import F, G, F_prime, G_prime, H


@njit(fastmath=True)
def propagate_delta_error(a_tilde, 
                          b_tilde, 
                          wavefunction_error, 
                          M, 
                          M_pinv=None):
    r"""Propagates the errors from the wavefunction get a maximum propagated error to the phaseshift.
       NOTE: This computes equation 52 in the paper.
    
    Parameters
    ----------
    a_tilde : float
        The emulated K-matrix coefficient a from the ROM.
    b_tilde : float
        The emulated K-matrix coefficient b from the ROM.
    wavefunction_error : float
        The norm-error on the wave function, || chi_FOM - chi_ROM ||.
    M : numpy array (matrix)
        The left-hand side of the least-squares matching calculation
    M_pinv : numpy array (matrix) (optional)
        The pseudo-inverse left-hand side of the least-squares matching calculation. When given, the
        pseudo-inverse of M will not be taken, making the calculation more efficient.
    
    Returns
    -------
    max_propagated_error : float
        The maximum error on the phase shift, $delta_l$.
    """
    ab_error = propagate_ab_error(wavefunction_error, M, M_pinv=M_pinv)
    denominator = np.sqrt(a_tilde ** 2 + b_tilde ** 2)
    
    max_propagated_error = ab_error / denominator
    return max_propagated_error

@njit(fastmath=True)
def true_delta_error(a,
                     a_tilde, 
                     b,
                     b_tilde, 
                     wavefunction_error, 
                     M, 
                     M_pinv=None):
    r"""Propagates the errors from the wavefunction get a maximum propagated error to the phaseshift.
       NOTE: This computes equation 52 in the paper.
    
    Parameters
    ----------
    a : float
        The emulated K-matrix coefficient a from the FOM.
    a_tilde : float
        The emulated K-matrix coefficient a from the ROM.
    b : float
        The emulated K-matrix coefficient b from the FOM.
    b_tilde : float
        The emulated K-matrix coefficient b from the ROM.
    wavefunction_error : float
        The norm-error on the wave function, || chi_FOM - chi_ROM ||.
    M : numpy array (matrix)
        The left-hand side of the least-squares matching calculation
    M_pinv : numpy array (matrix) (optional)
        The pseudo-inverse left-hand side of the least-squares matching calculation. When given, the
        pseudo-inverse of M will not be taken, making the calculation more efficient.
    
    Returns
    -------
    max_propagated_error : float
        The maximum error on the phase shift, $delta_l$.
    """
    ab_error = propagate_ab_error(wavefunction_error, M, M_pinv=M_pinv)
    ab_matrix = np.array([a - a_tilde, b - b_tilde])
    denominator = np.linalg.norm(ab_matrix)
    
    max_propagated_error = ab_error / denominator
    return max_propagated_error

@njit(fastmath=True)
def propagate_ab_error(wavefunction_error, 
                       M, 
                       M_pinv=None):
    r"""Propagates the errors from the wavefunction get a maximum propagated error to $a$ and $b$.
       NOTE: This computes equation 53 in the paper.
    
    Parameters
    ----------
    wavefunction_error : float
        The norm-error on the wave function, || y_FOM - y_ROM ||.
    M : numpy array (matrix)
        The left-hand side of the least-squares matching calculation
    M_pinv : numpy array (matrix) (optional)
        The pseudo-inverse left-hand side of the least-squares matching calculation. When given, the
        pseudo-inverse of M will not be taken, making the calculation more efficient.
    
    Returns
    -------
    propagated_error : float
        The maximum error on $a$ and $b$.
    """
    if M_pinv is None:
        M_pinv = np.linalg.pinv(M)
    
    error_matrix = M_pinv * wavefunction_error
    propagated_error = np.linalg.norm(error_matrix)
    
    return propagated_error

def true_ab_error(a, 
                  a_tilde, 
                  b, 
                  b_tilde, 
                  ret_all_errors=False):
    r"""True error from $a$ and $b$.
    
    Parameters
    ----------
    a : float
        The FOM coefficient $a$ for calculating the K matrix.
    a_tilde : float
        The ROM coefficient $a$ for calculating the K matrix.
    b : float
        The FOM coefficient $b$ for calculating the K matrix.
    b_tilde : float
        The ROM coefficient $b$ for calculating the K matrix.
    ret_all_errors : bool (optional)
        When `True` and errors on $a$ and $b$ will be returned as well, after the original return
        value, `error`.
    
    Returns
    -------
    error : float
        The exact error on the phase shift as calculated from $a$ and $b$.
    """
    a_error = a - a_tilde
    b_error = b - b_tilde
    
    error_matrix = np.array((a_error, b_error))
    error = np.linalg.norm(error_matrix)
    
    if ret_all_errors:
        return error, a_error, b_error
    else:
        return error


def match_using_inverse_log(r_mesh, 
                            u, 
                            p, 
                            l: int, 
                            r_match=8, 
                            u_pr=None, 
                            derivative_accuracy: int = 0, 
                            scattered_wave: bool = False,
                            use_T_matrix: bool = False):
    r"""Matches a given wave function to its asymptotic limit.
    
    While this matching process is generally well known and widely used among the nuclear physics community, 
    this implementation is specifically from Thompson and Nunes's book, "Nuclear Reactions for Astrophysics: 
    Principles, Calculation and Applications of Low-Energy Reactions". There are better ways to implement 
    this (and in the past I have done a better implementation). This is left in a sub-optimal state to 
    increase readability.
    NOTE: Specifically Table 3.1 in Thompson and Nunes is phenomenal.
    
    Parameters
    ----------
    r_mesh : array-like
        The mesh for the problem
    u : array-like
        Either the full wave function, psi, or the scattered wave function, chi
    p : number
        Here k is p, or the momentum.
        NOTE: p = hbar * k, but hbar = 1, so p = k
    r_match : number
        The matching radius for the matching procedure. Typically r_match is the last point in the 
        mesh.
        NOTE: If the u[match_index] is not found then the last point in the mesh will be used
    l : int
        The relative angular momentum quantum number l, or \ell, for the problem.
    energy : number
        The energy of the incoming wave
    u_pr : array-like, (optional)
        The derivative of the given wave function, u. If derivative_accuracy is left as its default 
        value of 0, then u_pr will be used. If derivative_accuracy is not 0 then u_pr will be found 
        numerically with the accuracy that was input. When specified, u_pr will override the 
        numerical derivative specified by derivative_accuracy.
    derivative_accuracy : int (optional)
        The desired accuracy for the derivative, u_pr. If left u_pr is given, then a numerical 
        derivative will *not* be taken.
        NOTE: Implemented accuracies: 2, 4, 6, 8
    scattered_wave : bool (optional)
        A boolean flag to tell the function whether or not the input wave function, u, is the full 
        wave function, psi, or the scattered wave function, chi.
    
    Returns
    -------
    return_dict : dict
        A dictionary containing:
        {r, u_scaled, u_pr_scaled, delta_l, scale, K_l, T_l, S_l, R_l}
    """
    
    ###   ###   ###   variable setup   ###   ###   ###
    phi = F(r_mesh, p, l) / p
    phi_pr = F_prime(r_mesh, p, l) / p
    if scattered_wave:
        chi = u
        psi = chi + phi  # this is an unmatched psi
    elif not scattered_wave:
        psi = u
    
    if u_pr is None:
        if derivative_accuracy == 0:
            raise ValueError(f"Must specify a nonzero derivative_accuracy value if there is no u_pr input.")
        allowed_accuracies = [2, 4, 6, 8]
        if derivative_accuracy not in allowed_accuracies:
            raise ValueError(f"Invalid accuracy. Allowed accuracies are: {allowed_accuracies}.")
    else:
        derivative_accuracy = 0
    
    ###   ###   ###   handling derivatives   ###   ###   ###
    # NOTE: From this "block" onward, r_mesh goes to r because the mesh will be different if a numerical derivative is taken
    if derivative_accuracy != 0:
        n = len(r_mesh)
        dr = r_mesh[1] - r_mesh[0]  # this assumes uniform mesh !!
        accuracy_of_2 = diags([-1, 0, 1],
                              [-1, 0, 1], shape=(n, n)) / (2 * dr)
        accuracy_of_4 = diags([1, -8, 0, 8, -1],
                              [-2, -1, 0, 1, 2], shape=(n, n)) / (12 * dr)
        accuracy_of_6 = diags([-1, 9, -45, 0, 45, -9, 1],
                              [-3, -2, -1, 0, 1, 2, 3], shape=(n, n)) / (60 * dr)
        accuracy_of_8 = diags([3, -32, 168, -672, 0, 672, -168, 32, -3],
                              [-4, -3, -2, -1, 0, 1, 2, 3, 4],
                              shape=(n, n)) / (840 * dr)
        derivative_matrix = eval(f"accuracy_of_{derivative_accuracy}.toarray()")  # "construct" the desired derivative matrix
        
        if scattered_wave:
            chi_pr = np.dot(derivative_matrix, chi)[derivative_accuracy: -derivative_accuracy]
        psi_pr = np.dot(derivative_matrix, psi)[derivative_accuracy: -derivative_accuracy]
        r = r_mesh[derivative_accuracy: -derivative_accuracy]
        
        # fix lengths of arrays so that arrays all have same length
        if scattered_wave:
            chi = chi[derivative_accuracy: -derivative_accuracy]
        # chi_pr = chi_pr[derivative_accuracy: -derivative_accuracy]
        psi = psi[derivative_accuracy: -derivative_accuracy]
        # psi_pr = psi_pr[derivative_accuracy: -derivative_accuracy]
        phi = F(r, p, l) / p
        phi_pr = F_prime(r, p, l) / p
    else:
        r = r_mesh
        
        if scattered_wave:
            chi_pr = u_pr
            psi_pr = chi_pr + phi_pr
        else:
            psi_pr = u_pr
    
    ###   ###   ###   getting r_match index in mesh   ###   ###   ###
    match_index = np.max(np.where(r <= r_match))
    try:
        u[match_index]
    except:
        # It is difficult to end up here. But if your value of r_match is in the last 
        # `derivative_accuracy` number of points then the value will be lost.
        match_index = -1
        print("r_match out of range after derivative. Using index = -1.")
    
    
    ###   ###   ###   the "matching" part of the matching function   ###   ###   ###
    R_l = psi[match_index] / (r_match * psi_pr[match_index])  # inverse log derivative
    
    K_l = - (F(r, p, l)[match_index] - r_match * R_l * F_prime(r, p, l)[match_index]) \
          / (G(r, p, l)[match_index] - r_match * R_l * G_prime(r, p, l)[match_index])  # K-matrix
    if (np.imag(K_l) > 0):
        print("K-matrix has imaginary element")  # quick check
    
    delta_l_from_K_l = np.arctan(K_l)  # calculate the phase shift from the K-matrix element
    
    # check to make sure there's some constancy
    S_l = (H(r, p, l, plus=False, derivative=False)[match_index] - r_match * R_l * H(r, p, l, plus=False, derivative=True)[match_index]) \
        / (H(r, p, l, plus= True, derivative=False)[match_index] - r_match * R_l * H(r, p, l, plus= True, derivative=True)[match_index])  # S-matrix
    S_l_from_K_l = (1 + 1j * K_l) / (1 - 1j * K_l)
    
    diff = np.abs(S_l - S_l_from_K_l)
    if diff > 1e-14:
        print(S_l, S_l_from_K_l)
        raise ValueError("Matched S-matrix and K-matrix are not consistent!")
    
    delta_l_from_S_l = np.real(np.log(S_l) / 2.j)
    if np.abs(delta_l_from_K_l - delta_l_from_S_l) > 1e-14:
        print(delta_l_from_K_l, delta_l_from_S_l)
        raise ValueError(f"delta_l from S-matrix ({delta_l_from_S_l:.14}) and K-matrix ({delta_l_from_K_l:.14}) are not consistent!")
    
    T_l = (1j / 2) * (1 - S_l)  # T-matrix
    delta_l_from_T_l = np.arctan(T_l / (1 + 1j * T_l))
    delta_l_from_T_l = np.real(delta_l_from_T_l)
    
    if not np.isclose(S_l, 1 + 2 * 1j * T_l):
        print(S_l, 1 + 2 * 1j * T_l)
        raise ValueError("S matrix element and T matrix element are not consistent with each other!")
    if not np.isclose(delta_l_from_T_l, delta_l_from_K_l):
        print(delta_l_from_T_l, delta_l_from_K_l)
        raise ValueError("Phaseshift from T_l and K_l are not consistent with each other!")
    if not np.isclose(np.abs(1 + 2 * 1j * T_l), 1):
        print(np.abs(1 + 2 * 1j * T_l), 1)
        raise ValueError("T matrix element not right :(")
    
    # given the previous consistency checks, this won't change much at all
    if use_T_matrix:
        delta_l = delta_l_from_T_l
    else:
        delta_l = delta_l_from_K_l
    
    
    ###   ###   ###   apply K (or T) matrix element   ###   ###   ###
    if scattered_wave:
        if use_T_matrix:
             # this is a complex equation due to the Hankle
            u_an = (F(r, p, l)[match_index] + np.real(T_l * H(r, p, l, plus=True)[match_index]))
            scale = u_an / psi[match_index]
            
            u_scaled = scale * psi - phi
            u_pr_scaled = scale * psi_pr - phi_pr
        else:
            u_an = (F(r, p, l)[match_index] + K_l * G(r, p, l)[match_index]) / p
            scale = u_an / psi[match_index]
            
            u_scaled = scale * psi - phi
            u_pr_scaled = scale * psi_pr - phi_pr
    else:  # for the full wave function
        if use_T_matrix:
            u_an = (F(r, p, l)[match_index] + (T_l * H(r, p, l, plus=True)[match_index]))
            scale = u_an / psi[match_index]
            
            u_scaled = psi * scale
            u_pr_scaled = psi_pr * scale
        else:
            u_an = (F(r, p, l)[match_index] + K_l * G(r, p, l)[match_index]) / p
            scale = u_an / psi[match_index]
            
            u_scaled = psi * scale
            u_pr_scaled = psi_pr * scale
    
    return_dict = {"r": r,
                   "scaled_wave_function": u_scaled,
                   "u_pr_scaled": u_pr_scaled,
                   "delta_l": delta_l,
                   "scale": scale,
                   "K_l": K_l,
                   "T_l": T_l,
                   "S_l": S_l,
                   "R_l": R_l}
    return return_dict


def matching_using_least_squares(r,
                                 wave_function, 
                                 p,
                                 l: int,
                                 matching_indices: int = 2,
                                 zeta: bool = 1,
                                 F_over_all_r=None,
                                 return_pieces: bool = False):
    r"""Least-squares matching procedure. Does not use an inverse logarithmic derivative.
    NOTE: This computes equation 14/30 in the paper using equations 15-20.
    
    This function can be used in place of `match`, which uses an inverse log derivative in order to 
    calculate matrix elements.
    
    Parameters
    ----------
    r : array-like
        The mesh of the problem.
    ode_solution : array-like
        Either the scattered wave, chi (corresponding to zeta=1) or the full wave, phi (corresponding
        to zeta = 0).
        # This solution should have the last two values be `a` and `b`, as described in the paper.
    p : number
        The center-of-mass momentum of the problem. Note, p = hbar k; hbar = 1.
    l : int
        The relative orbital angular momentum quantum number for the problem.
    matching_indices : array-like (optional)
        An array-like object that contains number indices that will be used for the matching process. 
        These indices will be used starting from the last point in the wave function. If no argument 
        is provided then the last 2 points in the mesh will be used.
    zeta : bool (0 or 1)
        The "toggle" between the homogeneous and inhomogeneous radial Schrodinger equation.
    F_over_all_r : array like (optional)
        The Coulomb function $F(r p)$ over all r. When provided, p
    return_pieces : bool (optional)
        When `True`, the components required to calculate the matrix elements will be returned
        after `return_dict`.
    
    Returns
    -------
    return_dict : dict
        A dictionary containing:
        {K_l, S_l, T_l, delta_l, scaled_wave_function}
    """
    # F is calculated over all r, and not just matching r, for its use in calculating the `scaled_wave_function`.
    if F_over_all_r is None:
        F_over_all_r = F(r, p, l)
    
    t = matching_indices  # define the size of the least squares problem
    
    lhs = np.empty((t, 2))  # `t` for t matching indices, `2` for F(r) and G(r).
    lhs[:, 0] = F_over_all_r[-t:]
    lhs[:, 1] = G(r[-t:], p, l)
    lhs /= p
    
    rhs = wave_function[-t:]
    
    a_b, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    a, b = a_b
    a += zeta
    
    K_l = b / a
    delta_l = np.arctan2(b, a)
    S_l = (1 + 1j * K_l) / (1 - 1j * K_l)
    T_l = K_l / (1 - 1j * K_l)
    
    scaled_wave_function = (wave_function + 
                            zeta * ((p - 1) * wave_function - 
                                    (a - 1) * F_over_all_r)
                            ) / a
    if zeta == 1:
        scaled_wave_function /= p
    
    return_dict = {}
    return_dict["a"] = a
    return_dict["b"] = b
    return_dict["K_l"] = K_l
    return_dict["S_l"] = S_l
    return_dict["T_l"] = T_l
    return_dict["delta_l"] = delta_l
    return_dict["scaled_wave_function"] = scaled_wave_function
    if return_pieces:
        return return_dict, lhs, rhs
    else:
        return return_dict
