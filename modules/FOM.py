# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)


###   ###   ###   imports   ###   ###   ###
import numpy as np
from scipy.sparse import spdiags, diags
from scipy.linalg import solve_banded
from functools import cached_property

import sys
sys.path.append("./../modules")
from Constants import *
from SpecialFunctions import analytic_phi, F, G


class MatrixNumerovSolver:
    def __init__(self,
                 potential,
                 use_ab: bool = False,
                 energy=50.,
                 mass=neutron_mass / 2,
                 zeta: bool = 1,
                 y_0=0.,
                 ypr_0=0.):
        r"""Full Order Model (FOM) solver for the Radial Schrodinger Equation using the matrix Numerov Method.
        
        This FOM takes advantage of the affine decomposition of the nuclear potential to accelerate 
        computation. Given a potential that has attributes as given in `Potential.py` in this repository, 
        values will be inherited from the passed potential.
        
        Parameters
        ----------
        potential : Potential object
            The potential for which y (chi or psi) will be found on. This must be in coordinate space. This should be 
            an Potential instance from Potential.py.
        use_ab : bool (optional)
            A toggle to include the K matrix coefficients, $a$ and $b$. The wavefunction calculation
            in either case is numerically identical. This is `False` by default.
        energy : number (optional)
            The center-of-mass energy for the scattering problem. This should be in units of MeV.
        mass : number (optional)
            The reduced mass of the scattered nuclei. This should be in MeV.
        zeta : bool (optional)
            When zeta is 0, the homogeneous RSE will be used. Otherwise, when zeta is 1 the inhomogeneous 
            RSE will be used. For numerical stability, better orthonormalization zeta=1 is the default.
        y_0 : number (optional)
            The initial value of the y (chi or psi). For regular solutions, this should always be `0.`
        ypr_0 : number (optional)
            The initial value of the derivative of y (chi or psi). For the inhomogeneous RSE, `ypr_0=0.` is used.
            When `zeta=0` and `ypr_0=0.` (the default for `ypr_0`), the initial conditions will automatically
            be corrected to `ypr_0=1`, based on the given Potential's mesh.
        """
        ###   ###   ###   variable definitions   ###   ###   ###
         # definitions taken from the potential
        self.potential = potential  # from Potential.py (or another file that uses the same structure)
        self.number_of_parameters = self.potential.number_of_parameters + 1  # because this inserts in `1.` before theta
        self.l = self.potential.l  # relative orbital angular momentum quantum number
        # mesh details
        self.r = self.potential.r
        self.dr = self.potential.dr
        self.n = self.potential.n
        
        
        # definitions that are solver specific
        self.epsilon = 1e-24  # numerical epsilon
        self.energy = energy  # center-of-mass energy
        self.mass = mass      # reduced mass of the system
        self.p2 = (2 * mass * energy) / hbarc ** 2  # fm ^(-2)
        self.p = self.p2 ** 0.5  # p = hbar k but hbar here is 1
        self.y_0 = y_0        # initial value of y (chi or psi)
        self.ypr_0 = ypr_0    # derivative at initial value of y (chi or psi)
        
        # this is how much larger the returned array is than the coordinate space mesh given
        self.use_ab = use_ab
        if self.use_ab:  # + 2 for a and b
            self.n_adjustment = 2
            self.n_slice = -2
        else:  # not adjusted only calculating the wavefunction
            self.n_adjustment = 0
            self.n_slice = None
        
        self.zeta = zeta
        if (self.zeta == 0) and (self.ypr_0 <= self.epsilon):  # set initial condition to 1 (or, 1 * dr)
            self.ypr_0 = self.dr
        
        self.G = G(self.r, self.p, self.l)
        self.F = F(self.r, self.p, self.l)  # using this in place of phi would be more consistent with the paper. It would also involve throwing factors of p (k) around.
        self.phi = analytic_phi(self.r, energy=self.energy, l=self.l)
        
        if self.use_ab:
            self.bandwidth = 4
        else:
            self.bandwidth = 3
        ###   ###   ###   affine pieces   ###   ###   ###
        # potential pieces
        self.parameter_independent_array = np.copy(potential.parameter_independent_array)  # this is a copy just to be safe, but it should never be mutated
        # other pieces
        self.p_sq = (2 * self.mass / hbarc ** 2) * self.energy
        
        self.l_prestore_over_rsq = self.l * (self.l + 1) / (self.r ** 2)
        if self.r[0] <= self.epsilon:
            self.l_prestore_over_rsq[0] = 0  # little crude, but y_0 = 0 for regular solutions
        self.s_prestore = self.phi * (2 * self.mass / hbarc ** 2)
        self.g_naught = self.l_prestore_over_rsq - self.p_sq
        # for A
        self.one_ten_one = (self.dr ** 2 / 12) * np.array([[1.], [10.], [1.]])
        # for b (s)
        self.coeffs_one_ten_one = (self.dr ** 2 / 12) * diags((1., 10., 1.), (-2, -1, 0), 
                                                                 shape=(self.n, self.n))
    ###################################################################################################################
    
    def theta_args(self,
                   theta : dict = None,
                   default_theta: dict = None):
        r"""Transforms the dict theta into a np.array theta with 1. added as zero-th component.
        
        Examples:
        theta_args({"V_r": 200., "V_s": -91.85})  ->  np.array([1., 200., -91.85.])
        
        theta_args({"V_s": -50})  ->  np.array([1., 200., -50.])
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter. 
            Used for testing.

        Returns
        -------
        theta_vec : numpy array
            A numpy array of the parameters, in the same order as the default theta (inherited from 
            the potential). This inserts `1.` as the zero-th component to take advantage the potential's 
            affine structure.
        """
        if theta is None:
            theta = self.potential.default_theta.copy()
        # attempt to set theta to default
        # this is to prevent the order of parameters from getting mixed up
        if default_theta is None:
            default_theta = self.potential.default_theta.copy()
        else:
            default_theta = default_theta
        
        used_theta = {}
        for parameter in list(default_theta.keys()):  # go through all parameters (by name)
            if parameter not in theta:
                used_theta[parameter] = default_theta[parameter]  # assign default value if parameter isn't specified
            else:
                used_theta[parameter] = theta[parameter]  # assign parameter from given value if it is given
        
        theta_vec = np.array(list(used_theta.values()))  # playing with datatypes to "array-ify" this 
        theta_vec = np.append([1.], theta_vec)  # add a constant, 1., as the zero-th component of the parameters
        return theta_vec
    
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    #--   ---    parameter-independent functions    ---   --#
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    @cached_property  # This should not be the case if we want to change the energy !
    def const_s(self):
        r""" Calculates the parameter-independent piece of Numerov $s$. 
        NOTE: This computes equation 24d.
        NOTE: This is _not_ the $s$ used as the right-hand side of the RSE.
        
        Returns
        -------
        s : numpy array
            The parameter-independent piece of Numerov $s$.
        """
        s = np.zeros((self.n, self.number_of_parameters))
        s[0, 0] = self.y_0
        s[1, 0] = self.ypr_0
        
        # this is only non-zero for the inhomogeneous RSE
        if self.zeta == 1:
            for q in np.arange(1, self.number_of_parameters):
                s[:, q] = self.s_prestore * self.parameter_independent_array[:, q - 1]
        
        return s
    
    @cached_property  # This should not be the case if we want to change the energy !
    def const_g(self):
        r"""Calculates the parameter-independent component of Numerov $g$.
        NOTE: This computes equation 24b.
        
        Returns
        -------
        g : numpy array
            The constant component of numerov $g$.
        """
        minus_g = np.empty((self.n, self.number_of_parameters))
        
        for q in np.arange(1, self.number_of_parameters):
            minus_g[:, q] = (2 * self.mass / hbarc ** 2) * self.potential.parameter_independent_array[:, q - 1]
        minus_g[:, 0] = self.g_naught
        
        g = -minus_g
        return g
    
    
    @cached_property  # This should not be the case if we want to change the energy !
    def const_A(self):
        r"""Calculates the parameter-independent piece of the matrix, "A bar", used for solve_banded().
        NOTE: This is equivalent to equations 26b.
        
        Returns
        -------
        A_bar : numpy array
            The rectangular matrix that has the lower-, main-, and upper-diagonals of the matrix A. 
            This is in a form for use with `scipy.linalg.solve_banded()`.
        """
        # equation 25b
        A_bar = np.zeros((self.bandwidth, 
                          self.n + self.n_adjustment, 
                          self.number_of_parameters))
        
        if self.use_ab:
            for q in np.arange(self.number_of_parameters):
                A_bar[1:, :self.n_slice, q] = self.one_ten_one * self.const_g[:, q]
            # apply equation 26b
            A_bar[1, :self.n_slice, 0] += 1
            A_bar[2, :self.n_slice, 0] += -2
            A_bar[3, :self.n_slice, 0] += 1
            
            # edit last two rows and columns to calculate $a$ and $b$.
            #     B   i  a  =
            A_bar[0, -1, 0] = -self.G[-2]
            A_bar[1, -1, 0] = -self.G[-1]
            A_bar[1, -2, 0] = -self.F[-2]
            A_bar[2, -2, 0] = -self.F[-1]
            A_bar[2, -3, 0] = 0.
            A_bar[3, -3, 0] = 1.
            A_bar[3, -4, 0] = 1.
        
        else:
            for q in np.arange(self.number_of_parameters):
                A_bar[:, :self.n_slice, q] = self.one_ten_one * self.const_g[:, q]
            # apply equation 26b
            A_bar[0, :self.n_slice, 0] += 1
            A_bar[1, :self.n_slice, 0] += -2
            A_bar[2, :self.n_slice, 0] += 1
        
        return A_bar
    
    @cached_property  # This should not be the case if we want to change the energy !
    def const_A_matrix(self):
        r"""Calculates the parameter-independent component of the matrix $A$ .
        NOTE: This is equivalent to equations 25b
        
        Returns
        -------
        A_matrix : numpy array
            The square matrix representation of `const_A`.
        """
        A_matrix = np.empty((self.n + self.n_adjustment, 
                             self.n + self.n_adjustment, 
                             self.number_of_parameters))
        
        for q in np.arange(self.number_of_parameters):
            if self.use_ab:
                A_matrix[:, :, q] = spdiags(self.const_A[:, :, q], (1, 0, -1, -2)).toarray()
            else:
                A_matrix[:, :, q] = spdiags(self.const_A[:, :, q], (0, -1, -2)).toarray()
        return A_matrix
    
    @cached_property  # This should not be the case if we want to change the energy !
    def const_b(self):
        r"""Calculates the parameter-independent component of the right-hand side of the ODE
        NOTE: This computes equation 27b. Here, "b" is used in place of "bold s" to avoid confusion with 
              the $s$ in Numerov's method.
        
        Returns
        -------
        b : numpy array
            The parameter-independent calculation for the right-hand side of the ODE.
        """
        # define everything except for the initial conditions
        b = np.empty((self.n + self.n_adjustment, self.number_of_parameters))
        b[:self.n_slice, :] = self.coeffs_one_ten_one @ self.const_s
        
        # use initial conditions
        b[0, :] = self.y_0
        b[0, 1:] = 0.
        b[1, 0] = self.ypr_0
        b[1, 1:] = 0.
        
        # set last values
        if self.use_ab:
            b[-self.n_adjustment:, :] = 0.
        
        # define some initial conditions
        b[2, 0] += 2 * (1 - 5 * (self.dr ** 2 / 12) * self.const_g[1, 0]) * self.ypr_0
        if self.l == 1:
            b[2, 0] += 1 / 6 * self.ypr_0
        b[3, 0] += -(1 + 1 * (self.dr ** 2 / 12) * self.const_g[1, 0]) * self.ypr_0
        
        return b
    
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    #--   ---     parameter-dependent  functions    ---   --#
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    def affine_s(self,
                 theta=None,
                 theta_vec=None,
                 default_theta=None,
                 i_start=None,
                 i_end=None):
        r"""Calculates the parameter-dependent $s$ using the affine decomposition of the potential.
        NOTE: This computes equation 24c.
        
        This function is defined but never used. Instead `affine_b` is used in its place to handle
        the initial conditions of y.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        i_start : int (optional)
            The desired first index for $s$.
        i_end : int (optional)
            The desired last index for $s$.
        
        Returns
        -------
        s : numpy array
            The one-dimensional vector, $s$.
        """
        s = np.zeros(self.n)
        if theta_vec is None:
            theta_vec = self.theta_args(theta=theta, default_theta=default_theta)
        
        s[:-2] = self.const_s[i_start:i_end, :-self.n_slice] @ theta_vec
        return s
    
    def affine_g(self,
                 theta: dict = None,
                 theta_vec=None,
                 default_theta: dict = None,
                 i_start: int = None,
                 i_end: int = None):
        r"""Calculates the parameter-dependent $g$ using the affine decomposition of the potential.
        NOTE: This computes equation 24a.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        i_start : int (optional)
            The desired first index for $g$
        i_end : int (optional)
            The desired last index for $g$
        
        Returns
        -------
        g : numpy array
            The one-dimensional vector, $g$.
        """
        if theta_vec is None:
            theta_vec = self.theta_args(theta=theta, default_theta=default_theta)
        
        g = (self.const_g[i_start:i_end, :] @ theta_vec)
        return g
    
    def affine_mathcal_G(self,
                         coeff,
                         theta: dict = None,
                         theta_vec=None,
                         default_theta: dict = None,
                         i_start: int = None,
                         i_end: int = None):
        r"""Calculates the parameter-dependent "mathcal-G".
        NOTE: This computes equation 12.

        Parameters
        ----------
        coeff : number
            The coefficient, Xi, for the calculation. Usually either 1 or -5.
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        i_start : int (optional)
            The desired first index for $g$
        i_end : int (optional)
            The desired last index for $g$

        Returns
        -------
        mathcal_G : numpy array
            The one-dimensional array, "mathcal G", or $\mathcal{G}$.
        """
        parameter_independent_g = self.affine_g(theta=theta, 
                                                theta_vec=theta_vec, 
                                                default_theta=default_theta, 
                                                i_start=i_start, i_end=i_end)
        
        mathcal_G = 1 + coeff * (self.dr ** 2 / 12) * parameter_independent_g
        return mathcal_G
    
    def affine_A(self,
                 theta: dict = None,
                 theta_vec=None,
                 ret_matrix: bool = False,
                 default_theta: dict = None,
                 i_start: int = None,
                 i_end: int = None):
        r"""Calculates the matrix, $A(theta)$ using the affine structure of the potential.
        NOTE: This computes equation 25a when `ret_matrix=False`
              and  is equation 26a when `ret_matrix=True`.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        ret_matrix : bool (optional)
            Boolean flag to determine if the `A_bar` or `A_matrix` should be returned from the function.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        i_start : int (optional)
            The desired first index for $A$
        i_end : int (optional)
            The desired last index for $A$
        
        Returns
        -------
        A_bar or A_matrix : numpy matrix
            When `ret_matrix = False`, `A_bar` is returned. This is a rectangular matrix used for 
            scipy.linalg.solve_banded(). When `ret_matrix = True`, `A_matrix` is returned. This is a 
            square matrix used for matrix calculations (such as projections).
        """
        if theta_vec is None:
            theta_vec = self.theta_args(theta=theta, default_theta=default_theta)
        
        A_bar = (self.const_A[i_start:i_end, :] @ theta_vec)
        
        if ret_matrix:
            if self.use_ab:
                A_matrix = spdiags(A_bar, (1, 0, -1, -2)).toarray()
            else:
                A_matrix = spdiags(A_bar, (0, -1, -2)).toarray()
            return A_matrix
        return A_bar
    
    def affine_b(self,
                 theta: dict = None,
                 theta_vec=None,
                 default_theta: dict = None,
                 i_start: int = None,
                 i_end: int = None):
        r"""Calculates the parameter-dependent right-hand side of the ODE using the affine structure of the potential.
        NOTE: This computes equation 27b.
        NOTE: $b$ is used here instead of $S$ to avoid confusion with the $s$ in Numerov's method.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        i_start : int (optional)
            The desired first index for right-hand side of the ODE.
        i_end : int (optional)
            The desired last index for right-hand side of the ODE.

        Returns
        -------
        b : numpy array
            The one-dimensional right-hand side of the ODE.
        """
        if theta_vec is None:
            theta_vec = self.theta_args(theta=theta, default_theta=default_theta)
        b = np.empty(self.n + self.n_adjustment)
        
        # use initial conditions
        b[0] = self.y_0
        b[1] = self.ypr_0
        b[i_start:i_end] = (self.const_b[i_start:i_end, :] @ theta_vec)  # calculate the rest
        return b
    
    
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    #--   ---   ---   the actual  "solver" part of the solver   ---   ---   --#
    ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
    def solve(self,
              theta: dict = None,
              theta_vec=None,
              default_theta: dict = None,
              ret_bar: bool = False,
              ret_matrix: bool = False):
        r"""Solves the (in)homogeneous RSE at a given parameter, `theta`.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. When no theta argument is given, the 
            potential's best-fit parameter will be used.
        theta_vec : numpy array
            Numpy array of the parameters in theta, as in the output of `self.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        ret_bar : bool (optional)
            Boolean flag that when `True` will make the function return the left-hand side of the 
            ODE as a rectangular matrix for scipy.linalg.solve_banded(). 
            For testing.
        ret_matrix : bool (optional)
            Boolean flag that when `True` will make the function return the left-hand
            side of the ODE as a square matrix. 
            For testing.
        
        Returns
        -------
        y (chi or psi) : numpy array
            The scattered wave solution to the inhomogeneous Schrodinger equation when self.zeta=1,
            and the full wave solution to the homogeneous Schrodinger equation when self.zeta=0.
        """
        A_band = self.affine_A(theta=theta, theta_vec=theta_vec, default_theta=default_theta)[:, 2:]  # trim initial conditions
        b_band = self.affine_b(theta=theta, theta_vec=theta_vec, default_theta=default_theta)[2:]  # trim initial conditions
        if ret_matrix:
            if self.use_ab:
                matrix = spdiags(A_band, (1, 0, -1, -2)).toarray()
            else:
                matrix = spdiags(A_band, (0, -1, -2)).toarray()
            return matrix, b_band
        elif ret_bar:
            return A_band, b_band
        
        y = np.empty(self.n + self.n_adjustment)
        y[0] = self.y_0
        y[1] = self.ypr_0
        if self.use_ab:
            y[2:] = solve_banded((2, 1), A_band, b_band)
        else:
            y[2:] = solve_banded((2, 0), A_band, b_band)
        
        return y
#