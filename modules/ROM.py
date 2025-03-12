# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)


###   ###   ###   imports   ###   ###   ###
import numpy as np
import scipy
import time
from tqdm import tqdm
from functools import cached_property
from modules.gram_schmidt import MGS, unoptimized_MGS
from modules.Matching import matching_using_least_squares, propagate_delta_error
from modules.Constants import *


def apply_range_factor(lecs: dict, 
                       range_factor: float = 0.9, 
                       kept_lecs=None, 
                       potential=None):
    r"""Conveniently builds the range for emulating across, using the ROM class.
    
    Builds parameter_bounds based on a given range_value (with default of 0.9) for a given dict of LECs 
    and their best-fit, or "default" values. Intended for easy of use to construct a space for the 
    emulator class.
    
    Parameters
    ----------
    lecs : dict
        The low-energy constants that will be used in emulator. The default values for which the lecs 
        will be given a range from. These should be their best-fit values. The names should be the keys, 
        and the defaults should be the values.
    range_factor : float
        The value by which the lecs will be given a range from. This is in the form of 
        (1 - range_factor) * best-fit_value and  (1 + range_factor) * best-fit_value.
        
        "Examples":
        range_factor = 1:
            values varied between 0 and 2 * value
        range_factor = 0.5
            values varied between
            0.5 * value and 1.5 * value
    potential : Potential class object (optional)
        The potential that the emulator will be constructed with. When this argument is used, the lecs 
        that do not contribute to the given channel (or that should  not be emulated  over, such as 
        the constant component of the potential) will not be used.
        NOTE: It is recommended that this argument is used.
    
    Returns
    -------
    parameter_bounds: dict of length-two lists
        A dictionary where each entry contains a list of exactly two values, the min and max values 
        by which the emulator will vary parameters during training.
    """
    if kept_lecs is None:
        kept_lecs = list(lecs.keys())
    
    default_values = list(lecs.values())
    parameter_bounds = {}
    for i, parameter in enumerate(lecs.keys()):
        if parameter in kept_lecs:
            vary_value = range_factor * default_values[i]
            parameter_bounds[parameter] = [default_values[i] - vary_value,
                                       default_values[i] + vary_value]
    
    if potential is not None:
        # grab the LECs that contrubute
        noncontributing_lecs = []
        for i, array in enumerate(potential.parameter_independent_array.T):
            lec = potential.parameter_names[i]
            if np.max(np.abs(array)) < 1e-24:  # crude method of seeing if the array is flat zero or not
                noncontributing_lecs.append(lec)
        
        for lec in noncontributing_lecs:
            parameter_bounds.pop(lec)
        
        # remove constant piece if it hasn't already been
        if potential.const_piece and (potential.parameter_names[0] in parameter_bounds):
            parameter_bounds.pop(potential.parameter_names[0])
    
    return parameter_bounds


class Emulator:
    def __init__(self,
                 parameter_bounds: dict,
                 solver_class,
                 param_pts: int = 100,
                 snapshot_max: int = 10,
                 snapshot_parameters=None,
                 cutoff_accuracy: float = None,
                 ignore_error_bounds: bool = False,
                 orthonormal_basis: bool = True,
                 greedy: bool = True,
                 sampling_method: str = "LHS",
                 seed: int = 12345,
                 emulation_method: str = "LSPG-ROM",
                 error_estimation_method: str = "LSPG-ROM",
                 orth_cutoff: float = 1e-8,
                 use_scaled_estimated_error: bool = True,
                 custom_parameter_space: dict = None,
                 use_einsum_paths: bool = False,
                 use_practically: bool = False,
                 matching_radius=11.,
                 verbose: bool = False):
        r"""Implementation of the Galerkin-ROM (G-ROM) and Least-Squares Petrov-Galerkin-ROM (LSPG-ROM).
        This is intended for use with the all-at-once Numerov solver.
        
        This class can be used for a multitude of functionalities:
            - G-ROM emulation
            - LSPG-ROM emulation
            - Constructing a basis using a greedy algorithm ***
            - Emulating with a given basis
        And it can be used in a variety of cases:
            - Parameter spaces in an arbitrary number of parameters
            - Different, built-in parameter spaces
                - Arbitrary dimensions
                    - Latin HyperCube (LHS) sampling
                    - Random Gaussian sampling
                - One-dimension:
                    - Linear sampling
                    - Error function sampling, erf()
                    - All arbitrary-dimension sampling methods
            - Custom parameter spaces
        And it has some potentially handy built-in features:
            - Toggles:
                - Using an orthonormal basis
                - Using error bounds (via calculations singular values) (saves time when testing)
                - Using the scaled estimated error (will prevent the "one extra FOM calculation")
                - Using with or without an excessive number of FOM calculations (via `use_practically`) (saves time when testing)
        
        Examples of constructing the class:
            r = np.linspace(0, 12, 1000)
            potential = Potential.Potential("minnesota", r)  # required for solver
            solver = FOM.MatrixNumerov_ab(potential)  # required for emulator
            
            parameter_bounds = ROM.apply_range_factor(potential.default_theta, 0.5, potential=potential)  # parameter space for training emulator
            non_greedy_emulator = ROM.Emulator(parameter_bounds, solver, greedy=False)
            
            non_greedy_emulator.train()
        
        Parameters
        ----------
        parameter_bounds : dictionary of length-2 lists
            Currently a dictionary that must contain an affine parameter of the Minnesota potential, 
            either "V_r", or "V_s" argument of this dictionary should be the minimum and maximum values
            that those parameters will be varied between. This can be conveniently constructed 
            using apply_range_factor().
            Examples for Minnesota potential:
            `parameter_bounds={"V_s": [-400., 0.]}  # only varies V_s, V_r held constant
            `parameter_bounds={"V_r": [0., 200.], "V_s": [-200., 0.]}  # both V_r and V_s varied
        solver_class : FOM object
            The solver class used from FOM.py. This should be the `MatrixNumerovSolver`, and the solver
            can have any augmentation of its arguments (namely, `use_ab` and `zeta`).
        param_pts : int (optional)
            The number of points to be used in the parameter space.
            NOTE: This may be increased if the given snapshot parameters do not already exist in the
                  parameter space
        snapshots_max : int (optional)
            The maximum number of snapshots allowed in the emulator
        snapshot_parameters : array-like object of "parameter" dicts (optional)
            Snapshot parameters for the emulator. This will be used to start the greedy emulator.
            These should be given as an dictionary of arrays.
            Examples:
                snapshot_parameters=[{"V_s": -300}]
                snapshot_parameters=[{"V_s": -300, "V_r": 0},
                                     {"V_s": 100, "V_r: 400}]
        cutoff_accuracy : number (optional)
            When not None, the greedy training will stop once the MAXIMUM of the (scaled) estimated 
            error is at or below this value. The scaled estimated error is used when 
            `use_scaled_estimated_error=True`.
        ignore_error_bounds : bool (optional)
            When `True`, the error bounds will not be calculated. This eliminates the need for the 
            calculation of singular values. Can be set to `True` when testing.
        orthonormal_basis : bool (optional)
            When `True`, the basis will be orthonormalized. This should pretty much always be set to `True`.
        greedy : bool (optional)
            Whether the emulator will use the greedy algorithm to construct snapshot_parameters or not.
        sampling_method : string (optional)
            This will dictate the method that the parameter space will be sampled. The two good current 
            options  for this are "linear", where the parameter space will be uniformly discretized, 
            and "LHS" (or Latin Hypercube Sampling), where the space will be randomly sampled in a space 
            filling way using a latin hypercube sampler.
        seed : int (optional)
            The seed that will be used for random sampling (particularly the LHS sampling).
        emulation_method : str (optional)
            The method used for the emulation of the wave function. The two current options are "G-ROM" 
            and "LSPG-ROM".
        error_estimation_method : string (optional)
            The method used for the emulation of the error estimate. The two current options are "G-ROM" 
            and "LSPG-ROM".
        orth_cutoff : number
            The number for which atol & rtol will be set to for orthonormalization of snapshots.
        use_scaled_estimated_error : bool (optional)
            When `True`, the scaled estimated error will be calculated using one extra FOM calculation.
        custom_parameter_space : dict (optional)
            When provided (and when sampling_method="custom") the provided dict will be used as the 
            parameter space.
        use_einsum_paths : bool (optional)
            When `True` (and when one manually sets `emulator_class.use_einsum_calculations=True`), 
            the paths for the einsum calculations will be used (rather than calculated on the fly).
        use_practically : bool (optional)
            When `True`, the greedy emulator will omit calculations that don't want to be done in practice. 
            Such calculations are calculating exact errors or exact values.
        matching_radius : number (optional)
            The radius used to match the wave functions. This should be given in units of fm.
        verbose : bool (optional)
            When `True`, many print statements and plots will be shown.
        """
        # let's inherit some stuff
        self.solver = solver_class
        self.number_of_parameters = self.solver.number_of_parameters  # for convenience
        self.energy = self.solver.energy
        self.mass = self.solver.mass
        self.p = self.solver.p
        
        self.zeta = self.solver.zeta
        
        self.potential = self.solver.potential
        self.potential_name = self.potential.name
        self.l = self.potential.l
        
        ###   ###   ###   coordinate-space mesh   ###   ###   ###
        self.r = self.potential.r
        self.dr = self.potential.dr
        self.n = self.potential.n
        self.r_match = matching_radius
        
        ###   ###   ###   ###   ###   ###   ###   ###   ###   
        #--   ---   ---   Emulator  stuff   ---   ---   --#
        ###   ###   ###   ###   ###   ###   ###   ###   ###
        self.parameter_bounds = parameter_bounds
        self.seed = seed
        
        self.default_theta = self.potential.default_theta.copy()
        
        ###   ###   ###   Code Handling   ###   ###   ###
        self.use_practically = use_practically
        self.emulation_method = emulation_method
        self.error_estimation_method = error_estimation_method
        self.snapshot_parameters = snapshot_parameters
        self.greedy = greedy
        self.snapshot_max = snapshot_max
        self.sampling_method = sampling_method
        self.custom_parameter_space = custom_parameter_space
        self.param_pts = param_pts
        self.orthonormal_basis = orthonormal_basis
        self.orth_tolerance = 1e-8  # can be changed after initialization if desired.
        self.use_einsum_calculations = False  # can be changed after initialization if desired.
        self.cutoff_accuracy = cutoff_accuracy
        self.use_scaled_estimated_error = use_scaled_estimated_error
        self.ignore_error_bounds = ignore_error_bounds
        self.orth_cutoff = orth_cutoff
        self.use_einsum_paths = use_einsum_paths
        self.verbose = verbose
        
        self.matching_indices = 2  # the number of points used at the end of the coordinate-space mesh use for matching
        
        # for propagation of errors to a max upper bound on phaseshifts
        self.M = np.empty((self.matching_indices, 2))
        self.M[:, 0] = self.solver.F[-self.matching_indices:]
        self.M[:, 1] = self.solver.G[-self.matching_indices:]
        self.M_pinv = np.linalg.pinv(self.M)
        
        if (self.custom_parameter_space is not None) and (self.sampling_method != "custom"):
            print("Warning: Custom parameter space is provided but not used.")
            print("         To use custom parameter space use sampling_method=\"custom\"")
        
        self.trained = False
        self.training_time = None
        self.n_adj = self.solver.n_adjustment  # redefine (and shorten) for convenience
        self.snapshots = np.zeros((self.n + self.n_adj, self.snapshot_max))
        self.dagger_snapshots = self.snapshots.T.conjugate()  # for convenience
        self.truncated_snapshots = self.snapshots[2:, :]  # for convenience
        self.truncated_dagger_snapshots = self.dagger_snapshots[:, 2:]  # for convenience
        
        default_theta = self.potential.default_theta.copy()
        if self.snapshot_parameters is not None:
            self.preset_snapshots = len(self.snapshot_parameters)
            if self.greedy:
                assert len(self.snapshot_parameters) <= self.snapshot_max, "More snapshot parameters given than snapshot max"
            # if initial parameters are given then we want to calculate the snapshot basis immediately
            for i, snapshot in enumerate(self.snapshot_parameters):
                self.snapshots[:, i] = self.FOM(snapshot, default_theta=default_theta)
            # we also want to orthonormalize the basis if that flag is on
            if self.orthonormal_basis:
                self.snapshots[:, :len(self.snapshot_parameters)] = unoptimized_MGS(self.snapshots[:, :len(self.snapshot_parameters)])
            
            # add the provided snapshot parameters to the object's parameter space
            for snapshot_parameter in self.snapshot_parameters:
                snapshot_theta = self.solver.theta_args(snapshot_parameter)
                
                counter = 0
                for i in np.arange(self.param_pts):
                    parameter_in_space = self.parameters_at_index(i)
                    theta_in_space = self.solver.theta_args(parameter_in_space)
                    if all(theta_in_space == snapshot_theta):
                        counter += 1
                if counter == 0:
                    if self.verbose:
                        print(f"Snapshot parameter {snapshot_parameter} not in space. Adding...")
                    for parameter in snapshot_parameter.keys():
                        self.parameter_space[parameter] = np.append(self.parameter_space[parameter], 
                                                                    snapshot_parameter[parameter])
                        if self.sampling_method == "linear":
                            self.parameter_space[parameter] = np.sort(self.parameter_space[parameter])
            self.param_pts = len(list(self.parameter_space.values())[0])
            
            temp_params = np.empty(self.snapshot_max, dict)
            temp_params[:len(self.snapshot_parameters)] = self.snapshot_parameters
            self.snapshot_parameters = temp_params
        else:
            # if no initial parameters are given, then we can just set this to zero
            self.preset_snapshots = 0
        
        ###   ###   ###   to be (possibly) calculated   ###   ###   ###
        self.min_singular_values = None
        self.max_singular_values = None
        self.exact_elements = None  # K_l, delta_l, T_l, S_l
        self.emulated_elements = None
        self.propagated_max_phaseshift_errors = None
        self.exact_errors = None
        self.estimated_errors = None
        self.scaled_estimated_errors = None
        self.upper_error_bounds = None
        self.lower_error_bounds = None
        
        # projections for error estimator (NOTE: the solver calculates things from i=0, so we trim i=0,1 since those are from initial conditions)
        self.b_const = self.solver.const_b[2:, :].T  # a i
        self.b_const_dagger = self.b_const.conjugate()  # a i  (we "transpose" along the mesh, which is just the mesh here)
        
        self.A_const = np.transpose(self.solver.const_A_matrix[2:, 2:], (2, 0, 1))  # a i j
        self.A_const_dagger = np.transpose(self.A_const, (0, 2, 1)).conjugate()  # a j i
        
        # since these don't need to be super fast, they are just left as einsum calls
        self.A_dagger_on_A = np.einsum("aik,bkj->abij",
                                       self.A_const_dagger,
                                       self.A_const, optimize="greedy")
        self.b_dagger_on_b = np.einsum("ai,bi->ab",
                                       self.b_const_dagger,
                                       self.b_const, optimize="greedy")
        self.A_dagger_on_b = np.einsum("aji,bi->abj",
                                       self.A_const_dagger,
                                       self.b_const, optimize="greedy")
        self.b_dagger_on_A = np.einsum("aj,bji->abi",
                                       self.b_const_dagger,
                                       self.A_const, optimize="greedy")
        
        self.offline_projections(self.preset_snapshots)
    #
    
    ####################################################################################################################
    def train(self):
        r"""Trains the emulator with a convenient single function call. For convenience.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        if self.use_practically:
            self.train_with_greedy(use_practically=True)
        else:
            if not self.ignore_error_bounds:
                self.calculate_singular_values()
            self.calculate_exact_values()
            self.train_with_greedy(use_practically=False)
    
    def FOM(self, 
            theta: dict = None,
            theta_vec=None,
            default_theta: dict = None):
        r"""Wrapper for the FOM solver. For convenience.
        
        Parameters
        ----------
        theta : dict
            The parameter "vector", theta.  When no theta argument is given, the potential's best-fit 
            parameter will be used.
            Example:
            Emulator.solve(theta={"V_r": -50., "V_s":-200.})
        theta_vec : numpy array (optional)
            Numpy array of the parameters in theta, as in the output of `self.solver.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        
        Returns
        -------
        solver_result : numpy array
            A high-fidelity solution to the radial schrodinger equation based on the given potential 
            and solver.
        """
        if default_theta is None:
            default_theta = self.potential.default_theta.copy()
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta=theta, 
                                               default_theta=default_theta)
        
        solver_result = self.solver.solve(theta=theta, default_theta=default_theta, theta_vec=theta_vec)
        return solver_result
    
    @cached_property
    def parameter_space(self, 
                        seed: int = None):
        r"""Generates the parameter space for emulator training.
        
        Samples the space of the given parameters over the range defined by self.parameter_range.
        This is a @cached_property because it does not need to be rebuilt constantly.
        NOTE: Unless there is good reason otherwise, one should use `self.method=LHS`.
              (And a good reason is 1 dimensional parameter spaces.)
        
        Parameters
        ----------
        seed : int
            The seed for random sampling. Particularly using using a Latin HyperCube sampling method.
        
        Returns
        -------
        sampled_space : dictionary
            A dictionary where each key's value is a numpy array of the given space sampled via the 
            method defined by self.sampling_method.
        """
        if seed is None:
            seed = self.seed
        
        sampled_space = {}
        for parameter in self.parameter_bounds:
            assert len(self.parameter_bounds[parameter]) == 2, \
                f"Parameter range min and max only must be given. Must be fixed for {parameter}"
            if self.sampling_method != "linear-multid":
                sampled_space[parameter] = np.empty(self.param_pts)  # make a new dict element for the varied parameter
            else:
                sampled_space[parameter] = []
        
        if self.sampling_method == "linear":  # NOTE: This only works for 1D
            for i, param_i in enumerate(self.parameter_bounds):
                sampled_space[param_i] = np.linspace(*self.parameter_bounds[param_i], self.param_pts)
        elif self.sampling_method == "linear-multid":  # NOTE: this will only sample _on the axes_ of each dimension !!!
            for i, param_i in enumerate(self.parameter_bounds):
                for j, param_j in enumerate(self.parameter_bounds):
                    new_parameter_list = list(np.linspace(*self.parameter_bounds[param_j], self.param_pts))
                    sampled_space[param_i] += new_parameter_list
                sampled_space[param_i] = np.array(sampled_space[param_i])
            self.param_pts *= len(self.parameter_bounds)
        elif self.sampling_method == "LHS":
            from scipy.stats.qmc import LatinHypercube
            
            sampler = LatinHypercube(d=len(self.parameter_bounds), seed=seed)
            sample = sampler.random(n=self.param_pts)
            
            for i, param in enumerate(self.parameter_bounds):
                min_param, max_param = self.parameter_bounds[param]
                sampled_space[param] = (max_param - min_param) * sample[:, i] + min_param
            # since the LHS sampling is a multidimensional sampler, we can just sample once, then we
            # can exit the function without continuing the for loop.
            return sampled_space
        elif self.sampling_method == "gaussian":
            for i, param in enumerate(self.parameter_bounds):
                given_value = np.mean(self.parameter_bounds[param])
                sampled_space[param] = np.random.normal(loc=given_value,
                                                        scale=np.abs(self.parameter_bounds[param][1] -
                                                                     self.parameter_bounds[param][0]),
                                                        size=self.param_pts)
        elif self.sampling_method == "error-function":  # NOTE: This only works for 1D 
            error_function = scipy.special.erfinv(np.linspace(-1, 1, self.param_pts + 2)[1:-1])  # lose the infinities
            error_function /= np.max(error_function)  # "map" range to (-1, 1), it's antisymmetric so that's handy
            error_function = 0.5 * error_function + 0.5  # "map" range to (0, 1)
            for i, param in enumerate(self.parameter_bounds):
                x_min, x_max = self.parameter_bounds[param]
                sampled_space[param] = (x_max - x_min) * error_function + x_min  # "map" yet again to our problem
        elif self.sampling_method == "custom":
            sampled_space = self.custom_parameter_space
        else:
            raise ValueError("Unknown sampling method")
        return sampled_space
    
    def parameters_at_index(self, 
                            i: int):
        r"""Obtains the i-th parameter in the parameter space as a dictionary.
        
        Parameters
        ----------
        i : int
            The index of the parameter space that will be returned.

        Returns
        -------
        param_dict : dictionary
            A dictionary that only has the i-th element of each parameter.
        """
        param_dict = {}
        for parameter in self.parameter_space:
            param_dict[parameter] = self.parameter_space[parameter][i]
        return param_dict
    
    def offline_projections(self, 
                            n_basis: int):
        r"""Void function to project snapshots onto the ODE.
        
        This computes all offline projections for the offline-online decomposition of the emulator 
        and error estimation.
        
        Parameters
        ----------
        n_basis : int
            The size of the snapshot basis at time of projections.
        
        Returns
        -------
        None
        """
        # just to make sure:
        self.truncated_snapshots = self.snapshots[2:, :]
        self.dagger_snapshots = self.snapshots.T.conjugate()
        self.truncated_dagger_snapshots = self.truncated_snapshots.T.conjugate()
        
        # projections for online stage of emulator:
        self.offline_projection_A = np.einsum("ui,aij,jv->auv",
                                              self.truncated_dagger_snapshots[:n_basis, :],
                                              self.A_const,
                                              self.truncated_snapshots[:, :n_basis],
                                              optimize="greedy")
        self.offline_projection_b = np.einsum("ui,ai->au", 
                                              self.truncated_dagger_snapshots[:n_basis, :],
                                              self.b_const,
                                              optimize="greedy")
        
        # projections for error estimation
        if (self.use_practically
            and 
            (self.emulation_method == "G-ROM" or self.error_estimation_method == "G-ROM")) \
            or \
            (not self.use_practically):
            self.offline_projection_X_A_A_X = np.einsum("ui,aik,bkj,jv->abuv",
                                                        self.truncated_dagger_snapshots[:n_basis, :],
                                                        self.A_const_dagger,
                                                        self.A_const,
                                                        self.truncated_snapshots[:, :n_basis],
                                                        optimize="greedy")
            self.offline_projection_X_A_b = np.einsum("ui,aij,bj->abu",
                                                      self.truncated_dagger_snapshots[:n_basis, :],
                                                      self.A_const_dagger,
                                                      self.b_const,
                                                      optimize="greedy")
            #
        # LSPG-ROM stuff:
        if (self.use_practically
            and 
            (self.emulation_method == "LSPG-ROM" or self.error_estimation_method == "LSPG-ROM")) \
            or \
            (not self.use_practically):
            self.offline_projection_A_X = np.einsum("aij,ju->aui",
                                                    self.A_const,
                                                    self.truncated_snapshots[:, :n_basis],
                                                    optimize="greedy")
            self.offline_projection_b_X = np.einsum("aj,ju->au",
                                                    self.b_const,
                                                    self.truncated_snapshots[:, :n_basis],
                                                    optimize="greedy")
            self.Y_nonorth = np.zeros((self.n + self.n_adj - 2, self.number_of_parameters * (n_basis + 1)))
            for a in np.arange(self.number_of_parameters):
                self.Y_nonorth[:, (a * n_basis):((a + 1) * n_basis)] = self.offline_projection_A_X[a, :n_basis, :].T
            self.Y_nonorth[:, (self.number_of_parameters * n_basis):] = np.copy(self.b_const).T
            
            self.Y = unoptimized_MGS(self.Y_nonorth)
            self.Y_dagger = self.Y.conjugate().T
            self.Y_truncated = scipy.linalg.orth(self.Y_nonorth, rcond=self.orth_cutoff)  # orth used here because we need to truncate Y
            self.Y_truncated_dagger = self.Y_truncated.conjugate().T
            if self.verbose:
                compression = (1 - (self.Y_truncated_dagger.shape[0] / self.Y_dagger.shape[0]))
                print(f"truncated Y compression rate of {100 * compression:.4}%")
            
            self.Y_dagger_A_X = np.einsum("wi,aij,ju->awu",
                                          self.Y_truncated_dagger,
                                          self.A_const,
                                          self.truncated_snapshots[:, :n_basis], optimize="greedy")
            self.Y_dagger_b = np.einsum("wi,ai->aw",
                                        self.Y_truncated_dagger,
                                        self.b_const, optimize="greedy")
            #
        ###   ###   ###   time for the einsum paths   ###   ###   ###
        if self.use_einsum_paths:
            # define some dummy variables
            theta_vec = np.ones(self.number_of_parameters)
            coefficients = np.ones(n_basis)
            Coefficients = np.tensordot(coefficients, coefficients, axes=0)
            Theta = np.tensordot(theta_vec, theta_vec, axes=0)
            
            # now get the einsum paths constructed
            # GROM emulator: (doesn't use einsum anyway)
            if (self.use_practically and (self.error_estimation_method == "G-ROM")) \
                    and (not self.use_practically):
                # GROM error estimator:
                self.GROM_term_1 = np.einsum_path("abuv,ab,uv->",
                                                  self.offline_projection_X_A_A_X,
                                                  Theta,
                                                  Coefficients, 
                                                  optimize="greedy")
                self.GROM_term_2 = np.einsum_path("abu,ab,u->",
                                                  self.offline_projection_X_A_b,
                                                  Theta,
                                                  coefficients, optimize="greedy")
                
                self.GROM_term_3 = np.einsum_path("ab,ab->",
                                                  self.b_dagger_on_b,
                                                  Theta, optimize="greedy")
            # PG emulator:
            if (self.use_practically or 
                ((self.error_estimation_method == "LSPG-ROM") 
                    or self.emulation_method == "LSPG-ROM")):
                self.PG_emulator_lhs = np.einsum_path("awu,a->wu",
                                                      self.Y_dagger_A_X,
                                                      theta_vec, optimize="greedy")
                self.PG_emulator_rhs = np.einsum_path("aw,a->w",
                                                      self.Y_dagger_b,
                                                      theta_vec, optimize="greedy")
                # PG error estimator:
                self.PG_error_estimator_lhs = np.einsum_path("awu,a,u->w",
                                                             self.Y_dagger_A_X,
                                                             theta_vec,
                                                             coefficients, optimize="greedy")
                self.PG_error_estimator_rhs = np.einsum_path("aw,a->w",
                                                             self.Y_dagger_b,
                                                             theta_vec, optimize="greedy")
    
    def calculate_coefficients_GROM(self,
                                    theta: dict,
                                    theta_vec=None,
                                    default_theta: dict = None,
                                    n_basis: int = None,
                                    use_einsum_calculations: bool = False,
                                    ret_pieces=False):
        r"""Calculates the coefficient vector for the G-ROM emulator.
        NOTE: This computes equations 34, 35, 36
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. This will be the location that the emulator 
            will find a wave function at.
        theta_vec : numpy array (optional)
            Numpy array of the parameters in theta, as in the output of `self.solver.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        n_basis : int (optional)
            The size of the snapshot basis.
        use_einsum_calculations : bool (optional)
            A flag to toggle whether einsum calculations or tensordot calculations will be used. These 
            calculations are the same (up to numerical noise), and it has been seen that (for my 
            specific machine) the tensordot implementation is significantly faster. 
        ret_pieces : bool (optional)
            When `True`, the components, `A_tilde`, `b_tilde`, and `coefficients` will be returned. 
            Otherwise only `coefficients` will be returned.
        
        Returns
        -------
        coefficients : numpy array
            The coefficient vector of the emulator at the given theta.
        A_tilde : numpy array (matrix)
            The reduced matrix for the G-ROM emulation.
            This will _only_ be returned when `ret_pieces=True`.
        b_tilde : numpy array
            The reduced right-hand side for the G-ROM emulation.
            This will _only_ be returned when `ret_pieces=True`.
        """
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta=theta, default_theta=default_theta)
        if use_einsum_calculations:
            A_tilde = np.einsum("auv,a->uv",
                                self.offline_projection_A[:, :n_basis, :n_basis],
                                theta_vec, 
                                optimize="greedy")
            b_tilde = np.einsum("au,a->u",
                                self.offline_projection_b[:, :n_basis],
                                theta_vec,
                                optimize="greedy")
        else:
            A_tilde = (self.offline_projection_A[:, :n_basis, :n_basis].T @ theta_vec).T
            b_tilde = self.offline_projection_b[:, :n_basis].T @ theta_vec
        
        # get coefficients
        coefficients = np.linalg.solve(A_tilde, b_tilde)
        if ret_pieces:
            return A_tilde, b_tilde, coefficients
        else:
            return coefficients
    
    def estimate_error_GROM(self, 
                            coefficients, 
                            theta, 
                            theta_vec=None,
                            default_theta=None, 
                            use_einsum_calculations=False,
                            ret_pieces=False):
        r"""Calculates the estimated error using projections associated with the G-ROM.
        NOTE: This computes equation 51.
        
        Affine implementation of the error estimator. ALL the calculations in this function are of 
        size n_basis (the size of coefficients) or size n_theta (the number of parameters for the 
        potential + 1).
        
        Parameters
        ----------
        coefficients : numpy array
            The coefficients for the emulator. Likely from `self.calculate_coefficients_GROM()` or
            `self.calculate_coefficients_LSPG()`.
        theta : dict 
            The parameters for the emulator.
        theta_vec : numpy array (optional)
            Numpy array of the parameters in theta, as in the output of `self.solver.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        use_einsum_calculations : bool (optional)
            When True, the calculations for this function will use np.einsum(). When False, the equivalent
            calculations will occur but using np.dot() and np.tensordot(). This option is mainly here 
            to keep the readability of np.einsum().
        
        Returns
        -------
        epsilon : number
            The estimated error for the given emulation.
            NOTE: that this result is sensitive to round-off errors.
        """
        # make sure that we have everything we need for the tensor contractions
        # coefficient stuff
        coefficients_dagger = coefficients.conjugate().T
        Coefficients = np.tensordot(coefficients_dagger, coefficients, axes=0)
        # theta stuff
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta=theta, default_theta=default_theta)
        theta_dagger = theta_vec.conjugate().T
        Theta = np.tensordot(theta_dagger, theta_vec, axes=0)
        
        if use_einsum_calculations:
            if self.use_einsum_paths:
                term_1 = np.einsum("abuv,ab,uv->",
                                   self.offline_projection_X_A_A_X,
                                   Theta,
                                   Coefficients, optimize=self.GROM_term_1[0])
                term_2 = np.einsum("abu,ab,u->",
                                   self.offline_projection_X_A_b,
                                   Theta,
                                   coefficients, optimize=self.GROM_term_2[0])
                term_2 = - 2 * np.real(term_2)
                
                term_3 = np.einsum("ab,ab->",
                                   self.b_dagger_on_b,
                                   Theta, optimize=self.GROM_term_3[0])
            else:
                term_1 = np.einsum("abuv,ab,uv->",
                                   self.offline_projection_X_A_A_X,
                                   Theta,
                                   Coefficients, optimize="greedy")
                term_2 = np.einsum("abu,ab,u->",
                                   self.offline_projection_X_A_b,
                                   Theta,
                                   coefficients, optimize="greedy")
                term_2 = - 2 * np.real(term_2)
                
                term_3 = np.einsum("ab,ab->",
                                   self.b_dagger_on_b,
                                   Theta, optimize="greedy")
        else:
            term_1 = theta_dagger @ (coefficients_dagger @ self.offline_projection_X_A_A_X @ coefficients) @ theta_vec
            term_2 = theta_vec @ (theta_dagger @ self.offline_projection_X_A_b @ coefficients)
            term_2 *= -2
            term_3 = theta_dagger @ self.b_dagger_on_b @ theta_vec
        
        if ret_pieces:
            return term_1, term_2, term_3
        epsilon_sq = term_1 + term_2 + term_3
        epsilon = np.abs(epsilon_sq) ** 0.5  # abs to make sure that a positive value is given (G-ROM gets round-off errors)
        return epsilon
    
    def calculate_coefficients_LSPG(self,
                                    theta: dict,
                                    theta_vec=None,
                                    default_theta: dict = None,
                                    n_basis: int = None,
                                    use_einsum_calculations: bool = False):
        r"""Calculated the coefficient vector for the LSPG-ROM emulator.
        NOTE: This computes equations 42, 43, 44.
        
        Parameters
        ----------
        theta : dict (optional)
            Dictionary object for the parameters in theta. This will be the location that the emulator 
            will find a wave function at.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        n_basis : int (optional)
            The size of the snapshot basis.
        use_einsum_calculations : bool (optional)
            A flag to toggle whether einsum calculations or tensordot calculations will be used. These 
            calculations are the same (up to numerical noise), and it has been seen that (for my specific 
            machine) the tensordot implementation is significantly faster. 
        
        Returns
        -------
        coefficients : numpy array
            The coefficient vector of the emulator at the given theta.
        lhs : numpy array
            The left-hand side of the emulator's ODE, $\tilde{A}$.
        rhs : numpy array
            The right-hand side of the emulator's ODE, $\tilde{b}$
        residual : numpy array
            The residual vector of from the least-squares solver. This output allows for the error
            estimation for the LSPG-ROM to be quite fast.
        """
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta=theta, default_theta=default_theta)
        if n_basis is None:
            n_basis = self.snapshot_max
        
        if use_einsum_calculations:
            if self.use_einsum_paths:
                lhs = np.einsum("awu,a->wu",
                                self.Y_dagger_A_X[:, :, :n_basis],
                                theta_vec, 
                                optimize=self.PG_emulator_lhs[0])
                rhs = np.einsum("aw,a->w",
                                self.Y_dagger_b,
                                theta_vec, 
                                optimize=self.PG_emulator_rhs[0])
            else:
                lhs = np.einsum("awu,a->wu",
                                self.Y_dagger_A_X[:, :, :n_basis],
                                theta_vec, 
                                optimize="greedy")
                rhs = np.einsum("aw,a->w",
                                self.Y_dagger_b,
                                theta_vec, 
                                optimize="greedy")
        else:
            lhs = np.tensordot(self.Y_dagger_A_X[:, :, :n_basis], theta_vec, axes=(0, 0))
            rhs = theta_vec @ self.Y_dagger_b
        coefficients, residual, rank, s = np.linalg.lstsq(lhs, rhs, rcond=None)
        return coefficients, lhs, rhs, residual
    
    def estimate_error_LSPG(self, 
                            coefficients, 
                            theta=None, 
                            theta_vec=None,
                            default_theta=None, 
                            use_einsum_calculations=False):
        r"""Estimated the error using Y (from the LSPG-ROM equations).
        NOTE: This computes equation 
        
        The error estimator for any emulation method (of the two current ones, "G-ROM" and "LSPG-ROM"). 
        This calculation will use the equation Y.dagger A(theta) X coefficients = Y.dagger b(theta).
        For this function to work, either theta and b0_b1 should be given, _or_ $A$ and $b$ should be given.
        
        Parameters
        ----------
        coefficients : numpy array
            The coefficients for the snapshot basis vectors for the emulation at the given parameter.
        theta : dict (optional)
            The parameter(s) for the error estimation
        theta_vec : numpy array (optional)
            Numpy array of the parameters in theta, as in the output of `self.solver.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        A : numpy array (or scipy sparse matrix?) (optional)
            The matrix for A x = b.
        b : numpy array (or scipy sparse matrix?) (optional)
            The lhs vector for A x = b.
        
        Returns
        -------
        epsilon : number
            The estimated error, sqrt(epsilon_sq) using the LSPG-ROM projections for the calculation.
        """
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta, default_theta=default_theta)
        
        if use_einsum_calculations:
            if self.use_einsum_paths:
                lhs = np.einsum("awu,a,u->w",
                                self.Y_dagger_A_X,
                                theta_vec,
                                coefficients, optimize=self.PG_error_estimator_lhs[0])
                rhs = np.einsum("aw,a->w",
                                self.Y_dagger_b,
                                theta_vec, optimize=self.PG_error_estimator_rhs[0])
            else:
                lhs = np.einsum("awu,a,u->w",
                                self.Y_dagger_A_X,
                                theta_vec,
                                coefficients, optimize="greedy")
                rhs = np.einsum("aw,a->w",
                                self.Y_dagger_b,
                                theta_vec, optimize="greedy")
        else:
            lhs = (self.Y_dagger_A_X.T @ theta_vec).T @ coefficients
            rhs = self.Y_dagger_b.T @ theta_vec
        
        epsilon_sq = np.linalg.norm(lhs - rhs)  # this is a norm across the "semi-reduced" space, w
        epsilon = epsilon_sq ** 0.5
        
        return epsilon
    
    
    def calculate_singular_values(self):
        r"""Calculated the singular values of the FOM matrix $A$ across the given parameter space.
        
        Calculates the minimum and maximum singular values for the matrices used in the parameter space. 
        If the parameter space has not yet been sampled, then it will be in this function. This function 
        will call an SVD on each point in every dimension of the parameter space. This SVD will almost 
        certainly take the longest to run of any part of this emulator.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if (self.min_singular_values is not None) and (self.max_singular_values is not None):
            print("Singular values already calculated.")
            return self.min_singular_values, self.max_singular_values
        else:
            self.min_singular_values = np.zeros(self.param_pts)
            self.max_singular_values = np.zeros(self.param_pts)
            # self.A_condition_numbers = np.zeros(self.param_pts)
        
        if self.verbose:
            iterator = tqdm(np.arange(self.param_pts))
        else:
            iterator = np.arange(self.param_pts)
        for i in iterator:
            A = self.solver.solve(theta=self.parameters_at_index(i), ret_matrix=True)[0]
            
            singular_values = scipy.linalg.svdvals(A)
            
            self.min_singular_values[i] = np.min(singular_values)
            self.max_singular_values[i] = np.max(singular_values)
            # self.A_condition_numbers[i] = np.linalg.cond(A)
        # self.condition_numbers = self.A_condition_numbers
        self.condition_numbers = self.max_singular_values / self.min_singular_values
    
    def calculate_exact_values(self, 
                               matching_indices: int = None):
        r"""Void function that calculates the exact solutions of the wave function across the parameter space.
        
        Calculates exact matrix elements and phase shifts across the parameter spaces given in the 
        initialization of this class. This will populate `self.exact_elements`.
        
        Parameters
        ----------
        matching_indices : int
            The number of points used (at the end of the coordinate space) for matching via
            least squares.
        
        Returns
        -------
        None
        """
        if matching_indices is None:
            matching_indices = self.matching_indices
        if self.exact_elements is not None:
            print("Exact elements already calculated.")
        else:
            self.exact_elements = {"delta_l": np.empty(self.param_pts),
                                   "K_l": np.empty(self.param_pts),
                                   "T_l": np.empty(self.param_pts, dtype=complex),
                                   "S_l": np.empty(self.param_pts, dtype=complex),
                                   "a": np.empty(self.param_pts),
                                   "b": np.empty(self.param_pts)}
            
            default_theta = self.potential.default_theta
            if self.verbose:
                print("Calculating matrix elements:")
                iterator = tqdm(np.arange(self.param_pts))
            else:
                iterator = np.arange(self.param_pts)
            for i in iterator:
                exact_chi = self.FOM(self.parameters_at_index(i), default_theta=default_theta)
                if len(exact_chi) != self.n:
                    exact_chi = exact_chi[:-self.n_adj]
                matching_output = matching_using_least_squares(self.r, 
                                                               exact_chi, 
                                                               self.p, 
                                                               self.l, 
                                                               matching_indices=matching_indices, 
                                                               zeta=self.zeta)
                
                self.exact_elements["delta_l"][i] = matching_output["delta_l"]
                self.exact_elements["K_l"][i] = matching_output["K_l"]
                self.exact_elements["T_l"][i] = matching_output["T_l"]
                self.exact_elements["S_l"][i] = matching_output["S_l"]
                self.exact_elements["a"][i] = matching_output["a"]
                self.exact_elements["b"][i] = matching_output["b"]
    
    def propagate_error_to_phaseshift(self,
                                      param_i: int,
                                      n_basis: int,
                                      use_exact_error=False):
        """Propagates the error from the wave function to an upper bound on the error of the phaseshift.
        
        Parameters:
        param_i : int
            The index of the parameter being used in the parameter space.
            
        Returns
        -------
        None
        """
        if use_exact_error:
            chi_error = self.exact_errors[n_basis - 1, param_i]
        else:
            chi_error = self.scaled_estimated_errors[n_basis - 1, param_i]
        
        a_tilde = self.emulated_elements["a"][n_basis - 1, param_i]
        b_tilde = self.emulated_elements["b"][n_basis - 1, param_i]
        
        max_delta_error = propagate_delta_error(a_tilde,
                                                b_tilde,
                                                chi_error, 
                                                self.M, 
                                                M_pinv=self.M_pinv)
        
        self.propagated_max_phaseshift_errors[n_basis - 1, param_i] = max_delta_error
    
    def calculate_emulated_elements(self, 
                                    emulated_chi, 
                                    param_i: int, 
                                    n_basis: int,
                                    matching_indices: int = None):
        """Void function to calculate exact errors in the emulator and populate self.emulated_elements.
        
        Calculates emulated matrix elements and phase shifts across the parameter spaces given in 
        the initialization of this class. This will populate `self.emulated_elements["element"][:, i]`.
        
        Parameters
        ----------
        emulated_chi : list-like
            The emulated scattered wave
        param_i : int
            The index of the parameter used for the emulated wave
        n_basis : int
            The size of the snapshot basis used in the emulation
        matching_indices : int (optional)
            The number of indices used for the least squares matching process
        
        Returns
        -------
        None
        """
        if matching_indices is None:
            matching_indices = self.matching_indices
        emul_chi = np.copy(emulated_chi)
        if len(emul_chi) != self.n:
                    emul_chi = emul_chi[:-self.n_adj]
        
        matching_output = matching_using_least_squares(self.r, 
                                                       emul_chi, 
                                                       self.p, 
                                                       self.l, 
                                                       matching_indices=matching_indices)
        self.emulated_elements["delta_l"][n_basis - 1, param_i] = matching_output["delta_l"]
        self.emulated_elements["K_l"][n_basis - 1, param_i] = matching_output["K_l"]
        self.emulated_elements["T_l"][n_basis - 1, param_i] = matching_output["T_l"]
        self.emulated_elements["S_l"][n_basis - 1, param_i] = matching_output["S_l"]
        self.emulated_elements["a"][n_basis - 1, param_i] = matching_output["a"]
        self.emulated_elements["b"][n_basis - 1, param_i] = matching_output["b"]
    
    def calculate_errors(self, 
                         emulated_chi,
                         estimated_error,
                         param_i: int,
                         n_basis: int, 
                         theta_vec=None,
                         default_theta: dict = None, 
                         ignore_error_bounds: bool = False, 
                         calculate_elements: bool = True, 
                         use_practically: bool = False):
        """Void function that calculates the estimated and exact errors for the emulator.
        
        Parameters
        ----------
        n_basis : int
            The size of the snapshot basis used in the emulation
        param_i : int
            The index of the parameter space that will be used in the calculation.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter. Used for testing.
        ignore_error_bounds : bool (optional)
            When `True`, the error bounds will not be calculated. In this case, the singular values 
            will not be accessed, saving some computation time.
        calculate_elements : bool (optional)
            When `True`, the matrix elements and phase shift will be calculated across the
            parameter space.
        use_practically : bool (optional)
            When `True`, the exact errors will be calculated. Depending on the FOM implementation 
            this may be a cumbersome calculation.
        
        Returns
        -------
        None
        """
        if default_theta is None:
            default_theta = self.potential.default_theta.copy()
        current_parameter = self.parameters_at_index(param_i)
        if theta_vec is None:
            theta_vec = self.solver.theta_args(current_parameter)
        if not use_practically:
            exact_chi = self.FOM(current_parameter, 
                                 theta_vec=theta_vec,
                                 default_theta=default_theta)
            # calculate vectors for estimated and exact errors
            exact_residual = exact_chi - emulated_chi
            
            exact_error = np.linalg.norm(exact_residual)  # calculate norms of error vectors
            # also grab the emulated matrix elements and phase shift while we've spent the time to emulate chi
            if calculate_elements:
                self.calculate_emulated_elements(emulated_chi, 
                                                 param_i, 
                                                 n_basis=n_basis)
            self.exact_errors[n_basis - 1, param_i] = exact_error
        self.estimated_errors[n_basis - 1, param_i] = estimated_error
        
        # sometimes if the parameter corresponds to a snapshot in the basis, then this may trigger due to numerics.
        if (not ignore_error_bounds) and (not use_practically):
            # calculate error bounds
            upper_bound = estimated_error / self.min_singular_values[param_i]
            lower_bound = estimated_error / self.max_singular_values[param_i]
            
            if not (current_parameter in self.snapshot_parameters):
                # make sure the errors are within the error bounds
                if (estimated_error > upper_bound) or (estimated_error < lower_bound):
                    print(self.solver.theta_args(current_parameter), self.solver.theta_args(self.snapshot_parameters))
                    print(f"estimated error {current_parameter} not in bounds. \n\tupper_bound = {upper_bound}\n\testimated error = {estimated_error}\n\tlower_bound = {lower_bound}")
                if (exact_error > upper_bound) or (exact_error < lower_bound):
                    print(self.solver.theta_args(current_parameter), self.solver.theta_args(self.snapshot_parameters))
                    print(f"estimated error {current_parameter} not in bounds. \n\tupper_bound = {upper_bound}\n\testimated error = {estimated_error}\n\tlower_bound = {lower_bound}")
            self.upper_error_bounds[n_basis - 1, param_i] = upper_bound
            self.lower_error_bounds[n_basis - 1, param_i] = lower_bound
    
    def scale_estimated_error(self, 
                              n_basis: int, 
                              theta: dict, 
                              one_extra_FOM_calculation):
        """Scales the estimated error by the factor between the estimated error and exact error.
        
        The scaling of the estimated error and the exact error should (and does, in the current
        implementation) occur based on the wave functions at the point of the worst estimated
        error. This is so that the calculation can be used in the next iteration of the greedy
        algorithm.
        
        Parameters
        ----------
        n_basis : int
            The number of snapshots currently in the basis. This input value should be starting from 1, 
            not 0. So if there are five basis vectors, the input should be 5.
        theta : dict
            The parameter value at which the emulator will emulate with the given basis.
        one_extra_FOM_calculation : array-like
            An extra FOM calculation for calculating exact errors.
        
        Returns
        -------
        None
        """
        one_extra_ROM_calculation = self.emulate(theta, 
                                                 n_basis=n_basis,
                                                 emulation_method=self.emulation_method,
                                                 error_estimation_method=self.error_estimation_method)
        one_extra_error_calculation = np.linalg.norm(one_extra_FOM_calculation - one_extra_ROM_calculation)
        scale_factor = one_extra_error_calculation / np.max(self.estimated_errors[n_basis - 1, :])
        self.scaled_estimated_errors[n_basis - 1, :] = scale_factor * self.estimated_errors[n_basis - 1, :]
    
    
    def train_with_greedy(self, 
                          use_practically: bool = False):
        """Constructs the emulator basis using the greedy algorithm.
        NOTE: This encompass section IV B in the paper.
        
        "The" greedy algorithm. This will use the sampled parameter space and construct a snapshot 
        basis in an iterative manner, checking estimated errors and choosing the optimal parameter 
        to place the new parameters.
        
        Parameters
        ----------
        use_practically : bool (optional)
            When `True`, the minimum number of FOM calculations will be kept to the bare minimum, and
            the errors in the scattering matrices and phase shifts will not be calculated.
            
        Returns
        -------
        None
        """
        # first see if the emulator has already been trained
        if self.trained:
            if self.verbose:
                print("Emulator already trained")
            return
        elif not self.greedy:
            if self.verbose:
                print("Emulator is not defined as greedy.")
            return
        
        # if the emulator isn't trained, then let's go ahead and train it
        else:
            # create arrays for storing everything we want
            start_of_training = time.time()
            self.exact_errors = np.empty((self.snapshot_max, self.param_pts))
            self.estimated_errors = np.empty((self.snapshot_max, self.param_pts))
            self.scaled_estimated_errors = np.empty((self.snapshot_max, self.param_pts))
            if not self.ignore_error_bounds:
                self.upper_error_bounds = np.empty((self.snapshot_max, self.param_pts))
                self.lower_error_bounds = np.empty((self.snapshot_max, self.param_pts))
            self.emulated_elements = {"delta_l": np.empty((self.snapshot_max, self.param_pts)),
                                      "K_l": np.empty((self.snapshot_max, self.param_pts)),
                                      "T_l": np.empty((self.snapshot_max, self.param_pts), dtype=complex),
                                      "S_l": np.empty((self.snapshot_max, self.param_pts), dtype=complex),
                                      "a": np.empty((self.snapshot_max, self.param_pts)),
                                      "b": np.empty((self.snapshot_max, self.param_pts))}
            self.propagated_max_phaseshift_errors = np.empty((self.snapshot_max, self.param_pts))
        
        # seer if we have any initial snapshots
        if self.snapshot_parameters is None:  # if we do then let's scan the parameter space once to see if we've met accuracy goals initially
            self.snapshot_parameters = np.empty(self.snapshot_max, dict)
            next_parameter = {}
            for parameter in self.parameter_bounds:
                # grab the first parameter for LHS comparisons
                next_parameter[parameter] = (self.parameter_space[parameter])[0]
                self.snapshot_parameters[0] = next_parameter
            
            one_extra_FOM = None
            self.scale_estimated_error(self.preset_snapshots, 
                                       next_parameter,
                                       one_extra_FOM_calculation=self.snapshots[:, self.preset_snapshots])
        else:  # If there's no initial snapshots then let's set the non-used arrays to some default value (1e-16).
            #    1e-16 is used to allow for log plots to be used without warnings, otherwise its arbitrary.
            self.exact_errors[:self.preset_snapshots, :] = 1e-16
            self.estimated_errors[:self.preset_snapshots, :] = 1e-16
            self.scaled_estimated_errors[:self.preset_snapshots, :] = 1e-16
            
            if not self.ignore_error_bounds:
                self.upper_error_bounds[:self.preset_snapshots, :] = 1e-16
                self.lower_error_bounds[:self.preset_snapshots, :] = 1e-16
            
            self.emulated_elements["delta_l"][:self.preset_snapshots] = 1e-16
            self.emulated_elements["K_l"][:self.preset_snapshots] = 1e-16
            self.emulated_elements["T_l"][:self.preset_snapshots] = 1e-16
            self.emulated_elements["S_l"][:self.preset_snapshots] = 1e-16
            self.emulated_elements["a"][:self.preset_snapshots] = 1e-16
            self.emulated_elements["b"][:self.preset_snapshots] = 1e-16
            
            self.propagated_max_phaseshift_errors[:self.preset_snapshots, :] = 1e-16
            
            # if we have given snapshots, then we need to determine where to place the next "initial" snapshot
            if self.verbose:
                print(f"Calculating errors with given snapshots ({self.preset_snapshots}).")
            
            # do one scan of the parameter space to "get the ball rolling" on the greedy algorithm
            for param_i in np.arange(self.param_pts):
                current_parameter = self.parameters_at_index(param_i)
                theta_vec = self.solver.theta_args(current_parameter)
                
                emulated_chi, estimated_error = self.emulate(current_parameter,
                                                             n_basis=self.preset_snapshots,
                                                             theta_vec=theta_vec,
                                                             default_theta=self.default_theta,
                                                             estimate_error=True,
                                                             emulation_method=self.emulation_method,
                                                             error_estimation_method=self.error_estimation_method,
                                                             use_einsum_in_emulation=self.use_einsum_calculations,
                                                             use_einsum_in_error_estimation=self.use_einsum_calculations)
                self.calculate_errors(emulated_chi,
                                      estimated_error,
                                      param_i,
                                      self.preset_snapshots,
                                      ignore_error_bounds=self.ignore_error_bounds, 
                                      use_practically=use_practically,
                                      calculate_elements=True)
                self.calculate_emulated_elements(emulated_chi,
                                                 param_i,
                                                 self.preset_snapshots,
                                                 matching_indices=self.matching_indices)
                self.propagate_error_to_phaseshift(param_i, 
                                                   self.preset_snapshots, 
                                                   use_exact_error=True)
            
            index_of_worst_error = np.argsort(self.estimated_errors[self.preset_snapshots - 1])[-1]
            next_parameter = {}
            for parameter in self.parameter_space:
                next_parameter[parameter] = self.parameter_space[parameter][index_of_worst_error]
                
            one_extra_FOM = self.FOM(next_parameter)
            if self.use_scaled_estimated_error:
                self.scale_estimated_error(self.preset_snapshots, 
                                           next_parameter,
                                           one_extra_FOM_calculation=one_extra_FOM)
            if self.verbose:
                print(f"Parameter of worst estimated error: {next_parameter}")
                plotted_parameter = list(self.parameter_space.keys())[0]
                self.iterative_plot(plotted_parameter, 
                                    self.preset_snapshots - 1)
            
            if self.preset_snapshots == self.snapshot_max:
                print("Given number of snapshots is the same as the given max number of snapshots!")
                return
            self.snapshot_parameters[self.preset_snapshots] = next_parameter
        
        
        
        """   ###   ###   ###   ###   ###   ###   ###   ###   ###   ###
        #--   ---   ---   The Greedy Algorithm's Loop   ---   ---   --#
        ###   ###   ###   ###   ###   ###   ###   ###   ###   ###   """
        
        # iterate over snapshots
        for i in np.arange(self.preset_snapshots, self.snapshot_max):
            next_param_vec = self.solver.theta_args(next_parameter)  # "vector-ify" the next snapshot parameter
            snapshot_num = i + 1  # bookkeeping
            if self.verbose:
                print(f"\n\nCalculating snapshot {snapshot_num}")
            
            
            # calculate new proposed snapshot
            if one_extra_FOM is None:  # see if we have an extra FOM to use
                # if we don't then let's get one
                proposed_new_snapshot = self.FOM(next_parameter, 
                                                 theta_vec=next_param_vec,
                                                 default_theta=self.default_theta)
            else:
                # if we do then let's use it
                proposed_new_snapshot = one_extra_FOM
            
            # add new snapshot to the basis and orthonormalize it
            if self.orthonormal_basis:
                # optimize with optimized Modified-Gram-Schmidt
                new_snapshots = MGS(self.snapshots[:, :i], 
                                    proposed_new_snapshot,
                                    atol=self.orth_cutoff, 
                                    rtol=self.orth_cutoff)  # take tolerances from the class
                if new_snapshots.shape[1] != snapshot_num:
                    # you only end up here if the proposed snapshot is orthonormalized out!
                    if self.verbose:
                        print("Proposed new snapshot got orthonormalized out.")
                    
                    new_snapshot_orthonormalized_out = True  # this will trigger the greedy algorithm to terminate early
                    
                    # scale estimated error if that's something we care about
                    if self.use_scaled_estimated_error:
                        self.scale_estimated_error(snapshot_num - 1, next_parameter, proposed_new_snapshot)
                    break
                else:
                    new_snapshot_orthonormalized_out = False
                self.snapshots[:, :snapshot_num] = new_snapshots
                
                # see if MGS isn't happy anymore
                orthonormalization_error = snapshot_num - np.sum(np.diag(self.snapshots.T @ self.snapshots))
                if orthonormalization_error >= self.orth_tolerance:
                    if self.verbose:
                        print(f"MGS errors are above tolerance ({orthonormalization_error} >= {self.orth_tolerance}). Reorthonormalizing entire basis")
                    self.snapshots = unoptimized_MGS(self.snapshots)  # this is the un-optimized MGS, and will reorthonormalize the _entire_ basis.
            
            # if we're not orthonormalizing the basis, for some reason, then this is as easy as just writing the snapshot to the matrix.
            else:
                self.snapshots[:, i] = proposed_new_snapshot
                new_snapshot_orthonormalized_out = False
            
            
            self.offline_projections(snapshot_num)  # project current snapshot basis onto FOM ODE for emulation (a.k.a. the offline stage)
            
            
            # iterate across parameter space and check errors
            for param_i in np.arange(self.param_pts):
                current_parameter = self.parameters_at_index(param_i)  # get current parameter
                theta_vec = self.solver.theta_args(current_parameter)  # "vector-ify" the current parameter
                
                # calculate emulated wave function (and its associated estimated error)
                emulated_chi, estimated_error = self.emulate(current_parameter,
                                                             n_basis=snapshot_num,
                                                             theta_vec=theta_vec,
                                                             default_theta=self.default_theta,
                                                             estimate_error=True,
                                                             emulation_method=self.emulation_method,
                                                             error_estimation_method=self.error_estimation_method,
                                                             use_einsum_in_emulation=self.use_einsum_calculations,
                                                             use_einsum_in_error_estimation=self.use_einsum_calculations)
                # calculate the exact errors and store in errors in their associated arrays 
                self.calculate_errors(emulated_chi,
                                      estimated_error,
                                      param_i,
                                      snapshot_num,
                                      ignore_error_bounds=self.ignore_error_bounds, 
                                      use_practically=use_practically,
                                      calculate_elements=True)
                # calculate emulated matrix elements and phaseshift and store in errors in their associated arrays 
                self.calculate_emulated_elements(emulated_chi,
                                                 param_i,
                                                 snapshot_num,
                                                 matching_indices=self.matching_indices)
                # calculate maximum error bound in the phaseshift and store in errors in its associated arrays
                self.propagate_error_to_phaseshift(param_i, 
                                                   snapshot_num, 
                                                   use_exact_error=True)
            
            # find location of the largest error in parameter space
            index_of_worst_error = np.argmax(self.estimated_errors[i])
            
            # get the next parameter (as a dict object, as a parameter is treated in this class)
            next_parameter = {}
            for parameter in self.parameter_space:
                next_parameter[parameter] = self.parameter_space[parameter][index_of_worst_error]
            
            if self.verbose:
                print(f"Parameter of worst estimated error: {next_parameter}")
            
            # calculate "one-extra-FOM-calculation" that will be used in the next iteration if able.
            if self.use_scaled_estimated_error:
                one_extra_FOM = self.FOM(next_parameter)
                self.scale_estimated_error(snapshot_num, 
                                           next_parameter, 
                                           one_extra_FOM)
            
            # determine if any break conditions have been met
            cutoff_accuracy_met = False
            if self.cutoff_accuracy is None:  # for no accuracy goal
                if next_parameter in self.snapshot_parameters:
                    if next_parameter in self.snapshot_parameters:
                        if self.verbose:
                            print("Parameter of max error is already in parameter space")
                            print(f"{next_parameter} in {self.snapshot_parameters}")
                        cutoff_accuracy_met = True
                    elif ((np.max(self.estimated_errors) <= self.cutoff_accuracy)
                          or
                          (self.use_scaled_estimated_error  # check the scaled estimated error if the emulator is told to (better for accuracy of errors).
                           and
                           np.max(self.scaled_estimated_errors) <= self.cutoff_accuracy)):
                        if self.verbose:
                            print("***   ***   Cutoff accuracy reached across parameter space   ***   ***")
                    self.snapshot_max = snapshot_num  # override the snapshot_max (bookkeeping)
                    
                    
                    # take only successful runs in arrays (more bookkeeping)
                    cut_index = snapshot_num + 1
                    self.snapshot_parameters = self.snapshot_parameters[:cut_index]
                    self.exact_errors = self.exact_errors[:cut_index, :]
                    self.estimated_errors = self.estimated_errors[:cut_index, :]
                    self.scaled_estimated_errors = self.scaled_estimated_errors[:cut_index, :]
                    if not self.ignore_error_bounds:
                        self.upper_error_bounds = self.upper_error_bounds[:cut_index, :]
                        self.lower_error_bounds = self.lower_error_bounds[:cut_index, :]
                    for element in self.emulated_elements:
                        self.emulated_elements[element] = self.emulated_elements[element][:cut_index, :]
                    if self.verbose:
                        plotted_parameter = list(self.parameter_bounds.keys())[0]
                        self.iterative_plot(plotted_parameter, i)
                    break
                else:
                    if snapshot_num < self.snapshot_max:
                        self.snapshot_parameters[snapshot_num] = next_parameter
            else:  # for when an accuracy goal is provided
                if (next_parameter in self.snapshot_parameters) or (
                        not self.use_scaled_estimated_error and
                        np.max(self.estimated_errors[i, :]) <= self.cutoff_accuracy) \
                        or (self.use_scaled_estimated_error
                            and
                            np.max(self.scaled_estimated_errors[i, :]) <= self.cutoff_accuracy):
                    if next_parameter in self.snapshot_parameters:
                        if self.verbose:
                            print("Parameter of max error is already in parameter space.")
                            print("Reported errors of last iteration may be incorrect.")
                    elif (not self.use_scaled_estimated_error and
                          np.max(self.estimated_errors[i, :]) <= self.cutoff_accuracy) \
                            or (self.use_scaled_estimated_error
                                and
                                np.max(self.scaled_estimated_errors[i, :]) <= self.cutoff_accuracy):
                        if self.verbose:
                            print("***   ***   Cutoff accuracy reached across parameter space   ***   ***")
                    cutoff_accuracy_met = True
                    self.snapshot_max = snapshot_num + 1  # override the snapshot_max
                    
                    # take only successful runs in arrays
                    cut_index = snapshot_num + 1
                    self.snapshot_parameters = self.snapshot_parameters[:cut_index]
                    self.exact_errors = self.exact_errors[:cut_index, :]
                    self.estimated_errors = self.estimated_errors[:cut_index, :]
                    self.scaled_estimated_errors = self.scaled_estimated_errors[:cut_index, :]
                    if not self.ignore_error_bounds:
                        self.upper_error_bounds = self.upper_error_bounds[:cut_index, :]
                        self.lower_error_bounds = self.lower_error_bounds[:cut_index, :]
                    for element in self.emulated_elements:
                        self.emulated_elements[element] = self.emulated_elements[element][:cut_index, :]
                    if self.verbose:
                        # don't forget to define the plotted parameter !
                        plotted_parameter = list(self.parameter_bounds.keys())[0]
                        self.iterative_plot(plotted_parameter, i)
                    break
                else:
                    if snapshot_num < self.snapshot_max:
                        self.snapshot_parameters[snapshot_num] = next_parameter
                #
            if self.verbose:
                plotted_parameter = list(self.parameter_bounds.keys())[0]
                self.iterative_plot(plotted_parameter, i)
        
        # scale estimated error if that's something we care about
        if self.use_scaled_estimated_error:
            if new_snapshot_orthonormalized_out:
                self.scale_estimated_error(snapshot_num - 1,  # the basis size is one less than expected at this point
                                           next_parameter, 
                                           one_extra_FOM)
            else:
                self.scale_estimated_error(snapshot_num, 
                                           next_parameter, 
                                           one_extra_FOM)
        
        """   ###   ###   ###    ###   ###    ###   ###    ###   ###   ###
        #--   ---   ---   End of Greedy Algorithm's Loop   ---   ---   --#
        ###   ###   ###   ###    ###   ###    ###   ###    ###   ###   """
        
        
        ###   ###   data management   ###   ###
        # if the loop ended early then we should clean up the array sizes
        if new_snapshot_orthonormalized_out or cutoff_accuracy_met:
            if new_snapshot_orthonormalized_out:
                self.snapshot_max = snapshot_num - 1  # override the snapshot_max
            elif cutoff_accuracy_met:
                self.snapshot_max = snapshot_num # override the snapshot_max
            # take only successful runs in arrays
            self.snapshots = self.snapshots[:, :self.snapshot_max]
            self.truncated_snapshots = self.snapshots[2:, :]
            self.dagger_snapshots = self.snapshots.T.conjugate()
            self.truncated_dagger_snapshots = self.truncated_snapshots.T.conjugate()
            
            cut_index = self.snapshot_max
            self.snapshot_parameters = self.snapshot_parameters[:cut_index]
            self.exact_errors = self.exact_errors[:cut_index, :]
            self.estimated_errors = self.estimated_errors[:cut_index, :]
            self.scaled_estimated_errors = self.scaled_estimated_errors[:cut_index, :]
            if not self.ignore_error_bounds:
                self.upper_error_bounds = self.upper_error_bounds[:cut_index, :]
                self.lower_error_bounds = self.lower_error_bounds[:cut_index, :]
            for element in self.emulated_elements:
                self.emulated_elements[element] = self.emulated_elements[element][:cut_index, :]
            #
        self.trained = True
        end_of_training = time.time()
        self.training_time = end_of_training - start_of_training
    
    def emulate(self, 
                theta: dict, 
                default_theta: dict = None, 
                theta_vec=None,
                n_basis: int = None, 
                ret_pieces: bool = False,
                estimate_error: bool = False,
                emulation_method=None, 
                error_estimation_method=None,
                use_einsum_in_emulation: bool = False, 
                use_einsum_in_error_estimation: bool = False):
        """Emulator using the G-ROM or LSPG-ROM, with or without error estimation.
        
        Emulates the wave function for the given emulator type (G-ROM or LSPG-ROM), with or without
        estimating the error. Einsum calculations can be used, but typically it is faster to use
        tensordot (notably, the einsum code is _far_ more readable, hence why it's included).
        
        Parameters
        ----------
        theta : dict
            The parameter value at which the emulator will emulate with the given basis.
        theta_vec : numpy array (optional)
            Numpy array of the parameters in theta, as in the output of `self.solver.theta_args(theta)`.
        default_theta : dict (optional)
            Dictionary containing the default arguments of the parameter.
            Used for testing.
        n_basis : int or None (optional)
            The number of snapshots currently in the basis. This input value should be starting 
            from 1, not 0. So if there are five basis vectors, the input should be 5.
        ret_pieces : bool (optional)
            When True, emulate() will return the "pieces" that are used to solve for coefficients, 
            as well as coefficients itself.
        estimate_error : bool (optional)
            When True, emulate() will return the emulated wave as well as an error estimate.
        emulation_method : str (optional)
            The two options are "LSPG-ROM" (the default) and "G-ROM"
        error_estimation_method : str (optional)
            The method that the error estimator will use for the calculations. The two current 
            options are "G-ROM" and "LSPG-ROM".
        
        Returns
        -------
        ("A', "b", "coefficients") : tuple
            When ret_pieces = `True` then emulate() will return the "pieces" in the 
            equation A coefficients = b.
        emulated_chi : numpy array
            The emulated scattered wave as according to the given emulation_method.
        (emulated_chi, estimated_error) : tuple
            The emulated scattered wave as according to the given emulation_method as well as the 
            estimated error calculated using the given argument for error_estimation_method.
        """
        
        if emulation_method is None:
            emulation_method = self.emulation_method
        if error_estimation_method is None:
            error_estimation_method = self.error_estimation_method
        if n_basis is None:
            n_basis = self.snapshot_max
        if default_theta is None:
            default_theta = self.potential.default_theta.copy()
        if theta_vec is None:
            theta_vec = self.solver.theta_args(theta=theta, default_theta=default_theta)
        
        # if use_einsum_in_error_estimation:
        if emulation_method == "G-ROM":
            A_tilde, b_tilde, coefficients = self.calculate_coefficients_GROM(theta,
                                                                              theta_vec=theta_vec,
                                                                              default_theta=self.default_theta,
                                                                              n_basis=n_basis,
                                                                              use_einsum_calculations=use_einsum_in_emulation,
                                                                              ret_pieces=True)
            if ret_pieces:
                return A_tilde, b_tilde, coefficients
            emulated_chi = self.truncated_snapshots[:, :n_basis] @ coefficients
            
            if estimate_error:
                if error_estimation_method == "G-ROM":
                    estimated_error = self.estimate_error_GROM(coefficients, 
                                                               theta,
                                                               theta_vec=theta_vec,
                                                               default_theta=self.default_theta,
                                                               use_einsum_calculations=use_einsum_in_error_estimation)
                elif error_estimation_method == "LSPG-ROM":
                    estimated_error = self.estimate_error_LSPG(coefficients, 
                                                               theta=theta,
                                                               theta_vec=theta_vec,
                                                               default_theta=self.default_theta,
                                                               use_einsum_calculations=use_einsum_in_error_estimation) ** 2
                else:
                    raise ValueError("Unknown error estimation method. Must be \"G-ROM\" or \"LSPG-ROM\".")
                # use the information we know in the emulated wave
                emulated_chi = np.append(np.array([self.solver.y_0, self.solver.ypr_0]), emulated_chi)
                return emulated_chi, estimated_error
            
            emulated_chi = np.append(np.array([self.solver.y_0, self.solver.ypr_0]), emulated_chi)
            return emulated_chi
        elif emulation_method == "LSPG-ROM":
            coefficients, lhs, rhs, residual = self.calculate_coefficients_LSPG(theta,
                                                                                n_basis=n_basis,
                                                                                theta_vec=theta_vec,
                                                                                use_einsum_calculations=use_einsum_in_emulation)
            if ret_pieces:
                return lhs, rhs, coefficients, residual
            emulated_chi = self.truncated_snapshots[:, :n_basis] @ coefficients
            
            if estimate_error:
                if error_estimation_method == "LSPG-ROM":
                    """
                    Here we can take advantage of the least-squares solver residual return value. 
                    This _is_ a norm across the FOM space, not the ROM space, but the time saved by
                    not doing more tensor contractions.
                    """
                    estimated_error = np.linalg.norm(residual) ** 0.5
                elif error_estimation_method == "G-ROM":
                    estimated_error = self.estimate_error_GROM(coefficients, theta)
                # use the information we know in the emulated wave
                emulated_chi = np.append(np.array([self.solver.y_0, self.solver.ypr_0]), emulated_chi)
                return emulated_chi, estimated_error
            
            # use the information we know in the emulated wave
            emulated_chi = np.append(np.array([self.solver.y_0, self.solver.ypr_0]), emulated_chi)
            return emulated_chi
        else:
            raise ValueError(f"Emulation_method {emulation_method} not known.")
    
    
    ###   ###   ###   ###   ###   ###   ###   ###    ###    ###    ###   ###   ###   ###   ###   ###   ###
    #--   ---   These remaining functions are for visualizing emulator accuracy and convergence.          
    #           They are not required for use of the emulator.                                   ---   --#
    ###   ###   ###   ###   ###   ###   ###   ###    ###    ###    ###   ###   ###   ###   ###   ###   ###
    def iterative_plot(self, 
                       plotted_parameter: str, 
                       i: int,
                       show_plot: bool = True):
        """Function for making plots in the greedy algorithm to visualize training.
        
        This plotting function is mainly for avoid code duplication. It can also be called from outside
        the training, if one is curious about the emulator state after some amount of iterations.
        
        Parameters
        ----------
        plotted_parameter : string
            The name of the parameter that will be used in the plot. The choice for this in the greedy
            algorithm is to simply take the first parameter value that is varied.
        i : int
            The iteration of snapshot that will be used. For example, if i=2, then the first 3 snapshots 
            be used, as iterations start at 0.
        show_plot : bool (optional)
            When `True`, the plot will be shown. When `False`, `plt.show()` will not be called.
        
        Returns
        -------
        None
        """
        from matplotlib import pyplot as plt
        
        plt.figure(dpi=200, figsize=(7, 5))
        plt.title(f"Emulator Errors for {i + 1} Snapshots", fontsize=20)
        
        plt.plot(self.parameter_space[plotted_parameter],
                 self.estimated_errors[i, :], color="purple", label="Estimated Error")
        if self.use_scaled_estimated_error:
            plt.plot(self.parameter_space[plotted_parameter],
                     self.scaled_estimated_errors[i, :], color="red", label="Scaled Estimated Error")
        if not self.use_practically:
            plt.plot(self.parameter_space[plotted_parameter],
                     self.exact_errors[i, :], color="orange", label="Exact Error")
        if not self.ignore_error_bounds:
            plt.fill_between(self.parameter_space[plotted_parameter],
                             self.lower_error_bounds[i, :],
                             self.upper_error_bounds[i, :], 
                             color="grey", alpha=0.3, label="Error Bounds")
        for snapshot_parameter in self.snapshot_parameters[:(i + 1)]:
            plt.axvline(snapshot_parameter[plotted_parameter], color="green")
        
        if self.cutoff_accuracy is not None:
            plt.axhline(self.cutoff_accuracy, color="black", linestyle="--", linewidth=1)
        
        plt.xlabel("Parameter Space [MeV]", fontsize=16)
        plt.ylabel("Error", fontsize=16)
        plt.yscale("log")
        plt.legend()
        if show_plot:
            plt.show()
    
    def emulation_errors_at_theta(self, 
                                  theta: dict = None,
                                  n_basis: int = None, 
                                  basis_scaling=1,
                                  dpi=500, 
                                  show: bool = True, 
                                  plot_title_1=None, 
                                  plot_title_2=None,
                                  print_errors=False):
        """Plots the FOM, ROM, and basis for a given theta.
        
        Parameters
        ----------
        theta : dict (optional)
            The parameter value used for the plotting. If no value is given then the best-fit
            parameter value for the potential will be used.
        n_basis : int
            The size of the snapshot basis. 
            
        """
        from matplotlib import pyplot as plt
        
        if theta is None:
            theta = self.potential.default_theta.copy()
        if n_basis is None:
            n_basis = self.snapshot_max
        
        FOM_chi = self.FOM(theta, default_theta=self.default_theta)
        ROM_chi, estimated_error = self.emulate(theta, default_theta=self.default_theta, n_basis=n_basis, estimate_error=True)
        if self.n_adj > 0:
            FOM_chi = FOM_chi[:-self.n_adj]
            ROM_chi = ROM_chi[:-self.n_adj]
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=dpi, height_ratios = [3, 2])
        if plot_title_1 is None:
            ax[0].set_title("Visualizing Emulator Results", fontsize=20)
        else:
            ax[0].set_title(plot_title_1, fontsize=20)
        # snapshots
        snapshot_alpha = 0.2
        
        for i in np.arange(n_basis):
            if self.n_adj == 0:
                snapshots = self.snapshots[:, i]
            else:
                snapshots = self.snapshots[:-self.n_adj, i]
            if i == 0:
                if self.solver.zeta == 1:
                    ax[0].plot(self.r, basis_scaling * snapshots,
                                alpha=snapshot_alpha, color="green", linestyle="-.", linewidth=1.8,
                                label=r"$\chi_{\rm{basis}}(r)$")
                else:
                    ax[0].plot(self.r, basis_scaling * snapshots,
                                alpha=snapshot_alpha, color="green", linestyle="-.", linewidth=1.8,
                                label=r"$\psi_{\rm{basis}}(r)$")
            else:
                ax[0].plot(self.r, basis_scaling * snapshots,
                           alpha=snapshot_alpha, color="green", linestyle="-.", linewidth=1.8)
        
        try:
            if self.solver.zeta == 1:
                ax[0].plot(self.r, FOM_chi,
                           alpha=0.8, color="grey", linestyle="-", linewidth=3, label=r"$\chi(r)$")
                ax[0].plot(self.r, ROM_chi,
                           alpha=0.8, color="purple", linestyle=(1, (3, 3)), linewidth=3, label=r"$\tilde{\chi}(r)$")
            else:
                ax[0].plot(self.r, FOM_chi,
                           alpha=0.8, color="grey", linestyle="-", linewidth=3, label=r"$\psi(r)$")
                ax[0].plot(self.r, ROM_chi,
                           alpha=0.8, color="purple", linestyle=(1, (3, 3)), linewidth=3, label=r"$\tilde{\psi}(r)$")
        except:
            ax[0].plot(self.r, FOM_chi,
                       alpha=0.8, color="grey", linestyle="-", linewidth=3)
            ax[0].plot(self.r, ROM_chi,
                      alpha=0.8, color="purple", linestyle=(1, (3, 3)), linewidth=3)
        
        # ax[0].set_xlabel("$r$ (fm)", fontsize=14)
        ax[0].set_xlim(self.r[0], self.r[-1])
        try:
            ax[0].set_ylabel(r"Wave Function [fm$^{-1}$]", fontsize=14)
        except:
            None
        # ax[0].set_ylim(-0.6, 0.6)
        ax[0].legend(fontsize=8.5, framealpha=0)
        
        # error plot
        if plot_title_2 is None:
            ax[1].set_title("Error in Emulated Wavefunction", fontsize=16)
        else:
            ax[1].set_title(plot_title_2, fontsize=16)
        
        ax[1].plot(self.r, np.abs(FOM_chi - ROM_chi),
                   alpha=1.0, color="purple", linestyle="-", linewidth=3, label="Before Matching")
        
        ax[1].set_xlabel("$r$ [fm]", fontsize=14)
        ax[1].set_xlim(self.r[0], self.r[-1])
        try:
            if self.solver.zeta == 1:
                ax[1].set_ylabel(r"$|\chi(r) \, - \, \tilde{\chi}(r)|$", fontsize=14)
            else:
                ax[1].set_ylabel(r"$|\psi(r) \, - \, \tilde{\psi}(r)|$", fontsize=14)
        except:
            None
        # ax[1].set_ylim(1e-8, 5e-5)
        ax[1].set_yscale("log")
        
        if print_errors:
            exact_error = np.linalg.norm(FOM_chi - ROM_chi)
            print(f"    exact error = {exact_error:.4e}")
            print(f"estimated error = {estimated_error:.4e}")
        if show:
            plt.show()
        else:
            return fig, ax
    
    def convergence_plot(self, 
                         dpi=300, 
                         initialize_plot=True, 
                         plot_title=None, 
                         decorate=True, show=True,
                         use_scaled_estimated_errors=True,
                         max_label=r"$||\varepsilon||_{max}$", max_color="purple", max_linestyle="-",
                         avg_label=r"$||\varepsilon||_{average}$", avg_color="orange", avg_linestyle="-",
                         ylim=None):
        from matplotlib import pyplot as plt
        
        max_error = []
        avg_error = []
        snapshots = []
        start = 1 if self.preset_snapshots == 0 else self.preset_snapshots
        for i, snapshot in enumerate(np.arange(start - 1, self.snapshot_max)):
            if use_scaled_estimated_errors:
                max_error.append(np.max(self.scaled_estimated_errors[snapshot, :]))
                avg_error.append(np.mean(self.scaled_estimated_errors[snapshot, :]))
            else:
                max_error.append(np.max(self.scaled_estimated_errors[snapshot, :]))
                avg_error.append(np.mean(self.scaled_estimated_errors[snapshot, :]))
            snapshots.append(snapshot + 1)
        
        if initialize_plot:
            plt.figure(dpi=dpi)
            if plot_title:
                plt.title(plot_title, fontsize=14)
            else:
                plt.title("Convergence Plot", fontsize=14)
        
        plt.plot(snapshots, max_error, color=max_color, linestyle=max_linestyle, label=max_label,
                 linewidth=2, marker=".", markersize=15)
        plt.plot(snapshots, avg_error, color=avg_color, linestyle=avg_linestyle, label=avg_label,
                 linewidth=2, marker="*", markersize=15)
        
        if decorate:
            plt.xlabel("Basis Size", fontsize=12)
            plt.xticks(snapshots)
            plt.xlim(np.min(snapshots) - 0.03, np.max(snapshots) + 0.03)
            plt.ylabel("Corrected Estimated Error", fontsize=12)
            plt.yscale("log")
            if ylim is not None:
                plt.ylim(*ylim)
            plt.legend(loc="upper right", framealpha=0, fontsize=10)
        if show:
            plt.show()
    
    def rainbow_plot(self, 
                     plot_parameter,
                     plot_elements=False,
                     dpi=300, 
                     x_buffer=0, 
                     ylims=None,
                     plot_title=None, 
                     y_label=None,
                     show=True, 
                     messy=False):
        from matplotlib import pyplot as plt
        from matplotlib import colormaps
        cmap = colormaps.get_cmap('rainbow')
        
        start = 0 if self.preset_snapshots == 0 else self.preset_snapshots - 1  # or else this pulls the last index
        color_values = np.linspace(0, 1, self.snapshot_max - start)[::-1]
        
        fig, ax = plt.subplots(1, 1, dpi=dpi, figsize=(7, 5))
        for counter, i in enumerate(np.arange(start, self.snapshot_max)):
            mapped_color_value = color_values[counter]
            
            if not plot_elements:
                # error plots
                if plot_title is None:
                    if messy:
                        ax.set_title("Iterative Errors", fontsize=14)
                    else:
                        ax.set_title(f"Iterative Estimated Errors for {self.emulation_method}", fontsize=14)
                else:
                    ax.set_title(plot_title, fontsize=14)
                
                if messy:
                    # ax.fill_between(self.parameter_space[plot_parameter],
                    #                 self.upper_error_bounds[i, :],
                    #                 self.lower_error_bounds[i, :],
                    #                 color=cmap(mapped_color_value), alpha=0.2, zorder=0)
                    ax.plot(self.parameter_space[plot_parameter], self.exact_errors[i, :],
                            color=cmap(mapped_color_value), linewidth=2, alpha=0.5, linestyle="--", zorder=1)
                    if self.use_scaled_estimated_error:
                        ax.plot(self.parameter_space[plot_parameter], self.scaled_estimated_errors[i, :],
                                color=cmap(mapped_color_value), linewidth=2, alpha=0.5, linestyle=":", zorder=2)
                    else:
                        scale_factor = np.max(self.exact_errors[i, :]) / np.max(self.estimated_errors[i, :])
                        ax.plot(self.parameter_space[plot_parameter], scale_factor * self.estimated_errors[i, :],
                                color=cmap(mapped_color_value), linewidth=2, alpha=0.5, linestyle=":", zorder=2)
                ax.plot(self.parameter_space[plot_parameter], self.estimated_errors[i, :],
                        color=cmap(mapped_color_value), linewidth=2.2, alpha=0.8, zorder=3)
        
        # accuracy threshold line
        if self.cutoff_accuracy is not None:
            ax.axhline(self.cutoff_accuracy, color="black", linestyle=(1, (3, 5)), linewidth=1, label="Accuracy Cutoff")
        
        ax.set_xlabel(f"${plot_parameter}$ [MeV]")
        ax.set_xlim(np.min(self.parameter_space[plot_parameter]) - x_buffer,
                    np.max(self.parameter_space[plot_parameter]) + x_buffer)
        ax.set_yscale("log")
        if ylims is None:
            ax.set_ylim(1e-10, 1e4)
        else:
            ax.set_ylim(*ylims)
        if messy:
            if y_label is None:
                ax.set_ylabel("Errors")
            
            # these have all of their plots as plt.plot(x, 1e32 + y, label="some-label") because it's just so that
            # the colors are gray for the legend,
            # and so there isn't a ton of labels made for each iteration
            legend_color = "silver"
            ax.plot(self.parameter_space[plot_parameter], 1e32 + self.exact_errors[start, :],
                    color=legend_color, linewidth=1.5, alpha=1.0, linestyle="--", label="Exact Errors")
            if self.use_scaled_estimated_error:
                ax.plot(self.parameter_space[plot_parameter], 1e32 + self.scaled_estimated_errors[start, :],
                        color=legend_color, linewidth=1.5, alpha=1.0, linestyle=":", label="Scaled Estimated Errors")
            else:
                scale_factor = np.max(self.exact_errors[start, :]) / np.max(self.estimated_errors[start, :])
                ax.plot(self.parameter_space[plot_parameter], 1e32 + scale_factor * self.estimated_errors[start, :],
                        color=legend_color, linewidth=1.5, alpha=1.0, linestyle=":", label="Scaled Estimated Errors")
            ax.plot(self.parameter_space[plot_parameter], 1e32 + self.estimated_errors[start, :],
                    color=legend_color, linewidth=2.5, alpha=1.0, label="Estimated Errors")
            ax.legend(loc="lower left", fontsize=8, framealpha=0.95)
        else:
            if y_label is None:
                ax.set_ylabel("Estimated Error (fm$^{-1}$)")
        if y_label is not None:
            ax.set_ylabel(y_label)
        if show:
            plt.show()
