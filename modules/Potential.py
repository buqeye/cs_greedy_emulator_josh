# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)

###   ###   ###   imports   ###   ###   ###
import yaml
import numpy as np
from functools import cached_property


path_to_potential_data = "./potential_data/"


"""
This class will create a potential in coordinate-space. This potential class can "generate" the 
Minnesota potential, or a chiral potential. The chiral potential is using generated using a cython 
& C++ code that was originally built by Ingo Tews and slightly altered by Dr. Drischler. This chiral 
potential is merely imported and used.
"""


def chiral_affine_outside_of_class(x, 
                                   potId: int = 213, 
                                   l: int = 0, 
                                   ll: int = 0, 
                                   j: int = 0, 
                                   S: int = 0, 
                                   channel=0):
    r"""Affine implementation of GT+ Chiral Potential. Used for testing.
    
    This is a modified function built by Dr. Drischler. It builds the chiral potential using 
    Ingo Tews cpp and cython code that was slightly modified by Dr. Drischler.
    
    Parameters
    ----------
    x : number
        The coordinate space point that this function will be evaluated at. Notably, this function 
        is _not_ vectorized.
    chan : Channel object
        This is a Channel object as constructed in chiral_constriction/Channel. This interfaces with 
        the c++ code so the data types matter significantly.
    potId : int
        The "ID" for the potential. It should be made up of three ints "strung" together.
        potID = int1 int2 int3
        where,
        int1 is the [order] ({"LO": 0, "NLO": 1, "N2L0": 2)
        int2 is the [cutoff]
        int3 is the [sfr cutoff]
        
    Returns
    -------
    ret: numpy array
        An array of the chiral potential outputs _without_ their parameter coefficients.
    """
    from chiral_construction.Channel import Channel
    from chiral_construction import chiralPot
    
    chan = Channel(S=S, L=l, LL=ll, J=j, channel=channel)  # build the "channel" object from Dr. Drischler's code
    
    # Interface with the c++ via the channel object.
    channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
    
    ret = np.zeros(12, dtype=np.double)  # get the array that chiralPot will populate
    chiralPot.Vrlocal_affine(x, potId, channel, ret)  # populate ret & coeffs for the potential
    return ret


class Potential:
    def __init__(self,
                 label: str,
                 r,
                 l: int = 0,
                 ll: int = 0,
                 j: int = 0,
                 S: int = 0,
                 potId: int = 213,
                 channel: int = 0,
                 chiral_flags=None):
        r"""Nuclear Potential(s) for use with the FOM & ROM classes.
        
        Currently, the Minnesota potential (with and without an arbitrary zero constant), 
        as well as the local GT+ Chiral potential are implemented. This class takes advantage of the
        nuclear potential's affine structure to accelerate computation.
        
        Parameters
        ----------
        label : string
            The name of the potential being defined. Currently, only "minnesota" and "chiral".
            One can use `name=minnesota-no-const` to access the minnesota potential _without_ the
            arbitrary zero constant term.
        r : array-like
            The coordinate-space mesh for the potential.
        l : int (optional)
            The orbital angular momentum quantum number, \ell.
        ll : int (optional)
            The final orbital angular momentum quantum number, \ell'.
        j : int (optional)
            The total orbital angular momentum quantum number, j.
        S : int (optional)
            The spin quantum number of the system, S.
        potId : int (optional)
            Specification of the chiral potential leading order as well as the real- and momentum-
            space cutoffs.
        channel: int (optional)
            Denoted the type of scattering:
                -1 for nn scattering
                 0 for np scattering  (NOTE: This code has only been used for np scattering)
                 1 for pp scattering
        chiral_flags: list
            A list of strings to denote specifications for the chiral potential, such as
            what set of LECs to use.
        """
        ###   ###   ###   variable definitions   ###   ###   ###
        self.name = label
        
        ###   ###   ###   initial error checks   ###   ###   ###
        self.acceptable_potentials = ["minnesota", "minnesota-no-const", "chiral"]
        if self.name not in self.acceptable_potentials:
            raise ValueError(f"Potential must be one of the following: {self.acceptable_potentials}")
        
        ###   ###   ###   coordinate-space mesh   ###   ###   ###
        self.r = r
        self.n = len(self.r)
        self.dr = r[1] - r[0]  # This assumes a uniform mesh. That may not always be the case
        
        self.l = l
        self.ll = ll
        self.j = j
        self.S = S
        self.potId = potId
        self.chiral_flags = chiral_flags
        
        ###   ###   ###   define potential based on input   ###   ###   ###
        if self.name == "minnesota":
            # define values according to the best-fit Minnesota potential parameters
            # non-affine parameters:
            kappa_r = 1.487  # fm ^(-2)
            kappa_s = 0.465  # fm ^(-2)
            
            with open(f"{path_to_potential_data}/minnesota/minnesota-coeffs.yaml", "r") as file:
                self.import_parameters = yaml.safe_load(file)
            
            self.theta = self.import_parameters.copy()
            self.parameter_names = list(self.import_parameters.keys())
            self.parameter_defaults = list(self.import_parameters.values())
            
            self.const_piece = True 
            self.number_of_parameters = 3
            # build Gaussians once since this is (treated as) an affine potential
            self.parameter_independent_array = np.empty((self.n, self.number_of_parameters))
            self.parameter_independent_array[:, 0] = np.zeros_like(self.r)  # arbitrary zero constant term
            self.parameter_independent_array[:, 1] = np.exp(-kappa_r * self.r ** 2)
            self.parameter_independent_array[:, 2] = np.exp(-kappa_s * self.r ** 2)
        
        elif self.name == "minnesota-no-const":
            """This is a legacy version of the Minnesota potential.
            This potential is the same as `self.name = "minnesota", but without the flat-zero constant term.
            """
            
            # define values according to the best-fit Minnesota potential parameters
            # non-affine parameters:
            kappa_r = 1.487  # fm ^(-2)
            kappa_s = 0.465  # fm ^(-2)
            
            with open(f"{path_to_potential_data}/minnesota/minnesota-no-const-coeffs.yaml", "r") as file:
                self.import_parameters = yaml.safe_load(file)
            
            self.theta = self.import_parameters
            self.parameter_names = list(self.import_parameters.keys())
            self.parameter_defaults = list(self.import_parameters.values())
            
            self.const_piece = False
            self.number_of_parameters = 2
            # build gaussians once since this is (treated as) an affine potential
            self.parameter_independent_array = np.empty((self.n, self.number_of_parameters))
            self.parameter_independent_array[:, 0] = np.exp(-kappa_r * self.r ** 2)
            self.parameter_independent_array[:, 1] = np.exp(-kappa_s * self.r ** 2)
            
            self.theta = self.import_parameters.copy()
        
        elif self.name == "chiral":
            # The first thing to do is get the relevant functions defined for chiral interactions:
            from chiral_construction.Channel import Channel
            from chiral_construction import chiralPot
            
            self.potId = potId
            self.channel = channel
            
            # dicts for pulling the right yaml file
            potId_map_order = {"0": "0",
                               "1": "1",
                               "2": "2"}  # 0 for 0th leading-order (LO), 1 for 1st leading-order (NLO), 2 for 2nd leading-order (N2LO)
            potId_map_cutoff = {"0": "0.8",
                                "1": "1.0",
                                "2": "1.2",
                                "3": "0.9",
                                "4": "1.1"}  # 0.8fm=0, 1.0fm=1, 1.2fm=2, 0.9fm=3, 1.1fm=4
            potId_map_SFRcutoff = {"2": "800",
                                   "3": "1000",
                                   "4": "1200",
                                   "5": "1400"}  # 800MeV=2, 1000MeV=3, 1200MeV=4, 1400MeV=5.
            self.order = potId_map_order[str(self.potId)[0]]
            self.cutoff = potId_map_cutoff[str(self.potId)[1]]
            self.SFR_cutoff = potId_map_SFRcutoff[str(self.potId)[2]]
            if (self.cutoff == "0.9") or (self.cutoff == "1.1"):
                print(f"Warning: The potential using potId={self.potId} may not exist or are shown to be obsolete.")
            
            # use `Channel` object to "talk" between python and cython
            self.channel = Channel(S=self.S, L=self.l, LL=self.ll, J=self.j, channel=self.channel)
            
            old_lagrangian_lecs = ["C0", "CS", "CT",
                                   "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                   "CNN", "CPP"]  # Because the "legacy" ordering has to be used to fix some stuff.
            lagrangian_lecs = ["C0", "CS", 
                               "CNN", "CPP", "CT",
                               "C1", "C2", "C3", "C4", "C5", "C6", "C7"]  # these are just in a _slightly_ different order than old_lagrangian_lecs
            auxiliary_lecs = ["d0", "d11", "d22", 
                              "dNN", "dPP",
                              "d1", "d2", "d3", "d4", "d5", "d6", "d7"]  # these are here to get the lagrangian and spectroscopic lecs to play nice together
            spectroscopic_lecs = ["d0",
                                  "d11_np", "d22_np", "d22_nn", "d22_pp",
                                  "D_1S0", "D_3S1",
                                  "d2",
                                  "D_3D1", "Dpr_3D1",
                                  "D_1P1",
                                  "D_3P0", "Dpr_3P0",
                                  "D_3P1", "Dpr_3P1",
                                  "D_3P2", "Dpr_3P2",
                                  "d7",
                                  "D_3F2", "Dpr_3F2"]
            
            def chiral_affine(r_value, chan, potId: int = self.potId):
                r"""Constructs the GT+ Chiral potential's parameter independent components.
                
                This is a modified function built by Dr. Drischler. It builds the chiral potential 
                using Ingo Tews cpp and cython code that was slightly modified by Dr. Drischler.
                This finds the parameter independent "basis" vectors for the lagrangian lecs.
                
                Parameters
                ----------
                r_value : number
                    The coordinate space point that this function will be evaluated at. Notably, 
                    this function is _not_ vectorized.
                chan : Channel
                    This is a Channel object as constructed in chiral_constriction/Channel. This 
                    interfaces with the C++ code so the data types matter significantly.
                potId : int
                    The "ID" for the potential. It should be made up of three ints "strung" together.
                    potID = int1 int2 int3
                    where,
                    int1 is the [order] ({"LO": 0, "NLO": 1, "N2L0": 2)
                    int2 is the [cutoff]
                    int3 is the [sfr cutoff]
                    
                Returns
                -------
                ret: numpy array
                    An array of the chiral potential outputs _without_ their parameter coefficients.
                """
                # Interface with the c++ via the `Channel` object.
                channel = chiralPot.Channel(S=chan.S, L=chan.L, LL=chan.LL, J=chan.J, channel=chan.channel)
                ret = np.zeros(12, dtype=np.double)  # get the array that chiralPot will populate
                chiralPot.Vrlocal_affine(r_value, potId, channel, ret)  # populate ret
                return ret
            
            # fun with yamls --
            file_to_open = f"localGT+_lecs_order_{self.order}_R0_{self.cutoff}_lam_{self.SFR_cutoff}.yaml"
            with open(f"{path_to_potential_data}/chiral/{file_to_open}", "r") as file:
                temp_import_parameters = yaml.safe_load(file)
                # filter out non-parameter values from yalm file
                del temp_import_parameters["potId"]
                del temp_import_parameters["order"]
                del temp_import_parameters["R0"]
                del temp_import_parameters["lambda"]
                # write the parameters, including C0
                temp_import_parameters = {**{"C0": 1.}, **temp_import_parameters}  # get C0 to be included & first
            
            # define theta-independent pieces
            self.const_piece = True
            lagrangian_chiral_components = []
            
            # r must be looped over since `chiral_affine` is not vectorized
            for x in self.r:
                chiral_affine_output = chiral_affine(x, self.channel)
                lagrangian_chiral_components.append(chiral_affine_output)
            
            # fix orderings of LECs
            old_lagrangian_parameter_independent_array = np.array(lagrangian_chiral_components)
            old_lagrangian_nu_dict = {}  # use a dict to keep each nu "coupled" to its parameter (for fixing order)
            for i, legacy_parameter in enumerate(old_lagrangian_lecs):
                old_lagrangian_nu_dict[legacy_parameter] = old_lagrangian_parameter_independent_array[:, i]
            if self.r[0] <= 1e-24:  # some epsilon
                # the c++ code will return problematic things when r=0, so let's fix that
                old_lagrangian_parameter_independent_array[0, 0] = 0
            
            # real quick get the parameters in the "right" order
            self.import_parameters = {}
            self.lagrangian_parameter_independent_array = []
            for parameter in lagrangian_lecs:
                self.import_parameters[parameter] = temp_import_parameters[parameter]
                self.lagrangian_parameter_independent_array.append(old_lagrangian_nu_dict[parameter])
            self.lagrangian_parameter_independent_array = np.array(self.lagrangian_parameter_independent_array).T
            
            # okay, now that that's finally done, let's define some stuff
            self.parameter_names = list(self.import_parameters.keys())
            self.parameter_defaults = list(self.import_parameters.values())
            
            # now let's see if there was anything particularly defined to make this potential correctly
            self.chiral_flags = chiral_flags
            if self.chiral_flags is None:
                # "base case" of spectroscopic lecs
                self.parameter_names = lagrangian_lecs
                self.parameter_defaults = self.parameter_defaults
                self.parameter_independent_array = self.lagrangian_parameter_independent_array
            else:
                if "use-lagrangian-lecs" in self.chiral_flags:
                    self.parameter_names = lagrangian_lecs
                    self.parameter_defaults = self.parameter_defaults
                    self.parameter_independent_array = self.lagrangian_parameter_independent_array
                elif "use-auxiliary-lecs" in self.chiral_flags:
                    self.parameter_names = auxiliary_lecs
                    self.parameter_defaults = (lagrangian_to_auxiliary
                                               @ self.parameter_defaults)
                    self.parameter_independent_array = (lagrangian_to_auxiliary
                                                        @ self.lagrangian_parameter_independent_array.T).T
                elif "use-spectroscopic-lecs" in self.chiral_flags:
                    self.parameter_names = spectroscopic_lecs
                    self.parameter_defaults = (lagrangian_to_spectroscopic
                                               @ self.parameter_defaults)
                    self.parameter_independent_array = (lagrangian_to_spectroscopic
                                                        @ self.lagrangian_parameter_independent_array.T).T
                else:
                    raise ValueError("lecs not defined.\nUsable chiral_flags for lec definitions: \"use-lagrangian-lecs\", \"use-auxiliary-lecs\", \"use-spectroscopic-lecs\"")
                self.theta = {}
                for i, parameter in enumerate(self.parameter_names):
                    self.theta[parameter] = self.parameter_defaults[i]
            self.number_of_parameters = len(self.parameter_defaults)
            self.const_piece = True
        else:
            raise ValueError("Potential not recognized")
        
        # define a default theta (_that is not to be overwritten!_)
        self._default_theta = {}
        for value, parameter in zip(self.parameter_defaults, self.parameter_names):
            self._default_theta[parameter] = value
    ###################################################################################################################
    @cached_property
    def default_theta(self):
        r"""An attempt at a fail-safe of accidentally writing over the default_theta value.
        
        We want to keep default_theta un-altered as it is the theta value at _the_ best fit potential
        parameters. The variable `self.theta` is intended for altering the parameters.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        default_theta : dict
            The best-fit parameters of the potential
        """
        return self._default_theta.copy()
    
    def potential_array(self, 
                        theta: dict = None):
        r"""Builds an array based on the potential defined in the function.
        
        Parameters
        ----------
        theta : dict
            The parameter "vector" (dict), theta.
            This is a dict so that the parameters can be passed in any order, and for the ease of only 
            altering one parameters. Taking in a list/array of number is error prone.
        
        Returns
        -------
        the_potential_array : numpy array
            The one-dimensional array of the potential at the defined theta argument. If no theta argument 
            is given, then the best-fit parameters of the defined potential.
        """
        if theta is None:
            theta = {}  # this will just get populated with the default theta parameters
        
        used_theta = self.default_theta.copy()  # set default so that one doesn't need to provide _all_ parameters
        for param_key in theta:
            used_theta[param_key] = theta[param_key]  # alter any parameters given in function call
        self.theta = used_theta
        
        theta_array = np.array(list(self.theta.values()))  # there is probably a clearer way to go from dict.values() -> np.array()
        the_potential_array = self.parameter_independent_array @ theta_array
        
        return the_potential_array


class DoItYourselfPotential:
    def __init__(self,
                 r,
                 l: int = 0):
        r"""A "build your own" nuclear potential. Given to help guide implementing new NN potentials.
        
        r : array-like
            The coordinate-space mesh for the potential.
        l : int (optional)
            The orbital angular momentum of one nucleon.
        """
        self.name = "custom potential"
        
        # these are needed in the other imports (FOM and ROM)
        self.r = r  # (uniform) mesh
        self.n = len(self.r)
        self.dr = r[1] - r[0]  # This assumes a uniform mesh. That may not always be the case
        self.l = l  # orbital angular momentum quantum number
        self.number_of_parameters = ...  # (this needs to be an int)
        
        # define values according to "the" minnesota potential
        # self.theta = self.import_parameters.copy()
        self.parameter_names = ...  # list of strings; ex: ["V_0", "V_r", "V_s"]
        self.parameter_defaults = ...  # list of (real) numbers; ex: [0., 200., -91.85]
        
        self.const_piece = True  # (most NN potentials have a constant component)
        # The parameter_independent_array should be the defined such that (parameter_independent_array @ parameters) = the_potential 
        self.parameter_independent_array = np.empty((self.n, self.number_of_parameters))
        self.parameter_independent_array[:, 0] = ...
        self.parameter_independent_array[:, 1] = ...
        ...
        self.parameter_independent_array[:, self.number_of_parameters] = ...
        
        self._default_theta = {}
        for name, parameter in zip(self.parameter_names, self.parameter_defaults):
            self._default_theta[name] = parameter
    ###################################################################################################################
    
    @property
    def default_theta(self):
        r"""An attempt at a fail-safe of accidentally writing over the default_theta value
        
        We want to keep default_theta un-altered as it is the theta value at _the_ best fit potential
        parameters. The variable `self.theta` is intended for altering the parameters.
        """
        return self._default_theta.copy()
    
    def potential_array(self, 
                        theta: dict = None):
        r"""Builds an array based on the potential defined in the function.
        
        Parameters
        ----------
        theta : dict
            The parameter "vector" (dict), theta.
            This is a dict so that the parameters can be passed in any order, and for the ease of only 
            altering one parameters. Taking in a list/array of number is error prone.
        
        Returns
        -------
        the_potential_array : numpy array
            The one-dimensional array of the potential at the defined theta argument. If no theta argument 
            is given, then the best-fit parameters of the defined potential.
        """
        if theta is None:
            theta = {}  # this will just get populated with the default theta parameters
        
        used_theta = self.default_theta.copy()  # set default so that one doesn't need to provide _all_ parameters
        for param_key in theta:
            used_theta[param_key] = theta[param_key]  # alter any parameters given in function call
        self.theta = used_theta
        
        theta_array = np.array(list(self.theta.values()))  # there is probably a clearer way to go from dict.values() -> np.array()
        the_potential_array = self.parameter_independent_array @ theta_array
        
        return the_potential_array




###   these functions are moved below the classes as they are the last thing that one should consider when looking at this file.   ###
def chiral_lagrangian_to_auxiliary():
    """Transformation matrix to go from "lagrangian" LECs to "auxiliary" LECs
    basis: {C0, CT, CS, C1, C2, ..., C7, CNN, CPP}
    """
    #               C0 CT CS C1  C2 C3  C4  C5  C6  C7  CNN  CPP
    mat = np.array([[1, 0, 0, 0,  0, 0,  0,  0,  0,   0,  0,  0],
                    [0, 1, 0, 0,  1, 0,  0,  0,  0,   0,  0,  0],
                    [0, 1, 0, 0, -3, 0,  0,  0,  0,   0,  0,  0],
                    [0, 1, 1, 0, -3, 0,  0,  0,  0,   0,  0,  0],
                    [0, 1, 0, 1, -3, 0,  0,  0,  0,   0,  0,  0],
                    [0, 0, 0, 0,  0, 1, -3,  1, -3,   0,  0,  0],
                    [0, 0, 0, 0,  0, 0,  0,  0,  0,   0,  1, -3],
                    [0, 0, 0, 0,  0, 1,  1, -3, -3,   0,  0,  0],
                    [0, 0, 0, 0,  0, 1,  1,  1,  1,   0,  0,  0],
                    [0, 0, 0, 0,  0, 1, -3, -3,  9,   0,  0,  0],
                    [0, 0, 0, 0,  0, 0,  0,  0,  0, 0.5,  0,  0],
                    [0, 0, 0, 0,  0, 0,  0,  0,  0,   0,  1,  1]])
    return mat

def chiral_auxiliary_to_spectroscopic():
    """Transformation matrix to go from "auxiliary" LECs to "spectroscopic" LECs
    basis: {d0, d11, d22, dNN, dPP, d1, d2, ..., d7}
    """
    #                    d0 d11 d22 dNN dPP  d1     d2  d3  d4 d5     d6      d7
    lin_comb = np.array([[1,  0,  0,  0,  0,  0,      0, 0,  0, 0,      0,      0],
                         [0,  1,  0,  0,  0,  0,      0, 0,  0, 0,      0,      0],
                         [0,  0,  1,  0,  0,  0,      0, 0,  0, 0,      0,      0],
                         [0,  0,  0,  1,  0,  0,      0, 0,  0, 0,      0,      0],
                         [0,  0,  0,  0,  1,  0,      0, 0,  0, 0,      0,      0],
                         [0,  0,  0,  0,  0,  0,      0, 1,  0, 0,      0,     -1],
                         [0,  0,  0,  0,  0,  1,  1 / 3, 0,  0, 0,      0,      0],
                         [0,  0,  0,  0,  0,  0,      1, 0,  0, 0,      0,      0],
                         [0,  0,  0,  0,  0, -1,  1 / 3, 0,  0, 0,      0,      0],
                         [0,  0,  0,  0,  0,  1, 1 / 15, 0,  0, 0,  3 / 5,      0],
                         [0,  0,  0,  0,  0,  0,     -1, 0,  0, 1,      0,      0],
                         [0,  0,  0,  0,  0,  0,      0, 0, -1, 0,      0,      1],
                         [0,  0,  0,  0,  0,  0,      0, 0,  1, 0,  2 / 5, -1 / 5],
                         [0,  0,  0,  0,  0,  0,      0, 0, -1, 0,      0,     -1],
                         [0,  0,  0,  0,  0,  0,      0, 0,  1, 0,  1 / 5,  3 / 5],
                         [0,  0,  0,  0,  0,  0,      0, 0, -1, 0,      0, -1 / 5],
                         [0,  0,  0,  0,  0,  0,      0, 0,  1, 0, -1 / 5, 7 / 25],
                         [0,  0,  0,  0,  0,  0,      0, 0,  0, 0,      0,      1],
                         [0,  0,  0,  0,  0,  0,      0, 0, -1, 0,      0,  1 / 5],
                         [0,  0,  0,  0,  0,  0,      0, 0,  1, 0,  4 / 5, 3 / 25]])
    return lin_comb

def map_between_auxiliary_to_lagrangian(theta: dict, 
                                        lagrangian_to_auxiliary: bool = True):
    r"""Maps dictionary "theta" between "auxiliary" LECs to "lagrangian" LECs.
    
    Parameters
    ----------
    theta : dict
        A dictionary of parameters, as used by Potential, FOM, & ROM. For this function the parameters
        must be either in "lagrangian" LECs or "auxiliary" LECs.
    lagrangian_to_auxiliary : bool (optional)
        When `True`, `theta` is assumed to be in the "lagrangian" LECs and will me mapped to the
        "auxiliary" LECs. When `False, "auxiliary" will be mapped to "lagrangian".
    
    Returns
    -------
    mapped_theta : dict
        A dictionary of parameters, as used fby Potential, FOM, ROM. This is the result after mapping
        to either "auxiliary" LECs or "lagrangian" LECs, depending on the setting of `lagrangian_to_auxiliary`.
    """
    # define dummy dicts defined for dedicated determination of "data", or "DDDDDDD" for short
    lagrangian_dict = {"C0": 1., "CS": 1, "CNN": 1, "CPP": 1, "CT": 1,
                       "C1": 1, "C2": 1, "C3": 1, "C4": 1, "C5": 1, "C6": 1, "C7": 1}
    auxiliary_dict = {"d0": 1, "d11": 1, "d22": 1, "dNN": 1, "dPP": 1,
                      "d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1, "d6": 1, "d7": 1}
    
    # if the given parameters were "lagrangian" LECs, then we want to map them to "auxiliary" LECs
    if lagrangian_to_auxiliary:
        # get a (sorted) fully populated dict of parameters
        theta_copy = {}  # define empty dict, to populate
        for parameter in lagrangian_dict:
            if parameter not in theta:
                theta_copy[parameter] = 0
            else:
                theta_copy[parameter] = theta[parameter]
        
        # now map them
        mapped_parameter_values = (chiral_lagrangian_to_auxiliary() @
                                   np.array(list(theta_copy.values())))
        mapped_theta = {}
        for i, parameter in enumerate(auxiliary_dict.keys()):
            parameter_value = mapped_parameter_values[i]
            if parameter_value != 0.:
                mapped_theta[parameter] = parameter_value
    
    # if the given parameters were "auxiliary" LECs, then we want to map them to "lagrangian" LECs
    else:
        # get a (sorted) fully populated dict of parameters
        theta_copy = {}
        for parameter in auxiliary_dict:
            if parameter not in theta:
                theta_copy[parameter] = 0
            else:
                theta_copy[parameter] = theta[parameter]
        
        # now map them
        mapped_parameter_values = (np.linalg.inv(chiral_lagrangian_to_auxiliary()) @
                                   np.array(list(theta_copy.values())))
        mapped_theta = {}
        for i, parameter in enumerate(lagrangian_dict.keys()):
            parameter_value = mapped_parameter_values[i]
            if parameter_value != 0.:
                mapped_theta[parameter] = parameter_value
    return mapped_theta

def map_auxiliary_to_spectroscopic(theta: dict, 
                                   auxiliary_to_spectroscopic: bool = True, 
                                   kept_parameters=None):
    r"""Maps dictionary "theta" between "auxiliary" LECs to "spectroscopic" LECs
    
    Parameters
    ----------
    theta : dict
        A dictionary of parameters, as used by Potential, FOM, ROM. For this function the parameters
        must be either in "spectroscopic" LECs or "auxiliary" LECs.
    auxiliary_to_spectroscopic : bool (optional)
        When `True`, `theta` is assumed to be in the "auxiliary" LECs and will me mapped to the
        "spectroscopic" LECs. When `False, "spectroscopic" will me mapped to "auxiliary".
    kept_parameters : list (optional)
        When given a list of strings of LEC names, only the specified LECs in the list will be used.
        This is to avoid "unexpected" LECs appearing from the mapping, as the mapping here is _not_ 
        one-to-one.
    
    Returns
    -------
    mapped_theta : dict
        A dictionary of parameters, as used by Potential, FOM, ROM. This is the result after mapping
        to either "spectroscopic" LECs or "auxiliary" LECs, depending on the setting of `auxiliary_to_spectroscopic`.
    """
    # define dummy dicts defined for dedicated determination of "data", or "DDDDDDD" for short
    auxiliary_dict = {"d0": 1, "d11": 1, "d22": 1, "dNN": 1, "dPP": 1,
                      "d1": 1, "d2": 1, "d3": 1, "d4": 1, "d5": 1, "d6": 1, "d7": 1}
    spectroscopic_dict = {"d0": 1,
                          "d11_np": 1, 
                          "d22_np": 1,
                          "d22_nn": 1, 
                          "d22_pp": 1,
                          "D_1S0": 1,
                          "D_3S1": 1,
                          "d2": 1,
                          "D_3D1": 1, 
                          "Dpr_3D1": 1,
                          "D_1P1": 1,
                          "D_3P0": 1, 
                          "Dpr_3P0": 1,
                          "D_3P1": 1, 
                          "Dpr_3P1": 1,
                          "D_3P2": 1, 
                          "Dpr_3P2": 1,
                          "d7": 1,
                          "D_3F2": 1, 
                          "Dpr_3F2": 1}
    # if the given parameters were "lagrangian" LECs, then we want to map them to "auxiliary" LECs
    if auxiliary_to_spectroscopic:
        if kept_parameters is None:
            kept_parameters = list(spectroscopic_dict.keys())
        # get a (sorted) fully populated dict of parameters
        theta_copy = {}
        for parameter in auxiliary_dict:
            if parameter in theta:
                theta_copy[parameter] = theta[parameter]
            else:
                theta_copy[parameter] = 0  # to avoid matrix multiplication errors
        
        # now map them
        mapped_parameter_values = (chiral_auxiliary_to_spectroscopic() @
                                   np.array(list(theta_copy.values())))
        # and save them to a new dict, with the new LECs names
        mapped_theta = {}
        for i, parameter in enumerate(spectroscopic_dict.keys()):
            parameter_value = mapped_parameter_values[i]
            if (parameter_value != 0) and (parameter in kept_parameters):
                mapped_theta[parameter] = parameter_value
    else:
        if kept_parameters is None:
            kept_parameters = list(auxiliary_dict.keys())
        # get a (sorted) fully populated dict of parameters
        theta_copy = {}
        for parameter in spectroscopic_dict:
            if parameter not in theta:
                theta_copy[parameter] = 0
            else:
                theta_copy[parameter] = theta_copy[parameter]
        
        # now map them
        mapped_parameter_values = (np.linalg.pinv(chiral_auxiliary_to_spectroscopic()) @
                                   np.array(list(theta_copy.values())))  # the determined individual could get around using a pseudo-inverse here
        # and save them to a new dict, with the new LECs names
        mapped_theta = {}
        for i, parameter in enumerate(auxiliary_dict.keys()):
            parameter_value = mapped_parameter_values[i]
            if (parameter_value != 0) and (parameter in kept_parameters):
                mapped_theta[parameter] = parameter_value
    
    return mapped_theta

def map_lagrangian_to_spectroscopic(theta, 
                                    lagrangian_to_spectroscopic=True, 
                                    kept_parameters=None):
    r"""Wrapper for map_between_auxiliary_to_lagrangian() and map_auxiliary_to_spectroscopic() for convenience.
    
    Parameters
    ----------
    theta : dict
        A dictionary of parameters, as used by Potential, FOM, ROM. For this function the parameters
        must be either in "lagrangian" LECs or "spectroscopic" LECs.
    lagrangian_to_spectroscopic : bool (optional)
        When `True`, `theta` is assumed to be in the "lagrangian" LECs and will me mapped to the
        "spectroscopic" LECs. When `False, "spectroscopic" will be mapped to "lagrangian".
    kept_parameters : list (optional)
        When given a list of strings of LEC names, only the specified LECs in the list will be used.
        This is to avoid "unexpected" LECs appearing from the mapping, as the mapping here is _not_ 
        one-to-one.
    
    Returns
    -------
    mapped_theta : dict
        A dictionary of parameters, as used by Potential, FOM, ROM. This is the result after mapping
        to either "spectroscopic" LECs or "lagrangian" LECs, depending on the setting of `lagrangian_to_spectroscopic`.
    """
    if lagrangian_to_spectroscopic:
        lagrangian_theta = theta.copy()
        auxiliary_theta = map_between_auxiliary_to_lagrangian(lagrangian_theta,
                                                              lagrangian_to_auxiliary=True)
        spectroscopic_theta = map_auxiliary_to_spectroscopic(auxiliary_theta,
                                                             auxiliary_to_spectroscopic=True,
                                                             kept_parameters=kept_parameters)
        return spectroscopic_theta
    else:
        spectroscopic_theta = theta.copy()
        auxiliary_theta = map_auxiliary_to_spectroscopic(spectroscopic_theta,
                                                         auxiliary_to_spectroscopic=False,
                                                         kept_parameters=kept_parameters)
        lagrangian_theta = map_between_auxiliary_to_lagrangian(auxiliary_theta,
                                                               lagrangian_to_auxiliary=False)
        return lagrangian_theta

"""
Here are the matrices needed for mapping between the different chiral lecs. It's more convenient to 
look at (and import/call) matrices instead of functional-versions of these matrices. Plus I can't mess 
up the order of matrices when going between lagrangian and spectroscopic lecs.
"""
lagrangian_to_auxiliary = chiral_lagrangian_to_auxiliary()
auxiliary_to_lagrangian = np.linalg.inv(chiral_lagrangian_to_auxiliary())

auxiliary_to_spectroscopic = chiral_auxiliary_to_spectroscopic()
spectroscopic_to_auxiliary = np.linalg.pinv(chiral_auxiliary_to_spectroscopic())  # the determined individual could get around using a pseudo-inverse here

lagrangian_to_spectroscopic = (chiral_auxiliary_to_spectroscopic() @
                               chiral_lagrangian_to_auxiliary())
spectroscopic_to_lagrangian = (np.linalg.inv(chiral_lagrangian_to_auxiliary()) @
                               np.linalg.pinv(chiral_auxiliary_to_spectroscopic()))  # the determined individual could get around using a pseudo-inverse here


# this is for importing convenience
mapping_dict = {"lagrangian_to_auxiliary": lagrangian_to_auxiliary,
                "auxiliary_to_lagrangian": auxiliary_to_lagrangian,
                "auxiliary_to_spectroscopic": auxiliary_to_spectroscopic,
                "spectroscopic_to_auxiliary": spectroscopic_to_auxiliary,
                "lagrangian_to_spectroscopic": lagrangian_to_spectroscopic,
                "spectroscopic_to_lagrangian": spectroscopic_to_lagrangian}
