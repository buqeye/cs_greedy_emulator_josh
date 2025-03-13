# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)

###   ###   ###   imports   ###   ###   ###
import numpy as np
from scipy.integrate import solve_ivp
from modules.Constants import *
from modules.SpecialFunctions import analytic_phi
from modules.Potential import chiral_affine_outside_of_class


def callable_chiral(r_value, theta, l=0, ll=0, j=0, S=0, channel=0):
    old_lagrangian_lecs = ["C0", "CS", "CT",
                           "C1", "C2", "C3", "C4", "C5", "C6", "C7",
                           "CNN", "CPP"]  # Because the "legacy" ordering has to be used to fix some stuff.
    lagrangian_lecs = ["C0", "CS", "CNN", "CPP", "CT",
                       "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    default_lec_for_podId_213 = {'C0': 1.0,
                                 'CS': 5.4385,
                                 'CNN': 0.04344,
                                 'CPP': 0.062963,
                                 'CT': 0.27672,
                                 'C1': -0.14084,
                                 'C2': 0.04243,
                                 'C3': -0.12338,
                                 'C4': 0.11018,
                                 'C5': -2.11254,
                                 'C6': 0.15898,
                                 'C7': -0.26994}
    theta_vec = []
    for lec in old_lagrangian_lecs:
        if lec in theta:
            theta_vec.append(theta[lec])
        else:
            theta_vec.append(default_lec_for_podId_213[lec])
    potential_at_r = chiral_affine_outside_of_class(r_value, l=l, ll=ll, j=j, S=S, channel=channel) @ theta_vec
    
    return potential_at_r


def callable_minnesota(r_value, theta, l=0, ll=0, j=0, S=0, channel=0):
    V_r = theta["V_r"]
    V_s = theta["V_s"]
    
    kappa_r = 1.487  # fm ^(-2)
    kappa_s = 0.465  # fm ^(-2)
    potential_at_r = (V_r * np.exp(-kappa_r * r_value ** 2) + V_s * np.exp(-kappa_s * r_value ** 2))
    return potential_at_r


def chi_solve_ivp_eqn(r_value,
                      u,
                      theta,
                      callable_potential,
                      l=0,
                      ll=0,
                      j=0,
                      S=0,
                      channel=0,
                      energy=50,
                      zeta=0,
                      mass=neutron_mass / 2):
        """Expression for the full radial schrodinger equation
        with the potential, V, as the minnesota potential with an output style that scipy's solve_ivp() likes.
        """
        p2 = (2 * mass * energy) / hbarc ** 2  # fm ^(-2)
        # p = np.sqrt(p2)
        
        V_at_r = (2 * mass / hbarc ** 2) * callable_potential(r_value,
                                                              theta,
                                                              l=l, ll=ll, j=j, S=S)  # fm
        
        return np.array([
            u[1],
            ((l * (l + 1)) / (r_value ** 2) +
             (V_at_r - p2)) * u[0] +
            zeta * V_at_r * analytic_phi(r_value, energy=energy, mass=mass, l=l)
        ])


class FOM:
    def __init__(self,
                 r,
                 l,
                 ll=0,
                 j=0,
                 S=0,
                 energy=50,
                 mass=neutron_mass / 2,
                 potential_name="minnesota",
                 zeta=1,
                 atol=1e-9,
                 rtol=1e-9):
        self.r = np.copy(r)
        if self.r[0] < 1e-24:
            # print("adjusting first index for scipy's sake")
            self.r[0] = 1e-24
        self.l = l
        self.ll = ll
        self.j = j
        self.S = S
        self.energy = energy
        self.mass = mass
        self.potential_name = potential_name
        if self.potential_name == "minnesota":
            self.potential = callable_minnesota
        elif self.potential_name == "chiral":
            self.potential = callable_chiral
        else:
            raise ValueError("potential not known")
        
        self.p2 = (2 * self.mass * self.energy) / hbarc ** 2  # fm ^(-2)
        self.p = np.sqrt(self.p2)  # p = hbar k but hbar here is 1
        
        self.zeta = zeta
        self.atol = atol
        self.rtol = rtol
    
    def solve(self, theta, return_derivative=False):
        sol = solve_ivp(chi_solve_ivp_eqn,
                        (self.r[0], self.r[-1]),
                        (0, (1 - self.zeta)),
                        args=(theta, self.potential, self.l, self.ll, self.j, self.S, 0, self.energy, self.zeta),
                        method="RK45",
                        atol=self.atol,
                        rtol=self.rtol,
                        t_eval=self.r)
        
        chi = sol.y[0]
        if return_derivative:
            chi_pr = sol.y[1]
            return chi, chi_pr
        return chi
