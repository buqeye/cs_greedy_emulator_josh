# Author: Joshua Maldonado
# GitHub: https://github.com/Ub3rJosh
# Email: jm998521@ohio.edu (joshuamaldonado4432@gmail.com)


###   ###   ###   imports   ###   ###   ###
import numpy as np
import pandas
import sys
sys.path.append("./modules")
import Potential, FOM, ROM, Matching


test_values_dict = {}


# define necessary variables for testing
r, dr = np.linspace(0, 12, 1000, retstep=True)  
r[0] = 1e-24  # 1e-24 used here instead of zero to avoid warnings in the test.
energies = [25., 50., 100., 200.]
angular_momenta = [0, 1, 2]


print("testing...")


###   ###   test the minnesota potential   ###   ###
# NOTE: Only the 1S0 channel is physically meaningful. The other "channels" are purely to test the numerics.
best_fit_minnesota_theta = {"V_0": 0., "V_r": 200., "V_s": -91.85}
best_fit_minnesota_theta_vec = [1., 0., 200., -91.85]

for l in angular_momenta:
    for energy in energies:
        minnesota_potential = Potential.Potential("minnesota", r, l=l)
        assert (minnesota_potential.default_theta == best_fit_minnesota_theta), \
            "Best-Fit parameter for Minnesota potential does not pass."
        
        ###    ###   look at solvers   ###   ###
        ###   inhomogeneous solver   ###
        # initialize solvers
        minnesota_solver_numerov_inhomo = FOM.MatrixNumerovSolver(minnesota_potential, 
                                                                  energy=energy, zeta=1)
        minnesota_solver_numerov_ab_inhomo = FOM.MatrixNumerovSolver(minnesota_potential, 
                                                                     energy=energy, zeta=1, use_ab=True)
        
        # calculate wave function
        minnesota_numerov_chi = minnesota_solver_numerov_inhomo.solve(best_fit_minnesota_theta)
        minnesota_numerov_ab_chi = minnesota_solver_numerov_ab_inhomo.solve(best_fit_minnesota_theta)[:-2]  # trim off a, b from solution
        
        # inverse log matching
        minnesota_numerov_chi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                        minnesota_numerov_chi, 
                                                                                        minnesota_solver_numerov_inhomo.p, 
                                                                                        l=l, 
                                                                                        r_match=11, 
                                                                                        derivative_accuracy=8,
                                                                                        scattered_wave=True)
        minnesota_numerov_ab_chi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                            minnesota_numerov_ab_chi, 
                                                                                            minnesota_solver_numerov_ab_inhomo.p, 
                                                                                            l=l, 
                                                                                            r_match=11, 
                                                                                            derivative_accuracy=8,
                                                                                            scattered_wave=True)
        # least-squares matching
        minnesota_numerov_chi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                            minnesota_numerov_chi,
                                                                                            minnesota_solver_numerov_inhomo.p,
                                                                                            l=l,
                                                                                            zeta=1)
        minnesota_numerov_ab_chi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                                minnesota_numerov_ab_chi,
                                                                                                minnesota_solver_numerov_ab_inhomo.p,
                                                                                                l=l,
                                                                                                zeta=1)
        
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={1}, ab=False, method=inv log"] = [minnesota_numerov_chi_inv_log_matching_output["S_l"],
                                                                                                        minnesota_numerov_chi_inv_log_matching_output["K_l"],
                                                                                                        minnesota_numerov_chi_inv_log_matching_output["T_l"],
                                                                                                        minnesota_numerov_chi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={1}, ab=False, method=lst sq"] = [minnesota_numerov_chi_lst_sq_matching_output["S_l"],
                                                                                                       minnesota_numerov_chi_lst_sq_matching_output["K_l"],
                                                                                                       minnesota_numerov_chi_lst_sq_matching_output["T_l"],
                                                                                                       minnesota_numerov_chi_lst_sq_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={1}, ab=True, method=inv log"] = [minnesota_numerov_ab_chi_inv_log_matching_output["S_l"],
                                                                                                       minnesota_numerov_ab_chi_inv_log_matching_output["K_l"],
                                                                                                       minnesota_numerov_ab_chi_inv_log_matching_output["T_l"],
                                                                                                       minnesota_numerov_ab_chi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={1}, ab=True, method=lst sq"] = [minnesota_numerov_ab_chi_lst_sq_matching_output["S_l"],
                                                                                                      minnesota_numerov_ab_chi_lst_sq_matching_output["K_l"],
                                                                                                      minnesota_numerov_ab_chi_lst_sq_matching_output["T_l"],
                                                                                                      minnesota_numerov_ab_chi_lst_sq_matching_output["delta_l"]]
        
        
        
        ###   homogenous solver   ###
        # initialize solvers
        minnesota_solver_numerov_homo = FOM.MatrixNumerovSolver(minnesota_potential, 
                                                                energy=energy, zeta=0)
        minnesota_solver_numerov_ab_homo = FOM.MatrixNumerovSolver(minnesota_potential, 
                                                                   energy=energy, zeta=0, use_ab=True)
        
        # calculate wave function
        minnesota_numerov_psi = minnesota_solver_numerov_homo.solve(best_fit_minnesota_theta)
        minnesota_numerov_ab_psi = minnesota_solver_numerov_ab_homo.solve(best_fit_minnesota_theta)[:-2]  # trim off a, b from solution
        
        # inverse log matching
        minnesota_numerov_psi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                        minnesota_numerov_psi, 
                                                                                        minnesota_solver_numerov_inhomo.p, 
                                                                                        l=l, 
                                                                                        r_match=11, 
                                                                                        derivative_accuracy=8,
                                                                                        scattered_wave=False)
        minnesota_numerov_ab_psi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                            minnesota_numerov_ab_psi, 
                                                                                            minnesota_solver_numerov_ab_inhomo.p, 
                                                                                            l=l, 
                                                                                            r_match=11, 
                                                                                            derivative_accuracy=8,
                                                                                            scattered_wave=False)
        # least-squares matching
        minnesota_numerov_psi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                            minnesota_numerov_psi,
                                                                                            minnesota_solver_numerov_inhomo.p,
                                                                                            l=l,
                                                                                            zeta=0)
        minnesota_numerov_ab_psi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                                minnesota_numerov_ab_psi,
                                                                                                minnesota_solver_numerov_ab_inhomo.p,
                                                                                                l=l,
                                                                                                zeta=0)
        
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={0}, ab=False, method=inv log"] = [minnesota_numerov_psi_inv_log_matching_output["S_l"],
                                                                                                        minnesota_numerov_psi_inv_log_matching_output["K_l"],
                                                                                                        minnesota_numerov_psi_inv_log_matching_output["T_l"],
                                                                                                        minnesota_numerov_psi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={0}, ab=False, method=lst sq"] = [minnesota_numerov_psi_lst_sq_matching_output["S_l"],
                                                                                                       minnesota_numerov_psi_lst_sq_matching_output["K_l"],
                                                                                                       minnesota_numerov_psi_lst_sq_matching_output["T_l"],
                                                                                                       minnesota_numerov_psi_lst_sq_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={0}, ab=True, method=inv log"] = [minnesota_numerov_ab_psi_inv_log_matching_output["S_l"],
                                                                                                       minnesota_numerov_ab_psi_inv_log_matching_output["K_l"],
                                                                                                       minnesota_numerov_ab_psi_inv_log_matching_output["T_l"],
                                                                                                       minnesota_numerov_ab_psi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Minnesota, l={l}, E={energy}MeV, zeta={0}, ab=True, method=lst sq"] = [minnesota_numerov_ab_psi_lst_sq_matching_output["S_l"],
                                                                                                      minnesota_numerov_ab_psi_lst_sq_matching_output["K_l"],
                                                                                                      minnesota_numerov_ab_psi_lst_sq_matching_output["T_l"],
                                                                                                      minnesota_numerov_ab_psi_lst_sq_matching_output["delta_l"]]
#


###   ###   test the GT+ chiral potential   ###   ###
try:
    import chiral_construction.chiralPot
except:
    print("")
    print("Chiral potential import test failed.")
    print("")
    print("Chiral potential is not constructed properly.")
    print("Please compile using instruction on the GitHub page's README: ")
    print("https://github.com/Ub3rJosh/greedy-emulator?tab=readme-ov-file#compiling-the-gt-chiral-potential")
    print("")


best_fit_chiral_theta = {'C0': 1.0, 'CS': 5.4385, 'CNN': 0.04344, 'CPP': 0.062963, 'CT': 0.27672, 'C1': -0.14084, 'C2': 0.04243, 'C3': -0.12338, 'C4': 0.11018, 'C5': -2.11254, 'C6': 0.15898, 'C7': -0.26994}
best_fit_chiral_theta_vec = [1., 1., 5.4385, 0.04344, 0.062963, 0.27672, -0., 0.04243, -0.12338, 0.11018, -2.11254, 0.15898, -0.26994]

for l in angular_momenta:
    for energy in energies:
        chiral_potential = Potential.Potential("chiral", r, l=l, ll=l, S=l, j=0)
        assert (chiral_potential.default_theta == best_fit_chiral_theta), \
            "Best-Fit parameter for Chiral potential does not pass."
        
        ###    ###   look at solvers   ###   ###
        ###   inhomogeneous solver   ###
        # initialize solvers
        chiral_solver_numerov_inhomo = FOM.MatrixNumerovSolver(chiral_potential, 
                                                               energy=energy, zeta=1)
        chiral_solver_numerov_ab_inhomo = FOM.MatrixNumerovSolver(chiral_potential, 
                                                                  energy=energy, zeta=1, use_ab=True)
        
        # calculate wave function
        chiral_numerov_chi = chiral_solver_numerov_inhomo.solve(best_fit_chiral_theta)
        chiral_numerov_ab_chi = chiral_solver_numerov_ab_inhomo.solve(best_fit_chiral_theta)[:-2]  # trim off a, b from solution
        
        # inverse log matching
        chiral_numerov_chi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                      chiral_numerov_chi, 
                                                                                      chiral_solver_numerov_inhomo.p, 
                                                                                      l=l, 
                                                                                      r_match=11, 
                                                                                      derivative_accuracy=8,
                                                                                      scattered_wave=True)
        chiral_numerov_ab_chi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                         chiral_numerov_ab_chi, 
                                                                                         chiral_solver_numerov_ab_inhomo.p, 
                                                                                         l=l, 
                                                                                         r_match=11, 
                                                                                         derivative_accuracy=8,
                                                                                         scattered_wave=True)
        # least-squares matching
        chiral_numerov_chi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                          chiral_numerov_chi,
                                                                                          chiral_solver_numerov_inhomo.p,
                                                                                          l=l,
                                                                                          zeta=1)
        chiral_numerov_ab_chi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                             chiral_numerov_ab_chi,
                                                                                             chiral_solver_numerov_ab_inhomo.p,
                                                                                             l=l,
                                                                                             zeta=1)
        
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={1}, ab=False, method=inv log"] = [chiral_numerov_chi_inv_log_matching_output["S_l"],
                                                                                                     chiral_numerov_chi_inv_log_matching_output["K_l"],
                                                                                                     chiral_numerov_chi_inv_log_matching_output["T_l"],
                                                                                                     chiral_numerov_chi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={1}, ab=False, method=lst sq"] = [chiral_numerov_chi_lst_sq_matching_output["S_l"],
                                                                                                    chiral_numerov_chi_lst_sq_matching_output["K_l"],
                                                                                                    chiral_numerov_chi_lst_sq_matching_output["T_l"],
                                                                                                    chiral_numerov_chi_lst_sq_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={1}, ab=True, method=inv log"] = [chiral_numerov_ab_chi_inv_log_matching_output["S_l"],
                                                                                                    chiral_numerov_ab_chi_inv_log_matching_output["K_l"],
                                                                                                    chiral_numerov_ab_chi_inv_log_matching_output["T_l"],
                                                                                                    chiral_numerov_ab_chi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={1}, ab=True, method=lst sq"] = [chiral_numerov_ab_chi_lst_sq_matching_output["S_l"],
                                                                                                   chiral_numerov_ab_chi_lst_sq_matching_output["K_l"],
                                                                                                   chiral_numerov_ab_chi_lst_sq_matching_output["T_l"],
                                                                                                   chiral_numerov_ab_chi_lst_sq_matching_output["delta_l"]]
        
        
        ###   homogenous solver   ###
        # initialize solvers
        chiral_solver_numerov_homo = FOM.MatrixNumerovSolver(chiral_potential, 
                                                             energy=energy, zeta=0)
        chiral_solver_numerov_ab_homo = FOM.MatrixNumerovSolver(chiral_potential, 
                                                                energy=energy, zeta=0, use_ab=True)
        
        # calculate wave function
        chiral_numerov_psi = chiral_solver_numerov_homo.solve(best_fit_chiral_theta)
        chiral_numerov_ab_psi = chiral_solver_numerov_ab_homo.solve(best_fit_chiral_theta)[:-2]  # trim off a, b from solution
        
        # inverse log matching
        chiral_numerov_psi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                      chiral_numerov_psi, 
                                                                                      chiral_solver_numerov_inhomo.p, 
                                                                                      l=l, 
                                                                                      r_match=11, 
                                                                                      derivative_accuracy=8,
                                                                                      scattered_wave=False)
        chiral_numerov_ab_psi_inv_log_matching_output = Matching.match_using_inverse_log(r, 
                                                                                         chiral_numerov_ab_psi, 
                                                                                         chiral_solver_numerov_ab_inhomo.p, 
                                                                                         l=l, 
                                                                                         r_match=11, 
                                                                                         derivative_accuracy=8,
                                                                                         scattered_wave=False)
        # least-squares matching
        chiral_numerov_psi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                          chiral_numerov_psi,
                                                                                          chiral_solver_numerov_inhomo.p,
                                                                                          l=l,
                                                                                          zeta=0)
        chiral_numerov_ab_psi_lst_sq_matching_output = Matching.matching_using_least_squares(r,
                                                                                             chiral_numerov_ab_psi,
                                                                                             chiral_solver_numerov_ab_inhomo.p,
                                                                                             l=l,
                                                                                             zeta=0)
        
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={0}, ab=False, method=inv log"] = [chiral_numerov_psi_inv_log_matching_output["S_l"],
                                                                                                     chiral_numerov_psi_inv_log_matching_output["K_l"],
                                                                                                     chiral_numerov_psi_inv_log_matching_output["T_l"],
                                                                                                     chiral_numerov_psi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={0}, ab=False, method=lst sq"] = [chiral_numerov_psi_lst_sq_matching_output["S_l"],
                                                                                                    chiral_numerov_psi_lst_sq_matching_output["K_l"],
                                                                                                    chiral_numerov_psi_lst_sq_matching_output["T_l"],
                                                                                                    chiral_numerov_psi_lst_sq_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={0}, ab=True, method=inv log"] = [chiral_numerov_ab_psi_inv_log_matching_output["S_l"],
                                                                                                    chiral_numerov_ab_psi_inv_log_matching_output["K_l"],
                                                                                                    chiral_numerov_ab_psi_inv_log_matching_output["T_l"],
                                                                                                    chiral_numerov_ab_psi_inv_log_matching_output["delta_l"]]
        test_values_dict[f"pot=Chiral, l={l}, E={energy}MeV, zeta={0}, ab=True, method=lst sq"] = [chiral_numerov_ab_psi_lst_sq_matching_output["S_l"],
                                                                                                   chiral_numerov_ab_psi_lst_sq_matching_output["K_l"],
                                                                                                   chiral_numerov_ab_psi_lst_sq_matching_output["T_l"],
                                                                                                   chiral_numerov_ab_psi_lst_sq_matching_output["delta_l"]]
#



unit_test_key = pandas.read_csv("./potential_data/Unit_Test_Key.csv", index_col=0).transpose().to_dict()

for key in test_values_dict:
    if (np.allclose(list(test_values_dict[key]), np.complex128(list(unit_test_key[key].values())), 
                    rtol=1e-8, atol=1e-8)):
         print("passed:  ", end="")
    print(f"{key}")

print("")
print("All tests passed.")
print("")
