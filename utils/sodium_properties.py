import numpy as np
from scipy.optimize import brentq

def calculate_Na_rho_l(T):
    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 86. 

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    f     = 275.32
    g     = 511.58
    h     = 0.5
    rho_c = 219
    T_c   = 2503.7

    # kg / m^3
    return rho_c + f*(1 - T/T_c) + g*(1 - T/T_c)**h

def calculate_Na_rho_v(T):
    rho_l       = calculate_Na_rho_l(T)
    DHg         = calculate_Na_h_fg(T)
    gamma_sigma = calculate_gamma_sigma(T)

    # kg / m^3
    return 1 / (DHg/(T*gamma_sigma) + 1/rho_l)

def calculate_rho_cc(T_vapour):
    pc = 1300
    Tc = 818
    R = 361.7
    hfg = 4.182e6

    rho = pc / (R * T_vapour) * np.exp(hfg/R * (1/Tc - 1/T_vapour))
    
    return rho

def calculate_Na_viscosity_l(T_liquid):
    
    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 207.

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    # unit is Pa * s
    return np.exp(-6.4406 - 0.3958 * np.log(T_liquid) + 556.835 / T_liquid)

def calculate_Na_viscosity_v(T_vapour):
    
    # Constants and formula taken from:
    # MODELING OF TRANSIENT HEAT PIPE OPERATION - NASA GRANT NAG-1-392
    # BY Gene T. Colwell, George W, Woodruff 
    # page 190.

    # Should probably be for saturated vapour(?)
    
    # Unit is N * s / m^2
    return 6.083e-9 * T_vapour + 1.2606e-5

def calculate_Na_h_fg(T_vapour):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 65.

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    T_crit_Na = 2503.7

    # Original unit is in kJ / kg, converting it to J / kg.
    return 1e3 * (393.37 * (1 - T_vapour / T_crit_Na) + 4398.6 * (1 - T_vapour / T_crit_Na)**(0.29302))

def calculate_gamma_sigma(T):
    a = 11.9463
    b = -12633.73
    c = -0.4672

    # Pa / K
    return 1e6 * (-b/T**2 + c/T)*np.exp(a + b/T + c*np.log(T))

def calculate_Na_surface_tension(T_liquid):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 65.

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    sigma_0 = 240.5
    n = 1.126
    T_crit_Na = 2503.7

    # Original unit is mN / m, converting it to N / m. 
    return 1e-3 * sigma_0 * (1 - T_liquid / T_crit_Na)**(n)

def calculate_Na_pressure_v(T_vapour):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz 
    # page 55.

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    A = 11.9463
    B = 12633.73
    C = 0.4672

    # Original unit is in MPa, converting it to Pa
    return 1e6 * np.exp(A - B / T_vapour - C * np.log(T_vapour))

def calculate_Na_temperature_v(p_vapour):

    # Numerical inversion of calculate_Na_pressure_v().
    # Pressure in Pa. Valid range: ~2923 Pa (864 K) to ~25.6 MPa (2503.7 K).
    # Returns NaN outside valid range.

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    T_min, T_max = 200, 2503.7
    P_min = calculate_Na_pressure_v(T_min)
    P_max = calculate_Na_pressure_v(T_max)

    def invert_single(p):
        if not np.isfinite(p) or p < P_min or p > P_max:
            return np.nan
        if p == P_min:
            return T_min
        if p == P_max:
            return T_max
        return brentq(lambda T: calculate_Na_pressure_v(T) - p, T_min, T_max)

    return np.vectorize(invert_single)(np.asarray(p_vapour, dtype=float))

def calculate_Na_thermal_conductivity_l(T_liquid):

    # Constants and formula taken from:
    # Thermodynamic and Transport Properties of Sodium Liquid and Vapor by J. K. Fink and L. Leibowitz
    # ANL/RE-95/2, page 181, equation (1).

    # https://researchdata.brighton.ac.uk/id/eprint/258/3/Thermodynamic%20and%20transport%20porperties%20of%20sodium%20liquid%20and%20vapor.pdf

    A = 124.67
    B = -0.11381
    C = 5.5226e-5
    D = -1.1842e-8

    # Units is W/(m * K).
    return A + B * T_liquid + C * T_liquid**2 + D * T_liquid**3


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    T = np.linspace(500, 2000, 100)

    rho_exp = calculate_Na_rho_v(T)
    rho_cc = calculate_rho_cc(T)

    plt.figure(figsize=(12, 8))

    plt.plot(T, rho_exp, color="black")
    plt.plot(T, rho_cc, color="red")
    plt.yscale("log")
    plt.grid(alpha=0.4)
    # plt.ylim([0, 1e-5])
    plt.show()






    # R_UNIVERSAL = 8.314462618          # J / (mol K)
    # M_NA = 0.02298976928               # kg / mol
    # R_NA = R_UNIVERSAL / M_NA          # J / (kg K)
    # GAMMA_MONOATOMIC = 5.0 / 3.0


    # def check_sodium_vapour_ideal_gas(T_vapour, u=None, z=None):
    #     """
    #     Check whether saturated sodium vapour behaves like an ideal gas,
    #     based on the user's own property correlations.

    #     Parameters
    #     ----------
    #     T_vapour : array_like
    #         Vapour temperature [K].
    #     u : array_like or None
    #         Optional vapour velocity [m/s]. If given, Mach numbers are computed.
    #     z : array_like or None
    #         Optional axial coordinate [m] for plotting/reporting.

    #     Returns
    #     -------
    #     results : dict
    #         Dictionary with fields:
    #             T               [K]
    #             p               [Pa]
    #             rho             [kg/m^3]
    #             mu              [Pa s]
    #             R_eff           [J/(kg K)]
    #             Z               [-]
    #             eos_rel_error   [-]
    #             a_T             [m/s]   = sqrt(gamma * R_NA * T)
    #             a_prho          [m/s]   = sqrt(gamma * p / rho)
    #             Ma_T            [-]     optional, if u is given
    #             Ma_prho         [-]     optional, if u is given
    #     """

    #     T = np.asarray(T_vapour, dtype=float)

    #     if np.any(T <= 0.0):
    #         raise ValueError("All temperatures must be positive.")

    #     p = calculate_Na_pressure_v(T)
    #     rho = calculate_Na_rho_v(T)
    #     mu = calculate_Na_viscosity_v(T)

    #     # Effective gas constant implied by the correlations
    #     R_eff = p / (rho * T)

    #     # Compressibility factor relative to ideal sodium vapour
    #     Z = p / (rho * R_NA * T)

    #     # Relative EOS error compared to ideal-gas law
    #     eos_rel_error = (p - rho * R_NA * T) / p

    #     # Two equivalent ideal-gas sound-speed forms if Z = 1
    #     a_T = np.sqrt(GAMMA_MONOATOMIC * R_NA * T)
    #     a_prho = np.sqrt(GAMMA_MONOATOMIC * p / rho)

    #     results = {
    #         "T": T,
    #         "p": p,
    #         "rho": rho,
    #         "mu": mu,
    #         "R_eff": R_eff,
    #         "Z": Z,
    #         "eos_rel_error": eos_rel_error,
    #         "a_T": a_T,
    #         "a_prho": a_prho,
    #     }

    #     if u is not None:
    #         u = np.asarray(u, dtype=float)
    #         if u.shape != T.shape:
    #             raise ValueError("u must have the same shape as T_vapour.")

    #         results["u"] = u
    #         results["Ma_T"] = np.abs(u) / a_T
    #         results["Ma_prho"] = np.abs(u) / a_prho

    #     if z is not None:
    #         z = np.asarray(z, dtype=float)
    #         if z.shape != T.shape:
    #             raise ValueError("z must have the same shape as T_vapour.")
    #         results["z"] = z

    #     return results


    # def print_ideal_gas_report(results, z_tolerance_levels=(0.01, 0.03, 0.05, 0.10)):
    #     """
    #     Print a compact report on how close the vapour is to ideal-gas behaviour.
    #     """

    #     T = results["T"]
    #     Z = results["Z"]
    #     eos_rel_error = results["eos_rel_error"]
    #     R_eff = results["R_eff"]

    #     max_abs_dev_Z = np.max(np.abs(Z - 1.0))
    #     mean_abs_dev_Z = np.mean(np.abs(Z - 1.0))
    #     max_abs_eos_err = np.max(np.abs(eos_rel_error))
    #     mean_abs_eos_err = np.mean(np.abs(eos_rel_error))

    #     print("=== Sodium Vapour Ideal-Gas Check ===")
    #     print(f"R_Na (theoretical)      = {R_NA:.6f} J/(kg K)")
    #     print(f"R_eff range from fits   = [{R_eff.min():.6f}, {R_eff.max():.6f}] J/(kg K)")
    #     print(f"Z range                 = [{Z.min():.6f}, {Z.max():.6f}]")
    #     print(f"max |Z - 1|             = {max_abs_dev_Z:.6e}")
    #     print(f"mean |Z - 1|            = {mean_abs_dev_Z:.6e}")
    #     print(f"max |EOS rel. error|    = {max_abs_eos_err:.6e}")
    #     print(f"mean |EOS rel. error|   = {mean_abs_eos_err:.6e}")
    #     print(f"T range                 = [{T.min():.3f}, {T.max():.3f}] K")

    #     for tol in z_tolerance_levels:
    #         ok = np.all(np.abs(Z - 1.0) <= tol)
    #         print(f"Within ±{100*tol:.1f}% in Z?   {'yes' if ok else 'no'}")

    #     if np.all(np.abs(Z - 1.0) <= 0.03):
    #         print("\nConclusion: the ideal-gas EOS looks very good over this temperature range.")
    #     elif np.all(np.abs(Z - 1.0) <= 0.10):
    #         print("\nConclusion: the ideal-gas EOS looks usable, but deviations are not negligible.")
    #     else:
    #         print("\nConclusion: the ideal-gas EOS is not especially accurate over this range.")

    #     print("\nNote: this checks ideal-gas consistency of p(T) and rho(T).")
    #     print("It does not by itself prove that gamma = 5/3 is exact.")


    # def plot_ideal_gas_check(results):
    #     """
    #     Plot diagnostics for axial 1D vapour model.
    #     """

    #     T = results["T"]
    #     Z = results["Z"]
    #     eos_rel_error = results["eos_rel_error"]
    #     a_T = results["a_T"]
    #     a_prho = results["a_prho"]

    #     x = results.get("z", np.arange(T.size))
    #     xlabel = "Axial position [m]" if "z" in results else "Axial cell index"

    #     fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    #     axs[0, 0].plot(x, T, linewidth=2)
    #     axs[0, 0].set_title("Vapour Temperature")
    #     axs[0, 0].set_xlabel(xlabel)
    #     axs[0, 0].set_ylabel("T [K]")
    #     axs[0, 0].grid(True)

    #     axs[0, 1].plot(x, Z, linewidth=2, label="Z = p / (rho R T)")
    #     axs[0, 1].axhline(1.0, linestyle="--", linewidth=1.5)
    #     axs[0, 1].set_title("Compressibility Factor")
    #     axs[0, 1].set_xlabel(xlabel)
    #     axs[0, 1].set_ylabel("Z [-]")
    #     axs[0, 1].grid(True)
    #     axs[0, 1].legend()

    #     axs[1, 0].plot(x, eos_rel_error, linewidth=2)
    #     axs[1, 0].axhline(0.0, linestyle="--", linewidth=1.5)
    #     axs[1, 0].set_title("Ideal-Gas EOS Relative Error")
    #     axs[1, 0].set_xlabel(xlabel)
    #     axs[1, 0].set_ylabel(r"$(p - \rho R T)/p$ [-]")
    #     axs[1, 0].grid(True)

    #     axs[1, 1].plot(x, a_T, linewidth=2, label=r"$a=\sqrt{\gamma R T}$")
    #     axs[1, 1].plot(x, a_prho, linewidth=2, label=r"$a=\sqrt{\gamma p/\rho}$")
    #     axs[1, 1].set_title("Sound-Speed Comparison")
    #     axs[1, 1].set_xlabel(xlabel)
    #     axs[1, 1].set_ylabel("a [m/s]")
    #     axs[1, 1].grid(True)
    #     axs[1, 1].legend()

    #     if "u" in results:
    #         fig2, ax = plt.subplots(figsize=(8, 5))
    #         ax.plot(x, results["Ma_T"], linewidth=2, label=r"$Ma=u/\sqrt{\gamma R T}$")
    #         ax.plot(x, results["Ma_prho"], linewidth=2, label=r"$Ma=u/\sqrt{\gamma p/\rho}$")
    #         ax.set_title("Mach Number")
    #         ax.set_xlabel(xlabel)
    #         ax.set_ylabel("Ma [-]")
    #         ax.grid(True)
    #         ax.legend()

    #     plt.tight_layout()
    #     plt.show()

    # T_test = np.linspace(700.0, 1200.0, 200)

    # results = check_sodium_vapour_ideal_gas(T_test)
    # print_ideal_gas_report(results)
    # plot_ideal_gas_check(results)