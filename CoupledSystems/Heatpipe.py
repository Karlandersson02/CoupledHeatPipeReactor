
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

from models.heat_discretised_model import heatpipe_discretised
from models.vapor_discretised_model import vapour_discretised
from models.liquid_discretised_model import liquid_discretised
from visualisation.visualise_discretised_results import display_temperature_distribution

import seaborn as sns

# Physical constants
gamma = 7 / 5
R_g   = 361.6  # J kg^-1 K^-1, specific gas constant for Na

class Heatpipe:

    def __init__(self, data):
        self.data = data

        self.heatpipe_discretised = heatpipe_discretised(data)

        self.calculated_quantities = dict()

    def setup_fluid_models(self):
        T = self.get_heatpipe_temperature()
        self.data["T_HP"] = T
        self.vapour_discretised = vapour_discretised(self.data)
        self.liquid_discretised = liquid_discretised(self.data)

    def get_heatpipe_temperature(self):
        T = self.calculated_quantities.get("heatpipe_T")
        if T is None:
            self.heatpipe_discretised.solve()
            T = self.heatpipe_discretised.get_temperature_profile()

            self.calculated_quantities["heatpipe_T"] = T

        return T
    
    def get_vapour_temperature(self):
        T = self.calculated_quantities.get("vapour_T")
        if T is None:
            self.vapour_discretised.solve_numeric()
            T = self.vapour_discretised.get_temperature()
            P = self.vapour_discretised.get_pressure_numeric()

            self.calculated_quantities["vapour_T"] = T
            self.calculated_quantities["vapour_P_numeric"] = P

        return T

    def get_vapour_pressure_drop_profile_analytic(self):
        P = self.calculated_quantities.get("vapour_P_analytic")
        if P is None:
            P = self.vapour_discretised.get_pressure_drop_analytic()

            self.calculated_quantities["vapour_P_analytic"] = P

        return P
    
    def get_vapour_pressure_drop_profile_numeric(self):
        P = self.calculated_quantities.get("vapour_P_numeric")
        if P is None:
            self.vapour_discretised.solve_numeric()
            P = self.vapour_discretised.get_pressure_numeric()
            T = self.vapour_discretised.get_temperature()

            P = P - P[0]

            self.calculated_quantities["vapour_T"] = T
            self.calculated_quantities["vapour_P_numeric"] = P

        return P
    
    def get_liquid_pressure_drop_profile(self):
        P = self.calculated_quantities.get("liquid_P")
        if P is None:
            P = self.liquid_discretised.get_pressure_drop_profile()

            self.calculated_quantities["liquid_P"] = P

        return P
    
if __name__ == "__main__":

    data = {
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wick": 0.0005,
        "delta_wall": 0.001,
        "l_evap": 0.1,
        "l_adiabatic": 0.3,
        "l_cond": 0.1,

        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 30,
        "N_adiabatic": 90,
        "N_cond": 30,

        "adiabatic_radial_flux": False,
        "Temperature_BC": False,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "T_op": 800,

        "k_wick": 45.0,
        "k_wall": 21.7,

        "P_C": 2476,
        "T_C": 856
    }

    Q_total = 1000
    Q = np.repeat(np.array([Q_total / data["N_evap"]], dtype=float), data["N_evap"])
    data["Q"] = Q
    HP = Heatpipe(data)
    HP.setup_fluid_models()
    # P = HP.get_vapour_pressure_drop_profile_analytic()
    T = HP.get_vapour_temperature()
    train_history = HP.vapour_discretised.train_history
    T, u = train_history[-1][HP.vapour_discretised.N_Z-1:], train_history[-1][:HP.vapour_discretised.N_Z-1]
    # rey = HP.vapour_discretised.calculate_reynolds(T, u)


    # Q_total2 = 1720
    # Q2 = np.repeat(np.array([Q_total2 / data["N_evap"]], dtype=float), data["N_evap"])
    # data["Q"] = Q2
    # HP2 = Heatpipe(data)
    # HP2.setup_fluid_models()
    # # P2 = HP2.get_vapour_pressure_drop_profile_analytic()
    # T2 = HP2.get_vapour_temperature()
    # train_history2 = HP2.vapour_discretised.train_history
    # T2, u2 = train_history2[-1][HP2.vapour_discretised.N_Z-1:], train_history2[-1][:HP2.vapour_discretised.N_Z-1]
    # # rey2 = HP2.vapour_discretised.calculate_reynolds(T2, u2)

    # mdot = u * HP.vapour_discretised.calculate_rho(T) * HP.vapour_discretised.r_vapour**2 * np.pi
    # mdot2 = u2 * HP.vapour_discretised.calculate_rho(T2) * HP.vapour_discretised.r_vapour**2 * np.pi

    # print(u[-1], u2[-1])

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    ax.plot(T, color="blue")
    # ax.plot(u2, color="red")

    plt.show()


    # Sonic velocity field and Mach-like ratio u/c_0
    # c  = np.sqrt(gamma * R_g * T)
    # c2 = np.sqrt(gamma * R_g * T2)

    # u_over_c  = u  / c
    # u_over_c2 = u2 / c2

    # # Axial coordinate — evaporator section only
    # z_evap = np.linspace(0, data["l_evap"], data["N_evap"])

    # sns.set_theme(style="whitegrid")

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # for ax, uc, Q_val, label in zip(
    #     axes,
    #     [u_over_c, u_over_c2],
    #     [Q_total, Q_total2],
    #     [rf"$Q = {Q_total}$ W", rf"$Q = {Q_total2}$ W"]
    # ):
    #     ax.plot(z_evap * 1e2, uc[:data["N_evap"]], linewidth=2, label=label)
    #     ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5,
    #             alpha=0.7, label=r"Sonic limit ($u/c_0 = 1$)")
    #     ax.set_xlabel(r"$z$ [dm]", fontsize=16)
    #     ax.set_title(label, fontsize=13)
    #     ax.tick_params(labelsize=12)
    #     ax.grid(True, linestyle="--", alpha=0.6)
    #     ax.legend(fontsize=12)

    # axes[0].set_ylabel(r"$u \, / \, c_0 \; [-]$", fontsize=16)

    # fig.suptitle(r"Mach number proxy $u/c_0$ along evaporator", fontsize=15)
    # plt.tight_layout()
    # plt.show()