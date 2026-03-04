
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.heat_discretised_model import heatpipe_discretised
from models.vapor_discretised_model import vapour_discretised
from models.liquid_discretised_model import liquid_discretised
from visualisation.visualise_discretised_results import display_temperature_distribution

class Heatpipe:

    def __init__(self, data):
        self.data = data

        self.heatpipe_discretised = heatpipe_discretised(data)

        self.calculated_quantities = dict()

    def setup_fluid_models(self):
        T = self.get_heatpipe_temperature()
        self.data["T_HP"] = T
        self.vapour_discretised = vapour_discretised(data)
        self.liquid_discretised = liquid_discretised(data)

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
        "T_op": 870,

        "k_wick": 45.0,
        "k_wall": 21.7,

        "P_C": 2476,
        "T_C": 856
    }

    Q_total = 1050
    Q_total2 = 1052
    Q = np.repeat(np.array([Q_total / data["N_evap"]], dtype=float), data["N_evap"])
    Q2 = np.repeat(np.array([Q_total2 / data["N_evap"]], dtype=float), data["N_evap"])
    
    data["Q"] = Q
    data2 = data
    data2["Q"] = Q2

    HP = Heatpipe(data)
    HP.setup_fluid_models()
    HP2 = Heatpipe(data2)
    HP2.setup_fluid_models()

    T = HP.get_vapour_temperature()
    T2 = HP2.get_vapour_temperature()

    train_history = HP.vapour_discretised.train_history
    train_history2 = HP2.vapour_discretised.train_history

    T, u = train_history[-1][HP.vapour_discretised.N_Z-1:], train_history[-1][:HP.vapour_discretised.N_Z-1]
    T2, u2 = train_history2[-1][HP2.vapour_discretised.N_Z-1:], train_history2[-1][:HP2.vapour_discretised.N_Z-1]

    rey = HP.vapour_discretised.calculate_reynolds(T, u)
    rey2 = HP2.vapour_discretised.calculate_reynolds(T2, u2)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)



    # display_temperature_distribution(T, data)
    
    # mpl.rcParams["font.size"] = 22

    # fig = plt.figure(figsize=(16,9))
    # ax = fig.add_subplot(111)
    # ax_twin = ax.twinx()

    # ax.grid(alpha=0.4)
    # ax.plot(P_numeric, color="red", label=rf"Numeric: $\Delta P =$ {P_numeric[-1]:.0f}")
    # ax.plot(P_analytic, color="blue", label=rf"Analytic: $\Delta P =$ {P_analytic[-1]:.0f}")
    # ax.plot(HP.vapour_discretised.P_guess, color="green")
    # ax.plot(T, color="black")
    # ax.plot(reynolds, color="black")
    # ax_twin.plot(lam, color="red")
    # ax.plot(T)

    # ax.set_ylabel(r"$P$")
    # ax.set_xlabel(r"$n$")

    # ax.legend()

    # plt.show()