
import numpy as np
import matplotlib.pyplot as plt

from models.heat_discretised_model import heatpipe_discretised
from models.vapor_discretised_model import vapour_discretised
from models.liquid_discretised_model import liquid_discretised

class Heatpipe:

    def __init__(self, data):
        self.data = data

        self.heatpipe_discretised = heatpipe_discretised(data)
        self.vapour_discretised = vapour_discretised(data)
        self.liquid_discretised = liquid_discretised(data)

        self.calculated_quantities = dict()

    def get_heatpipe_temperature(self):
        T = self.calculated_quantities.get("heatpipe_T")
        if T is None:
            self.heatpipe_discretised.solve()
            T = self.heatpipe_discretised.temperature

            self.calculated_quantities["heatpipe_T"] = T

        return T
    
    def get_vapour_temperature(self):
        T = self.calculated_quantities.get("vapour_T")
        if T is None:
            T_heatpipe = self.get_heatpipe_temperature()
            self.vapour_discretised.T_HP = T_heatpipe

            self.vapour_discretised.solve()
            T = self.vapour_discretised.temperature

            T = self.vapour_discretised.temperature
            P = self.vapour_discretised.pressure
            self.calculated_quantities["vapour_T"] = T
            self.calculated_quantities["vapour_P"] = P

        return T

    def get_vapour_pressure_drop_profile(self, mode="analytic"):
        P = self.calculated_quantities.get("vapour_P" + "_" + mode)
        if P is None:
            T_heatpipe = self.get_heatpipe_temperature()
            self.vapour_discretised.T_HP = T_heatpipe

            if mode == "analytic":
                P = self.vapour_discretised.analytical_pressure_drop()
            elif mode == "numeric":
                self.vapour_discretised.solve()
                T = self.vapour_discretised.temperature
                P = self.vapour_discretised.pressure
                self.calculated_quantities["vapour_T"] = T

            self.calculated_quantities["vapour_P" + "_" + mode] = P

        return P
    
    def get_liquid_pressure_drop_profile(self):
        P = self.calculated_quantities.get("liquid_P")
        if P is None:
            T_heatpipe = self.get_heatpipe_temperature()
            self.liquid_discretised.T_HP = T_heatpipe

            self.liquid_discretised.calculate_pressure_drop_profile()

            P = self.liquid_discretised.pressure
            self.calculated_quantities["liquid_P"] = P

        return P
    
if __name__ == "__main__":

    data = {
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wick": 0.0005,
        "delta_wall": 0.001,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,

        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 20,
        "N_adiabatic": 10,
        "N_cond": 110,

        "adiabatic_radial_flux": False,
        "Temperature_BC": False,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "T_op": 850,

        "k_wick": 45.0,
        "k_wall": 21.7,

        "P_C": 2476,
        "T_C": 856
    }

    Q_total = 1000
    Q = np.repeat(np.array([Q_total / data["N_evap"]], dtype=float), data["N_evap"])
    
    data["Q"] = Q

    HP = Heatpipe(data)

    P = HP.get_liquid_pressure_drop_profile()
    
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    ax.plot(P, color="black")

    plt.show()