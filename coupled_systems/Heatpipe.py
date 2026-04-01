import numpy as np

from models.heatpipe.heat_discretised_model import HeatpipeDiscretised
from models.heatpipe.vapor_discretised_model import vapour_discretised
from models.heatpipe.liquid_discretised_model import liquid_discretised

class Heatpipe:

    def __init__(self, data):
        self.data = data

        self.heatpipe_discretised = HeatpipeDiscretised(data)

        self.calculated_quantities = dict()

    def setup_fluid_models(self):
        T = self.get_heatpipe_temperature()
        self.data["T_HP"] = T
        self.vapour_discretised = vapour_discretised(self.data)
        # self.liquid_discretised = liquid_discretised(self.data)

    def get_heatpipe_temperature(self):
        T = self.calculated_quantities.get("heatpipe_T")
        if T is None:
            T = self.heatpipe_discretised.linear_solve()

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

    # def get_vapour_pressure_drop_profile_analytic(self):
    #     P = self.calculated_quantities.get("vapour_P_analytic")
    #     if P is None:
    #         P = self.vapour_discretised.get_pressure_drop_analytic()

    #         self.calculated_quantities["vapour_P_analytic"] = P

    #     return P
    
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
    
    def solve_vapour_temperature_iteratively(self, Q_start, Q_stop, Q_step):
        Q_cases = np.arange(Q_start, Q_stop, Q_step)
        for ix, Q_tot in enumerate(Q_cases):
            Q_profile = np.repeat(np.array([Q_tot / self.data["N_evap"]]), self.data["N_evap"])
            self.data["Q"] = Q_profile

            self.heatpipe_discretised = heatpipe_discretised(self.data)
            self.heatpipe_discretised.solve()
            T = self.heatpipe_discretised.get_temperature_profile()
            self.data["T_HP"] = T

            if ix == 0:
                self.vapour_discretised = vapour_discretised(self.data)
            else:
                self.vapour_discretised = vapour_discretised(self.data, next_initial_temperature, next_initial_velocity)

            self.vapour_discretised.solve_numeric()
            next_initial_temperature = self.vapour_discretised.get_temperature()
            next_initial_velocity = self.vapour_discretised.get_velocity()

        P = self.vapour_discretised.get_pressure_numeric()
        self.calculated_quantities["vapour_T"] = next_initial_temperature
        self.calculated_quantities["vapour_u"] = next_initial_velocity
        self.calculated_quantities["vapour_P_numeric"] = P

