import numpy as np
from scipy.optimize import root, newton_krylov

class vapor_discretised:
    def __init__(self, data):
        self.r_outer  = data.get("r_outer")
        self.delta_wick  = data.get("delta_wick")
        self.delta_wall  = data.get("delta_wall")
        self.r_vapour = self.r_outer - self.delta_wick - self.delta_wall

        self.l_evap  = data.get("l_evap")
        self.l_adiabatic  = data.get("l_adiabatic")
        self.l_cond  = data.get("l_cond")
        self.l_tot = self.l_evap + self.l_adiabatic + self.l_cond

        self.N_evap = data.get("N_evap")
        self.N_adiabatic = data.get("N_adiabatic")
        self.N_cond = data.get("N_cond")
        self.N_Z = self.N_evap + self.N_adiabatic + self.N_cond

        # heat transfer coefficient
        self.h_vap = data.get("h_vap")

        # Latent heat
        self.h_fg_Na = data.get("h_fg")
        self.viscosity_Na = data.get("viscosity_Na")

        self.T = data.get("T_HP")

        self.T_C = data.get("T_C")
        self.rho_C = data.get("rho_C")

    def solve_vapor_discretised(self, ) -> np.ndarray:
        m_dot = self.calculate_mass_flow()

        T_v = self.solve_vapor_heat_drop(m_dot)

        P_v = self.calculate_pressure_drop(T_v)

        return T_v, P_v
    
    def calculate_mass_flow(self):
        return np.array([0])

    def solve_vapor_heat_drop(self, m_dot):
        return np.array([0])

    def calculate_pressure_drop(self, T_v):
        return np.array([0])
    
if __name__ == "__main__":
    data = {}
    vapor = vapor_discretised(data) 
    T_v, P_v = vapor.solve_vapor_discretised()
