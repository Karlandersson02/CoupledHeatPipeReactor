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

    def solve_vapor_discretised(self, ) -> tuple:
        
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

    def coupled_system(self, ui: np.ndarray, Ti: np.ndarray) -> np.ndarray:
        T = np.zeros(len(Ti) + 2)
        T[1:-1] = Ti
        T[0] = Ti[0]
        T[-1] = Ti[-1]
        Tim1 = T[:-2]
        Tip1 = T[2:]
        Tbar = (Ti + Tim1)/2

        u = np.zeros(len(ui) + 2)
        u[1:-1] = ui
        uim1 = u[:-2]
        uip1 = u[2:]

        rho = self.rho(T)
        rhobar = self.rho(Tbar)
        rhoi = rho[1:-1]
        rhoim1 = rho[:-2]
        rhoip1 = rho[2:]

        dxi = np.zeros_like(Ti)
        dxi[:self.N_evap] = self.l_evap / self.N_evap
        dxi[self.N_evap:self.N_adiabatic] = self.l_adiabatic / self.N_adiabatic
        dxi[(self.N_evap + self.N_adiabatic):] = self.l_cond / self.N_cond

        Rei = rhoi * ui * 2*self.r_vapour / self.viscosity_Na
        lami = np.zeros_like(Ti)
        lami[Rei <= 2200] = 64 / Rei[Rei <= 2200]
        lami[Rei > 3000] = 0.316 / Rei[Rei > 3000]**0.25
        lami[2200 < Rei <= 3000] = Rei[2200 < Rei <= 3000] * 1.70088e-5 - 0.00832838               # interpolation (behöver dubbelkollas)

        r2 = (rhoi * (ui + uip1)/2 * ui - rhoim1 * (uim1 + ui)/2 * uim1) + (rhobar * self.h_fg_Na) / Tbar * (Ti - Tim1) + dxi * lami / (2*2*self.r_vapour) * rhobar * ui * np.abs(ui)

        return r2 # temp

if __name__ == "__main__":
    data = {}
    vapor = vapor_discretised(data) 
    T_v, P_v = vapor.solve_vapor_discretised()
