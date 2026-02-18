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

        self.N_wick = data.get("N_wick")
        self.N_wall = data.get("N_wall")
        self.N_R  = self.N_wick + self.N_wall

        # heat transfer coefficient
        self.h_vap = data.get("h_vap")
        self.viscosity_Na = data.get("viscosity_Na")

        self.T_HP = data.get("T_HP")

        self.T_C = data.get("T_C")
        self.P_C = data.get("P_C")

        # Molar gas constant
        self.R = 8.314472 

        # Specific gas constant, 22.990 being the molar mass.
        self.R_Na = self.R / 22.990


    def solve_vapor_discretised(self) -> tuple[np.ndarray, np.ndarray]:
        Gamma, h_fg_Na = self.calculate_mass_flow_and_latent_heat()

        T_v = self.solve_vapor_heat_profile(Gamma, h_fg_Na)

        P_v = self.calculate_pressure_profile(T_v, h_fg_Na)

        return T_v, P_v
    

    def calculate_mass_flow_and_latent_heat(self) -> tuple[np.ndarray, float]:
        T_wick_lv_interface = self.T_HP[::self.N_R]
        T_evap_lv_interface = T_wick_lv_interface[0:self.N_evap]
        T_cond_lv_interface = T_wick_lv_interface[self.N_Z - self.N_cond: ]

        T_v = self.T_HP[-1]

        q_bis_surface = np.zeros(self.N_Z)
        q_bis_surface[0: self.N_evap]           =  self.h_vap * (T_wick_lv_interface[0: self.N_evap]           - T_v)
        q_bis_surface[self.N_Z - self.N_cond: ] = -self.h_vap * (T_wick_lv_interface[self.N_Z - self.N_cond: ] - T_v)

        # Heat transfer surface area density per unit volume.
        # a_W = 2 * np.pi * self.r_vapour * delta_Z / np.pi * self.r_vapour**2 * delta_Z
        a_W = 2 / self.r_vapour

        # Formula from: https://www.osti.gov/servlets/purl/94649
        T_crit_Na = 2503.7
        h_fg_Na = 393.37 * (1 - T_v / T_crit_Na) * 4398.6 * (1 - T_v / T_crit_Na)**(0.29302)

        Gamma = a_W * q_bis_surface / h_fg_Na

        return Gamma, h_fg_Na
    

    def calculate_rho(self, T: np.ndarray, h_fg_Na: float) -> np.ndarray:
        rho = ( self.P_C / (self.R * T) ) * np.exp(h_fg_Na * self.R * (1 / self.T_C - 1 / T))

        return rho


    def solve_vapor_heat_profile(self, Gamma: np.ndarray, h_fg_Na: float) -> np.ndarray:
        initial_guess = np.zeros(self.N_Z + self.N_Z + 1)

        coupled_system_lambda = lambda S: self.coupled_system(S[:(self.N_Z - 1)], S[(self.N_Z - 1):], Gamma, h_fg_Na) 

        sol_krylov = root(
            coupled_system_lambda, 
            initial_guess,
            method="krylov",
            options={
                "disp": True,
                "maxiter": 350,   # outer iterations
                "fatol": 1e-6,  # residual tolerance
                # You *can* set inner method here too, but leaving default shows "krylov" usage.
                # "method": "lgmres",
            },
        )

        print("\n[root/krylov] success:", sol_krylov.success)
        print("[root/krylov] ||F|| =", np.linalg.norm(sol_krylov.fun))

        return sol_krylov.x[:self.N_Z]
    

    def calculate_pressure_profile(self, T_v: np.ndarray, h_fg_Na: float) -> np.ndarray:
        rho_v = self.calculate_rho(T_v, h_fg_Na)

        return rho_v * self.R_Na * T_v


    def coupled_system(self, ui: np.ndarray, Ti: np.ndarray, Gami: np.ndarray, h_fg_Na: float) -> np.ndarray:
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

        rho = self.calculate_rho(T, h_fg_Na)
        rhobar = self.calculate_rho(Tbar, h_fg_Na)
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

        r2 = (rhoi * (ui + uip1)/2 * ui - rhoim1 * (uim1 + ui)/2 * uim1) + (rhobar * h_fg_Na) / Tbar * (Ti - Tim1) + dxi * lami / (2*2*self.r_vapour) * rhobar * ui * np.abs(ui)
        r1 = np.zeros(len(r2) + 2)
        r1[1:-1] = (uip1*rhoi - ui*rhoim1) - dxi * Gami
        r1[0] = ui[0]*rho[0] - dxi[0]*Gami[0]

        return np.concatenate([r1, r2])


if __name__ == "__main__":
    data = {}
    vapor = vapor_discretised(data) 
    T_v, P_v = vapor.solve_vapor_discretised()
