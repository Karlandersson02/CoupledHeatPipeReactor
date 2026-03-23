import numpy as np
from scipy.optimize import root, newton_krylov
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.sodium_properties import calculate_Na_rho_l, calculate_Na_viscosity_l, calculate_Na_h_fg

class liquid_discretised:
    def __init__(self, data):
        self.r_outer  = data.get("r_outer")
        self.delta_wick  = data.get("delta_wick")
        self.delta_gap   = data.get("delta_gap")
        self.delta_wall  = data.get("delta_wall")
        self.r_gap    = self.r_outer - self.delta_wall
        self.r_wick   = self.r_gap   - self.delta_gap
        self.r_vapour = self.r_wick  - self.delta_wick

        self.l_evap  = data.get("l_evap")
        self.l_adia  = data.get("l_adiabatic")
        self.l_cond  = data.get("l_cond")
        self.l_tot = self.l_evap + self.l_adia + self.l_cond

        self.N_wick = data.get("N_wick")
        self.N_wall = data.get("N_wall")
        self.N_R = self.N_wick + self.N_wall

        self.N_evap = data.get("N_evap")
        self.N_adia = data.get("N_adiabatic")
        self.N_cond = data.get("N_cond")
        self.N_Z = self.N_evap + self.N_adia + self.N_cond

        self.T_HP = data.get("T_HP")
        self.h_vap = data.get("h_vap")
        self.h_fg = calculate_Na_h_fg(self.T_HP[-1])

        self.Annular = data.get("Is_annular")
        self.K = data.get("K")


    def get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        T_v = self.T_HP[-1]

        A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        mdot = Qevap / self.h_fg

        mdot = np.concatenate([
            np.repeat(np.array([mdot/self.N_evap]), self.N_evap) * np.arange(self.N_evap),
            np.repeat(np.array([mdot]), self.N_adia),
            np.repeat(np.array([mdot/self.N_cond]), self.N_cond) * np.arange(self.N_cond)[::-1]
        ])
        
        return mdot


    def calculate_K_annular_wick(self):
        self.r_2 = self.r_wick
        self.r_1 = self.r_gap

        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K

    def get_pressure_drop_profile(self):
        if self.Annular == True:
            A_wick = np.pi * (self.r_gap**2 - self.r_wick**2) 
        else:
            A_wick = np.pi * (self.r_wick**2 - self.r_vapour**2)
        
        T_wick = np.mean(np.array(self.T_HP)[:-1].reshape(self.N_Z, self.N_R)[:, :self.N_wick], axis=1) 

        mu_l = calculate_Na_viscosity_l(T_wick)  
        rho_l = calculate_Na_rho_l(T_wick)

        if self.Annular == True:
            K = self.calculate_K_annular_wick()
        else:
            K = self.K

        delta_z = np.concatenate(
            [np.ones(self.N_evap) * self.l_evap / self.N_evap,
            np.ones(self.N_adia) * self.l_adia / self.N_adia,
            np.ones(self.N_cond) * self.l_cond / self.N_cond]
            )
        
        mdot = self.get_mdot()
        
        P = -np.cumsum(mu_l * np.array(mdot) / (rho_l * A_wick * K) * delta_z)
        self.pressure = P

        return P


if __name__ == "__main__":
    data = {
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wall": 0.001,
        "delta_gap": 0.005,
        "delta_wick": 0.0005,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,
        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 20,
        "N_adiabatic": 10,
        "N_cond": 110,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "k_wick": 45.0,
        "k_wall": 21.7,
        "P_C": 2476,
        "T_C": 856,
        "T_HP":[0],
        "Is_annular": True,
        "K":1e-10,
    }   

    lpd = liquid_discretised(data)

    P = lpd.get_pressure_drop_profile()