import numpy as np
from scipy.optimize import root, newton_krylov
import matplotlib.pyplot as plt
import matplotlib as mpl

from models.sodium_properties import calculate_Na_rho_l, calculate_Na_viscosity_l, calculate_Na_h_fg

class liquid_discretised:
    def __init__(self, data):
        self.r_outer  = data.get("r_outer")
        self.delta_wick  = data.get("delta_wick")
        self.delta_wall  = data.get("delta_wall")
        self.r_1 = self.r_outer - self.delta_wall
        self.r_2 = self.r_1 - self.delta_wick
        self.r_vapour = self.r_outer - self.delta_wall - self.delta_wick

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

    def get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        T_v = self.T_HP[-1]
        h_fg = calculate_Na_h_fg(T_v)

        A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        mdot = Qevap / h_fg

        mdot = np.concatenate([
            np.repeat(np.array([mdot/self.N_evap]), self.N_evap) * np.arange(self.N_evap),
            np.repeat(np.array([mdot]), self.N_adia),
            np.repeat(np.array([mdot/self.N_cond]), self.N_cond) * np.arange(self.N_cond)[::-1]
        ])
        
        return mdot


    def calculate_K_annular_wick(self):
        R_star = self.r_2 / self.r_1
        R_star_m = np.sqrt((1 - R_star**2) / (2 * np.log(1 / R_star)))

        fRe_l = 16 * (1 - R_star**2)**2 / (1 + R_star**2 - 2*R_star_m**2)

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K

    
    def calculate_pressure_drop(self):
        A_wick = np.pi * (self.r_1**2 - self.r_2**2) 
        
        T_wick = np.mean(np.array(self.T_HP)[:-1].reshape(self.N_Z, self.N_R)[:, :self.N_wick], axis=1) 

        mu_l = calculate_Na_viscosity_l(T_wick)  
        rho_l = calculate_Na_rho_l(T_wick)
        K = self.calculate_K_annular_wick()

        delta_z = np.concatenate(
            [np.ones(self.N_evap) * self.l_evap / self.N_evap,
            np.ones(self.N_adia) * self.l_adia / self.N_adia,
            np.ones(self.N_cond) * self.l_cond / self.N_cond]
            )
        
        mdot = self.get_mdot()
        deltaP = np.sum(mu_l * np.array(mdot) / (rho_l * A_wick * K) * delta_z)

        return deltaP
    

    def calculate_pressure_drop_profile(self):
        A_wick = np.pi * (self.r_1**2 - self.r_2**2) 
        
        T_wick = np.mean(np.array(self.T_HP)[:-1].reshape(self.N_Z, self.N_R)[:, :self.N_wick], axis=1) 

        mu_l = calculate_Na_viscosity_l(T_wick)  
        rho_l = calculate_Na_rho_l(T_wick)
        K = self.calculate_K_annular_wick()

        delta_z = np.concatenate(
            [np.ones(self.N_evap) * self.l_evap / self.N_evap,
            np.ones(self.N_adia) * self.l_adia / self.N_adia,
            np.ones(self.N_cond) * self.l_cond / self.N_cond]
            )
        
        mdot = self.get_mdot()
        P = -np.cumsum(mu_l * np.array(mdot) / (rho_l * A_wick * K) * delta_z)
        self.pressure = P
        print(K)


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
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "k_wick": 45.0,
        "k_wall": 21.7,
        "P_C": 2476,
        "T_C": 856
    }   

    lpd = liquid_discretised(data)

    P = lpd.calculate_pressure_drop_profile()