import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from models.sodium_properties import (
    calculate_Na_h_fg,
    calculate_Na_rho_l,
    calculate_Na_rho_v,
    calculate_Na_surface_tension,
    calculate_Na_viscosity_l,
    calculate_Na_viscosity_v,
    calculate_Na_thermal_conductivity_l,
)
class heat_pipe_limitations:
    def __init__(self, data):
        self.r_outer  = data.get("r_outer")
        self.r_gap  = data.get("r_gap")
        self.r_wick  = data.get("r_wick")
        self.r_1 = data.get("r_wall")
        self.r_2 = data.get("r_wick")
        self.r_vapour = data.get("r_vapour")

        self.l_evap  = data.get("l_evap")
        self.l_adia  = data.get("l_adia")
        self.l_cond  = data.get("l_cond")
        self.l_tot = self.l_evap + self.l_adia + self.l_cond

        self.N_wick = data.get("N_wick")
        self.N_wall = data.get("N_wall")
        self.N_R = self.N_wick + self.N_wall

        self.N_evap = data.get("N_evap")
        self.N_adia = data.get("N_adia")
        self.N_cond = data.get("N_cond")
        self.N_Z = self.N_evap + self.N_adia + self.N_cond

        self.k_wall = data.get("k_wall")
        self.k_wick = data.get("k_wick")

        self.r_pore = data.get("r_pore")
        self.porosity = data.get("porosity")
        self.T_op = data.get("T_op")
        self.mdot_HP = data.get("mdot_HP")

        self.R = 8.314472
        self.R_Na = self.R / 0.022990

    def calculate_analytical_heat_pipe_limitations(self):
        return 0
    
    def calculate_analytical_capillary_limit(self):
        """ 
        Based on eq. (4.10) in Heat Pipe science and technology by Amir Faghri.

        Builds on the assumptions of: 
            - Uniform heating distribution on evap and cond.
            - Laminar incompressable flow in both vapour and liquid.
            - Wet point close to the condeser end cap.
            - Constant K, A_wick and A_v along the heat pipe.
            - Neglecting pressure drop in liquid-vapour interface
            - Cylindrical heat pipe
            - Assuming perfect wetting, theta = 0. 

        The models used to capture the pressure drop are:
            Vapour: eq. (3.76) in Faghri
            Liquid: Darcy's law in 1D, given by eq. (3.7) in Faghri
        """

        T_low = 600; T_high = 1400
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        h_fg    = calculate_Na_h_fg(T_span)
        rho_l   = calculate_Na_rho_l(T_span)
        rho_v   = calculate_Na_rho_v(T_span)
        mu_v    = calculate_Na_viscosity_v(T_span)
        mu_l    = calculate_Na_viscosity_l(T_span)
        sigma_l = calculate_Na_surface_tension(T_span)

        
        # Calculating the pressure drop due to viscous forces in the liquid flow. 
        K = self.calculate_K_annular_wick()
    
        A_wick = np.pi * (self.r_1**2 - self.r_2**2)  

        L_cap_max = (0.5 * self.l_evap + self.l_adia + 0.5 * self.l_cond)

        F_l = mu_l * L_cap_max / (rho_l * A_wick * K * h_fg)

        # Calculating the pressure drop due to viscous forces in the vapour flow.
        
        # Due to a circular duct 
        fRe_zv = 16

        A_v = np.pi * self.r_vapour**2

        F_v_viscous = fRe_zv * mu_v / (2 * self.r_vapour**2 * A_v * rho_v * h_fg)

        # Calculating the pressure drop due to inertial forces in the vapour flow.
        cotter_recovery = 4.0 / np.pi**2          
        inertial_fraction = 1.0 - cotter_recovery  
        F_v_inertial = inertial_fraction / (8 * rho_v * self.r_vapour**4 * h_fg**2)

        # Calculating Q_max based on the pressure drops.
        Delta_p_cap_max = (2 * sigma_l / self.r_p)

        # Equation is off the quadratic form.
        a = F_v_inertial
        b = F_v_viscous + F_l
        Q_cap = - (b / (2 * a)) + np.sqrt((b / (2 * a))**2 + Delta_p_cap_max / a)

        plt.plot(T_span, Q_cap, label="Capillary limit")
        plt.xlabel("Temperature [Kelvin]")
        plt.ylabel("Heat transfer [W]")

        plt.show()

        return 0
    
    
    def calculate_analytical_boiling_limit(self):
        T_low = 600; T_high = 1400
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        h_fg    = calculate_Na_h_fg(T_span)
        rho_l   = calculate_Na_rho_l(T_span)
        rho_v   = calculate_Na_rho_v(T_span)
        sigma_l = calculate_Na_surface_tension(T_span)
        k_l     = calculate_Na_thermal_conductivity_l(T_span)

        nu_v = 1.0 / rho_v
        nu_l = 1.0 / rho_l
        k_eff = (1 - self.porosity) * self.k_wick + self.porosity * k_l

        def residual(Q, i):
            q_r   = Q / (np.pi * self.r_vapour**2)
            R_b   = np.sqrt((2 * sigma_l[i] * T_span[i] * k_l[i] * (nu_v[i] - nu_l[i]))
                            / (h_fg[i] * q_r))
            dT    = (2 * sigma_l[i] * T_span[i]) / (h_fg[i] * rho_v[i]) * (1/R_b - 1/self.r_pore)
            Q_rhs = (2 * np.pi * self.l_evap * k_eff[i] * dT) / np.log(self.r_wick / self.r_vapour)
            return Q - Q_rhs

        def find_bracket(residual, i, Q_min=1e-3, Q_max=1e12, n_search=500):
            Q_vals = np.logspace(np.log10(Q_min), np.log10(Q_max), n_search)
            r_vals = np.array([residual(Q, i) for Q in Q_vals])
            sign_changes = np.where(np.diff(np.sign(r_vals)))[0]
            if len(sign_changes) == 0:
                return None, None
            idx = sign_changes[0]
            return Q_vals[idx], Q_vals[idx + 1]

        Q_boil = np.zeros_like(T_span)
        for i in range(len(T_span)):
            a, b = find_bracket(residual, i)
            if a is None:
                Q_boil[i] = np.nan
            else:
                Q_boil[i] = brentq(residual, a=a, b=b, args=(i,))

        plt.semilogy(T_span, Q_boil, label="Boiling limit")
        plt.xlabel("Temperature [Kelvin]")
        plt.ylabel("Heat transfer [W]")

        plt.show()
        return 0
    
    
    def calculate_analytical_sonic_limit(self):
        T_low = 600; T_high = 1400
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        h_fg    = calculate_Na_h_fg(T_span)
        rho_v   = calculate_Na_rho_v(T_span)

        A_v = np.pi * self.r_vapour**2 
        gamma = 5/3

        Q_sonic = A_v * rho_v * h_fg * np.sqrt( (gamma * self.R_Na * T_span) / (2 * (gamma + 1)) )

        plt.plot(T_span, Q_sonic, label="Sonic limit")
        plt.xlabel("Temperature [Kelvin]")
        plt.ylabel("Heat transfer [W]")

        plt.show()
        return 0
    
    def calculate_analytical_entrainment_limit(self):
        T_low = 600; T_high = 1400
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        h_fg    = calculate_Na_h_fg(T_span)
        rho_v   = calculate_Na_rho_v(T_span)
        sigma_l = calculate_Na_surface_tension(T_span)

        A_v = np.pi * self.r_vapour**2 

        # Assuming that the area of the individual pore is a half sphere (best case scenario)
        #R_h_w = self.r_pore / 3

        # Assuming that the area of the individual pore is flat (worst case scenario)
        R_h_w = self.r_pore
        
        Q_entrainment = A_v * h_fg * np.sqrt( (sigma_l * rho_v) / (2 * R_h_w) )

        plt.plot(T_span, Q_entrainment, label="Entrainment limit")
        plt.xlabel("Temperature [Kelvin]")
        plt.ylabel("Heat transfer [W]")

        plt.show()

        return Q_entrainment
    
    def calculate_K_annular_wick(self):
        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K

if __name__ == "__main__":
    data = {
        "r_outer": 0.01410/2,
        "r_wall":   0.01410/2,
        "r_wick":   0.0130/2, 
        "r_vapour": 0.012310/2,
        "delta_wall": 0.,
        "l_evap": 0.3,
        "l_adia": 0.2,
        "l_cond": 0.3,
        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 20,
        "N_adia": 10,
        "N_cond": 110,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "k_wick": 45.0,
        "k_wall": 21.7,
        "P_C": 2476,
        "T_C": 856,
        "r_pore": 2.e-5,
        "porosity": 0.7,
        "mdot_HP": [1.5],
    }   

    HP_limits = heat_pipe_limitations(data)

    HP_limits.calculate_analytical_sonic_limit()


    

