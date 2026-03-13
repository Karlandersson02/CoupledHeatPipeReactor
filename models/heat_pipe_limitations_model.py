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

from CoupledSystems.Heatpipe import Heatpipe

class heat_pipe_limitations:
    def __init__(self, HP):
        self.HP = HP
        self.r_outer    = HP.data.get("r_outer")
        self.delta_wall = HP.data.get("delta_wall")
        self.delta_gap  = HP.data.get("delta_gap")
        self.delta_wick = HP.data.get("delta_wick")
        self.r_gap    = self.r_outer - self.delta_wall
        self.r_wick   = self.r_gap   - self.delta_gap
        self.r_vapour = self.r_wick  - self.delta_wick

        self.l_evap      = HP.data.get("l_evap")
        self.l_adiabatic = HP.data.get("l_adiabatic")
        self.l_cond      = HP.data.get("l_cond")
        self.l_tot       = self.l_evap + self.l_adiabatic+ self.l_cond

        self.N_wick   = HP.data.get("N_wick")
        self.N_wall   = HP.data.get("N_wall")
        self.N_R      = self.N_wick + self.N_wall

        self.N_evap      = HP.data.get("N_evap")
        self.N_adiabatic = HP.data.get("N_adiabatic")
        self.N_cond      = HP.data.get("N_cond")
        self.N_Z         = self.N_evap + self.N_adiabatic+ self.N_cond

        self.k_wall   = HP.data.get("k_wall")
        self.k_wick   = HP.data.get("k_wick")

        self.r_pore   = HP.data.get("r_pore")
        self.porosity = HP.data.get("porosity")
        self.T_op     = HP.data.get("T_op")
        self.mdot_HP  = HP.data.get("mdot_HP")

        self.R = 8.314472
        self.R_Na = self.R / 0.022990


    def plot_analytical_limits(self, T_low, T_high):
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        Q_sonic       = HP_limits.calculate_analytical_sonic_limit(T_span)
        Q_cap         = HP_limits.calculate_analytical_capillary_limit_Busse(T_span)
        Q_boil        = HP_limits.calculate_analytical_boiling_limit(T_span)
        Q_entrainment = HP_limits.calculate_analytical_entrainment_limit(T_span)

        plt.semilogy(T_span, Q_sonic,       label="Sonic")
        plt.semilogy(T_span, Q_cap,         label="Capillary")
        plt.semilogy(T_span, Q_boil,        label="Boiling")
        plt.semilogy(T_span, Q_entrainment, label="Entrainment")
        plt.legend()
        plt.xlabel("Temperature [K]")
        plt.ylabel("Heat transfer [W]")
        plt.show()

        Q_lim = np.min(np.vstack([Q_sonic, Q_cap, Q_boil, Q_entrainment]), axis=0)
        plt.semilogy(T_span, Q_lim, label="Operating limit")
        plt.legend()
        plt.xlabel("Temperature [K]")
        plt.ylabel("Heat transfer [W]")
        plt.show()
    

    def calculate_analytical_capillary_limit(self, T_span):
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

        h_fg    = calculate_Na_h_fg(T_span)
        rho_l   = calculate_Na_rho_l(T_span)
        rho_v   = calculate_Na_rho_v(T_span)
        mu_v    = calculate_Na_viscosity_v(T_span)
        mu_l    = calculate_Na_viscosity_l(T_span)
        sigma_l = calculate_Na_surface_tension(T_span)

        
        # Calculating the pressure drop due to viscous forces in the liquid flow. 
        K = self.calculate_K_annular_wick()
    
        A_wick = np.pi * (self.r_1**2 - self.r_2**2)  

        L_cap_max = (0.5 * self.l_evap + self.l_adiabatic+ 0.5 * self.l_cond)

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
        Delta_p_cap_max = (2 * sigma_l / self.r_pore)

        # Equation is off the quadratic form.
        a = F_v_inertial
        b = F_v_viscous + F_l
        Q_cap = - (b / (2 * a)) + np.sqrt((b / (2 * a))**2 + Delta_p_cap_max / a)

        # plt.plot(T_span, Q_cap, label="Capillary limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_cap
    

    def calculate_analytical_capillary_limit_Busse(self, T_span):
        """
        Uses Busse instead of Cotter and calculates the wet point after both pressure drop profiles has been computed.
        """
        def build_flat_profile(Qtot, N):
            Q = np.repeat(np.array([Qtot / N], dtype=float), N)
            return Q
        
        # Must have flux BC for this to work.
        self.HP.data["Temperature_BC"] = False

        # Initial condition for Q. Might worth looking in to.
        Q = 50

        # Initialize the Q vector to store results.
        Q_cap = np.array([])
        
        # Iterate over all the temperatures in the span.
        for T in T_span:
            print(f"Progress: {(T - np.min(T_span)) / (np.max(T_span) - np.min(T_span))} %")
            # Calculate the max capillary head for the current temperature.
            sigma_l = calculate_Na_surface_tension(T)
            Delta_p_cap_max = (2 * sigma_l / self.r_pore)

            # Set margin to 0 to force atleast two iterations, due to "< 0." and "> 0." in break condition 
            Delta_p_margin = 0.

            max_iter = 100
            for i in range(max_iter):
                # Changing the setting in the heat pipe.
                Q_array = build_flat_profile(Q, self.HP.data["N_evap"])
                self.HP.data["Q"] = Q_array
                self.HP.data["T_HP"] = None

                self.HP.data["T_op"] = T

                # Solve for the total pressure profile with a variable wet point.
                self.HP.setup_fluid_models()
                P_v = self.analytical_pressure_drop_Busse()
                P_l = self.HP.get_liquid_pressure_drop_profile()

                P_v -= P_v[0]
                P_l -= P_l[0]

                P_l_rev = P_l[::-1]

                shift_touch = np.min(P_v - P_l_rev)
                P_v_touch = P_v - shift_touch

                diff = P_v_touch - P_l_rev
                i_contact = np.argmin(np.abs(diff))

                shift_zero = P_v_touch[0]
                P_v_plot = P_v_touch - shift_zero
                P_l_plot = P_l_rev - shift_zero

                Delta_P_profile_v_l = np.concatenate([
                    P_v_plot[:i_contact + 1],
                    P_l_plot[:i_contact + 1][::-1]
                ])

                Total_Delta_P_drop_v_l = (Delta_P_profile_v_l[0] - Delta_P_profile_v_l[-1])

                # Calculate the new pressure margin
                new_Delta_p_margin = Delta_p_cap_max - Total_Delta_P_drop_v_l

                # Break if there is a sign difference between new and old margin (Crossed the limit)
                if Delta_p_margin * new_Delta_p_margin < 0.:
                    Q_cap = np.append(Q_cap, Q)
                    break

                # Change Q based on the margin
                if new_Delta_p_margin > 0.:
                    Q += 10

                if new_Delta_p_margin < 0.:
                    Q -= 10

                Delta_p_margin = new_Delta_p_margin

        plt.plot(T_span, Q_sonic, label="Sonic limit")
        plt.xlabel("Temperature [Kelvin]")
        plt.ylabel("Heat transfer [W]")
 
        plt.show()

        return Q_cap 
    
    
    def calculate_analytical_boiling_limit(self, T_span):

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

        # plt.semilogy(T_span, Q_boil, label="Boiling limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_boil
    
    
    def calculate_analytical_sonic_limit(self, T_span):

        h_fg    = calculate_Na_h_fg(T_span)
        rho_v   = calculate_Na_rho_v(T_span)

        A_v = np.pi * self.r_vapour**2 
        gamma = 5/3

        Q_sonic = A_v * rho_v * h_fg * np.sqrt( (gamma * self.R_Na * T_span) / (2 * (gamma + 1)) )

        # plt.plot(T_span, Q_sonic, label="Sonic limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_sonic
    

    def calculate_analytical_entrainment_limit(self, T_span):

        h_fg    = calculate_Na_h_fg(T_span)
        rho_v   = calculate_Na_rho_v(T_span)
        sigma_l = calculate_Na_surface_tension(T_span)

        A_v = np.pi * self.r_vapour**2 

        # Assuming that the area of the individual pore is a half sphere (best case scenario)
        #R_h_w = self.r_pore / 3

        # Assuming that the area of the individual pore is flat (worst case scenario)
        R_h_w = self.r_pore
        
        Q_entrainment = A_v * h_fg * np.sqrt( (sigma_l * rho_v) / (2 * R_h_w) )

        # plt.plot(T_span, Q_entrainment, label="Entrainment limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_entrainment
    

    def calculate_K_annular_wick(self):
        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K    
    

    def analytical_pressure_drop_Busse(self):
        T_v   = self.HP.calculated_quantities["heatpipe_T"][-1]
        h_fg  = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        mu_v  = calculate_Na_viscosity_v(T_v)

        Rv  = self.r_vapour
        d_v = 2.0 * Rv
        L_C = self.l_cond
        L_e = self.l_evap

        Q_tot = self.HP.liquid_discretised.get_mdot()[self.N_evap] * h_fg

        Re_re = Q_tot / (2 * np.pi * L_e * h_fg * mu_v)
        Re_rc = Q_tot / (2 * np.pi * L_C * h_fg * mu_v)

        # if Re_rc >= -2.25:
        #     raise ValueError(
        #         f"Busse (1967) invalid: Re_{{r,c}} = {Re_rc:.4f} <= -2.25."
        #     )

        def _alpha(Re_r):
            inner = 5.0 + 18.0 / Re_r
            disc  = inner**2 - 44.0 / 5.0
            if disc < 0:
                raise ValueError(
                    f"Busse (1967): negative discriminant at Re_r = {Re_r:.4f}."
                )
            return (15.0 / 22.0) * (inner + np.sqrt(disc))**0.5

        # --- Evaporator + adiabatic (combined, Busse 1967) ---
        F = (7.0/9.0 - 1.7 * Re_re / (36 + 10*Re_re) * np.exp(-7.5 * self.l_adiabatic / (Re_re * L_e)))
        
        dP_evap = ((-4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (L_e * (1.0 + Re_re * F)))
        
        dP_adiabatic = -(8.0 * mu_v * Q_tot * self.l_adiabatic) / (rho_v * np.pi * Rv**4 * h_fg)

        # --- Condenser pressure recovery (Busse 1967, eq. 11) ---
        dP_cond = -dP_evap + -dP_adiabatic - (4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (self.l_evap + 2*self.l_adiabatic + self.l_cond)

        # --- Distribute onto spatial grid ---
        dx   = np.zeros(self.N_Z, dtype=float)

        dx[:self.N_evap] = self.l_evap / (self.N_evap)
        dx[self.N_evap: self.N_evap + self.N_adiabatic] = self.l_adiabatic / (self.N_adiabatic)
        dx[self.N_evap + self.N_adiabatic:] = self.l_cond / (self.N_cond)
        
        dpdx = np.zeros(self.N_Z, dtype=float)

        x_evap  = np.linspace(0, L_e, self.N_evap)
        profile = (x_evap / L_e)**2
        norm    = np.sum(profile) * (L_e / self.N_evap)
        dpdx[:self.N_evap] = dP_evap * profile / norm

        dpdx[self.N_evap:self.N_evap + self.N_adiabatic] = (
            dP_adiabatic / (self.l_adiabatic)
        )

        j0 = self.N_evap + self.N_adiabatic
        x2     = L_e + self.l_adiabatic
        x_cond = np.linspace(x2, x2 + L_C, self.N_cond)
        xi     = 1.0 - (x_cond - x2) / L_C
        dP_cond_total = dP_cond  # scalar total
        # distribute as (1 - xi)^2 profile, normalised to integrate to dP_cond_total
        profile    = (xi)**2
        dpdx[j0:] = dP_cond_total * profile / (np.sum(profile) * dx[self.N_evap + self.N_adiabatic:])

        return np.cumsum(dpdx * dx) 

if __name__ == "__main__":
    data_Guoju_2 = {
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wall": 0.001,
        "delta_gap": 0.,
        "delta_wick": 0.0005,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,

        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 30,
        "N_adiabatic": 15,
        "N_cond": 165,

        "adiabatic_radial_flux": False,
        "Temperature_BC": False,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "T_op": 850,

        "k_wick": 66.2,
        "k_wall": 19.0,

        "P_C": 2476,
        "T_C": 856,

        "Is_annular": True,
        "K":1e-10,
        "r_pore": 2.e-5,
        "porosity": 0.7,
    }

    def build_flat_profile(Qtot, N):
        Q = np.repeat(np.array([Qtot / N], dtype=float), N)
        return Q

    Q = build_flat_profile(560, data_Guoju_2["N_evap"])
    data_Guoju_2["Q"] = Q
    HP = Heatpipe(data_Guoju_2)
    HP_limits = heat_pipe_limitations(HP)

    T_low = 800
    T_high = 850
    T_span = np.linspace(T_low, T_high, T_high - T_low + 1)
    HP_limits.calculate_analytical_capillary_limit_Busse(T_span)


    

