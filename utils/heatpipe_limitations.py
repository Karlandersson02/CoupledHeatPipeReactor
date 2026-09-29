import numpy as np
import matplotlib.pyplot as plt

import utils.sodium_properties as s_props
import data.dataclass as d_class

from scipy.optimize import brentq

from utils.heatpipe_decoupled import Heatpipe

R = 8.314472
R_Na = R / 0.022990

class HeatPipeLimitations:
    def __init__(self, heatpipe: Heatpipe, config: d_class.HeatpipeConfigResolved):
        self.HP = heatpipe
        self.cfg = config

    def plot_analytical_limits(self, T_low, T_high):
        T_span = np.linspace(T_low, T_high, T_high - T_low + 1)

        Q_sonic       = self.calculate_analytical_sonic_limit(T_span)
        Q_cap         = self.calculate_analytical_capillary_limit(T_span)
        Q_boil        = self.calculate_analytical_boiling_limit(T_span)
        Q_entrainment = self.calculate_analytical_entrainment_limit(T_span)

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

        h_fg    = s_props.calculate_Na_h_fg(T_span)
        rho_l   = s_props.calculate_Na_rho_l(T_span)
        rho_v   = s_props.calculate_Na_rho_v(T_span)
        mu_v    = s_props.calculate_Na_viscosity_v(T_span)
        mu_l    = s_props.calculate_Na_viscosity_l(T_span)
        sigma_l = s_props.calculate_Na_surface_tension(T_span)

        
        # Calculating the pressure drop due to viscous forces in the liquid flow.     
        if self.cfg.wick.Is_annular == True:
            K = self.calculate_K_annular_wick()
            A_wick = np.pi * (self.cfg.geometry.r_gap**2 - self.cfg.geometry.r_wick**2) 
        else:
            K = self.cfg.wick.K
            A_wick = np.pi * (self.cfg.geometry.r_wick**2 - self.cfg.geometry.r_vapour**2)

        L_cap_max = (0.5 * self.cfg.geometry.l_evap + self.cfg.geometry.l_adiabatic+ 0.5 * self.cfg.geometry.l_cond)

        F_l = mu_l * L_cap_max / (rho_l * A_wick * K * h_fg)

        # Calculating the pressure drop due to viscous forces in the vapour flow.
        
        # Due to a circular duct 
        fRe_zv = 16

        A_v = np.pi * self.cfg.geometry.r_vapour**2

        F_v_viscous = fRe_zv * mu_v / (2 * self.cfg.geometry.r_vapour**2 * A_v * rho_v * h_fg)

        # Calculating the pressure drop due to inertial forces in the vapour flow.
        cotter_recovery = 4.0 / np.pi**2          
        inertial_fraction = 1.0 - cotter_recovery  
        F_v_inertial = inertial_fraction / (8 * rho_v * self.cfg.geometry.r_vapour**4 * h_fg**2)

        # Calculating Q_max based on the pressure drops.
        Delta_p_cap_max = (2. * sigma_l / self.cfg.wick.r_pore)

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

        # Initial condition for Q. Might be worth looking in to.
        Q = 1000.
        Q_old = 1000.
        Delta_Q = 10.

        # Initialize the Q vector to store results.
        Q_cap = np.array([])
        
        # Iterate over all the temperatures in the span.
        for T in T_span:
            print(f"Progress: {(T - np.min(T_span))*100 / (np.max(T_span) - np.min(T_span))} %")
            # Calculate the max capillary head for the current temperature.
            sigma_l = s_props.calculate_Na_surface_tension(T)
            Delta_p_cap_max = (2 * sigma_l / self.cfg.wick.r_pore)

            # Set margin to 0 to force atleast two iterations, due to "< 0." in break condition.
            Delta_p_margin = 0.

            max_iter = 200
            for i in range(max_iter):
                # Changing the data in the heat pipe to reflect current Q and T.
                Q_array = build_flat_profile(Q, self.HP.data["N_evap"])
                self.HP.data["Q"] = Q_array
                self.HP.data["T_op"] = T

                # Recalculate the conduction model temperature profile with the current settings for Q and T.
                # Since the conduction model data cannot be changed once the heatpipe is initialized, 
                # the heat pipe object needs to be reinitialized each iteration. 
                # !!! Definetly needs to be changed if this approach gives enough of a different answer
                # compared to the version in Faghri 1995.!!!
                self.HP = Heatpipe(self.HP.data)
                self.HP.setup_fluid_models()

                # Solve for the total pressure profile with a variable wet point.
                P_v = self.analytical_pressure_drop_Busse()
                P_l = self.HP.liquid_discretised.get_pressure_drop_profile()

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
                print(Total_Delta_P_drop_v_l, Delta_p_cap_max, Q, Delta_p_margin * new_Delta_p_margin / np.abs(Delta_p_margin * new_Delta_p_margin))
                if Delta_p_margin * new_Delta_p_margin < 0.:
                    Q_cap = np.append(Q_cap, (Q + Q_old)/2)
                    break

                # Change Q based on the margin
                if new_Delta_p_margin > 0.:
                    Q_old = Q
                    Q += Delta_Q

                if new_Delta_p_margin < 0.:
                    Q_old = Q
                    Q -= Delta_Q

                Delta_p_margin = new_Delta_p_margin

        return Q_cap 
    
    
    # def calculate_analytical_boiling_limit(self, T_span):

    #     h_fg    = s_props.calculate_Na_h_fg(T_span)
    #     rho_l   = s_props.calculate_Na_rho_l(T_span)
    #     rho_v   = s_props.calculate_Na_rho_v(T_span)
    #     sigma_l = s_props.calculate_Na_surface_tension(T_span)
    #     k_l     = s_props.calculate_Na_thermal_conductivity_l(T_span)

    #     nu_v = 1.0 / rho_v
    #     nu_l = 1.0 / rho_l
    #     k_eff = (1 - self.cfg.wick.porosity) * self.cfg.material.k_wick + self.cfg.wick.porosity * k_l

    #     def residual(Q, i):
    #         q_r   = Q / (np.pi * self.cfg.geometry.r_vapour**2)
    #         R_b   = np.sqrt((2 * sigma_l[i] * T_span[i] * k_l[i] * (nu_v[i] - nu_l[i]))
    #                         / (h_fg[i] * q_r))
    #         dT    = (2 * sigma_l[i] * T_span[i]) / (h_fg[i] * rho_v[i]) * (1/R_b - 1/self.cfg.wick.r_pore)
    #         Q_rhs = (2 * np.pi * self.cfg.geometry.l_evap * k_eff[i] * dT) / np.log(self.cfg.geometry.r_wick / self.cfg.geometry.r_vapour)
    #         return Q - Q_rhs

    #     def find_bracket(residual, i, Q_min=1e-3, Q_max=1e13, n_search=500):
    #         Q_vals = np.logspace(np.log10(Q_min), np.log10(Q_max), n_search)
    #         r_vals = np.array([residual(Q, i) for Q in Q_vals])
    #         sign_changes = np.where(np.diff(np.sign(r_vals)))[0]
    #         if len(sign_changes) == 0:
    #             return None, None
    #         idx = sign_changes[0]
    #         return Q_vals[idx], Q_vals[idx + 1]

    #     Q_boil = np.zeros_like(T_span)
    #     for i in range(len(T_span)):
    #         a, b = find_bracket(residual, i)
    #         if a is None:
    #             Q_boil[i] = np.nan
    #         else:
    #             Q_boil[i] = brentq(residual, a=a, b=b, args=(i,))

    #     # plt.semilogy(T_span, Q_boil, label="Boiling limit")
    #     # plt.xlabel("Temperature [Kelvin]")
    #     # plt.ylabel("Heat transfer [W]")

    #     # plt.show()

    #     print(Q_boil)

    #     return Q_boil
    
    def calculate_analytical_boiling_limit(
        self,
        T_span,
        root="lower",              # "lower" matches your current first-bracket behaviour
        return_diagnostics=False,
    ):
        T_span = np.asarray(T_span, dtype=float)
        scalar_input = T_span.ndim == 0
        T_span = np.atleast_1d(T_span)

        def as_array(x, name):
            x = np.asarray(x, dtype=float)
            try:
                return np.broadcast_to(x, T_span.shape).astype(float, copy=False)
            except ValueError as exc:
                raise ValueError(
                    f"{name} returned shape {x.shape}, but expected something "
                    f"broadcastable to {T_span.shape}."
                ) from exc

        h_fg    = as_array(s_props.calculate_Na_h_fg(T_span), "h_fg")
        rho_l   = as_array(s_props.calculate_Na_rho_l(T_span), "rho_l")
        rho_v   = as_array(s_props.calculate_Na_rho_v(T_span), "rho_v")
        sigma_l = as_array(s_props.calculate_Na_surface_tension(T_span), "sigma_l")
        k_l     = as_array(s_props.calculate_Na_thermal_conductivity_l(T_span), "k_l")

        r_vapour = float(self.cfg.geometry.r_vapour)
        r_wick   = float(self.cfg.geometry.r_wick)
        r_pore   = float(self.cfg.wick.r_pore)
        l_evap   = float(self.cfg.geometry.l_evap)
        porosity = float(self.cfg.wick.porosity)
        k_wick   = float(self.cfg.material.k_wick)

        if r_vapour <= 0:
            raise ValueError("cfg.geometry.r_vapour must be positive.")
        if r_wick <= r_vapour:
            raise ValueError("cfg.geometry.r_wick must be larger than r_vapour.")
        if r_pore <= 0:
            raise ValueError("cfg.wick.r_pore must be positive.")
        if l_evap <= 0:
            raise ValueError("cfg.geometry.l_evap must be positive.")
        if not (0.0 <= porosity <= 1.0):
            raise ValueError("cfg.wick.porosity must be between 0 and 1.")

        area_vapour = np.pi * r_vapour**2
        log_ratio = np.log(r_wick / r_vapour)

        Q_boil = np.full_like(T_span, np.nan, dtype=float)

        diagnostics = {
            "invalid_property_points": 0,
            "no_real_root_points": 0,
            "non_finite_result_points": 0,
        }

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            v_v = 1.0 / rho_v
            v_l = 1.0 / rho_l

            k_eff = (1.0 - porosity) * k_wick + porosity * k_l

            # Your residual can be written as:
            #
            #     residual(Q) = Q - E sqrt(Q) + F
            #
            # where x = sqrt(Q) gives:
            #
            #     x^2 - E x + F = 0
            #
            B = 2.0 * sigma_l * T_span * k_l * (v_v - v_l) / h_fg
            C = 2.0 * sigma_l * T_span / (h_fg * rho_v)
            D = 2.0 * np.pi * l_evap * k_eff / log_ratio

            E = D * C / np.sqrt(B * area_vapour)
            F = D * C / r_pore

            valid_base = (
                np.isfinite(T_span)
                & np.isfinite(h_fg)
                & np.isfinite(rho_l)
                & np.isfinite(rho_v)
                & np.isfinite(sigma_l)
                & np.isfinite(k_l)
                & np.isfinite(k_eff)
                & np.isfinite(B)
                & np.isfinite(C)
                & np.isfinite(D)
                & np.isfinite(E)
                & np.isfinite(F)
                & (T_span > 0.0)
                & (h_fg > 0.0)
                & (rho_l > 0.0)
                & (rho_v > 0.0)
                & (sigma_l > 0.0)
                & (k_l > 0.0)
                & (k_eff > 0.0)
                & (B > 0.0)
                & (C > 0.0)
                & (D > 0.0)
                & (E > 0.0)
                & (F >= 0.0)
            )

            diagnostics["invalid_property_points"] = int(np.count_nonzero(~valid_base))

            # Discriminant condition:
            #
            #     E^2 - 4F >= 0
            #
            # Written this way to avoid unnecessary overflow in E**2.
            ratio = 4.0 * (F / E) / E
            has_real_root = valid_base & np.isfinite(ratio) & (ratio <= 1.0)

            diagnostics["no_real_root_points"] = int(
                np.count_nonzero(valid_base & ~has_real_root)
            )

            sqrt_disc = np.full_like(T_span, np.nan, dtype=float)
            sqrt_disc[has_real_root] = (
                E[has_real_root]
                * np.sqrt(np.maximum(0.0, 1.0 - ratio[has_real_root]))
            )

            x_upper = np.full_like(T_span, np.nan, dtype=float)
            x_lower = np.full_like(T_span, np.nan, dtype=float)

            x_upper[has_real_root] = 0.5 * (E[has_real_root] + sqrt_disc[has_real_root])

            # More stable than:
            # x_lower = 0.5 * (E - sqrt_disc)
            x_lower[has_real_root] = F[has_real_root] / x_upper[has_real_root]

            if root == "lower":
                x = x_lower
            elif root == "upper":
                x = x_upper
            else:
                raise ValueError("root must be either 'lower' or 'upper'.")

            Q = x**2

            valid_result = has_real_root & np.isfinite(Q) & (Q >= 0.0)
            Q_boil[valid_result] = Q[valid_result]

            diagnostics["non_finite_result_points"] = int(
                np.count_nonzero(has_real_root & ~valid_result)
            )

        print(diagnostics)

        if scalar_input:
            Q_boil = Q_boil.item()

        if return_diagnostics:
            return Q_boil, diagnostics

        return Q_boil
    

    def calculate_analytical_boiling_limit_new(
        self,
        T_span,
        R_b=1e-5,
        return_diagnostics=False,
    ):
        """
        Parameters
        ----------
        T_span : float or array-like
            Vapour/saturation temperature values [K].

        R_b : float, optional
            Critical bubble radius [m]. Default is 1e-7 m.

        return_diagnostics : bool, optional
            If True, also return a diagnostics dictionary.

        Returns
        -------
        Q_boil : float or np.ndarray
            Analytical boiling limit [W].

        diagnostics : dict, optional
            Returned only if return_diagnostics=True.
        """

        T_span = np.asarray(T_span, dtype=float)
        scalar_input = T_span.ndim == 0
        T_span = np.atleast_1d(T_span)

        def as_array(x, name):
            x = np.asarray(x, dtype=float)
            try:
                return np.broadcast_to(x, T_span.shape).astype(float, copy=False)
            except ValueError as exc:
                raise ValueError(
                    f"{name} returned shape {x.shape}, but expected something "
                    f"broadcastable to {T_span.shape}."
                ) from exc

        # ------------------------------------------------------------
        # Sodium properties
        # ------------------------------------------------------------
        h_fg    = as_array(s_props.calculate_Na_h_fg(T_span), "h_fg")
        rho_v   = as_array(s_props.calculate_Na_rho_v(T_span), "rho_v")
        sigma_l = as_array(s_props.calculate_Na_surface_tension(T_span), "sigma_l")
        k_l     = as_array(s_props.calculate_Na_thermal_conductivity_l(T_span), "k_l")

        # ------------------------------------------------------------
        # Geometry and wick/material properties
        # ------------------------------------------------------------
        r_vapour = float(self.cfg.geometry.r_vapour)
        r_wick   = float(self.cfg.geometry.r_wick)
        r_pore   = float(self.cfg.wick.r_pore)
        l_evap   = float(self.cfg.geometry.l_evap)
        porosity = float(self.cfg.wick.porosity)
        k_wick   = float(self.cfg.material.k_wick)

        # ------------------------------------------------------------
        # Input checks
        # ------------------------------------------------------------
        if r_vapour <= 0:
            raise ValueError("cfg.geometry.r_vapour must be positive.")
        if r_wick <= r_vapour:
            raise ValueError("cfg.geometry.r_wick must be larger than r_vapour.")
        if r_pore <= 0:
            raise ValueError("cfg.wick.r_pore must be positive.")
        if l_evap <= 0:
            raise ValueError("cfg.geometry.l_evap must be positive.")
        if R_b <= 0:
            raise ValueError("R_b must be positive.")
        if not (0.0 <= porosity <= 1.0):
            raise ValueError("cfg.wick.porosity must be between 0 and 1.")

        log_ratio = np.log(r_wick / r_vapour)

        Q_boil = np.full_like(T_span, np.nan, dtype=float)

        diagnostics = {
            "invalid_property_points": 0,
            "non_positive_superheat_points": 0,
            "non_finite_result_points": 0,
        }

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Effective wick conductivity
            k_eff_wick = (1.0 - porosity) * k_wick + porosity * k_l

            # Critical superheat
            delta_T_crit = (
                2.0 * sigma_l * T_span / (h_fg * rho_v)
                * (1.0 / R_b - 1.0 / r_pore)
            )

            # Boiling limit
            Q = (
                2.0 * np.pi * l_evap * k_eff_wick * delta_T_crit
                / log_ratio
            )

            valid_base = (
                np.isfinite(T_span)
                & np.isfinite(h_fg)
                & np.isfinite(rho_v)
                & np.isfinite(sigma_l)
                & np.isfinite(k_l)
                & np.isfinite(k_eff_wick)
                & np.isfinite(delta_T_crit)
                & np.isfinite(Q)
                & (T_span > 0.0)
                & (h_fg > 0.0)
                & (rho_v > 0.0)
                & (sigma_l > 0.0)
                & (k_l > 0.0)
                & (k_eff_wick > 0.0)
                & (log_ratio > 0.0)
            )

            diagnostics["invalid_property_points"] = int(np.count_nonzero(~valid_base))

            positive_superheat = valid_base & (delta_T_crit > 0.0)

            diagnostics["non_positive_superheat_points"] = int(
                np.count_nonzero(valid_base & ~positive_superheat)
            )

            valid_result = positive_superheat & np.isfinite(Q) & (Q >= 0.0)

            diagnostics["non_finite_result_points"] = int(
                np.count_nonzero(positive_superheat & ~valid_result)
            )

            Q_boil[valid_result] = Q[valid_result]

        print(diagnostics)

        if scalar_input:
            Q_boil = Q_boil.item()

        if return_diagnostics:
            return Q_boil, diagnostics

        return Q_boil

    
    
    def calculate_analytical_sonic_limit(self, T_span):

        h_fg    = s_props.calculate_Na_h_fg(T_span)
        rho_v   = s_props.calculate_Na_rho_v(T_span)

        A_v = np.pi * self.cfg.geometry.r_vapour**2 
        gamma = 5/3

        Q_sonic = A_v * rho_v * h_fg * np.sqrt( (gamma * R_Na * T_span) / (2 * (gamma + 1)) )

        # plt.plot(T_span, Q_sonic, label="Sonic limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_sonic
    

    def calculate_analytical_entrainment_limit(self, T_span):

        h_fg    = s_props.calculate_Na_h_fg(T_span)
        rho_v   = s_props.calculate_Na_rho_v(T_span)
        sigma_l = s_props.calculate_Na_surface_tension(T_span)

        A_v = np.pi * self.cfg.geometry.r_vapour**2 

        # Assuming that the area of the individual pore is a half sphere (best case scenario)
        R_h_w = self.cfg.wick.r_pore / 3

        # Assuming that the area of the individual pore is flat (worst case scenario)
        #R_h_w = self.cfg.wick.r_pore
        
        Q_entrainment = A_v * h_fg * np.sqrt( (sigma_l * rho_v) / (2 * R_h_w) )

        # plt.plot(T_span, Q_entrainment, label="Entrainment limit")
        # plt.xlabel("Temperature [Kelvin]")
        # plt.ylabel("Heat transfer [W]")

        # plt.show()

        return Q_entrainment
    

    def calculate_K_annular_wick(self):
        self.r_2 = self.cfg.geometry.r_wick
        self.r_1 = self.cfg.geometry.r_gap

        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K  
    

    def analytical_pressure_drop_Busse(self):
        T_v   = self.HP.calculated_quantities["heatpipe_T"][-1]
        h_fg  = s_props.calculate_Na_h_fg(T_v)
        rho_v = s_props.calculate_Na_rho_v(T_v)
        mu_v  = s_props.calculate_Na_viscosity_v(T_v)

        Rv  = self.cfg.geometry.r_vapour
        d_v = 2.0 * Rv
        L_C = self.cfg.geometry.l_cond
        L_e = self.cfg.geometry.l_evap

        Q_tot = self.HP.liquid_discretised.get_mdot()[self.cfg.mesh.N_evap] * h_fg

        print(f"Q_tot = {Q_tot}")

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
        F = (7.0/9.0 - 1.7 * Re_re / (36 + 10*Re_re) * np.exp(-7.5 * self.cfg.geometry.l_adiabatic / (Re_re * L_e)))
        
        dP_evap = ((-4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (L_e * (1.0 + Re_re * F)))
        
        dP_adiabatic = -(8.0 * mu_v * Q_tot * self.cfg.geometry.l_adiabatic) / (rho_v * np.pi * Rv**4 * h_fg)

        # --- Condenser pressure recovery (Busse 1967, eq. 11) ---
        dP_cond = -dP_evap + -dP_adiabatic - (4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (self.cfg.geometry.l_evap + 2*self.cfg.geometry.l_adiabatic + self.cfg.geometry.l_cond)

        # --- Distribute onto spatial grid ---
        dx   = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        dx[:self.cfg.mesh.N_evap] = self.cfg.geometry.l_evap / (self.cfg.mesh.N_evap)
        dx[self.cfg.mesh.N_evap: self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic] = self.cfg.geometry.l_adiabatic / (self.cfg.mesh.N_adiabatic)
        dx[self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic:] = self.cfg.geometry.l_cond / (self.cfg.mesh.N_cond)
        
        dpdx = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        x_evap  = np.linspace(0, L_e, self.cfg.mesh.N_evap)
        profile = (x_evap / L_e)**2
        norm    = np.sum(profile) * (L_e / self.cfg.mesh.N_evap)
        dpdx[:self.cfg.mesh.N_evap] = dP_evap * profile / norm

        dpdx[self.cfg.mesh.N_evap:self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic] = (
            dP_adiabatic / (self.cfg.geometry.l_adiabatic)
        )

        j0 = self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic
        x2     = L_e + self.cfg.geometry.l_adiabatic
        x_cond = np.linspace(x2, x2 + L_C, self.cfg.mesh.N_cond)
        xi     = 1.0 - (x_cond - x2) / L_C
        dP_cond_total = dP_cond  # scalar total
        # distribute as (1 - xi)^2 profile, normalised to integrate to dP_cond_total
        profile    = (xi)**2
        dpdx[j0:] = dP_cond_total * profile / (np.sum(profile) * dx[self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic:])

        return np.cumsum(dpdx * dx) 

if __name__ == "__main__":
    data_Guoju_2 = {
        "r_outer": .007 + 0.001 + 0.0005 + 0.0005,
        "delta_wick": 0.0005,
        "delta_gap": 0.0005,
        "delta_wall": 0.001,
        "l_evap": 0.1,
        "l_adiabatic": 0.05,
        "l_cond": 0.55,

        "N_wick": 15,
        "N_wall": 15,
        "N_evap": 20,
        "N_adiabatic": 10,
        "N_cond": 160,

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

    data123 = {
        "r_outer":  0.01410/2,
        "r_wall":   0.01410/2,
        "r_wick":   0.0130/2, 
        "r_vapour": 0.012310/2,
        "delta_wick": 0.0130/2 - 0.012310/2,
        "delta_gap": 0.,
        "delta_wall": 0.,
        "delta_wall": 0.,
        "l_evap": 0.3,
        "l_adiabatic": 0.2,
        "l_cond": 0.3,
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
        "Is_annular": False,
        "K":1e-10,
        "r_pore": 2.e-5,
        "porosity": 0.7,
        "mdot_HP": [1.5],
    }   

    def build_flat_profile(Qtot, N):
        Q = np.repeat(np.array([Qtot / N], dtype=float), N)
        return Q
    

    import json
    with open("./data/reactor_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["HeatPipe"]

    geom = d_class.HeatpipeGeometry(**data["geometry"])
    mesh = d_class.HeatpipeMesh(N_R=20, N_Z=50)
    mat = d_class.HeatpipeMaterial(**data["material"])
    bc = d_class.HeatpipeBC(**data["bc"])
    wick = d_class.HeatpipeWick(**data["wick"])
    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg = cfg.resolve_geometry()

    HP = Heatpipe(cfg)
    HP_limits = HeatPipeLimitations(HP, cfg)

    HP_limits.plot_analytical_limits(700, 1400)

    # T_low = 700
    # T_high = 900
    # T_span = np.linspace(T_low, T_high, T_high - T_low + 1)
    # Q_cap_analytic = HP_limits.calculate_analytical_capillary_limit(T_span)
    # Q_cap_Busse = HP_limits.calculate_analytical_capillary_limit_Busse(T_span)

    # plt.plot(T_span, Q_cap_Busse, label="Busse")
    # plt.plot(T_span, Q_cap_analytic, label="Analytic")
    # plt.xlabel("Temperature [Kelvin]")
    # plt.ylabel("Heat transfer [W]")

    # plt.show()


    

