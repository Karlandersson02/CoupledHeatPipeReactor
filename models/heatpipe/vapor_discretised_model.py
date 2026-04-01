import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import root, newton_krylov

from utils.sodium_properties import calculate_Na_h_fg, calculate_Na_viscosity_v, calculate_Na_pressure_v, calculate_Na_rho_v, calculate_Na_temperature_v

R_constant = 8.314472
R_constant_Na = R_constant / 0.022990

class vapour_discretised_2:
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

        self.h_vap = data.get("h_vap")

        self.T_HP = data.get("T_HP")
        self.h_fg_Na = calculate_Na_h_fg(self.T_HP[-1])
        self.viscosity_Na = calculate_Na_viscosity_v(self.T_HP[-1])

        self.T_C = data.get("T_C")
        self.P_C = data.get("P_C")

        self.R = 8.314472
        self.R_Na = self.R / 0.022990

        self.niter = 1000

    def _calculate_Gamma(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]

        T_v = self.T_HP[-1]

        q_bis_surface = np.zeros(self.N_Z)
        q_bis_surface[0: self.N_evap]          =  self.h_vap * (T_wick_lv_interface[0: self.N_evap]          - T_v)
        q_bis_surface[self.N_Z - self.N_cond:] =  self.h_vap * (T_wick_lv_interface[self.N_Z - self.N_cond:] - T_v)
        
        # Heat transfer surface area density per unit volume.
        # a_W = 2 * np.pi * self.r_vapour * delta_Z / np.pi * self.r_vapour**2 * delta_Z
        a_W = 2 / self.r_vapour

        Gamma = a_W * q_bis_surface / self.h_fg_Na

        return Gamma

    def _calculate_friction_factor(self, T, u):
        Rei = self._calculate_reynolds(T, u)

        lami = np.zeros_like(Rei)
        lami[Rei <= 2200] = 64 / Rei[Rei <= 2200]
        lami[Rei > 3000] = 0.316 / Rei[Rei > 3000]**0.25
        lami[(Rei > 2200) & (Rei <= 3000)] = Rei[(Rei > 2200) & (Rei <= 3000)] * 1.70088e-5 - 0.00832838
        lami[Rei < 1e-10] = 1e-10

        return lami

    def _calculate_reynolds(self, T, u):
        Tbar = 0.5 * (T[1:] + T[:-1])
        rhobar = calculate_Na_rho_v(Tbar)
        Rei = rhobar * np.abs(u) * 2 * self.r_vapour / self.viscosity_Na
        return Rei
    
    def _get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]
        T_v = self.T_HP[-1]

        A_int = 2 * np.pi * self.r_vapour * self.l_evap / self.N_evap
        Qevap = np.sum(self.h_vap * (T_wick_lv_interface[:self.N_evap] - T_v)) * A_int
        mdot_peak = Qevap / self.h_fg_Na

        mdot = np.concatenate([
            np.repeat(np.array([mdot_peak/self.N_evap]), self.N_evap) * np.arange(self.N_evap),
            np.repeat(np.array([mdot_peak]), self.N_adiabatic),
            np.repeat(np.array([mdot_peak/self.N_cond]), self.N_cond) * np.arange(self.N_cond)[::-1]
        ])

        return mdot
    
    def update_Gamma(self, T_i: np.ndarray, Gami: np.ndarray):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R] 
        a_W = 2 / self.r_vapour 
        Gami[-1: ] = (a_W / self.h_fg_Na) * self.h_vap * (T_wick_lv_interface[-1: ] - T_i[-1: ])
        return Gami

    def get_residuals(
        self,
        u: np.ndarray,
        T: np.ndarray,
        Gamma: np.ndarray
    ):
        u_full = np.zeros(len(u) + 2)
        u_full[1:-1] = u
        ui   = u
        uim1 = u_full[:-2]
        uip1 = u_full[2:]

        T_full = T
        Ti    = T_full[1:]
        Tim1  = T_full[:-1]
        Tbar  = 0.5 * (Ti + Tim1)

        # Pi   = calculate_Na_pressure_v(Ti)
        # Pim1 = calculate_Na_pressure_v(Tim1)

        rho_full = calculate_Na_rho_v(T_full)
        rhoi     = rho_full[1:]
        rhoim1   = rho_full[:-1]
        rhobar   = calculate_Na_rho_v(Tbar)

        dxi = np.zeros_like(T_full)
        dxi[:self.N_evap] = self.l_evap / self.N_evap
        dxi[self.N_evap:(self.N_evap + self.N_adiabatic)] = self.l_adiabatic / self.N_adiabatic
        dxi[(self.N_evap + self.N_adiabatic):] = self.l_cond / self.N_cond

        lami = self._calculate_friction_factor(T_full, u)

        r1 = np.zeros_like(T_full)

        r1[1:-1] = (
            (uip1[:-1] * rhoi[:-1] - ui[:-1] * rhoim1[:-1])
            - dxi[1:-1] * Gamma[1:-1]
        )

        r1[0] = (
            (u_full[1] * rho_full[0])
            - dxi[0] * Gamma[0]
        )

        r1[-1] = (
            (-u_full[-2] * rho_full[-2])
            - dxi[-1] * Gamma[-1]
        )

        r2 = (
            (rhoi * ui**2 - rhoim1 * uim1**2)
            # (rhoi * (ui + uip1) / 2 * ui - rhoim1 * (ui + uim1) / 2 * uim1)
            + self.h_fg_Na * rhobar * ((Ti - Tim1) / Tbar)
            # + (Pi - Pim1)
            + dxi[1:] * lami * (1.0 / (4.0 * self.r_vapour)) * rhobar * ui * np.abs(ui)
        )

        return np.concatenate([self.r2r1 * r1, r2])
    
    def _build_initial_velocity(self):
        mdot = self._get_mdot()
        rho_v = calculate_Na_rho_v(self.T_HP[-1])
        A_v = np.pi * self.r_vapour**2
        u_guess = mdot / (rho_v * A_v)
        
        return u_guess[1:]

    def _build_initial_temperature(self) -> np.ndarray:
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.N_R]

        # A good "cold end" estimate: average wick-LV interface temperature in condenser
        T_cold_wall = float(np.mean(T_wick_lv_interface[-self.N_cond:]))

        analytical_pressure_drop_q = self._analytical_pressure_drop_Busse()
        P_cond_end = calculate_Na_pressure_v(T_cold_wall)
        P_profile = np.array(analytical_pressure_drop_q) + P_cond_end - analytical_pressure_drop_q[-1]
        T_profile_analytic = calculate_Na_temperature_v(P_profile)

        return T_profile_analytic
    
    def _analytical_pressure_drop_Busse(self):
        T_v   = self.T_HP[-1]
        h_fg  = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        mu_v  = calculate_Na_viscosity_v(T_v)

        Rv  = self.r_vapour
        d_v = 2.0 * Rv
        L_C = self.l_cond
        L_e = self.l_evap

        Q_tot = self._get_mdot()[self.N_evap] * h_fg

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

    def analytical_pressure_drop(self):
        T_v  = self.T_HP[-1]
        h_fg = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        self.viscosity_Na = calculate_Na_viscosity_v(self.T_HP[-1])

        Rv  = self.r_vapour          # make sure this is in meters
        mu_v = self.viscosity_Na

        Av = np.pi * Rv**2
        mdot = self._build_initial_velocity() * rho_v * Av

        dmdx_evap = mdot[self.N_evap] / self.l_evap
        dmdx_adia = 0.0
        dmdx_cond = -mdot[self.N_evap] / self.l_cond

        dpdx = np.zeros_like(mdot, dtype=float)

        # Evaporator: q>0 => s=1, a=0
        dpdx[:self.N_evap] = -(1.0) * (mdot[:self.N_evap] * dmdx_evap) / (4.0 * rho_v * Rv**4)

        # Adiabatic: q=0 => s=0, a=1
        i0 = self.N_evap
        i1 = self.N_evap + self.N_adiabatic
        dpdx[i0:i1] = -(8.0 * mu_v * mdot[i0:i1]) / (rho_v * np.pi * Rv**4)

        # Condenser: q<0 => s=4/pi^2, a=0
        j0 = self.N_Z - self.N_cond
        dpdx[j0:] = -(4.0 / np.pi**2*1.0) * (mdot[j0:] * dmdx_cond) / (4.0 * rho_v * Rv**4)

        # Integrate dp/dx over x
        dx = self.l_tot / (self.N_Z - 1)
        # return np.sum(dpdx) * dx
        return np.cumsum(dpdx) * dx
    
    def converge_numeric(self, Gamma):
        initial_guess = np.ones(self.N_Z - 1 + self.N_Z)
        initial_guess[:self.N_Z-1] = self._build_initial_velocity()
        initial_guess[self.N_Z-1:] = self._build_initial_temperature()

        def residual_wrapper(initial_guess):
            return self.get_residuals(
                initial_guess[:(self.N_Z - 1)],
                initial_guess[(self.N_Z - 1):],
                Gamma
            )

        self.r2r1 = 50
        self.train_history = np.zeros((1, len(initial_guess)))
        def iteration_callback(x, f):
            self.train_history = np.append(self.train_history, x[None], axis=0)
            print(f"i: {len(self.train_history)-1}\t|F| = {np.linalg.norm(f, np.inf):.3g}\t|r1| = {np.linalg.norm(f[:self.N_Z], np.inf):.3g}\t|r2| = {np.linalg.norm(f[self.N_Z:], np.inf):.3g}")

        sol_krylov = newton_krylov(
            residual_wrapper,
            initial_guess,
            iter = self.niter,
            verbose = False,
            method = "lgmres",
            callback = iteration_callback
        )

        self.train_history = self.train_history[1:]

        T = sol_krylov[(self.N_Z - 1):]
        u = sol_krylov[:(self.N_Z - 1)]

        return T, u

    def solve_numeric(self):
        Gamma_hat = self._calculate_Gamma()

        T, u = self.converge_numeric(Gamma_hat)

        P = calculate_Na_pressure_v(T)

        self.T = T
        self.P_numeric = P
        self.u = u

    def get_temperature(self):
        return self.T
    
    def get_pressure_numeric(self):
        return self.P_numeric

    def get_velocity(self):
        return self.u

from dataclasses import dataclass
from project_data.heatpipe_dataclasses import *

from models.component import Component

class vapour_discretised(Component):
    def __init__(self, config: VapourConfigResolved):

        self.cfg = config
        self.r2r1 = 50

        # temporary

        mdot_flux0 = self._get_mdot()[self.cfg.mesh.N_evap] / (self.cfg.geometry.r_vapour**2 * np.pi)
        self.L0 = self.cfg.geometry.l_tot
        self.T0 = self.cfg.vapour_bc.T_HP[-1]
        self.P0 = calculate_Na_pressure_v(self.cfg.vapour_bc.T_HP[-1])
        self.rho0 = calculate_Na_rho_v(self.T0)
        self.U0 = mdot_flux0 / self.rho0
        self.Gamma0 = self.rho0 * self.U0 / self.L0

    def assemble(self):
        Gamma_hat = self._calculate_Gamma() / self.Gamma0
        self.Gamma_hat = Gamma_hat

    def initial_guess(self):
        X_initial = np.ones(self.cfg.mesh.N_Z - 1 + self.cfg.mesh.N_Z)
        X_initial[:self.cfg.mesh.N_Z-1] = self._build_initial_velocity() / self.U0
        X_initial[self.cfg.mesh.N_Z-1:] = self._build_initial_temperature() / self.T0
        return X_initial
    
    def post_process(self, X):
        u = X[:self.cfg.mesh.N_Z-1]
        T = X[self.cfg.mesh.N_Z-1:]
        return (u, T)
    
    def pack(self, X_tuple):
        return np.r_[*X_tuple]

    def get_residuals(self, X):
        u_hat, T_hat = X[:self.cfg.mesh.N_Z-1], X[self.cfg.mesh.N_Z-1:]

        T_full_hat = T_hat
        T_full = self.T0 * T_full_hat

        u_full_hat = np.zeros(len(u_hat) + 2)
        u_full_hat[1:-1] = u_hat
        u_full = self.U0 * u_full_hat

        ui_hat = u_hat
        ui = self.U0 * ui_hat

        uim1_hat = u_full_hat[:-2]
        uip1_hat = u_full_hat[2:]
        uim1 = self.U0 * uim1_hat
        uip1 = self.U0 * uip1_hat

        Ti_hat = T_full_hat[1:]
        Tim1_hat = T_full_hat[:-1]
        Tbar_hat = 0.5 * (Ti_hat + Tim1_hat)

        Ti = self.T0 * Ti_hat
        Tim1 = self.T0 * Tim1_hat
        Tbar = self.T0 * Tbar_hat

        Pi_hat = calculate_Na_pressure_v(Ti) / self.P0
        Pim1_hat = calculate_Na_pressure_v(Tim1) / self.P0

        rho_full = calculate_Na_rho_v(T_full)
        rhoi = rho_full[1:]
        rhoim1 = rho_full[:-1]
        rhobar = calculate_Na_rho_v(Tbar)

        rho_full_hat = rho_full / self.rho0
        rhoi_hat = rhoi / self.rho0
        rhoim1_hat = rhoim1 / self.rho0
        rhobar_hat = rhobar / self.rho0

        dxi = np.zeros_like(T_full)
        dxi[:self.cfg.mesh.N_evap] = self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        dxi[self.cfg.mesh.N_evap:(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic)] = self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic
        dxi[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):] = self.cfg.geometry.l_cond / self.cfg.mesh.N_cond

        dxi_hat = dxi / self.L0

        lami = self._calculate_friction_factor(T_full_hat, u_hat)

        r1_hat = np.zeros_like(T_full_hat)

        r1_hat[1:-1] = (
            (uip1_hat[:-1] * rhoi_hat[:-1] - ui_hat[:-1] * rhoim1_hat[:-1])
            - dxi_hat[1:-1] * (self.Gamma_hat[1:-1] * (self.Gamma0 * self.L0 / (self.rho0 * self.U0)))
        )

        r1_hat[0] = (
            (u_full_hat[1] * rho_full_hat[0])
            - dxi_hat[0] * (self.Gamma_hat[0] * (self.Gamma0 * self.L0 / (self.rho0 * self.U0)))
        )

        r1_hat[-1] = (
            (-u_full_hat[-2] * rho_full_hat[-2])
            - dxi_hat[-1] * (self.Gamma_hat[-1] * (self.Gamma0 * self.L0 / (self.rho0 * self.U0)))
        )

        hfg_over_U2 = calculate_Na_h_fg(self.cfg.vapour_bc.T_HP[-1]) / (self.U0**2)
        geom = self.L0 / (4.0 * self.cfg.geometry.r_vapour)


        r2_hat = (
            (rhoi_hat * ui_hat**2 - rhoim1_hat * uim1_hat**2)
            # (rhoi_hat * (ui_hat + uip1_hat) / 2 * ui_hat - rhoim1_hat * (ui_hat + uim1_hat) / 2 * uim1_hat)
            + hfg_over_U2 * rhobar_hat * ((Ti_hat - Tim1_hat) / Tbar_hat)
            # + (self.P0 / (self.rho0 * self.U0**2)) * (Pi_hat - Pim1_hat)
            + dxi_hat[1:] * lami * geom * rhobar_hat * ui_hat * np.abs(ui_hat)
        )

        return np.r_[self.r2r1 * r1_hat, r2_hat]

    def _calculate_Gamma(self):
        T_wick_lv_interface = np.array(self.cfg.vapour_bc.T_HP)[:-1:self.cfg.mesh.N_R]

        T_v = self.cfg.vapour_bc.T_HP[-1]

        q_bis_surface = np.zeros(self.cfg.mesh.N_Z)
        q_bis_surface[0: self.cfg.mesh.N_evap]          =  self.cfg.material.h_vap * (T_wick_lv_interface[0: self.cfg.mesh.N_evap]          - T_v)
        q_bis_surface[self.cfg.mesh.N_Z - self.cfg.mesh.N_cond:] =  self.cfg.material.h_vap * (T_wick_lv_interface[self.cfg.mesh.N_Z - self.cfg.mesh.N_cond:] - T_v)
        
        # Heat transfer surface area density per unit volume.
        # a_W = 2 * np.pi * self.r_vapour * delta_Z / np.pi * self.r_vapour**2 * delta_Z
        a_W = 2 / self.cfg.geometry.r_vapour

        Gamma = a_W * q_bis_surface / calculate_Na_h_fg(self.cfg.vapour_bc.T_HP[-1])

        return Gamma

    def _calculate_friction_factor(self, T, u):
        Rei = self._calculate_reynolds(T, u)

        lami = np.zeros_like(Rei)
        lami[Rei <= 2200] = 64 / Rei[Rei <= 2200]
        lami[Rei > 3000] = 0.316 / Rei[Rei > 3000]**0.25
        lami[(Rei > 2200) & (Rei <= 3000)] = Rei[(Rei > 2200) & (Rei <= 3000)] * 1.70088e-5 - 0.00832838
        lami[Rei < 1e-10] = 1e-10

        return lami

    def _calculate_reynolds(self, T_hat, u_hat):
        Tbar = self.T0 * 0.5 * (T_hat[1:] + T_hat[:-1])
        rhobar = calculate_Na_rho_v(Tbar)
        u = self.U0 * u_hat
        Rei = rhobar * np.abs(u) * 2 * self.cfg.geometry.r_vapour / calculate_Na_viscosity_v(Tbar)
        return Rei
    
    def _build_initial_velocity(self):
        mdot = self._get_mdot()
        rho_v = calculate_Na_rho_v(self.cfg.vapour_bc.T_HP[-1])
        A_v = np.pi * self.cfg.geometry.r_vapour**2
        u_guess = mdot / (rho_v * A_v)
        
        return u_guess[1:]

    def _build_initial_temperature(self) -> np.ndarray:
        T_wick_lv_interface = np.array(self.cfg.vapour_bc.T_HP)[:-1:self.cfg.mesh.N_R]

        # A good "cold end" estimate: average wick-LV interface temperature in condenser
        T_cold_wall = float(np.mean(T_wick_lv_interface[-self.cfg.mesh.N_cond:]))

        analytical_pressure_drop_q = self._analytical_pressure_drop_Busse()
        P_cond_end = calculate_Na_pressure_v(T_cold_wall)
        P_profile = np.array(analytical_pressure_drop_q) + P_cond_end - analytical_pressure_drop_q[-1]
        T_profile_analytic = calculate_Na_temperature_v(P_profile)

        return T_profile_analytic
    
    def _analytical_pressure_drop_Busse(self):
        T_v   = self.cfg.vapour_bc.T_HP[-1]
        h_fg  = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        mu_v  = calculate_Na_viscosity_v(T_v)

        Rv  = self.cfg.geometry.r_vapour
        d_v = 2.0 * Rv
        L_C = self.cfg.geometry.l_cond
        L_e = self.cfg.geometry.l_evap

        Q_tot = self._get_mdot()[self.cfg.mesh.N_evap] * h_fg

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

    def _analytical_pressure_drop_cotter(self):
        T_v  = self.cfg.vapour_bc.T_HP[-1]
        h_fg = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        self.viscosity_Na = calculate_Na_viscosity_v(self.cfg.vapour_bc.T_HP[-1])

        Rv  = self.cfg.geometry.r_vapour          # make sure this is in meters
        mu_v = self.viscosity_Na

        Av = np.pi * Rv**2
        mdot = self._build_initial_velocity() * rho_v * Av

        dmdx_evap = mdot[self.cfg.mesh.N_evap] / self.cfg.geometry.l_evap
        dmdx_adia = 0.0
        dmdx_cond = -mdot[self.cfg.mesh.N_evap] / self.cfg.geometry.l_cond

        dpdx = np.zeros_like(mdot, dtype=float)

        # Evaporator: q>0 => s=1, a=0
        dpdx[:self.cfg.mesh.N_evap] = -(1.0) * (mdot[:self.cfg.mesh.N_evap] * dmdx_evap) / (4.0 * rho_v * Rv**4)

        # Adiabatic: q=0 => s=0, a=1
        i0 = self.cfg.mesh.N_evap
        i1 = self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic
        dpdx[i0:i1] = -(8.0 * mu_v * mdot[i0:i1]) / (rho_v * np.pi * Rv**4)

        # Condenser: q<0 => s=4/pi^2, a=0
        j0 = self.cfg.mesh.N_Z - self.cfg.mesh.N_cond
        dpdx[j0:] = -(4.0 / np.pi**2*1.0) * (mdot[j0:] * dmdx_cond) / (4.0 * rho_v * Rv**4)

        # Integrate dp/dx over x
        dx = self.cfg.geometry.l_tot / (self.cfg.mesh.N_Z - 1)
        # return np.sum(dpdx) * dx
        return np.cumsum(dpdx) * dx

    def _get_mdot(self):
        T_wick_lv_interface = np.array(self.cfg.vapour_bc.T_HP)[:-1:self.cfg.mesh.N_R]
        T_v = self.cfg.vapour_bc.T_HP[-1]

        A_int = 2 * np.pi * self.cfg.geometry.r_vapour * self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        Qevap = np.sum(self.cfg.material.h_vap * (T_wick_lv_interface[:self.cfg.mesh.N_evap] - T_v)) * A_int
        mdot_peak = Qevap / calculate_Na_h_fg(self.cfg.vapour_bc.T_HP[-1])

        mdot = np.concatenate([
            np.repeat(np.array([mdot_peak/self.cfg.mesh.N_evap]), self.cfg.mesh.N_evap) * np.arange(self.cfg.mesh.N_evap),
            np.repeat(np.array([mdot_peak]), self.cfg.mesh.N_adiabatic),
            np.repeat(np.array([mdot_peak/self.cfg.mesh.N_cond]), self.cfg.mesh.N_cond) * np.arange(self.cfg.mesh.N_cond)[::-1]
        ])

        return mdot

if __name__ == "__main__":

    import json
    from models.heatpipe.heat_discretised_model import HeatpipeDiscretised
    from utils.solver import Solver

    with open("./project_data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(**data["mesh"])
    mat = HeatpipeMaterial(**data["material"])
    pipe_bc = HeatpipeBC(**data["bc"])
    cfg = HeatpipeConfig(geom, mesh, mat, pipe_bc)
    cfg = cfg.resolve()

    heatpipe = HeatpipeDiscretised(cfg)
    T_HP = heatpipe.linear_solve()

    vap_bc = VapourBC(T_HP=T_HP)
    vap_cfg = VapourConfig(geom, mesh, mat, vap_bc)
    vap_cfg = vap_cfg.resolve()

    vapour = vapour_discretised(vap_cfg)
    solver = Solver([vapour])
    solver.newton_krylov()

    u_vap, T_vap = solver.solution

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    x = np.linspace(0, vapour.cfg.geometry.l_tot, len(T_vap))
    ax.plot(x, T_vap)

    plt.show()