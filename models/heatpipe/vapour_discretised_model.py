import numpy as np

from utils.sodium_properties import calculate_Na_h_fg, calculate_Na_viscosity_v, calculate_Na_pressure_v, calculate_Na_rho_v, calculate_Na_temperature_v
from models.component import Component
from data.dataclass import *

R_constant = 8.314472
R_constant_Na = R_constant / 0.022990


class VapourDiscretised(Component):
    def __init__(self, config: HeatpipeConfigResolved, T_HP):

        self.cfg = config
        self.T_HP = T_HP

        self.r2r1 = 5e2
        self.update_Gamma = False

    def assemble(self):
        Gamma = self._calculate_Gamma()
        self.Gamma = Gamma

    def initial_guess(self):
        X_initial = np.ones(self.cfg.mesh.N_Z - 1 + self.cfg.mesh.N_Z)
        X_initial[:self.cfg.mesh.N_Z-1] = self._build_initial_velocity()
        X_initial[self.cfg.mesh.N_Z-1:] = self._build_initial_temperature()
        return X_initial
    
    def post_process(self, X):
        return self.unpack(X)
    
    def unpack(self, X):
        u = X[:self.cfg.mesh.N_Z-1]
        T = X[self.cfg.mesh.N_Z-1:]
        return u, T
    
    def pack(self, X_tuple):
        return np.r_[*X_tuple]
    
    def get_residuals(self, X):
        u, T = X[:self.cfg.mesh.N_Z-1], X[self.cfg.mesh.N_Z-1:]

        T_full = T

        u_full = np.zeros(len(u) + 2)
        u_full[1:-1] = u

        ui = u
        uim1 = u_full[:-2]
        uip1 = u_full[2:]

        Ti = T_full[1:]
        Tim1 = T_full[:-1]
        Tbar = 0.5 * (Ti + Tim1)

        Pi = calculate_Na_pressure_v(Ti)
        Pim1 = calculate_Na_pressure_v(Tim1)

        rho_full = calculate_Na_rho_v(T_full)
        rhoi = rho_full[1:]
        rhoim1 = rho_full[:-1]
        rhobar = calculate_Na_rho_v(Tbar)

        dxi = np.zeros_like(T_full)
        dxi[:self.cfg.mesh.N_evap] = self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        dxi[self.cfg.mesh.N_evap:(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic)] = self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic
        dxi[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):] = self.cfg.geometry.l_cond / self.cfg.mesh.N_cond

        lami = self._calculate_friction_factor(T_full, u)
        if self.update_Gamma:
            self.Gamma = self._update_Gamma_function(Ti, self.Gamma)

        r1 = np.zeros_like(T_full)

        r1[1:-1] = (
            (uip1[:-1] * rhoi[:-1] - ui[:-1] * rhoim1[:-1])
            - dxi[1:-1] * self.Gamma[1:-1]
        )

        r1[0] = (
            (u_full[1] * rho_full[0])
            - dxi[0] * self.Gamma[0]
        )

        r1[-1] = (
            (-u_full[-2] * rho_full[-2])
            - dxi[-1] * self.Gamma[-1]
        )

        r2 = (
            (rhoi * ui**2 - rhoim1 * uim1**2)
            # (rhoi * (ui + uip1) / 2 * ui - rhoim1 * (ui + uim1) / 2 * uim1)
            + calculate_Na_h_fg(self.T_HP[-1]) * rhobar * ((Ti - Tim1) / Tbar)
            # + (Pi - Pim1)
            + dxi[1:] * lami * (1.0 / (4.0 * self.cfg.geometry.r_vapour)) * rhobar * ui * np.abs(ui)
        )

        return np.r_[self.r2r1 * r1, r2]

    def _update_Gamma_function(self, T_i: np.ndarray, Gamma_i: np.ndarray):
        T_wick_lv_interface = np.asarray(self.T_HP)[:-1:self.cfg.mesh.N_R]
        a_W = 2.0 / self.cfg.geometry.r_vapour

        Gamma_i[-1:] = (
            a_W / calculate_Na_h_fg(self.T_HP[-1])
        ) * self.cfg.material.h_vap * (T_wick_lv_interface[-1:] - T_i[-1:])

        return Gamma_i

    def _calculate_Gamma(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.cfg.mesh.N_R]

        T_v = self.T_HP[-1]

        q_bis_surface = np.zeros(self.cfg.mesh.N_Z)
        q_bis_surface[0: self.cfg.mesh.N_evap] = self.cfg.material.h_vap * (
            T_wick_lv_interface[0: self.cfg.mesh.N_evap] - T_v
        )
        q_bis_surface[self.cfg.mesh.N_Z - self.cfg.mesh.N_cond:] = self.cfg.material.h_vap * (
            T_wick_lv_interface[self.cfg.mesh.N_Z - self.cfg.mesh.N_cond:] - T_v
        )
        
        a_W = 2 / self.cfg.geometry.r_vapour

        Gamma = a_W * q_bis_surface / calculate_Na_h_fg(self.T_HP[-1])

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
        Rei = rhobar * np.abs(u) * 2 * self.cfg.geometry.r_vapour / calculate_Na_viscosity_v(Tbar)
        return Rei
    
    def _build_initial_velocity(self):
        mdot = self._get_mdot()
        rho_v = calculate_Na_rho_v(self.T_HP[-1])
        A_v = np.pi * self.cfg.geometry.r_vapour**2
        u_guess = mdot / (rho_v * A_v)
        
        return u_guess[1:]

    def _build_initial_temperature(self) -> np.ndarray:
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.cfg.mesh.N_R]

        T_cold_wall = float(np.mean(T_wick_lv_interface[-self.cfg.mesh.N_cond:]))

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

        F = (7.0/9.0 - 1.7 * Re_re / (36 + 10*Re_re) * np.exp(-7.5 * self.cfg.geometry.l_adiabatic / (Re_re * L_e)))
        
        dP_evap = ((-4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (L_e * (1.0 + Re_re * F)))
        
        dP_adiabatic = -(8.0 * mu_v * Q_tot * self.cfg.geometry.l_adiabatic) / (rho_v * np.pi * Rv**4 * h_fg)

        dP_cond = -dP_evap + -dP_adiabatic - (4.0/np.pi) * (mu_v * Q_tot) / (rho_v * Rv**4 * h_fg) * (
            self.cfg.geometry.l_evap + 2*self.cfg.geometry.l_adiabatic + self.cfg.geometry.l_cond
        )

        dx = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        dx[:self.cfg.mesh.N_evap] = self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        dx[self.cfg.mesh.N_evap: self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic] = (
            self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic
        )
        dx[self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic:] = self.cfg.geometry.l_cond / self.cfg.mesh.N_cond
        
        dpdx = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        x_evap  = np.linspace(0, L_e, self.cfg.mesh.N_evap)
        profile = (x_evap / L_e)**2
        norm    = np.sum(profile) * (L_e / self.cfg.mesh.N_evap)
        dpdx[:self.cfg.mesh.N_evap] = dP_evap * profile / norm

        dpdx[self.cfg.mesh.N_evap:self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic] = (
            dP_adiabatic / self.cfg.geometry.l_adiabatic
        )

        j0 = self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic
        x2 = L_e + self.cfg.geometry.l_adiabatic
        x_cond = np.linspace(x2, x2 + L_C, self.cfg.mesh.N_cond)
        xi = 1.0 - (x_cond - x2) / L_C
        dP_cond_total = dP_cond
        profile = (xi)**2
        dpdx[j0:] = dP_cond_total * profile / (
            np.sum(profile) * dx[self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic:]
        )

        return np.cumsum(dpdx * dx) 

    def _analytical_pressure_drop_cotter(self):
        T_v  = self.T_HP[-1]
        h_fg = calculate_Na_h_fg(T_v)
        rho_v = calculate_Na_rho_v(T_v)
        self.viscosity_Na = calculate_Na_viscosity_v(self.T_HP[-1])

        Rv = self.cfg.geometry.r_vapour
        mu_v = self.viscosity_Na

        Av = np.pi * Rv**2
        mdot = self._build_initial_velocity() * rho_v * Av

        dmdx_evap = mdot[self.cfg.mesh.N_evap] / self.cfg.geometry.l_evap
        dmdx_adia = 0.0
        dmdx_cond = -mdot[self.cfg.mesh.N_evap] / self.cfg.geometry.l_cond

        dpdx = np.zeros_like(mdot, dtype=float)

        dpdx[:self.cfg.mesh.N_evap] = -(1.0) * (mdot[:self.cfg.mesh.N_evap] * dmdx_evap) / (4.0 * rho_v * Rv**4)

        i0 = self.cfg.mesh.N_evap
        i1 = self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic
        dpdx[i0:i1] = -(8.0 * mu_v * mdot[i0:i1]) / (rho_v * np.pi * Rv**4)

        j0 = self.cfg.mesh.N_Z - self.cfg.mesh.N_cond
        dpdx[j0:] = -(4.0 / np.pi**2*1.0) * (mdot[j0:] * dmdx_cond) / (4.0 * rho_v * Rv**4)

        dx = self.cfg.geometry.l_tot / (self.cfg.mesh.N_Z - 1)
        return np.cumsum(dpdx) * dx

    def _get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.cfg.mesh.N_R]
        T_v = self.T_HP[-1]

        A_int = 2 * np.pi * self.cfg.geometry.r_vapour * self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        Qevap = np.sum(self.cfg.material.h_vap * (T_wick_lv_interface[:self.cfg.mesh.N_evap] - T_v)) * A_int
        mdot_peak = Qevap / calculate_Na_h_fg(self.T_HP[-1])

        mdot = np.concatenate([
            np.repeat(np.array([mdot_peak/self.cfg.mesh.N_evap]), self.cfg.mesh.N_evap) * np.arange(self.cfg.mesh.N_evap),
            np.repeat(np.array([mdot_peak]), self.cfg.mesh.N_adiabatic),
            np.repeat(np.array([mdot_peak/self.cfg.mesh.N_cond]), self.cfg.mesh.N_cond) * np.arange(self.cfg.mesh.N_cond)[::-1]
        ])

        return mdot

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import json
    from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = HeatpipeGeometry(**data["geometry"])
    geom.delta_wick = 0.00025
    geom.delta_gap = 0.00025
    mesh = HeatpipeMesh(**data["mesh"])
    mesh.N_R = 20
    mesh.N_Z = 50
    mat = HeatpipeMaterial(**data["material"])
    wick = HeatpipeWick(**data["wick"])
    pipe_bc = HeatpipeBC(**data["bc"])
    pipe_bc.Q = 1000
    cfg = HeatpipeConfig(geom, mesh, mat, wick, pipe_bc)
    cfg = cfg.resolve_geometry()

    heatpipe = HeatpipeDiscretised(cfg)
    T_HP = heatpipe.linear_solve()

    vapour = VapourDiscretised(cfg, T_HP)
    solver = Solver([vapour])
    solver.newton_krylov()

    u_vap, T_vap = solver.solution

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    x = np.linspace(0, vapour.cfg.geometry.l_tot, len(T_vap))
    ax.plot(x, T_vap, color="black")

    plt.show()