import numpy as np

import data.dataclass as d_class
import utils.sodium_properties as s_props

from models.component import Component

class VapourDiscretised(Component):
    def __init__(self, config: d_class.HeatpipeConfigResolved, T_HP=None):
        self.cfg = config
        self.T_HP = None
        self.dx = self._calculate_dx()
        self.loss_mult = 1

        if T_HP is not None and not np.isscalar(T_HP):
            self.set_T_HP(T_HP)

    def set_T_HP(self, T_HP):
        self.T_HP = T_HP.copy().reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

    def assemble(self):
        return

    def initial_guess(self):
        T_int = self._get_interface_temperature()

        if self.cfg.mesh.N_evap > 0:
            T_hot = float(np.mean(T_int[:self.cfg.mesh.N_evap]))
        else:
            T_hot = float(T_int[0])

        if self.cfg.mesh.N_cond > 0:
            T_cold = float(np.mean(T_int[-self.cfg.mesh.N_cond:]))
        else:
            T_cold = float(T_int[-1])

        T_v0 = np.linspace(T_hot, T_cold, self.cfg.mesh.N_Z)
        u0 = self._build_initial_velocity(T_v0)

        return self.pack((u0, T_v0))

    def post_process(self, X):
        return self.unpack(X)

    def unpack(self, X):
        u = X[:self.cfg.mesh.N_Z - 1]
        T_v = X[self.cfg.mesh.N_Z - 1:]
        return u, T_v

    def pack(self, X_tuple):
        return np.r_[*X_tuple]

    def get_residuals(self, X):
        u, T_v = self.unpack(X)

        u_full = np.zeros(self.cfg.mesh.N_Z + 1, dtype=float)
        u_full[1:-1] = u

        ui = u
        uim1 = u_full[:-2]
        uip1 = u_full[2:]

        Ti = T_v[1:]
        Tim1 = T_v[:-1]
        Tbar = 0.5 * (Ti + Tim1)

        p = s_props.calculate_Na_pressure_v(T_v)
        dp = p[1:] - p[:-1]

        rho_full = np.asarray(self.calculate_rho(T_v), dtype=float)
        rhoi = rho_full[1:]
        rhoim1 = rho_full[:-1]
        rhobar = np.asarray(self.calculate_rho(Tbar), dtype=float)

        h_fg_bar = np.asarray(self.calculate_hfg(Tbar), dtype=float)
        lami = self._calculate_friction_factor(T_v, u)
        Gamma = self._calculate_Gamma(T_v)

        Dh = 2 * self.cfg.geometry.r_vapour

        r1 = np.zeros_like(T_v)

        r1[1:-1] = (
            (uip1[:-1] * rhoi[:-1] - ui[:-1] * rhoim1[:-1])
            - self.dx[1:-1] * Gamma[1:-1]
        )

        r1[0] = (
            u_full[1] * rho_full[0]
            - self.dx[0] * Gamma[0]
        )

        r1[-1] = (
            -u_full[-2] * rho_full[-2]
            - self.dx[-1] * Gamma[-1]
        )

        r2 = (
            self.loss_mult * (rhoi * ui**2 - rhoim1 * uim1**2)
            # loss_mult * (rhoi * (ui + uip1)/2*ui - rhoim1 * (ui + uim1)/2*uim1)
            # + h_fg_bar * rhobar * ((Ti - Tim1) / Tbar)
            + dp
            + self.dx[1:] * lami * (1.0 / (2*Dh)) * rhobar * ui * np.abs(ui)
        )

        return np.r_[r1, r2]
    
    def calculate_rho(self, T):
        return s_props.calculate_Na_rho_v(T)
        # return s_props.calculate_rho_cc(T)
    
    def calculate_hfg(self, T):
        return s_props.calculate_Na_h_fg(T)
        return np.full_like(T, 4.182e6)
    
    def calculate_viscosity(self, T):
        return s_props.calculate_Na_viscosity_v(T)
        # return np.full_like(T, 1.80e-5)

    def _calculate_dx(self):
        dx = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        dx[:self.cfg.mesh.N_evap] = self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        dx[self.cfg.mesh.N_evap:(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic)] = (
            self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic
        )
        dx[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):] = (
            self.cfg.geometry.l_cond / self.cfg.mesh.N_cond
        )

        return dx

    def _get_interface_temperature(self):
        if self.T_HP is None:
            raise ValueError("T_HP has not been set.")
        return self.T_HP[:, 0]

    # def _calculate_Gamma(self, T_v):
    #     T_int = self._get_interface_temperature()

    #     q_bis_surface = np.zeros(self.cfg.mesh.N_Z, dtype=float)

    #     evap = slice(0, self.cfg.mesh.N_evap)
    #     cond = slice(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond, self.cfg.mesh.N_Z)

    #     q_bis_surface[evap] = self.cfg.material.h_vap * (T_int[evap] - T_v[evap])
    #     q_bis_surface[cond] = self.cfg.material.h_vap * (T_int[cond] - T_v[cond])

    #     a_W = 2.0 / self.cfg.geometry.r_vapour
    #     h_fg = np.asarray(self.calculate_hfg(T_v), dtype=float)

    #     return a_W * q_bis_surface / h_fg

    def _calculate_Gamma(self, T_v):
        T_int = self._get_interface_temperature()

        q_bis_surface = self.cfg.material.h_vap * (T_int - T_v)

        a_W = 2.0 / self.cfg.geometry.r_vapour
        h_fg = np.asarray(self.calculate_hfg(T_v), dtype=float)

        return a_W * q_bis_surface / h_fg

    def _calculate_friction_factor(self, T_v, u):
        Re = self._calculate_reynolds(T_v, u)
        Re_safe = np.maximum(Re, 1e-10)

        lam = np.zeros_like(Re_safe)

        laminar = Re_safe <= 2200.0
        turbulent = Re_safe > 3000.0
        transitional = (~laminar) & (~turbulent)

        lam[laminar] = 64.0 / Re_safe[laminar]
        lam[turbulent] = 0.316 / Re_safe[turbulent] ** 0.25
        lam[transitional] = (
            Re_safe[transitional] * 1.70088e-5 - 0.00832838
        )

        return lam

    def _calculate_reynolds(self, T_v, u):
        Tbar = 0.5 * (T_v[1:] + T_v[:-1])
        rhobar = np.asarray(self.calculate_rho(Tbar), dtype=float)
        mu = np.asarray(self.calculate_viscosity(Tbar), dtype=float)

        return rhobar * np.abs(u) * 2.0 * self.cfg.geometry.r_vapour / mu

    def _build_initial_velocity(self, T_v):
        mdot_faces = self._calculate_mdot_faces(T_v)
        A_v = np.pi * self.cfg.geometry.r_vapour**2

        T_v_bar = (T_v[:-1] + T_v[1:]) / 2
        rho_bar = self.calculate_rho(T_v_bar)

        u_initial = mdot_faces / (rho_bar * A_v)

        return u_initial

    def _calculate_mdot_faces(self, T_v):
        Gamma = self._calculate_Gamma(T_v)
        A_v = np.pi * self.cfg.geometry.r_vapour**2

        mdot_cells = Gamma * A_v * self.dx
        return np.cumsum(mdot_cells[:-1])

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import json
    from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = d_class.HeatpipeGeometry(**data["geometry"])
    mesh = d_class.HeatpipeMesh(N_R=20, N_Z=50)
    mat = d_class.HeatpipeMaterial(**data["material"])
    wick = d_class.HeatpipeWick(**data["wick"])
    pipe_bc = d_class.HeatpipeBC(**data["bc"])
    cfg = d_class.HeatpipeConfig(geom, mesh, mat, wick, pipe_bc)
    cfg = cfg.resolve_geometry()

    heatpipe = HeatpipeDiscretised(cfg)
    T_HP = heatpipe.linear_solve()

    vapour = VapourDiscretised(cfg, T_HP)
    solver = Solver([vapour])
    solver.newton_krylov()

    u_vap, T_vap = solver.solution # type: ignore

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    x = np.linspace(0, vapour.cfg.geometry.l_tot, len(T_vap))
    ax.plot(x, T_vap, color="black")

    plt.show()