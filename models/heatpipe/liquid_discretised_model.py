import numpy as np
from scipy.optimize import root, newton_krylov
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils.sodium_properties import calculate_Na_rho_l, calculate_Na_viscosity_l, calculate_Na_h_fg
from project_data.heatpipe_dataclasses import *

class LiquidDiscretised:
    def __init__(self, config, T_HP):
        self.cfg = config
        self.T_HP = T_HP

    def get_mdot(self):
        T_wick_lv_interface = np.array(self.T_HP)[:-1:self.cfg.mesh.N_R]
        T_v = self.T_HP[-1]

        A_int = 2 * np.pi * self.cfg.geometry.r_vapour * self.cfg.geometry.l_evap / self.cfg.mesh.N_evap
        Qevap = np.sum(self.cfg.material.h_vap * (T_wick_lv_interface[:self.cfg.mesh.N_evap] - T_v)) * A_int
        mdot = Qevap / calculate_Na_h_fg(T_v)

        mdot = np.concatenate([
            np.repeat(np.array([mdot/self.cfg.mesh.N_evap]), self.cfg.mesh.N_evap) * np.arange(self.cfg.mesh.N_evap),
            np.repeat(np.array([mdot]), self.cfg.mesh.N_adiabatic),
            np.repeat(np.array([mdot/self.cfg.mesh.N_cond]), self.cfg.mesh.N_cond) * np.arange(self.cfg.mesh.N_cond)[::-1]
        ])
        
        return mdot


    def calculate_K_annular_wick(self):
        self.r_2 = self.cfg.geometry.r_wick
        self.r_1 = self.cfg.geometry.r_gap

        R_star = self.r_2 / self.r_1

        fRe_l = 16 * (1 - R_star)**2 / (1 + R_star**2 - (1 - R_star**2)/np.log(1/R_star))

        D_h = 2 * (self.r_1 - self.r_2)

        K = D_h**2 / (2 * fRe_l)

        return K

    def get_pressure_drop_profile(self):
        if self.cfg.wick.Is_annular == True:
            A_wick = np.pi * (self.cfg.geometry.r_gap**2 - self.cfg.geometry.r_wick**2) 
        else:
            A_wick = np.pi * (self.cfg.geometry.r_wick**2 - self.cfg.geometry.r_vapour**2)
        
        T_wick = np.mean(np.array(self.T_HP)[:-1].reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)[:, :self.cfg.mesh.N_wick], axis=1) 

        mu_l = calculate_Na_viscosity_l(T_wick)  
        rho_l = calculate_Na_rho_l(T_wick)

        if self.cfg.wick.Is_annular == True:
            K = self.calculate_K_annular_wick()
        else:
            K = self.cfg.wick.K

        delta_z = np.concatenate(
            [np.ones(self.cfg.mesh.N_evap) * self.cfg.geometry.l_evap / self.cfg.mesh.N_evap,
            np.ones(self.cfg.mesh.N_adiabatic) * self.cfg.geometry.l_adiabatic / self.cfg.mesh.N_adiabatic,
            np.ones(self.cfg.mesh.N_cond) * self.cfg.geometry.l_cond / self.cfg.mesh.N_cond]
            )
        
        mdot = self.get_mdot()
        
        P = -np.cumsum(mu_l * np.array(mdot) / (rho_l * A_wick * K) * delta_z)
        self.pressure = P

        return P


if __name__ == "__main__":
    # "T_HP":[0],
    # "Is_annular": True,
    # "K":1e-10,
    import json
    from models.heatpipe.heat_discretised_model import HeatpipeDiscretised
    from utils.solver import Solver
    with open("./project_data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]

    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(**data["mesh"])
    mat = HeatpipeMaterial(**data["material"])
    wick = HeatpipeWick(**data["wick"])
    bc = HeatpipeBC(**data["bc"])
    heatpipe_cfg = HeatpipeConfig(geom, mesh, mat, bc, wick)
    heatpipe_cfg = heatpipe_cfg.resolve()

    heatpipe = HeatpipeDiscretised(heatpipe_cfg)
    solver = Solver([heatpipe])
    solver.newton_krylov()

    T_HP = heatpipe.pack(solver.solution)

    data["wick"]["K"] = 1e-10
    data["wick"]["Is_annular"] = True
    liquid_cfg = LiquidConfig(geom, mesh, mat, wick)
    liquid_cfg = liquid_cfg.resolve()
    liquid = LiquidDiscretised(liquid_cfg, T_HP)

    P = liquid.get_pressure_drop_profile()

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    x = np.linspace(0, heatpipe.cfg.geometry.l_tot, len(P))
    ax.grid(alpha=0.4)
    ax.plot(x, P, color="black")
    ax.set_xlim([heatpipe.cfg.geometry.l_tot, 0])

    ax.set_xlabel("l")
    ax.set_ylabel("P")

    plt.show()