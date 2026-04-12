import numpy as np
import matplotlib.pyplot as plt

import json

from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.moderator.mesh_conduction_model import ModeratorDiscretisedMesh
from models.moderator.triangle_mesh import UnstructuredMesh

from models.component import Component
from project_data.heatpipe_dataclasses import *
from project_data.neutronics_dataclasses import *

class Reactor(Component):
    def __init__(self, cfg_FP, cfg_HP: HeatpipeConfigResolved, cfg_N: NeutronicsConfig):

        self.cfg_FP = cfg_FP
        self.cfg_HP = cfg_HP
        self.cfg_N = cfg_N

        self.heat_pipe_thermal_model = HeatpipeDiscretised(cfg_HP)
        self.fuel_pin_thermal_model = FuelPin(cfg_FP)
        self.neutron_flux_model = NeutronicsModel(cfg_N)
        
        self.moderator_eff_res = 0.03
        self.T_cond = 300.

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1
        self.N_FP = self.cfg_N.mesh.N_R  * self.cfg_N.mesh.N_Z
        self.N_N  = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

    def assemble(self):
        return
    
    def post_process(self, X):
        T_HP             = self.T_cond * X[:self.N_HP]
        T_FP             = self.T_cond * X[self.N_HP:(self.N_HP + self.N_FP)]
        phi_ng_hat_and_k = X[-self.N_N:]
        
        T_solid    = T_HP[:-1].reshape((self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R))
        T_vap      = T_HP[-1]
        T_FP       = T_FP.reshape((self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R))
        phi_ng_hat = phi_ng_hat_and_k[:-1].reshape((self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G))
        k          = phi_ng_hat_and_k[-1]

        return ((T_solid, T_vap), T_FP, (phi_ng_hat, k))

    def pack(self, X_tuple):
        ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = X_tuple
        X = np.r_[T_solid, T_vap, T_FP, phi_ng_hat, k]
        return X

    def initial_guess(self):
        X_initial = np.ones(self.N_var)
        return X_initial
    
    def get_residuals(self, X):
        T_HP = self.T_cond * X[:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1)]
        T_FP = self.T_cond * X[(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1):((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z)]
        phi_ng_hat = X[((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z):]

        # self.fuel_pin_thermal_model.initialize_discretization()

        T_FP_ave = np.mean(T_FP.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.mesh.N_R), axis=1)
        qr = self.calculate_qr(T_FP, phi_ng_hat[:-1]) 
        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        self.heat_pipe_thermal_model.cfg.bc.Q = Q_HP
        # self.heat_pipe_thermal_model.assemble()
        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP)
        
        self.fuel_pin_thermal_model.qr = qr
        self.fuel_pin_thermal_model.T_mod = T_mod
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M  = np.mean(T_mod)
        self.neutron_flux_model.T_HP = np.mean(T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap], dtype=float)
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP_ave, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP_ave)

        qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W / m

        power_rel = np.sum(qr_rel)

        return qr_rel * self.cfg_N.energy.power / (power_rel * self.cfg_FP.mesh.N_fuel)
            

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_edge_FP = T_FP[self.cfg_FP.mesh.N_R - 1::self.cfg_FP.mesh.N_R]
        T_edge_HP = T_HP[self.cfg_HP.mesh.N_R - 1:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap):self.cfg_HP.mesh.N_R]

        Delta_z = self.cfg_HP.geometry.l_evap / self.cfg_HP.mesh.N_evap
        R_seg = self.moderator_eff_res / Delta_z
        Q_HP = (T_edge_FP - T_edge_HP) / R_seg

        T_mod = T_edge_FP - Q_HP / (self.cfg_FP.material.h_mod * 2 * self.cfg_FP.geometry.r * np.pi * self.cfg_FP.geometry.l / self.cfg_FP.mesh.N_Z)

        return Q_HP, T_mod

    

if __name__ == "__main__":

    with open("./project_data/reactor_data.json", "r") as f:
        data = json.load(f)
    
    # Mesh dimensions
    N_R_HP, N_R_FP, N_Z = 30, 30, 50

    # Heat pipe
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg_HP = cfg_HP.resolve()

    # Fuel pin
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=cfg_HP.mesh.N_evap)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)
    cfg_FP    = cfg_FP.resolve()

    # Neutronics 
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = cfg_HP.mesh.N_evap,
        l   = cfg_HP.geometry.l_evap
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)
    
    # Reactor ------------------
    reactor = Reactor(cfg_FP, cfg_HP, cfg_N)

    from utils.solver import Solver

    solver = Solver([reactor])
    solver.newton_krylov()

    ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = solver.solution

    # Calculating Q_out
    T_edge_cond_HP = T_solid[(cfg_HP.mesh.N_evap + cfg_HP.mesh.N_adiabatic):, -1]
    Q_out = np.sum((T_edge_cond_HP - 300.) * cfg_HP.material.h_cond * cfg_HP.geometry.r_outer * 2 * np.pi * cfg_HP.geometry.l_cond / cfg_HP.mesh.N_cond)

    r_fp = reactor.fuel_pin_thermal_model.R[1::2]

    r_hp = reactor.heat_pipe_thermal_model.R[1::2]

    z_flux = reactor.neutron_flux_model.Z

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Reactor Profiles | $k_{{eff}} = ${k:.4f} | "
        f"$Q_{{in}} = ${cfg_N.energy.power:.0f} W | "
        f"$Q_{{out}} = ${Q_out:.0f} W",
        fontsize=14
    )

    # --- Top left: Neutronics ---
    axs[0, 0].plot(z_flux, phi_ng_hat[:, 0], linewidth=2)
    axs[0, 0].set_title("Axial Neutron Flux Profile")
    axs[0, 0].set_xlabel("Length [m]")
    axs[0, 0].set_ylabel("Flux")
    axs[0, 0].grid(True)

    # --- Top right: Heat pipe evaporator ---
    axs[0, 1].plot(r_hp, T_solid[0], linewidth=2)
    axs[0, 1].set_title("Heat Pipe Evaporator Radial Temperature")
    axs[0, 1].set_xlabel("Radius [m]")
    axs[0, 1].set_ylabel("Temperature [K]")
    axs[0, 1].grid(True)

    # --- Bottom left: Fuel pin ---
    axs[1, 0].plot(r_fp, T_FP[0], linewidth=2)
    axs[1, 0].set_title("Fuel Pin Radial Temperature Profile")
    axs[1, 0].set_xlabel("Radius [m]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 0].grid(True)

    # --- Bottom right: Heat pipe condenser ---
    axs[1, 1].plot(r_hp, T_solid[-1], linewidth=2)
    axs[1, 1].set_title("Heat Pipe Condenser Radial Temperature")
    axs[1, 1].set_xlabel("Radius [m]")
    axs[1, 1].set_ylabel("Temperature [K]")
    axs[1, 1].grid(True)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()