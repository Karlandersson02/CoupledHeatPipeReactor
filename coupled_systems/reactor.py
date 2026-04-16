import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

import json

from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.moderator.mesh_conduction_model import ModeratorDiscretisedMesh
from models.moderator.triangle_mesh import UnstructuredMesh

from models.component import Component
from data.dataclass import *

class Reactor(Component):
    def __init__(self, cfg_R: ReactorConfigResolved):

        self.cfg_R = cfg_R

        self.cfg_HP: HeatpipeConfigResolved = cfg_R.HP
        self.cfg_FP: FuelPinConfigResolved = cfg_R.FP
        self.cfg_N: NeutronicsConfigResolved = cfg_R.N

        self.heat_pipe_thermal_model = HeatpipeDiscretised(self.cfg_HP)
        self.fuel_pin_thermal_model = FuelPin(self.cfg_FP)
        self.neutron_flux_model = NeutronicsModel(self.cfg_N)
        
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
        T_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)
        T_HP *= self.T_cond
        T_FP *= self.T_cond
        
        T_solid    = T_HP[:-1].reshape((self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R))
        T_vap      = T_HP[-1]
        T_FP       = T_FP.reshape((self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R))
        phi_ng_hat = phi_ng_hat_and_k[:-1].reshape((self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G))
        k          = phi_ng_hat_and_k[-1]

        return ((T_solid, T_vap), T_FP, (phi_ng_hat, k))
    
    def unpack(self, X):
        T_HP             = X[:self.N_HP].copy()
        T_FP             = X[self.N_HP:(self.N_HP + self.N_FP)].copy()
        phi_ng_hat_and_k = X[-self.N_N:].copy()
        return T_HP, T_FP, phi_ng_hat_and_k

    def pack(self, X_tuple):
        X = np.r_[*X_tuple]
        return X

    def initial_guess(self):
        X_initial = np.ones(self.N_var)
        return X_initial
    
    def get_residuals(self, X):
        T_HP = self.T_cond * X[:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1)]
        T_FP = self.T_cond * X[(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1):((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_FP.mesh.N_R * self.cfg_FP.mesh.N_Z)]
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

def generate_config(data, N_R_HP: int, N_R_FP: int, N_Z: int):
    # Heat pipe config
    geom   = HeatpipeGeometry(**data["HeatPipe"]["geometry"])
    mesh   = HeatpipeMesh(N_R=N_R_HP, N_Z=N_Z)
    mat    = HeatpipeMaterial(**data["HeatPipe"]["material"])
    wick   = HeatpipeWick(**data["HeatPipe"]["wick"])
    bc     = HeatpipeBC(**data["HeatPipe"]["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)

    # Fuel pin config
    geom_FP   = FuelPinGeometry(**data["FuelPin"]["geometry"])
    mesh_FP   = FuelPinMesh(N_R=N_R_FP, N_Z=9*N_Z//20)
    energy_FP = FuelPinEnergy(**data["FuelPin"]["energy"])
    mat_FP    = FuelPinMaterial(**data["FuelPin"]["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)

    # Neutronics config
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = 9*N_Z//20,
        l   = data["FuelPin"]["geometry"]["l"]
    )
    energy = NeutronicsEnergy(
        N_G   = cfg_FP.energy.N_G,
        power = data["Reactor"]["power"]["thermal"] / data["Reactor"]["components"]["N_FP"]
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)

    # Reactor config
    cfg_R = ReactorConfig(cfg_HP, cfg_FP, cfg_N)
    cfg_R = cfg_R.resolve_geometry()

    return cfg_R

def generate_config_seq(data, Ns: Sequence[Sequence[int]]):

    cfgs = []

    for (N_R_HP, N_R_FP, N_Z) in Ns:
        cfg = generate_config(data, N_R_HP, N_R_FP, N_Z)
        cfgs.append(cfg)

    return cfgs

def plot_reactor_solutions(solver):
    reactors = solver.components

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    k_effs  = []
    Q_fracs = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(reactors)))[::-1] # type: ignore

    for i, reactor in enumerate(reactors):
        ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = solver.solutions[i]

        cfg_R = reactor.cfg_R

        T_edge_cond_HP1 = T_solid[(cfg_R.HP.mesh.N_evap + cfg_R.HP.mesh.N_adiabatic):, -1]
        Q_out = np.sum(
            (T_edge_cond_HP1 - 300.0)
            * cfg_R.HP.material.h_cond
            * cfg_R.HP.geometry.r_outer
            * 2 * np.pi
            * cfg_R.HP.geometry.l_cond
            / cfg_R.HP.mesh.N_cond
        )
        Q_in = cfg_R.N.energy.power

        k_effs.append(k)
        Q_fracs.append(Q_out / Q_in)

        z_flux = reactor.neutron_flux_model.Z
        z_hp = reactor.heat_pipe_thermal_model.Z[:cfg_R.HP.mesh.N_evap]
        r_hp = reactor.heat_pipe_thermal_model.R

        axs[0, 0].plot(z_flux, phi_ng_hat[:, 0],                   linewidth=2, color=colors[i], label=f"Solution {i+1}")
        axs[0, 1].plot(z_hp,   T_solid[:cfg_R.HP.mesh.N_evap, -1], linewidth=2, color=colors[i], label=f"Solution {i+1}")
        axs[1, 0].plot(z_flux, T_FP[:, -1],                        linewidth=2, color=colors[i], label=f"Solution {i+1}")
        axs[1, 1].plot(r_hp,   T_solid[-1],                        linewidth=2, color=colors[i], label=f"Solution {i+1}")

    k_eff_string = ", ".join(f"{k_eff:.3f}" for k_eff in k_effs)
    Q_string     = ", ".join(f"{Q_frac*100:.2f}%" for Q_frac in Q_fracs)

    fig.suptitle(
        f"Reactor Profiles\n" +
        r"$k_{eff} = $" + f"[{k_eff_string}]\n" +
        r"$Q_{frac} = $" + f"[{Q_string}]",
        fontsize=14
    )

    axs[0, 0].set_title("Axial Neutron Flux Profile")
    axs[0, 0].set_xlabel("Length [m]")
    axs[0, 0].set_ylabel("Flux")
    axs[0, 0].grid(True, alpha=0.4)
    axs[0, 0].legend()

    axs[0, 1].set_title("Heat Pipe Evaporator Radial Temperature")
    axs[0, 1].set_xlabel("Radius [m]")
    axs[0, 1].set_ylabel("Temperature [K]")
    axs[0, 1].grid(True, alpha=0.4)
    axs[0, 1].legend()

    axs[1, 0].set_title("Fuel Pin Radial Temperature Profile")
    axs[1, 0].set_xlabel("Radius [m]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 0].grid(True, alpha=0.4)
    axs[1, 0].legend()

    axs[1, 1].set_title("Heat Pipe Condenser Radial Temperature")
    axs[1, 1].set_xlabel("Radius [m]")
    axs[1, 1].set_ylabel("Temperature [K]")
    axs[1, 1].grid(True, alpha=0.4)
    axs[1, 1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig()
    plt.show()


if __name__ == "__main__":
    from utils.solver import Solver

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)
    
    delta_wick = data["HeatPipe"]["geometry"]["delta_wick"]
    delta_gap  = data["HeatPipe"]["geometry"]["delta_gap"]
    delta_wall = data["HeatPipe"]["geometry"]["delta_wall"]

    # print(delta_wick / (delta_gap + delta_wick + delta_wall) * 15)
    # print(delta_gap  / (delta_gap + delta_wick + delta_wall) * 15)
    # print(delta_wall / (delta_gap + delta_wick + delta_wall) * 15)

    Ns = [[15, 65, 20], [15, 65, 40], [30, 65, 20], [30, 65, 40], [60, 65, 20]]
    cfgs = generate_config_seq(data, Ns)

    # from visualisation.visualise_mesh import plot_reactor_schematic
    # plot_reactor_schematic(cfgs[-1])

    # Reactor ------------------
    reactors = [Reactor(cfg) for cfg in cfgs]

    solver = Solver(reactors, iterate=False, save_iterates=False)
    solver.fsolve()
    # print(solver.solutions)

    plot_reactor_solutions(solver)
