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
        
        self.moderator_eff_res = 0.0807
        self.T_cond = 300.

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1
        self.N_FP = self.cfg_N.mesh.N_R  * self.cfg_N.mesh.N_Z
        self.N_N  = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

    def set_variable_k(self, cond: bool):
        self.heat_pipe_thermal_model.variable_k = cond
        self.fuel_pin_thermal_model.variable_k  = cond

    def assemble(self):
        return
    
    def post_process(self, X):
        T_HP, T_FP, phi_ng_hat_and_k = self.unpack(X)
        T_HP *= self.T_cond
        T_FP *= self.T_cond
        
        T_solid    = T_HP[:-1].reshape((self.cfg_HP.mesh.N_Z, self.cfg_HP.mesh.N_R))
        T_vap      = T_HP[-1]
        T_FP       = T_FP.reshape((self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R))
        phi_ng_hat = phi_ng_hat_and_k[:-1]
        k          = phi_ng_hat_and_k[-1]

        # power normalisation
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP)
        phi_n_g_hat = phi_ng_hat.reshape((self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G))
        power_density = kappa * Sigma_f * phi_n_g_hat
        power = np.sum(power_density) * self.cfg_N.mesh.cross_sectional_area * self.cfg_N.mesh.delta_Z
        phi_n_g = phi_n_g_hat * self.cfg_N.energy.power / power

        return ((T_solid, T_vap), T_FP, (phi_n_g, k))
    
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

    plt.rcParams["font.size"] = 12
    plt.rcParams["font.family"] = "Computer Modern"
    plt.rcParams["text.usetex"] = True

    fig, axs = plt.subplots(3, 2, figsize=(11, 11), dpi=300)

    k_effs = []
    Q_fracs = []

    colors = plt.cm.viridis(np.linspace(0, 1, len(reactors)))[::-1]  # type: ignore

    for i, reactor in enumerate(reactors):
        ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = solver.solutions[i]

        cfg_R = reactor.cfg_R

        N_evap = int(cfg_R.HP.mesh.N_evap)
        N_cond_start = int(cfg_R.HP.mesh.N_evap + cfg_R.HP.mesh.N_adiabatic)

        k_cond = reactor.heat_pipe_thermal_model.variable_k
        k_cond_string = "variable" if k_cond else "fixed"

        # Condenser heat rejection
        T_edge_cond_HP = T_solid[N_cond_start:, -1]
        Q_out = np.sum(
            (T_edge_cond_HP - 300.0)
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
        z_hp = reactor.heat_pipe_thermal_model.Z[:N_evap]
        z_fp = reactor.fuel_pin_thermal_model.Z[:N_evap] if hasattr(reactor.fuel_pin_thermal_model, "Z") else z_hp

        r_hp = reactor.heat_pipe_thermal_model.R
        r_fp = reactor.fuel_pin_thermal_model.R

        # Choose one axial location in evaporator for radial plots
        evap_idx = 0

        # Top row: neutron flux
        axs[0, 0].plot(
            z_flux,
            phi_ng_hat[:, 0],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

        # Keep top-right empty or use it for another group if desired
        axs[0, 1].plot(
            z_flux,
            phi_ng_hat[:, -1],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

        # Middle row: axial temperature distributions
        axs[1, 0].plot(
            z_hp,
            T_solid[:N_evap, -1],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

        axs[1, 1].plot(
            z_fp,
            T_FP[:, -1],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

        # Bottom row: radial temperature distributions
        axs[2, 0].plot(
            r_hp,
            T_solid[evap_idx, :],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

        axs[2, 1].plot(
            r_fp,
            T_FP[evap_idx, :],
            linewidth=2,
            color=colors[i],
            label=f"k {k_cond_string}",
        )

    k_eff_string = ", ".join(f"{k_eff:.3f}" for k_eff in k_effs)
    Q_string = ", ".join(f"{100 * Q_frac:.2f}\\%" for Q_frac in Q_fracs)

    fig.suptitle(
        "Reactor Profiles\n"
        + rf"$k_{{eff}} = [{k_eff_string}]$" + "\n"
        + rf"$Q_{{\mathrm{{frac}}}} = [{Q_string}]$",
        fontsize=14,
    )

    # Top row
    axs[0, 0].set_title("Axial Neutron Flux Profile (Group 1)")
    axs[0, 0].set_xlabel("Length [m]")
    axs[0, 0].set_ylabel("Flux")
    axs[0, 0].grid(True, alpha=0.4)
    axs[0, 0].legend()

    axs[0, 1].set_title("Axial Neutron Flux Profile (Last Group)")
    axs[0, 1].set_xlabel("Length [m]")
    axs[0, 1].set_ylabel("Flux")
    axs[0, 1].grid(True, alpha=0.4)
    axs[0, 1].legend()

    # Middle row
    axs[1, 0].set_title("Heat Pipe Evaporator Axial Temperature")
    axs[1, 0].set_xlabel("Length [m]")
    axs[1, 0].set_ylabel("Temperature [K]")
    axs[1, 0].grid(True, alpha=0.4)
    axs[1, 0].legend()

    axs[1, 1].set_title("Fuel Pin Axial Temperature Profile")
    axs[1, 1].set_xlabel("Length [m]")
    axs[1, 1].set_ylabel("Temperature [K]")
    axs[1, 1].grid(True, alpha=0.4)
    axs[1, 1].legend()

    # Bottom row
    axs[2, 0].set_title("Heat Pipe Evaporator Radial Temperature")
    axs[2, 0].set_xlabel("Radius [m]")
    axs[2, 0].set_ylabel("Temperature [K]")
    axs[2, 0].grid(True, alpha=0.4)
    axs[2, 0].legend()

    axs[2, 1].set_title("Fuel Pin Radial Temperature")
    axs[2, 1].set_xlabel("Radius [m]")
    axs[2, 1].set_ylabel("Temperature [K]")
    axs[2, 1].grid(True, alpha=0.4)
    axs[2, 1].legend()

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig("./outputs/reactor_figures/reactor_data_variable_k_comparison_2.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from utils.solver import Solver

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)
    
    delta_wick = data["HeatPipe"]["geometry"]["delta_wick"]
    delta_gap  = data["HeatPipe"]["geometry"]["delta_gap"]
    delta_wall = data["HeatPipe"]["geometry"]["delta_wall"]

    # Ns = [[15, 65, 20], [15, 65, 40], [30, 65, 20], [30, 65, 40], [60, 65, 20]]
    Ns = [[30, 65, 40], [30, 65, 40]]
    cfgs = generate_config_seq(data, Ns)

    # from visualisation.visualise_mesh import plot_reactor_schematic
    # plot_reactor_schematic(cfgs[-1])

    # Reactor ------------------
    reactors = [Reactor(cfg) for cfg in cfgs]
    reactors[-1].set_variable_k(True)

    solver = Solver(reactors, iterate=True, save_iterates=True)
    solver.fsolve()

    ((T_solid, T_vap), T_FP, (phi_n_g, k)) = solver.solutions[-1]
    T_HP = np.r_[T_solid.reshape(-1), T_vap]
    T_FP = T_FP.reshape(-1)
    Q_mod = reactors[-1].calculate_HP_FP_boundary_cond(T_FP, T_HP)[1]

    plt.plot(Q_mod)
    plt.show()

    plot_reactor_solutions(solver)
