import numpy as np
import meshio # type: ignore

from scipy.optimize import fsolve, newton_krylov

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
    
    def solve(self, monolithic=True):

        X_initial = np.ones(self.N_var)

        sol, info, ier, mesg = fsolve(self.get_residuals, X_initial, full_output=True)

        print(info)
        print(ier)
        print(mesg)

        return sol 
    
    
    def get_residuals(self, X):
        T_HP = self.T_cond * X[:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1)]
        T_FP = self.T_cond * X[(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1):((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z)]
        phi_ng_hat = X[((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z):]

        self.fuel_pin_thermal_model.initialize_discretization()

        T_FP_ave = np.mean(T_FP.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.mesh.N_R), axis=0)
        qr = self.calculate_qr(T_FP_ave, phi_ng_hat[:-1]) 
        Q_HP, T_mod, T_edge_FP = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        self.heat_pipe_thermal_model.cfg.bc.Q = Q_HP
        self.heat_pipe_thermal_model.assemble()
        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP)
        
        self.fuel_pin_thermal_model.qr = qr
        self.fuel_pin_thermal_model.T_mod = T_edge_FP
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M  = np.mean(T_mod)
        self.neutron_flux_model.T_HP = np.mean(T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap])
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP_ave, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP_ave)

        qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W / m

        power_rel = np.sum(qr_rel)

        return qr_rel * self.cfg_N.energy.power / power_rel
    

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_edge_FP = T_FP[self.cfg_FP.mesh.N_R - 1::self.cfg_FP.mesh.N_R]
        T_edge_HP = T_HP[self.cfg_HP.mesh.N_R - 1:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap):self.cfg_HP.mesh.N_R]

        Q_HP = (T_edge_FP - T_edge_HP) / self.moderator_eff_res

        T_mod = (T_edge_FP + T_edge_HP) / 2

        return Q_HP, T_mod, T_edge_FP

    

if __name__ == "__main__":

    # HeatPipe ---------------
    data_HP = {
        "geometry": {
            "r_outer": .007 + 0.001 + 0.0005,
            "delta_wick": 0.0003,
            "delta_gap": 0.0002,
            "delta_wall": 0.001,
            "l_evap": 0.75,
            "l_adiabatic": 0.15,
            "l_cond": 0.6,
        },

        "mesh": {
            "N_R": 10,
            "N_evap": 10,
            "N_adiabatic": 2,
            "N_cond": 8,
        },

        "material": {
            "k_wick": 66.2,
            "k_gap": 80,
            "k_wall": 19.0,
            "h_vap": 1e6,
            "h_cond": 62.6,
        },

        "wick": {
            "Is_annular": True,
            "K":1e-10,
            "r_pore": 0.00002,
            "porosity": 0.7
        },

        "bc": {
            "Temperature_BC": True,
            "T_cond": 300,
            "T_op": 850,
            "Q": 1000,
        },
    }

    geom   = HeatpipeGeometry(**data_HP["geometry"])
    mesh   = HeatpipeMesh(**data_HP["mesh"])
    mat    = HeatpipeMaterial(**data_HP["material"])
    wick   = HeatpipeWick(**data_HP["wick"])
    bc     = HeatpipeBC(**data_HP["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg_HP = cfg_HP.resolve()

    # FuelPin ---------------
    N_Z_FP = cfg_HP.mesh.N_evap
    N_R_FP = 20
    data_FP = {
        "geometry": {
            "delta_gap": 1e-3,
            "delta_wall": 1e-3,
            "r": 1e-2,
            "l": cfg_HP.geometry.l_evap,
        },
        "mesh": {
            "N_R": N_R_FP,
            "N_Z": N_Z_FP,
        },

        "energy": {
            "N_G": 8,
        },

        "material": {
            "k_fuel": 15.0,
            "k_clad": 16.5,
            "h_gap": 1e5,
            "h_mod": 1e4,
        }
    }

    geom_FP   = FuelPinGeometry(**data_FP["geometry"])
    mesh_FP   = FuelPinMesh(**data_FP["mesh"])
    energy_FP = FuelPinEnergy(**data_FP["energy"])
    mat_FP    = FuelPinMaterial(**data_FP["material"])
    cfg_FP    = FuelPinConfig(geom_FP, mesh_FP, energy_FP, mat_FP)
    cfg_FP = cfg_FP.resolve()

    # Neutronics ---------------
    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = cfg_HP.mesh.N_evap,
        l = cfg_HP.geometry.l_evap
    )
    energy = NeutronicsEnergy(
        N_G = 8,
        power = 1000
    )
    cfg_N = NeutronicsConfig(mesh_N, energy)\
    
    # Reactor ------------------
    reactor = Reactor(cfg_FP, cfg_HP, cfg_N)
    # X = reactor.solve()

    from utils.solver import Solver

    solver = Solver([reactor])
    solver.fsolve()

    ((T_solid, T_vap), T_FP, (phi_ng_hat, k)) = solver.solution

    import matplotlib.pyplot as plt

    plt.plot(reactor.fuel_pin_thermal_model.R[1::2], T_FP[0])
    plt.show()

    plt.plot(T_solid[0])
    plt.show()

    plt.plot(phi_ng_hat[:, 0])
    plt.show()