import numpy as np
import meshio # type: ignore

from scipy.optimize import fsolve, newton_krylov

from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.moderator.mesh_conduction_model import ModeratorDiscretisedMesh
from models.moderator.triangle_mesh import UnstructuredMesh

from project_data.heatpipe_dataclasses import *
from project_data.neutronics_dataclasses import *

class Reactor:
    def __init__(self, data, cfg_HP: HeatpipeConfigResolved, cfg_N: NeutronicsConfig):

        self.cfg_HP = cfg_HP
        self.cfg_N = cfg_N

        # Models used
        self.heat_pipe_thermal_model = HeatpipeDiscretised(cfg_HP)
        self.fuel_pin_thermal_model = FuelPin(data)
        self.neutron_flux_model = NeutronicsModel(cfg_N)
        
        # if data.get("thermal_resistance") is None:
        #     mesh = meshio.read("./utils/hex_mesh.msh")

        #     points = mesh.points[:, :2]                
        #     triangles = mesh.cells_dict["triangle"]     
        #     mesh = UnstructuredMesh(points, triangles)

        #     self.moderator_thermal_model = ModeratorDiscretisedMesh(mesh=mesh, data=data)

        #     self.moderator_eff_res = self.moderator_thermal_model.calculate_effective_thermal_resistance()
        # else:
        #     self.moderator_eff_res = data.get("moderator_eff_res")

        self.moderator_eff_res = 0.03

        # Power
        self.power = 1000

        # Discretisation
        self.N_R_HP = cfg_HP.mesh.N_R
        self.N_R_FP = cfg_N.mesh.N_R

        # self.N_Z = data.get("N_Z")

        self.N_G = data.get("N_G")

        # Geometry
        self.r_FP = data.get("r_FP")

        # Boundary cond
        self.T_cond = 300.

        # Solver settings
        self.max_iter = 1000

    
    def solve(self, monolithic=True):

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_HP = self.N_R_HP         * self.cfg_HP.mesh.N_Z + 1
        self.N_FP = self.N_R_FP         * self.cfg_N.mesh.N_Z
        self.N_N  = self.cfg_N.mesh.N_Z * self.N_G + 1

        N_var = self.N_HP + self.N_FP + self.N_N

        initial_guess = np.ones(N_var)

        sol, info, ier, mesg = fsolve(self.get_residuals, initial_guess, full_output=True)

        print(info)
        print(ier)
        print(mesg)

        return sol 
    
    
    def get_residuals(self, X):
        T_HP = self.T_cond * X[:(self.N_R_HP * self.cfg_HP.mesh.N_Z + 1)]
        T_FP = self.T_cond * X[(self.N_R_HP * self.cfg_HP.mesh.N_Z + 1):((self.N_R_HP * self.cfg_HP.mesh.N_Z + 1) + self.N_R_FP * self.cfg_N.mesh.N_Z)]
        phi_ng_hat = X[((self.N_R_HP * self.cfg_HP.mesh.N_Z + 1) + self.N_R_FP * self.cfg_N.mesh.N_Z):]

        self.fuel_pin_thermal_model.initialize_discretization()

        T_FP_ave = np.mean(T_FP.reshape(self.cfg_N.mesh.N_Z, self.N_R_FP), axis=0)
        qr = self.calculate_qr(T_FP_ave, phi_ng_hat[:-1]) 
        Q_HP, T_mod, T_edge_FP = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        self.heat_pipe_thermal_model.cfg.bc.Q = Q_HP
        self.heat_pipe_thermal_model.assemble()
        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP)
        
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP, qr, T_edge_FP)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M = np.mean(T_mod)
        self.neutron_flux_model.T_HP = np.mean(T_HP[:self.N_R_HP * self.cfg_HP.mesh.N_evap])
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP_ave, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP_ave)

        qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W / m

        power_rel = np.sum(qr_rel)

        return qr_rel * self.power / power_rel
    

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_edge_FP = T_FP[self.N_R_FP - 1::self.N_R_FP]
        T_edge_HP = T_FP[self.N_R_HP - 1:(self.N_R_HP * self.cfg_N.mesh.N_Z):self.N_R_HP]

        Q_HP = (T_edge_FP - T_edge_HP) / self.moderator_eff_res

        T_mod = (T_edge_FP + T_edge_HP) / 2

        return Q_HP, T_mod, T_edge_FP

    

if __name__ == "__main__":
    data = {
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
            "N_wick": 3,
            "N_gap": 2,
            "N_wall": 5,
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

    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(**data["mesh"])
    mat = HeatpipeMaterial(**data["material"])
    wick = HeatpipeWick(**data["wick"])
    bc = HeatpipeBC(**data["bc"])
    cfg_HP = HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfg_HP = cfg_HP.resolve()

    N_Z_FP = cfg_HP.mesh.N_evap
    N_R_FP = 20
    data_FP = {
        # ---------------------------
        # Geometry / mesh
        # ---------------------------
        "N_R": N_R_FP,                 # radial cells (numerical choice)
        "N_Z": N_Z_FP,                 # axial cells (numerical choice)

        # Treat r_FP as OUTER fuel-pin radius, since your model has fuel + gap + clad
        "r_FP": 1e-2,           # [m] = 1 cm outer radius
        "l_FP": cfg_HP.geometry.l_evap,             # [m] active fuel length

        "delta_gap": 1e-3,

        "N_gap": 1,                # radial cells assigned to gas gap
        "N_clad": 5,               # radial cells assigned to cladding

        # ---------------------------
        # Neutronics
        # ---------------------------
        "N_G": 8,                  # 2-group model: [fast, thermal]

        # Approximate 2-group flux [n/m^2/s]
        # fast group = collapsed from non-thermal groups
        # thermal group = lowest-energy group
        "phi_g": np.tile(
            np.array([4.16e18, 5.47e17], dtype=float),
            (N_Z_FP, 1)
        ),

        # Approximate macroscopic fission cross section [1/m]
        "Sigma_f": np.array([
            9.4e-2,                # fast-group placeholder
            5.48e1                 # thermal-group estimate
        ], dtype=float),

        # Recoverable energy per fission
        "kappa": 3.204e-11,        # [J/fission]

        # ---------------------------
        # Thermal material properties
        # ---------------------------
        "k_fuel": 15.0,             # [W/m-K] simple UO2 operating-value placeholder
        "k_clad": 16.5,            # [W/m-K] Zircaloy near ~600 K
        "h_gap": 1e5,            # [W/m^2-K] reasonable mid-range gap conductance
        "h_moderator": 1e4,
        # ---------------------------
        # Coolant / moderator
        # ---------------------------
        "T_moderator": np.ones(N_Z_FP) * 1000.0       # [K]
    }

    mesh_N = NeutronicsMesh(
        N_R = N_R_FP,
        N_Z = cfg_HP.mesh.N_evap,
        l = cfg_HP.geometry.l_evap
    )
    energy = NeutronicsEnergy(
        N_G = 8,
        power = 1000
    )

    cfg_N = NeutronicsConfig(mesh_N, energy)
    
    reactor = Reactor(data_FP, cfg_HP, cfg_N)
    X = reactor.solve()

    T_cond = 300
    N_R_HP = cfg_HP.mesh.N_R

    T_HP = T_cond * X[:(N_R_HP * cfg_HP.mesh.N_Z + 1)]
    T_FP = T_cond * X[(N_R_HP * cfg_HP.mesh.N_Z + 1):((N_R_HP * cfg_HP.mesh.N_Z + 1) + N_R_FP * cfg_N.mesh.N_Z)]
    phi_ng_hat = X[((N_R_HP * cfg_HP.mesh.N_Z + 1) + N_R_FP * cfg_N.mesh.N_Z):]

    import matplotlib.pyplot as plt

    plt.plot(T_FP)
    plt.show()

    plt.plot(T_HP)
    plt.show()

    plt.plot(phi_ng_hat[:cfg_N.mesh.N_Z])
    plt.show()