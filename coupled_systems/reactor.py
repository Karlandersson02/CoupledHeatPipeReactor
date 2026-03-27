import numpy as np
import meshio # type: ignore

from scipy.optimize import fsolve, newton_krylov

from models.heatpipe.heat_discretised_model import heatpipe_discretised
from models.neutronics.axial_neutron_model import NeutronModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.mesh_heat_conduction.mesh_conduction_model import moderator_discretised_mesh
from models.mesh_heat_conduction.triangle_mesh import UnstructuredMesh


class Reactor:
    def __init__(self, data):

        # Models used
        self.heat_pipe_thermal_model = heatpipe_discretised(data)
        self.fuel_pin_thermal_model = FuelPin(data)
        self.neutron_flux_model = NeutronModel(data)
        
        if data.get("thermal_resistance") is None:
            mesh = meshio.read("meshHeatConduction/mesh.msh")

            points = mesh.points[:, :2]                
            triangles = mesh.cells_dict["triangle"]     
            mesh = UnstructuredMesh(points, triangles)

            self.moderator_thermal_model = moderator_discretised_mesh(mesh=mesh, data=data)

            self.moderator_eff_res = self.moderator_thermal_model.calculate_effective_thermal_resistance()
        else:
            self.moderator_eff_res = data.get("moderator_eff_res")

        # Power
        self.power = data.get("power")

        # Discretisation
        self.N_R_HP = data.get("N_wick_HP") + data.get("N_wall_HP")
        self.N_R_FP = data.get("N_R_FP")

        self.N_Z = data.get("N_Z")

        self.N_G = data.get("N_G")

        # Geometry
        self.r_FP = data.get("r_FP")

        # Boundary cond
        self.T_cond = 300.

        # Solver settings
        self.max_iter = data.get("max_iter")

    
    def solve(self, monolithic=True):

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_cond_HP = self.N_R_HP * self.N_Z + 1
        self.N_cond_FP = self.N_R_FP * self.N_Z
        self.N_flux = self.N_Z + 1

        N_var = self.N_cond_HP + self.N_cond_FP + self.N_flux

        initial_guess = np.ones(N_var)

        sol, info, ier, mesg = fsolve(self.get_residuals, initial_guess, full_output=True)

        print(info)
        print(ier)
        print(mesg)

        return sol 
    
    
    def get_residuals(self, X):       
        T_HP = self.T_cond * X[:(self.N_R_HP * self.N_Z + 1)]
        T_FP = self.T_cond * X[(self.N_R_HP * self.N_Z + 1):(self.N_R_HP * self.N_Z + 1 + self.N_R_FP * self.N_Z)]
        phi_ng_hat = X[(self.N_R_HP * self.N_Z + 1 + self.N_R_FP * self.N_Z):]

        T_FP_ave = np.mean(T_FP.reshape(self.N_Z, self.N_R_FP), axis=0)
        qr = self.calculate_qr(T_FP_ave, phi_ng_hat) 
        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)

        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP, Q_HP)
        
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP, qr, T_mod)
        
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat, T_FP_ave)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]
    

    def calculate_qr(self, T_FP_ave, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP_ave)

        qr_rel = np.sum(phi_ng_hat.reshape(self.N_Z, self.N_G) * Sigma_f * kappa * np.pi * self.fuel_pin_thermal_model.Delta_V) # W / m

        power_rel = np.sum(qr_rel)

        return qr_rel * self.power / power_rel
    

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_edge_FP = T_FP[self.N_R_FP - 1::self.N_R_FP]
        T_edge_HP = T_FP[self.N_R_HP - 1::self.N_R_HP]

        Q_HP = (T_edge_FP - T_edge_HP) / self.moderator_eff_res

        return Q_HP, T_edge_FP

    

if __name__ == "__main__":
    data = {
        # HP data
        "r_outer": .007 + 0.001 + 0.0005,
        "delta_wick": 0.0005,
        "delta_wall": 0.001,
        "l_evap": 0.3,
        "l_adiabatic": 0.15,
        "l_cond": 0.60,

        "N_wick": 20,
        "N_wall": 20,
        "N_evap": 20,
        "N_adiabatic": 10,
        "N_cond": 50,

        "adiabatic_radial_flux": False,
        "Temperature_BC": True,
        "h_vap": 1e6,
        "h_cond": 62.6,
        "T_cond": 300,
        "T_op": 850,

        "k_wick": 66.2,
        "k_wall": 19.0,

        "P_C": 2476,
        "T_C": 856,
    }