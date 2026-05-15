import numpy as np

import data.dataclass as d_class
import utils.material_properties as m_props

from models.heatpipe.solid_discretised_model import HeatpipeDiscretised
from models.neutronics.axial_neutron_model import NeutronicsModel
from models.fuel_pin.fuel_pin_model import FuelPin
from models.component import Component


class Reactor(Component):
    def __init__(self, cfg_R: d_class.ReactorConfigResolved):

        self.cfg_R = cfg_R

        self.cfg_HP = cfg_R.HP
        self.cfg_FP = cfg_R.FP
        self.cfg_N = cfg_R.N

        self.heat_pipe_thermal_model = HeatpipeDiscretised(self.cfg_HP)
        self.fuel_pin_thermal_model  = FuelPin(self.cfg_FP)
        self.neutron_flux_model      = NeutronicsModel(self.cfg_N)
        
        self.T_cond = 300.

        # heat transfer HP variables: N_R * N_Z + 1, heat transfer FP variables: N_R * N_Z + 1, neutron flux variables: N_Z + 1
        self.N_HP = self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1
        self.N_FP = self.cfg_FP.mesh.N_R  * self.cfg_FP.mesh.N_Z
        self.N_N  = self.cfg_N.mesh.N_Z  * self.cfg_N.energy.N_G + 1

        self.N_var = self.N_HP + self.N_FP + self.N_N

        # variable r_eff
        self.variable_r_eff    = False
        self.r_eff_temperature = 1100           # used if variable_r_eff is False

    def set_variable_k(self, cond: bool):
        self.heat_pipe_thermal_model.variable_k = cond
        self.fuel_pin_thermal_model.variable_k  = cond

    def set_interpolator_model(self, model):
        self.neutron_flux_model.interpolator_model = model

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
        phi_ng_hat_and_k = X[((self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_Z + 1) + self.cfg_N.mesh.N_R * self.cfg_N.mesh.N_Z):]

        Q_HP, T_mod = self.calculate_HP_FP_boundary_cond(T_FP, T_HP)
        
        T_HP_ave  = np.mean(T_HP[:self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap], dtype=float)
        T_FP_ave  = np.mean(T_FP.reshape(self.cfg_FP.mesh.N_Z, self.cfg_FP.mesh.N_R), axis=1)
        T_mod_ave = np.mean(T_mod)
        
        self.neutron_flux_model.T_FP = T_FP
        self.neutron_flux_model.T_M  = T_mod_ave
        self.neutron_flux_model.T_HP = T_HP_ave
        res_flux = self.neutron_flux_model.get_residuals(phi_ng_hat_and_k)

        self.heat_pipe_thermal_model.cfg.bc.Q = Q_HP
        res_cond_HP = self.heat_pipe_thermal_model.get_residuals(T_HP)

        qr = self.calculate_qr(T_FP, phi_ng_hat_and_k[:-1])
        
        self.fuel_pin_thermal_model.qr = qr
        self.fuel_pin_thermal_model.T_mod = T_mod
        res_cond_FP = self.fuel_pin_thermal_model.get_residuals(T_FP)

        return np.r_[res_cond_HP, res_cond_FP, res_flux]

    # def calculate_qr(self, T_FP, phi_ng_hat):
    #     _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP)

    #     qr_rel = np.sum(phi_ng_hat.reshape(self.cfg_N.mesh.N_Z, self.cfg_N.energy.N_G) * Sigma_f * kappa * self.fuel_pin_thermal_model.Delta_V, axis=1) # W
    #     power_rel = np.sum(qr_rel)

    #     return qr_rel * self.cfg_N.energy.power / (power_rel * self.cfg_FP.mesh.N_fuel)

    def calculate_qr(self, T_FP, phi_ng_hat):
        _, _, _, Sigma_f, _, _, kappa = self.neutron_flux_model.get_material_data(T_FP)

        phi_ng_hat = phi_ng_hat.reshape(
            self.cfg_N.mesh.N_Z,
            self.cfg_N.energy.N_G,
        )

        q_vol_z = np.sum(
            phi_ng_hat * Sigma_f * kappa,
            axis=1,
        )

        V_fuel = self.fuel_pin_thermal_model.Delta_V[:self.cfg_FP.mesh.N_fuel]

        qr_rel = q_vol_z * np.mean(V_fuel)
        power_rel = np.sum(q_vol_z) * np.sum(V_fuel)

        return qr_rel * self.cfg_N.energy.power / power_rel

    def calculate_HP_FP_boundary_cond(self, T_FP, T_HP):
        T_edge_FP = T_FP[self.cfg_FP.mesh.N_R - 1::self.cfg_FP.mesh.N_R]
        T_edge_HP = T_HP[self.cfg_HP.mesh.N_R - 1:(self.cfg_HP.mesh.N_R * self.cfg_HP.mesh.N_evap):self.cfg_HP.mesh.N_R]
        T_mod_ave = (T_edge_FP + T_edge_HP) / 2

        Delta_z = self.cfg_HP.geometry.l_evap / self.cfg_HP.mesh.N_evap

        if self.variable_r_eff:
            R_seg = m_props.moderator_R_eff(T_mod_ave) / Delta_z
        else:
            R_seg = m_props.moderator_R_eff(self.r_eff_temperature) / Delta_z

        Q_FP = (T_edge_FP - T_edge_HP) / R_seg
        Q_HP = Q_FP * 24 / 7
        T_mod1 = T_edge_FP - Q_FP / (self.cfg_FP.material.h_mod * 2 * self.cfg_FP.geometry.r * np.pi * self.cfg_FP.geometry.l / self.cfg_FP.mesh.N_Z)

        return Q_HP, T_mod1

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    from utils.solver import Solver
    from utils.iso_reactor_utils import generate_config_seq, plot_reactor_solutions, plot_reactor_temperature_schematic

    with open("./data/reactor_data.json", "r") as f:
        data = json.load(f)

    Ns = [[15, 65, 40]]
    cfgs = generate_config_seq(data, Ns)

    # from visualisation.visualise_mesh import plot_reactor_schematic
    # plot_reactor_schematic(cfgs[-1])

    # Reactor ------------------
    reactors = [Reactor(cfg) for cfg in cfgs]

    solver = Solver(reactors, iterate=True, save_iterates=True)
    solver.fsolve()

    plot_reactor_temperature_schematic(solver)

    # plot_reactor_solutions(solver)
