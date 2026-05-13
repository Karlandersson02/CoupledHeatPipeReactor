import numpy as np

import utils.sodium_properties as s_props

from models.heatpipe.solid_discretised_vapour_model import HeatpipeDiscretisedVapour
from models.heatpipe.vapour_discretised_model import VapourDiscretised
from models.component import Component
from data.dataclass import HeatpipeConfigResolved

class Heatpipe(Component):

    def __init__(self, cfg: HeatpipeConfigResolved):
        self.cfg = cfg

        self.solid  = HeatpipeDiscretisedVapour(cfg)
        self.vapour = VapourDiscretised(cfg, -1)

        # N_HP, N_u, N_v
        self.N_HP = self.cfg.mesh.N_Z * self.cfg.mesh.N_R
        self.N_u  = self.cfg.mesh.N_Z - 1
        self.N_v  = self.cfg.mesh.N_Z

        self.return_u = False

    def set_variable_k(self, cond: bool):
        self.solid.variable_k = cond

    def set_loss_mult_factor(self, factor):
        self.vapour.loss_mult = factor

    def assemble(self):
        self.solid.assemble()
        self.vapour.assemble()

    def initial_guess(self):
        X_HP_lin = self.solid.linear_solve()

        T_HP0 = X_HP_lin[:-1]
        T_v0_scalar = X_HP_lin[-1]

        self.vapour.set_T_HP(T_HP0)

        T_v0 = np.full(self.N_v, T_v0_scalar, dtype=float)
        u0 = self.vapour._build_initial_velocity(T_v0)

        return np.r_[T_HP0, u0, T_v0]

    def get_residuals(self, X):
        X_HP, u, T_v = self.unpack(X)
        X_Vap = self.vapour.pack((u, T_v))

        self.solid.set_T_v(T_v)
        self.vapour.set_T_HP(X_HP)

        res_HP  = self.solid.get_residuals(X_HP)
        res_Vap = self.vapour.get_residuals(X_Vap)

        return np.r_[res_HP, res_Vap]

    def post_process(self, X):
        T_HP, u, T_v = self.unpack(X)
        T_HP = T_HP.reshape((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))

        u_bar = self._interpolate_u(u)
        c_s = self._calculate_c_s(T_v)
        mach = u_bar / c_s

        if self.return_u:
            return T_HP, u, mach, T_v
        else:
            return T_HP, mach, T_v

    def unpack(self, X):
        T_HP = X[:self.N_HP]
        u    = X[self.N_HP:(self.N_HP + self.N_u)]
        T_v  = X[-self.N_v:]
        return T_HP, u, T_v
    
    def pack(self, X_tuple):
        return np.r_[*X_tuple]
    
    # def _calculate_c_s(self, T_v):
    #     gamma = 5 / 3
    #     c_s = np.sqrt(gamma * s_props.calculate_Na_pressure_v(T_v) / s_props.calculate_Na_rho_v(T_v))

    #     return c_s

    def _calculate_c_s(self, T_v):
        gamma = 5.0 / 3.0
        # p = self.vapour.calculate_rho(T_v) * 361.7 * T_v
        p = s_props.calculate_Na_pressure_v(T_v)
        rho = self.vapour.calculate_rho(T_v)
        return np.sqrt(gamma * p / rho)
    
    def _interpolate_u(self, u):
        u_full = np.r_[0, u, 0]
        u_bar = (u_full[:-1] + u_full[1:]) / 2
        return u_bar    

if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    from utils.heatpipe_utils import generate_cfgs_seq, plot_heatpipe_solutions
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_1000"]

    ncfgs = 5
    Qs = [1e3 for i in range(ncfgs)]
    Ns = [[28, 50] for i in range(len(Qs))]
    cfgs = generate_cfgs_seq(data, Ns, Qs)

    heatpipes = [Heatpipe(cfg) for cfg in cfgs]

    solver = Solver(heatpipes, iterate=False)
    solver.fsolve()


    plot_heatpipe_solutions(solver)