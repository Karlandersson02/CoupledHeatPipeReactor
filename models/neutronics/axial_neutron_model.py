import numpy as np
from pathlib import Path

import data.dataclass as d_class

from models.component import Component
from utils.interpolator import OpenMCTallyGridSurrogate

import numpy as np

class NeutronicsModel(Component):

    def __init__(self, config):
        self.cfg = config

        # temporary
        self.T_FP = np.full((self.cfg.mesh.N_Z*self.cfg.mesh.N_R), 900)
        self.T_HP = 900
        self.T_M  = 900

        self.T_HP_static  = 900
        self.T_M_static   = 900
        self.T_FP_static  = 900

        self.Z = np.linspace(0, self.cfg.mesh.l, self.cfg.mesh.N_Z)

        MODEL_PATH = Path("./utils/rgi_surrogate.joblib")
        interpolator_model = OpenMCTallyGridSurrogate()
        interpolator_model = interpolator_model.load(MODEL_PATH)

        self.interpolator_model = interpolator_model
        self.variable_neutron_data = False

    def set_variable_neutron_data(self, cond, T):
        self.variable_neutron_data = cond
        self.T_HP_static  = T
        self.T_M_static   = T
        self.T_FP_static  = T

    def initial_guess(self):
        phi_ng_initial = np.full((self.cfg.mesh.N_Z*self.cfg.energy.N_G), 1)
        k_initial = np.array([1])
        X_initial = np.concatenate([phi_ng_initial, k_initial])
        return X_initial
    
    def assemble(self):
        return
    
    def get_residuals(self, X):
        T = self.T_FP
        phi_ng, k = X[:-1], X[-1]

        phi_n_g = np.reshape(phi_ng, (self.cfg.mesh.N_Z, self.cfg.energy.N_G))
        phi_np1_g = np.vstack([phi_n_g[1:], np.zeros((1, self.cfg.energy.N_G))])
        phi_nm1_g = np.vstack([np.zeros((1, self.cfg.energy.N_G)), phi_n_g[:-1]])

        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = self.get_material_data(T)
        a_n_g, b_n_g, c_n_g = self._generate_abc(D)

        res_transport = (
            a_n_g * phi_n_g + b_n_g * phi_np1_g + c_n_g * phi_nm1_g + Sigma_t * phi_n_g
            - (np.sum(Sigma_s0 * phi_n_g[..., None], axis=1) + (chi / k) * (np.sum(nu * Sigma_f * phi_n_g, axis=1))[:, None])
        )
        res_transport = np.ravel(res_transport)

        res_anchor = self.cfg.mesh.N_Z * self.cfg.energy.N_G - np.dot(phi_ng, phi_ng)

        return np.r_[res_transport, res_anchor]

    def post_process(self, X):
        T = self.T_FP
        _, _, _, Sigma_f, _, _, kappa = self.get_material_data(T)
        
        phi_ng = X[:-1]
        phi_n_g = phi_ng.reshape((self.cfg.mesh.N_Z, self.cfg.energy.N_G))
        power_density = kappa * Sigma_f * phi_n_g
        power = np.sum(power_density) * self.cfg.mesh.cross_sectional_area * self.cfg.mesh.delta_Z
        phi_n_g *= self.cfg.energy.power / power
        return phi_n_g, X[-1]
    
    def unpack(self, X):
        return X[:-1], X[-1]
    
    def pack(self, X_tuple):
        X = np.r_[X_tuple[0].reshape(self.cfg.mesh.N_Z * self.cfg.energy.N_G), X_tuple[1]]
        return X

    def _generate_abc(self, D_n_g):

        D_nm1_g = np.zeros_like(D_n_g)
        D_np1_g = np.zeros_like(D_n_g)

        D_nm1_g[1:] = D_n_g[:-1]
        D_np1_g[:-1] = D_n_g[1:]

        alpha_n_g = np.zeros_like(D_n_g)
        alpha_np1_g = np.zeros_like(D_n_g)

        mask_nm1 = (D_n_g + D_nm1_g) > 0.0
        mask_np1 = (D_n_g + D_np1_g) > 0.0

        alpha_n_g[mask_nm1] = (
            (2.0 / self.cfg.mesh.delta_Z**2) *
            (D_n_g[mask_nm1] * D_nm1_g[mask_nm1]) /
            (D_n_g[mask_nm1] + D_nm1_g[mask_nm1])
        )

        alpha_np1_g[mask_np1] = (
            (2.0 / self.cfg.mesh.delta_Z**2) *
            (D_n_g[mask_np1] * D_np1_g[mask_np1]) /
            (D_n_g[mask_np1] + D_np1_g[mask_np1])
        )

        beta_NZ_g = (2.0 * D_n_g[-1] / self.cfg.mesh.delta_Z) / (self.cfg.mesh.delta_Z + 4.0 * D_n_g[-1])
        beta_1_g = (2.0 * D_n_g[0] / self.cfg.mesh.delta_Z) / (self.cfg.mesh.delta_Z + 4.0 * D_n_g[0])

        a_n_g = alpha_np1_g + alpha_n_g
        b_n_g = -alpha_np1_g
        c_n_g = -alpha_n_g

        # boundary adjustments
        a_n_g[0] = beta_1_g + alpha_np1_g[0]
        a_n_g[-1] = beta_NZ_g + alpha_n_g[-1]

        b_n_g[0] = -alpha_np1_g[0]
        b_n_g[-1] = np.zeros(self.cfg.energy.N_G)

        c_n_g[0] = np.zeros(self.cfg.energy.N_G)
        c_n_g[-1] = -alpha_n_g[-1]

        return a_n_g, b_n_g, c_n_g
    
    def get_axial_temperature(self, T):
        return T[::self.cfg.mesh.N_R]

    def get_material_data(self, T):
        if self.interpolator_model is None:
            return self.get_material_data_static(T)
        else:
            return self.get_material_data_interp(T)

    def get_material_data_static(self, T):
        T_center_axial = self.get_axial_temperature(T)

        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = self.calculate_parameters(T_center_axial)
        return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa
    
    def get_material_data_interp(self, T):
        if self.variable_neutron_data:
            T_reshaped = T.reshape((self.cfg.mesh.N_Z,self.cfg.mesh.N_R))
            T_mean = np.mean(T_reshaped, axis=1)
            X = np.array([[self.T_HP, T_FP, self.T_M] for T_FP in T_mean])
        else:
            X = np.array([[self.T_HP_static, self.T_FP_static, self.T_M_static] for i in range(self.cfg.mesh.N_Z)])

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Neutronics interpolator received non-finite temperatures: "
                f"T_HP={self.T_HP}, T_M={self.T_M}, "
                f"T_FP_mean_range=[{np.nanmin(X[:, 1]):.3f}, {np.nanmax(X[:, 1]):.3f}]"
            )

        # if self.variable_neutron_data:
        #     bounds = self.interpolator_model.axes  # type: ignore[attr-defined]
        #     if bounds is not None:
        #         out_of_range = []
        #         labels = ("T_HP", "T_FP", "T_M")
        #         for axis_idx, axis_values in enumerate(bounds):
        #             x_min = np.min(X[:, axis_idx])
        #             x_max = np.max(X[:, axis_idx])
        #             axis_min = np.min(axis_values)
        #             axis_max = np.max(axis_values)
        #             if x_min < axis_min or x_max > axis_max:
        #                 out_of_range.append(
        #                     f"{labels[axis_idx]}=[{x_min:.3f}, {x_max:.3f}] outside "
        #                     f"[{axis_min:.3f}, {axis_max:.3f}]"
        #                 )

        #         if out_of_range:
        #             raise ValueError(
        #                 "Neutronics interpolator temperatures exceeded the surrogate "
        #                 f"training domain: {', '.join(out_of_range)}"
        #             )

        params = self.interpolator_model.predict_dict(X)   # type: ignore

        Diffusivity = np.zeros((len(params), self.cfg.energy.N_G))
        Sigma_t = np.zeros((len(params), self.cfg.energy.N_G))
        Sigma_f = np.zeros((len(params), self.cfg.energy.N_G))
        Sigma_s0 = np.zeros((len(params), self.cfg.energy.N_G, self.cfg.energy.N_G))
        fission_number = np.zeros((len(params), self.cfg.energy.N_G))
        Chi = np.zeros((len(params), self.cfg.energy.N_G))
        kappa = np.zeros((len(params), self.cfg.energy.N_G))
        
        for i in range(len(params)):
            Diffusivity[i] = np.array(params[i]["diffusion_coefficient"]) * 1e-2
            Sigma_t[i] = np.array(params[i]["total_xs"]) * 1e2
            Sigma_f[i] = np.array(params[i]["fission_xs"]) * 1e2
            Sigma_s0[i] = (np.array(params[i]["scatter_matrix_xs"]) * 1e2)
            fission_number[i] = np.array(params[i]["nu"])
            Chi[i] = np.array(params[i]["chi"])
            kappa[i] = np.array(params[i]["kappa"]) * 1.602176634e-19

        return Diffusivity, Sigma_t, Sigma_s0, Sigma_f, fission_number, Chi, kappa
    
    # def _repeat_over_temperature(self, values, T):
    #     values = np.asarray(values, dtype=float)
    #     return np.repeat(values[None], len(T), axis=0)


    # def calculate_diffusivity(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         D = np.array([
    #             1.873379, 0.940634, 0.809891, 0.795359,
    #             0.801553, 0.732249, 0.630263, 1.438368
    #         ])  # [cm]
    #     else:
    #         D = np.array([
    #             1.2,   # group 1, fast [cm]
    #             0.35,  # group 2, thermal [cm]
    #         ])

    #     return self._repeat_over_temperature(D, T)


    # def calculate_Sigma_t(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         Sigma_t = np.array([
    #             0.212918, 0.385936, 0.433878, 0.439159,
    #             0.434345, 0.460764, 0.486712, 0.288787
    #         ])  # [cm^-1]
    #     else:
    #         Sigma_a1 = 0.01
    #         Sigma_a2 = 0.08
    #         Sigma_s12 = 0.04
    #         Sigma_s21 = 0.0

    #         Sigma_t = np.array([
    #             Sigma_a1 + Sigma_s12,
    #             Sigma_a2 + Sigma_s21,
    #         ])  # [cm^-1]

    #     return self._repeat_over_temperature(Sigma_t, T)


    # def calculate_Sigma_s0(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         Sigma_s0 = np.array([
    #             [1.861927e-01, 2.573515e-02, 9.995288e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
    #             [0.000000e+00, 3.692822e-01, 1.589830e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
    #             [0.000000e+00, 0.000000e+00, 4.148961e-01, 1.600025e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
    #             [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.163120e-01, 1.253294e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00],
    #             [0.000000e+00, 0.000000e+00, 0.000000e+00, 2.535302e-05, 4.066482e-01, 1.582596e-02, 8.742419e-07, 0.000000e+00],
    #             [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.355158e-02, 4.168228e-01, 6.309109e-04, 1.209800e-06],
    #             [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.073801e-01, 3.237823e-01, 0.000000e+00],
    #             [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 2.854831e-01, 0.000000e+00, 0.000000e+00],
    #         ])  # [cm^-1]
    #     else:
    #         Sigma_s12 = 0.04
    #         Sigma_s21 = 0.0

    #         Sigma_s0 = np.array([
    #             [0.0,       Sigma_s12],
    #             [Sigma_s21, 0.0      ],
    #         ])  # [cm^-1]

    #     return self._repeat_over_temperature(Sigma_s0, T)


    # def calculate_Sigma_f(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         Sigma_f = np.array([
    #             0.000751, 0.000286, 0.000963, 0.003956,
    #             0.006032, 0.023611, 0.043511, 0.0
    #         ])  # [cm^-1]
    #     else:
    #         Sigma_f = np.array([
    #             0.002,  # group 1, fast [cm^-1]
    #             0.05,   # group 2, thermal [cm^-1]
    #         ])

    #     return self._repeat_over_temperature(Sigma_f, T)


    # def calculate_nu(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         nu = np.array([
    #             2.714564, 2.444811, 2.433696, 2.434935,
    #             2.436692, 2.436700, 2.436700, 0.0
    #         ])  # [-]
    #     else:
    #         nuSigma_f1 = 0.005
    #         nuSigma_f2 = 0.12

    #         Sigma_f1 = 0.002
    #         Sigma_f2 = 0.05

    #         nu = np.array([
    #             nuSigma_f1 / Sigma_f1,
    #             nuSigma_f2 / Sigma_f2,
    #         ])  # [-]

    #     return self._repeat_over_temperature(nu, T)


    # def calculate_chi(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         chi = np.array([
    #             8.474541e-01, 1.516625e-01, 8.794743e-04, 3.858033e-06,
    #             0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00
    #         ])  # [-]
    #     else:
    #         chi = np.array([
    #             1.0,  # all fission neutrons born into group 1, fast
    #             0.0,
    #         ])  # [-]

    #     return self._repeat_over_temperature(chi, T)


    # def calculate_kappa(self, T):
    #     if self.cfg.energy.N_G == 8:
    #         kappa = np.array([
    #             1.963227e+08, 1.934084e+08, 1.934085e+08, 1.934054e+08,
    #             1.934054e+08, 1.934054e+08, 1.934054e+08, 0.000000e+00
    #         ])  # [eV/fission]
    #     else:
    #         kappa = np.array([
    #             2.0e8,
    #             2.0e8,
    #         ])  # [eV/fission]

    #     return self._repeat_over_temperature(kappa, T)

    # def calculate_parameters(self, T):
    #     D        = self.calculate_diffusivity(T) * 1e-2
    #     Sigma_t  = self.calculate_Sigma_t(T) * 1e2
    #     Sigma_s0 = self.calculate_Sigma_s0(T) * 1e2
    #     Sigma_f  = self.calculate_Sigma_f(T) * 1e2
    #     nu       = self.calculate_nu(T)
    #     chi      = self.calculate_chi(T)
    #     kappa    = self.calculate_kappa(T) * 1.602e-19

    #     return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa

if __name__ == "__main__":  
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    from utils.solver import Solver

    mesh = d_class.NeutronicsMesh(
        N_R = 50,
        N_Z = 100,
        l = 1.8
    )
    energy = d_class.NeutronicsEnergy(
        N_G = 8,
        power = 1000
    )
    cfg = d_class.NeutronicsConfig(mesh, energy)

    MODEL_PATH = Path("./utils/rgi_surrogate.joblib")
    interpolator_model = OpenMCTallyGridSurrogate()
    interpolator_model = interpolator_model.load(MODEL_PATH)

    neutronics_model = NeutronicsModel(cfg)
    neutronics_model.interpolator_model = interpolator_model # type: ignore

    solver = Solver([neutronics_model])
    solver.fsolve()

    phi_n_g, k = solver.solution      # type: ignore

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "Computer modern"
    mpl.rcParams["font.size"] = 22

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    colors = plt.cm.viridis(np.linspace(0, 1, 8))[::-1] # type: ignore
    ax.grid(alpha=0.4)
    for i in range(len(colors)):
        ax.plot(phi_n_g[:, i], color=colors[i], label=i)
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k:.2f}")

    plt.show()
