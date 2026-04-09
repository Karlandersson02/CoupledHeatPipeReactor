import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import fsolve, newton_krylov

from project_data.neutronics_dataclasses import *
from models.component import Component

MODEL_PATH = "./utils/rgi_surrogate.joblib"
from utils.interpolator import OpenMCTallyGridSurrogate
interpolator_model = OpenMCTallyGridSurrogate()
interpolator_model = interpolator_model.load(MODEL_PATH)

N_G = 8

def calculate_diffusivity(T):
    D = np.array([1.822733, 0.90954, 0.789696, 0.782178, 0.778223, 0.720307, 0.645907, 0.716544])
    return np.repeat(D[None], len(T), axis=0)


def calculate_Sigma_t(T):
    Sigma_t = np.array([0.218755, 0.399931, 0.448118, 0.4516, 0.4526, 0.472615, 0.477891, 0.465947])
    return np.repeat(Sigma_t[None], len(T), axis=0)


def calculate_Sigma_s0(T):
    Sigma_s0 = np.array([
        [1.915640e-01, 2.649477e-02, 7.612502e-06, 2.175001e-07, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 3.821866e-01, 1.748240e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 4.283324e-01, 1.897048e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 4.309142e-01, 1.830837e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.108383e-05, 4.177547e-01, 3.235313e-02, 2.333439e-06, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 7.035669e-03, 4.591851e-01, 1.059904e-03, 4.165211e-06],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 6.809375e-05, 1.054401e-01, 3.630264e-01, 3.095171e-05],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 4.201129e-01, 1.750470e-02, 0.000000e+00],
        ])
    return np.repeat(Sigma_s0[None], len(T), axis=0)


def calculate_Sigma_f(T):
    Sigma_f = np.array([5.195495e-04, 7.380079e-05, 1.986933e-04, 7.749989e-04, 1.227452e-03, 3.861481e-03, 6.068707e-03, 1.401790e-02])
    return np.repeat(Sigma_f[None], len(T), axis=0)


def calculate_nu(T):
    nu = np.array([2.747959, 2.448143, 2.433723, 2.435037, 2.436695, 2.4367, 2.4367, 2.4367])
    return np.repeat(nu[None], len(T), axis=0)


def calculate_chi(T):
    chi = np.array([8.456363e-01, 1.535990e-01, 7.580154e-04, 6.673239e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00])
    return np.repeat(chi[None], len(T), axis=0)


def calculate_kappa(T):
    kappa = np.array([1.968860e+08, 1.934114e+08, 1.934102e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08])
    return np.repeat(kappa[None], len(T), axis=0)

def calculate_parameters(T):
    D = calculate_diffusivity(T) * 1e-2
    Sigma_t = calculate_Sigma_t(T) * 1e2
    Sigma_s0 = calculate_Sigma_s0(T) * 1e2
    Sigma_f = calculate_Sigma_f(T) * 1e2
    nu = calculate_nu(T)
    chi = calculate_chi(T)
    kappa = calculate_kappa(T) * 1.602e-19
    return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa

class NeutronicsModel(Component):

    def __init__(self, config):
        self.cfg = config

        # temporary
        self.T_FP = np.full((self.cfg.mesh.N_Z*self.cfg.mesh.N_R), 900)
        self.T_HP = 900
        self.T_M = 900

    def initial_guess(self):
        phi_ng_initial = np.full((self.cfg.mesh.N_Z*self.cfg.energy.N_G), 1e12)
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

        res_anchor = self.cfg.mesh.N_Z * self.cfg.mesh.N_R - np.dot(phi_ng, phi_ng)

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
        T_center_axial = self.get_axial_temperature(T)

        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = calculate_parameters(T_center_axial)
        return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa
    
    # def get_material_data(self, T):
    #     T_center_axial = self.get_axial_temperature(T)

    #     X = np.array([[self.T_HP, T_FP, self.T_M] for T_FP in T_center_axial])
    #     params = interpolator_model.predict_dict(X)

    #     Diffusivity = np.zeros((len(params), self.cfg.energy.N_G))
    #     Sigma_t = np.zeros((len(params), self.cfg.energy.N_G))
    #     Sigma_f = np.zeros((len(params), self.cfg.energy.N_G))
    #     Sigma_s0 = np.zeros((len(params), self.cfg.energy.N_G, self.cfg.energy.N_G))
    #     fission_number = np.zeros((len(params), self.cfg.energy.N_G))
    #     Chi = np.zeros((len(params), self.cfg.energy.N_G))
    #     kappa = np.zeros((len(params), self.cfg.energy.N_G))
        
    #     for i in range(len(params)):
    #         Diffusivity[i] = np.array(params[i]["diffusion_coefficient"]) * 1e-2
    #         Sigma_t[i] = np.array(params[i]["total_xs"]) * 1e2
    #         Sigma_f[i] = np.array(params[i]["fission_xs"]) * 1e2
    #         Sigma_s0[i] = (np.array(params[i]["scatter_matrix_xs"]) * 1e2)
    #         fission_number[i] = np.array(params[i]["nu"])
    #         Chi[i] = np.array(params[i]["chi"])
    #         kappa[i] = np.array(params[i]["kappa"]) * 1.602176634e-19

    #     return Diffusivity, Sigma_t, Sigma_s0, Sigma_f, fission_number, Chi, kappa

if __name__ == "__main__":
    data = {}

    from utils.solver import Solver

    mesh = NeutronicsMesh(
        N_R = 50,
        N_Z = 100,
        l = 1
    )
    energy = NeutronicsEnergy(
        N_G = 8,
        power = 1000
    )
    cfg = NeutronicsConfig(mesh, energy)

    neutronics_model = NeutronicsModel(cfg)

    solver = Solver([neutronics_model])
    solver.fsolve()

    phi_n_g, k = solver.solution

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
    # ax.plot(phi[:, 1], color="blue")
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k:.2f}")

    plt.show()