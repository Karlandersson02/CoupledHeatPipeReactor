import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import fsolve

# MODEL_PATH = r"./utils/rgi_surrogate.joblib"
# from utils.interpolator import OpenMCTallyGridSurrogate
# interpolator_model = OpenMCTallyGridSurrogate()
# interpolator_model = interpolator_model.load(MODEL_PATH)

N_G = 8

def calculate_diffusivity(T):
    D = np.array([1.822733189502, 0.909539957965, 0.789695519368, 0.782178297954, 0.778223331717, 0.720306709623, 0.645906634649, 0.716544176406])
    return np.repeat(D[None] * 1e-2, len(T), axis=0)


def calculate_Sigma_t(T):
    Sigma_t = np.array([0.218755248796, 0.399931318974, 0.448117958797, 0.451600144497, 0.452600375002, 0.472615208883, 0.477890785379, 0.465947127704])
    return np.repeat(Sigma_t[None] * 1e2, len(T), axis=0)


def calculate_Sigma_s0(T):
    Sigma_s0 = np.array(
        [
            [1.915640419307e-01, 2.649476899088e-02, 7.612501864966e-06, 2.175000532847e-07, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00],
            [0.000000000000e+00, 3.821866087803e-01, 1.748239884516e-02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00],
            [0.000000000000e+00, 0.000000000000e+00, 4.283323644297e-01, 1.897047538109e-02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00],
            [0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 4.309142395936e-01, 1.830836691992e-02, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00],
            [0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.108383435515e-05, 4.177546754032e-01, 3.235312912299e-02, 2.333438811611e-06, 0.000000000000e+00],
            [0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 7.035668782444e-03, 4.591850559024e-01, 1.059903520475e-03, 4.165210863191e-06],
            [0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 6.809375303709e-05, 1.054400814073e-01, 3.630263684643e-01, 3.095170592595e-05],
            [0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 4.201128786015e-01, 1.750470327506e-02, 0.000000000000e+00]
        ]
   )
    return np.repeat(Sigma_s0[None] * 1e2, len(T), axis=0)


def calculate_Sigma_f(T):
    Sigma_f = np.array([5.195495025547e-04, 7.380079372461e-05, 1.986932777248e-04, 7.749988914091e-04, 1.227451921783e-03, 3.861481365247e-03, 6.068707316140e-03, 1.401789774873e-02])
    return np.repeat(Sigma_f[None] * 1e2, len(T), axis=0)


def calculate_nu(T):
    nu = np.array([2.747958695492, 2.448142676021, 2.433723274757, 2.435037061696, 2.436694668567, 2.436700028023, 2.436700023982, 2.436700023356])
    return np.repeat(nu[None], len(T), axis=0)


def calculate_chi(T):
    chi = np.array([8.456362989955e-01, 1.535990124090e-01, 7.580153564470e-04, 6.673239061383e-06, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00, 0.000000000000e+00])
    return np.repeat(chi[None], len(T), axis=0)


def calculate_kappa(T):
    kappa = np.array([1.968860125802e+08, 1.934114201002e+08, 1.934102493864e+08, 1.934054129759e+08, 1.934054046411e+08, 1.934054022163e+08, 1.934054018967e+08, 1.934054018472e+08])
    return np.repeat(kappa[None] * 1.602e-19, len(T), axis=0)

def calculate_parameters(T):              # SI
    D = calculate_diffusivity(T)
    Sigma_t = calculate_Sigma_t(T)
    Sigma_s0 = calculate_Sigma_s0(T)
    Sigma_f = calculate_Sigma_f(T)
    nu = calculate_nu(T)
    chi = calculate_chi(T)
    kappa = calculate_kappa(T)
    return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa

class NeutronModel:

    def __init__(self):
        # geometry
        self.N_R = 10
        self.N_Z = 200
        self.N_G = 8

        self.l = 2
        self.delta_Z = self.l / self.N_Z
        self.cross_sectional_area = 0.01**2 * np.pi # approximate

        # system properties
        self.power = 1000
        self.T_HP = 900
        self.T_M = 900

    def unpack_variables(self, X):
        T_n = X[:self.N_R * self.N_Z]
        phi_ng = X[self.N_R * self.N_Z:-1]
        k = X[-1]
        return T_n, phi_ng, k

    def calculate_abc(self, D_n_g):

        D_nm1_g = np.zeros_like(D_n_g)
        D_np1_g = np.zeros_like(D_n_g)

        D_nm1_g[1:] = D_n_g[:-1]
        D_np1_g[:-1] = D_n_g[1:]

        alpha_n_g = np.zeros_like(D_n_g)
        alpha_np1_g = np.zeros_like(D_n_g)

        mask_nm1 = (D_n_g + D_nm1_g) > 0.0
        mask_np1 = (D_n_g + D_np1_g) > 0.0

        alpha_n_g[mask_nm1] = (
            (2.0 / self.delta_Z**2) *
            (D_n_g[mask_nm1] * D_nm1_g[mask_nm1]) /
            (D_n_g[mask_nm1] + D_nm1_g[mask_nm1])
        )

        alpha_np1_g[mask_np1] = (
            (2.0 / self.delta_Z**2) *
            (D_n_g[mask_np1] * D_np1_g[mask_np1]) /
            (D_n_g[mask_np1] + D_np1_g[mask_np1])
        )

        beta_NZ_g = (2.0 * D_n_g[-1] / self.delta_Z) / (self.delta_Z + 4.0 * D_n_g[-1])
        beta_1_g = (2.0 * D_n_g[0] / self.delta_Z) / (self.delta_Z + 4.0 * D_n_g[0])

        a_n_g = alpha_np1_g + alpha_n_g
        b_n_g = -alpha_np1_g
        c_n_g = -alpha_n_g

        # boundary adjustments
        a_n_g[0] = beta_1_g + alpha_np1_g[0]
        a_n_g[-1] = beta_NZ_g + alpha_n_g[-1]

        b_n_g[0] = -alpha_np1_g[0]
        b_n_g[-1] = np.zeros(self.N_G)

        c_n_g[0] = np.zeros(self.N_G)
        c_n_g[-1] = -alpha_n_g[-1]

        return a_n_g, b_n_g, c_n_g
    
    def calculate_residuals(self, X):
        T, phi_ng, k = self.unpack_variables(X)
        phi_n_g = np.reshape(phi_ng, (self.N_Z, self.N_G))
        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = self.get_material_data(T)
        a_n_g, b_n_g, c_n_g = self.calculate_abc(D)

        phi_np1_g = np.vstack([phi_n_g[1:], np.zeros((1, self.N_G))])
        phi_nm1_g = np.vstack([np.zeros((1, self.N_G)), phi_n_g[:-1]])

        res_transport = (
            a_n_g * phi_n_g + b_n_g * phi_np1_g + c_n_g * phi_nm1_g + Sigma_t * phi_n_g
            - (np.sum(Sigma_s0 * phi_n_g[:, :, None], axis=1) + (chi / k) * (np.sum(nu * Sigma_f * phi_n_g, axis=1))[:, None])
        )
        res_transport = np.ravel(res_transport)

        # power_density = np.sum(kappa * Sigma_f * phi_n_g, axis=1)
        # power = np.sum(power_density) * self.cross_sectional_area * self.delta_Z

        # res_power = np.array([(self.power - power)])

        res_normal = self.N_G * self.N_Z - np.dot(phi_ng, phi_ng)

        return np.r_[res_transport, res_normal]

    def get_axial_temperature(self, T):
        return T[::self.N_R]

    def get_material_data(self, T):
        T_center_axial = self.get_axial_temperature(T)

        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = calculate_parameters(T_center_axial)
        return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa
    
    def solve(self):
        T_initial = np.full((self.N_Z*self.N_R), 900)
        phi_ng_initial = np.full((self.N_Z*self.N_G), 1)
        k_initial = np.array([1])
        X_initial = np.concatenate([phi_ng_initial, k_initial]) # ignore T for now

        counter = {"it": 0}
        def residuals_wrapper(X):
            counter["it"] += 1
            
            r = self.calculate_residuals(np.r_[T_initial, X])
            
            print(f"Iteration {counter['it']}: ||r|| = {np.linalg.norm(r):.3e}")
            
            return r
        
        res, info, ier, msg = fsolve(
            residuals_wrapper,
            X_initial,
            full_output=True
        )

        print("ier =", ier)
        print("msg =", msg)
        print("||res|| =", np.linalg.norm(info["fvec"]))

        if ier != 1:
            raise RuntimeError(f"fsolve did not converge: {msg}")
        
        _, _, _, Sigma_f, _, _, kappa = self.get_material_data(T_initial)
        phi_n_g = np.reshape(res[:-1], (self.N_Z, self.N_G))

        power_density = np.sum(kappa * Sigma_f * phi_n_g, axis=1)
        power = np.sum(power_density) * self.cross_sectional_area * self.delta_Z
        phi_n_g *= self.power / power
        
        k = res[-1]

        self.phi = phi_n_g
        self.k = k


    
    
    # def get_material_data(self, T):
    #     T_center_axial = self.get_axial_temperature(T)

    #     X = np.array([[self.T_HP, T_FP, self.T_M] for T_FP in T_center_axial])
    #     params = interpolator_model.predict_dict(X)

    #     Sigma_t = np.zeros((len(params), self.N_G))
    #     Sigma_f = np.zeros((len(params), self.N_G))
    #     Sigma_s0 = np.zeros((len(params), self.N_G, self.N_G))
    #     fission_number = np.zeros((len(params), self.N_G))
    #     Chi = np.zeros((len(params), self.N_G))
    #     kappa = np.zeros((len(params), self.N_G))
        
    #     for i in range(len(params)):
    #         Sigma_t[i] = np.array(params[i]["total_xs"]) * 1e2
    #         Sigma_f[i] = np.array(params[i]["fission_xs"]) * 1e2
    #         Sigma_s0[i] = (np.array(params[i]["scatter_matrix_xs"]) * 1e2)
    #         fission_number[i] = np.array(params[i]["nu"])
    #         Chi[i] = np.array(params[i]["chi"])
    #         kappa[i] = np.array(params[i]["kappa"]) * 1.602176634e-19

    #     return Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa


if __name__ == "__main__":

    neutron_model = NeutronModel()
    neutron_model.solve()
    phi = neutron_model.phi
    k_eff = neutron_model.k

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "Computer modern"
    mpl.rcParams["font.size"] = 22

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    colors = plt.cm.viridis(np.linspace(0, 1, 8))[::-1]
    ax.grid(alpha=0.4)
    for i in range(len(colors)):
        ax.plot(phi[:, i], color=colors[i], label=i)
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k_eff:.2f}")

    plt.show()