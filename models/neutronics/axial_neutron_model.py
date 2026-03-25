import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import fsolve
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

N_G = 8  # adjust as needed

MODEL_PATH = r"./utils/rgi_surrogate.joblib"
from utils.interpolator import OpenMCTallyGridSurrogate
interpolator_model = OpenMCTallyGridSurrogate()
interpolator_model = interpolator_model.load(MODEL_PATH)

# def calculate_diffusivity(T): # Temporary
#     return np.repeat(np.array([1.856146, 0.931944, 0.808143, 0.8032, 0.804013, 0.751048, 0.664298, 0.668599]) * 1e-2, len(T)).reshape(len(T), -1)

# def calculate_diffusivity(T):  # Temporary
#     D = np.repeat(np.array([1.856146, 0.931944, 0.808143, 0.8032,
#                   0.804013, 0.751048, 0.664298, 0.668599]) * 1e-2, len(T)).reshape(len(T), -1)
#     return D

def calculate_diffusivity(T):
    D = np.array([1.856146, 0.931944, 0.808143, 0.8032, 0.804013, 0.751048, 0.664298, 0.668599])
    return np.repeat(D[None] * 1e-2, len(T), axis=0)


def calculate_Sigma_t(T):
    Sigma_t = np.array([0.215433, 0.390558, 0.437022, 0.438771, 0.437527, 0.452985, 0.462072, 0.491347])
    return np.repeat(Sigma_t[None] * 1e2, len(T), axis=0)


def calculate_Sigma_s0(T):
    Sigma_s0 = np.array([
        [1.532002e-01, 2.560313e-02, 8.430881e-06, 2.465170e-08, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 3.415866e-01, 1.578866e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 3.946257e-01, 1.686322e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 3.961719e-01, 1.590331e-02, 0.000000e+00, 0.000000e+00, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 1.485168e-05, 3.886724e-01, 2.267349e-02, 6.326949e-07, 0.000000e+00],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 8.702828e-03, 4.265023e-01, 9.153768e-04, 3.121413e-06],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 3.510087e-05, 1.015273e-01, 3.854386e-01, 2.730068e-05],
        [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 4.610164e-01, 4.610164e-03, 7.207613e-03],
    ])
    return np.repeat(Sigma_s0[None] * 1e2, len(T), axis=0)


def calculate_Sigma_f(T):
    Sigma_f = np.array([5.863028e-04, 8.501323e-05, 2.374428e-04, 9.599783e-04, 1.514871e-03, 5.528794e-03, 9.530878e-03, 1.017026e-02])
    return np.repeat(Sigma_f[None] * 1e2, len(T), axis=0)


def calculate_nu(T):
    nu = np.array([2.746486, 2.448851, 2.433717, 2.435004, 2.436693, 2.4367, 2.4367, 2.4367])
    return np.repeat(nu[None], len(T), axis=0)


def calculate_chi(T):
    chi = np.array([8.453690e-01, 1.536567e-01, 9.669716e-04, 7.354006e-06, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00])
    return np.repeat(chi[None], len(T), axis=0)


def calculate_kappa(T):
    kappa = np.array([1.968885e+08, 1.934116e+08, 1.934106e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08, 1.934054e+08])
    return np.repeat(kappa[None] * 1.602e-19, len(T), axis=0)

class NeutronModel:

    def __init__(self):
        # geometry
        self.N_R = 10
        self.N_Z = 100
        self.N_G = 8

        self.l = 1.0
        self.delta_Z = self.l / self.N_Z
        self.cross_sectional_area = 0.01**2 * np.pi # approximate

        # system properties
        self.Power = 1000.0
        self.T_HP = 800
        self.T_M = 850

        Sigma_t, Sigma_f, Sigma_s0, nu, chi, _ = self.get_material_data(
            np.full(self.N_R * self.N_Z, 900.0)
        )

        n = 0
        nuSigma_f = nu[n] * Sigma_f[n]

        Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa = self.get_material_data(
            np.full(self.N_R * self.N_Z, 900.0)
        )

        n = 0

        Sig_t = Sigma_t[n]                    # (8,)
        S = Sigma_s0[n]                       # (8, 8)
        nuSig_f = fission_number[n] * Sigma_f[n]   # (8,)
        Chi_n = Chi[n]                        # (8,)

        L1 = np.diag(Sig_t) - S
        L2 = np.diag(Sig_t) - S.T
        F = np.outer(Chi_n, nuSig_f)

        eig1 = np.linalg.eigvals(np.linalg.inv(L1) @ F)
        eig2 = np.linalg.eigvals(np.linalg.inv(L2) @ F)

        print("eig using S   =", eig1)
        print("eig using S.T =", eig2)
        print("k_like using S   =", np.max(eig1.real))
        print("k_like using S.T =", np.max(eig2.real))

    def idx(self, n, g):
        return n * self.N_G + g

    def unpack_variables(self, X):
        T_n = X[:self.N_R * self.N_Z]
        phi_ng = X[self.N_R * self.N_Z:]
        return T_n, phi_ng

    def calculate_abc(self, X):
        T_center_axial = X[0:self.N_R * self.N_Z:self.N_R]

        D_n_g = calculate_diffusivity(T_center_axial)

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

    def get_axial_temperature(self, T):
        return T[::self.N_R]

    def get_material_data3(self, T):
        T_center_axial = self.get_axial_temperature(T)

        X = np.array([[self.T_HP, T_FP, self.T_M] for T_FP in T_center_axial])
        params = interpolator_model.predict_dict(X)

        Sigma_t = np.zeros((len(params), self.N_G))
        Sigma_f = np.zeros((len(params), self.N_G))
        Sigma_s0 = np.zeros((len(params), self.N_G, self.N_G))
        fission_number = np.zeros((len(params), self.N_G))
        Chi = np.zeros((len(params), self.N_G))
        kappa = np.zeros((len(params), self.N_G))
        
        for i in range(len(params)):
            Sigma_t[i] = np.array(params[i]["total_xs"]) * 1e2
            Sigma_f[i] = np.array(params[i]["fission_xs"]) * 1e2
            Sigma_s0[i] = (np.array(params[i]["scatter_matrix_xs"]) * 1e2)
            fission_number[i] = np.array(params[i]["nu"])
            Chi[i] = np.array(params[i]["chi"])
            kappa[i] = np.array(params[i]["kappa"]) * 1.602176634e-19

        return Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa
    
    def get_material_data(self, T):
        Sigma_t2, Sigma_f2, Sigma_s02, fission_number2, Chi2, kappa2 = self.get_material_data2(T)
        Sigma_t3, Sigma_f3, Sigma_s03, fission_number3, Chi3, kappa3 = self.get_material_data3(T)

        Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa = Sigma_t2, Sigma_f2, Sigma_s02, fission_number2, Chi2, kappa2

        return Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa

    def get_material_data2(self, T):
        T_axial = self.get_axial_temperature(T)
        # D = calculate_diffusivity(T_axial)
        Sigma_t = calculate_Sigma_t(T_axial)
        Sigma_s0 = calculate_Sigma_s0(T_axial)
        Sigma_f = calculate_Sigma_f(T_axial)
        nu = calculate_nu(T_axial)
        chi = calculate_chi(T_axial)
        kappa = calculate_kappa(T_axial)
        return Sigma_t, Sigma_f, Sigma_s0, nu, chi, kappa

    def build_loss_matrix(self, T):
        X_dummy = np.concatenate([T, np.zeros(self.N_Z * self.N_G)])
        a_n_g, b_n_g, c_n_g = self.calculate_abc(X_dummy)
        Sigma_t, _, Sigma_s0, _, _, _ = self.get_material_data(T)

        N = self.N_Z * self.N_G
        A = lil_matrix((N, N), dtype=np.float64)

        for n in range(self.N_Z):
            for g in range(self.N_G):
                i = self.idx(n, g)

                # diagonal leakage + total term
                A[i, i] += a_n_g[n, g] + Sigma_t[n, g]

                # axial couplings, same energy group
                if n < self.N_Z - 1:
                    A[i, self.idx(n + 1, g)] += b_n_g[n, g]
                if n > 0:
                    A[i, self.idx(n - 1, g)] += c_n_g[n, g]

                for gp in range(self.N_G):
                    A[i, self.idx(n, gp)] -= Sigma_s0[n, gp, g]

        return A.tocsr()

    def build_fission_source(self, phi_n_g, T):
        _, Sigma_f, _, fission_number, Chi, _ = self.get_material_data(T)

        nuSigma_f = fission_number * Sigma_f                     # shape (N_Z, N_G)
        production = np.sum(nuSigma_f * phi_n_g, axis=1)         # shape (N_Z,)
        source = Chi * production[:, None]                       # shape (N_Z, N_G)

        return source

    def normalize_flux_to_power(self, phi_n_g, T):
        _, Sigma_f, _, _, _, kappa = self.get_material_data(T)

        power_density = kappa * Sigma_f * phi_n_g
        total_power = np.sum(power_density * self.cross_sectional_area) * self.delta_Z

        if total_power <= 0.0:
            raise RuntimeError("Computed non-positive total power during normalization.")

        scale = self.Power / total_power
        return phi_n_g * scale

    def compute_eigen_residual(self, phi_n_g, k_eff, T):
        A = self.build_loss_matrix(T)
        rhs = (1.0 / k_eff) * self.build_fission_source(phi_n_g, T).ravel()
        lhs = A @ phi_n_g.ravel()
        return lhs - rhs

    def solve(self, max_iters=2000, tol_k=1e-5, tol_phi=1e-5, verbose=True):

        T = np.full((self.N_R * self.N_Z,), 900, dtype=np.float64)
        phi_n_g = np.ones((self.N_Z, self.N_G), dtype=np.float64)
        k_eff = 1.0

        phi_n_g = self.normalize_flux_to_power(phi_n_g, T)
        A = self.build_loss_matrix(T)

        for it in range(max_iters):
            phi_old = phi_n_g.copy()
            k_old = k_eff

            q_old = self.build_fission_source(phi_old, T)
            q_old_sum = np.sum(q_old)

            if q_old_sum <= 0.0:
                raise RuntimeError("Non-positive fission source encountered in power iteration.")

            rhs = (1.0 / k_old) * q_old.ravel()
            phi_new = spsolve(A, rhs).reshape(self.N_Z, self.N_G)

            phi_new[phi_new < 0.0] = 0.0

            q_new = self.build_fission_source(phi_new, T)
            q_new_sum = np.sum(q_new)

            if q_new_sum <= 0.0:
                raise RuntimeError("Non-positive updated fission source encountered in power iteration.")

            k_eff = k_old * (q_new_sum / q_old_sum)

            phi_n_g = self.normalize_flux_to_power(phi_new, T)

            # convergence metrics
            dk_rel = abs(k_eff - k_old) / max(abs(k_eff), 1e-16)
            dphi_rel = np.linalg.norm(phi_n_g - phi_old, np.inf) / max(np.linalg.norm(phi_n_g, np.inf), 1e-16)

            residual = self.compute_eigen_residual(phi_n_g, k_eff, T)
            res_inf = np.linalg.norm(residual, np.inf)

            if verbose:
                print(
                    f"it = {it:4d} | "
                    f"k_eff = {k_eff:.10f} | "
                    f"dk_rel = {dk_rel:.3e} | "
                    f"dphi_rel = {dphi_rel:.3e} | "
                    f"|R|_inf = {res_inf:.3e}"
                )

            if dk_rel < tol_k and dphi_rel < tol_phi:
                break
        else:
            raise RuntimeError("Power iteration did not converge within max_iters.")

        self.T = T
        self.phi = phi_n_g
        self.k_eff = k_eff
    




if __name__ == "__main__":

    neutron_model = NeutronModel()
    neutron_model.solve()
    phi = neutron_model.phi
    k_eff = neutron_model.k_eff

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
    # ax.plot(phi[:, 1], color="blue")
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k_eff:.2f}")

    plt.show()