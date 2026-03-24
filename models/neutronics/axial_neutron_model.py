import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import fsolve
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

N_G = 8  # adjust as needed

def calculate_diffusivity(T):
    T = np.asarray(T)
    return np.full((len(T), N_G), 0.5*1e-2)   # cm

def calculate_Sigma_t(T):
    T = np.asarray(T)
    return np.full((len(T), N_G), 1*1e2)   # 1/cm

def calculate_Sigma_f(T):
    T = np.asarray(T)
    return np.full((len(T), N_G), 0.1*1e2)   # 1/cm

def calculate_Sigma_s0(T):
    T = np.asarray(T, dtype=float)
    Sigma_s = np.zeros((len(T), N_G, N_G), dtype=float)

    # Reference temperature scaling for mild temperature dependence
    T_ref = 800.0
    temp_factor = np.sqrt(T_ref / T)[:, None, None]

    for g_from in range(N_G):
        for g_to in range(N_G):
            if g_to == g_from:
                # within-group scattering
                Sigma_s[:, g_from, g_to] = 0.20

            elif g_to > g_from:
                # downscatter: stronger than upscatter
                # farther jumps are weaker
                jump = g_to - g_from
                Sigma_s[:, g_from, g_to] = 0.08 / jump

            else:
                # upscatter: weaker, but allowed
                # farther jumps are much weaker
                jump = g_from - g_to
                Sigma_s[:, g_from, g_to] = 0.015 / jump

    # Apply mild temperature dependence
    Sigma_s *= temp_factor
    Sigma_s *= 1e2

    return Sigma_s

def calculate_fission_number(T):
    T = np.asarray(T)
    return np.full((len(T), N_G), 2.5)   # ν (neutrons per fission)

def calculate_Chi(T):
    T = np.asarray(T)
    
    chi = np.zeros((len(T), N_G))
    
    # all neutrons born in fast group (typical 2-group model)
    chi[:, 0] = 1
    
    return chi

def calculate_kappa(T):
    T = np.asarray(T)
    
    # ~200 MeV per fission in Joules
    kappa_value = 3.2e-11  # J/fission
    
    return np.full((len(T), N_G), kappa_value)

# def calculate_parameters(T):
#     D              = calculate_diffusivity(T)
#     Sigma_t        = calculate_Sigma_t(T)
#     Sigma_f        = calculate_Sigma_f(T)
#     Sigma_s0       = calculate_Sigma_s0(T)
#     fission_number = calculate_fission_number(T)
#     Chi            = calculate_Chi(T)
#     kappa          = calculate_kappa(T)

#     return D, Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa

from utils import interpolator
intepolator_model = interpolator()

def calculate_parameters():
    


class NeutronModel:

    def __init__(self):

        # geometry
        self.N_R = 10
        self.N_Z = 100
        self.N_G = 8

        self.l = 1.0
        self.delta_Z = self.l / self.N_Z

        # system properties
        self.Power = 1000.0

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

    def get_material_data(self, T):
        T_center_axial = self.get_axial_temperature(T)

        params = calculate_parameters(T_center_axial)

        _, Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa = params

        return Sigma_t, Sigma_f, Sigma_s0, fission_number, Chi, kappa

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
        total_power = np.sum(power_density) * self.delta_Z

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

        T = np.ones((self.N_R * self.N_Z,), dtype=np.float64) * 800
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

    colors = plt.cm.viridis(np.linspace(0, 1, 8))
    ax.grid(alpha=0.4)
    for i in range(len(colors)):
        ax.plot(phi[:, i], color=colors[i], label=i)
    ax.legend()
    # ax.plot(phi[:, 1], color="blue")
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k_eff:.2f}")

    plt.show()