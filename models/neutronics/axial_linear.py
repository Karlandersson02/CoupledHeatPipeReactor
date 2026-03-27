import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


def calculate_parameters(T):
    D = calculate_diffusivity(T)
    Sigma_t = calculate_Sigma_t(T)
    Sigma_s0 = calculate_Sigma_s0(T)
    Sigma_f = calculate_Sigma_f(T)
    nu = calculate_nu(T)
    chi = calculate_chi(T)
    kappa = calculate_kappa(T)
    return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa


class LinearNeutronModel:
    def __init__(self):
        self.N_R = 10
        self.N_Z = 100
        self.N_G = 8

        self.l = 2
        self.delta_Z = self.l / self.N_Z
        self.cross_sectional_area = np.pi * 0.01**2

        self.power = 1000.0
        self.T_HP = 900.0
        self.T_M = 900.0

    def get_axial_temperature(self, T):
        return T[::self.N_R]

    def get_material_data(self, T):
        T_center_axial = self.get_axial_temperature(T)
        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = calculate_parameters(T_center_axial)
        # D = D[:, ::-1]
        # Sigma_t = Sigma_t[:, ::-1]
        # Sigma_s0 = Sigma_s0[:, ::-1, ::-1]
        # Sigma_f = Sigma_f[:, ::-1]
        # nu = nu[:, ::-1]
        # chi = chi[:, ::-1]
        # kappa = kappa[:, ::-1]
        return D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa

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
            (2.0 / self.delta_Z**2)
            * (D_n_g[mask_nm1] * D_nm1_g[mask_nm1])
            / (D_n_g[mask_nm1] + D_nm1_g[mask_nm1])
        )

        alpha_np1_g[mask_np1] = (
            (2.0 / self.delta_Z**2)
            * (D_n_g[mask_np1] * D_np1_g[mask_np1])
            / (D_n_g[mask_np1] + D_np1_g[mask_np1])
        )

        beta_NZ_g = (2.0 * D_n_g[-1] / self.delta_Z) / (self.delta_Z + 4.0 * D_n_g[-1])
        beta_1_g = (2.0 * D_n_g[0] / self.delta_Z) / (self.delta_Z + 4.0 * D_n_g[0])

        a_n_g = alpha_np1_g + alpha_n_g
        b_n_g = -alpha_np1_g
        c_n_g = -alpha_n_g

        a_n_g[0] = beta_1_g + alpha_np1_g[0]
        a_n_g[-1] = beta_NZ_g + alpha_n_g[-1]

        b_n_g[0] = -alpha_np1_g[0]
        b_n_g[-1] = 0.0

        c_n_g[0] = 0.0
        c_n_g[-1] = -alpha_n_g[-1]

        return a_n_g, b_n_g, c_n_g

    def idx(self, z, g):
        return z * self.N_G + g

    def build_operators(self, T, scattering_mode="out_in"):
        """
        Build L and F for
            L phi = (1/k) F phi

        scattering_mode:
            "out_in" means Sigma_s[g_out, g_in]
            "in_out"  means Sigma_s[g_in, g_out]
        """
        D, Sigma_t, Sigma_s0, Sigma_f, nu, chi, kappa = self.get_material_data(T)
        a_n_g, b_n_g, c_n_g = self.calculate_abc(D)

        n_unknowns = self.N_Z * self.N_G

        L = sp.lil_matrix((n_unknowns, n_unknowns), dtype=np.float64)
        F = sp.lil_matrix((n_unknowns, n_unknowns), dtype=np.float64)

        for z in range(self.N_Z):
            for g in range(self.N_G):
                row = self.idx(z, g)

                # Leakage + total
                L[row, self.idx(z, g)] += a_n_g[z, g] + Sigma_t[z, g]

                if z < self.N_Z - 1:
                    L[row, self.idx(z + 1, g)] += b_n_g[z, g]

                if z > 0:
                    L[row, self.idx(z - 1, g)] += c_n_g[z, g]

                # Subtract scattering source contribution from L
                for gp in range(self.N_G):
                    if scattering_mode == "out_in":
                        # source into g from gp
                        sig_s = Sigma_s0[z, g, gp]
                    elif scattering_mode == "in_out":
                        # source into g from gp
                        sig_s = Sigma_s0[z, gp, g]
                    else:
                        raise ValueError("scattering_mode must be 'out_in' or 'in_out'")

                    L[row, self.idx(z, gp)] -= sig_s

                # Fission production operator
                # phi_gp -> chi_g * nu_gp * Sigma_f_gp
                for gp in range(self.N_G):
                    F[row, self.idx(z, gp)] += chi[z, g] * nu[z, gp] * Sigma_f[z, gp]

        return L.tocsr(), F.tocsr(), kappa

    def solve_eigenproblem(self, T=None, scattering_mode="out_in"):
        if T is None:
            T = np.full(self.N_R * self.N_Z, 900.0)

        L, F, kappa = self.build_operators(T, scattering_mode=scattering_mode)

        # Solve (F x) = k (L x), equivalently inv(L)F x = k x
        A = spla.LinearOperator(
            shape=L.shape,
            matvec=lambda x: spla.spsolve(L, F @ x),
            dtype=np.float64,
        )

        eigvals, eigvecs = spla.eigs(A, k=1, which="LR")
        k_eff = np.real(eigvals[0])
        phi = np.real(eigvecs[:, 0])

        # Fix overall sign
        if np.sum(phi) < 0.0:
            phi *= -1.0

        # Small negative roundoff can happen with eigensolvers; clip if tiny
        if np.min(phi) < 0.0:
            if np.min(phi) > -1e-12 * np.max(np.abs(phi)):
                phi = np.maximum(phi, 0.0)

        phi = phi.reshape(self.N_Z, self.N_G)

        # Normalize to requested power
        power_density = np.sum(kappa * calculate_Sigma_f(self.get_axial_temperature(T)) * phi, axis=1)
        current_power = np.sum(power_density) * self.cross_sectional_area * self.delta_Z

        if current_power <= 0.0:
            raise RuntimeError("Computed non-positive power; cannot normalize flux.")

        scale = self.power / current_power
        phi *= scale

        self.phi_n_g = phi
        self.k = k_eff
        self.T = T.copy()

        return phi, k_eff

    def check_residual(self, scattering_mode="out_in"):
        """
        Check ||L phi - (1/k) F phi|| after solve.
        """
        if not hasattr(self, "phi_n_g"):
            raise RuntimeError("Solve the model first.")

        L, F, _ = self.build_operators(self.T, scattering_mode=scattering_mode)
        phi_vec = self.phi_n_g.ravel()
        r = L @ phi_vec - (1.0 / self.k) * (F @ phi_vec)
        return np.linalg.norm(r), np.linalg.norm(r, ord=np.inf)
    

def compute_kinf(T):
    """
    Compute infinite-medium multiplication factor k_inf
    from homogenized multigroup cross sections.
    """

    # Get data (take first row since homogeneous in space)
    Sigma_t = calculate_Sigma_t(T)[0]          # (G,)
    Sigma_s = calculate_Sigma_s0(T)[0]         # (G, G)
    Sigma_f = calculate_Sigma_f(T)[0]          # (G,)
    nu      = calculate_nu(T)[0]               # (G,)
    chi     = calculate_chi(T)[0]              # (G,)

    # --- Build operators ---

    # Loss operator A = Σ_t - Σ_s (using in→out convention)
    # For diffusion form: A_g = Σ_t,g - sum_{g'} Σ_s(g -> g')
    # But matrix form must subtract full scattering matrix
    A = np.diag(Sigma_t) - Sigma_s

    # Fission operator F = χ ⊗ (νΣ_f)
    nuSigma_f = nu * Sigma_f
    F = np.outer(chi, nuSigma_f)

    # --- Solve eigenvalue problem ---
    # k_inf = dominant eigenvalue of A^{-1} F
    M = np.linalg.solve(A, F)
    eigs = np.linalg.eigvals(M)

    # Take largest real eigenvalue
    k_inf = np.max(np.real(eigs))

    return k_inf

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

if __name__ == "__main__":
    T = np.array([900.0])  # dummy input
    k_inf = compute_kinf(T)
    print("k_inf =", k_inf)

    model = LinearNeutronModel()

    T = np.full(model.N_R * model.N_Z, 900.0)

    # Try both if you are unsure about scattering orientation
    phi, k_eff = model.solve_eigenproblem(T, scattering_mode="in_out")
    l2, linf = model.check_residual(scattering_mode="in_out")

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "Computer modern"
    mpl.rcParams["font.size"] = 22

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    colors = plt.cm.viridis(np.linspace(0, 1, model.N_G))[::-1]
    ax.grid(alpha=0.4)

    for g in range(model.N_G):
        ax.plot(phi[:, g], color=colors[g], label=g)

    ax.legend()
    ax.set_yscale("log")
    ax.set_title(r"$k_{eff} = $" + f"{k_eff:.6f}")
    plt.show()