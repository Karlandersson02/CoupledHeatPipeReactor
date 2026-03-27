import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


N_G = 8


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

        self.l = 1
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
    

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

if __name__ == "__main__":
    model = LinearNeutronModel()

    T = np.full(model.N_R * model.N_Z, 900.0)

    # Try both if you are unsure about scattering orientation
    phi, k_eff = model.solve_eigenproblem(T, scattering_mode="in_out")
    l2, linf = model.check_residual(scattering_mode="in_out")

    print("k_eff =", k_eff)
    print("Residual L2   =", l2)
    print("Residual Linf =", linf)
    print("phi min =", phi.min())
    print("phi max =", phi.max())

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