import numpy as np

from models.component import Component
from data.dataclass import *

class FuelPin(Component):
    def __init__(self, config: FuelPinConfigResolved):

        self.cfg = config

        self.phi_g   = np.tile(np.array([4.16e18, 5.47e17], dtype=float), (self.cfg.mesh.N_Z, 1))
        self.Sigma_f = np.array([9.4e-2, 5.48e1], dtype=float)
        self.kappa   = 3.204e-11
        self.T_mod   = np.ones(self.cfg.mesh.N_Z) * 1000.0

        self.Delta_Z = self.cfg.geometry.l / self.cfg.mesh.N_Z
        
        self.calculate_qr()

    def initial_guess(self):
        X_initial = np.ones((self.cfg.mesh.N_Z * self.cfg.mesh.N_R, ))
        return X_initial

    def assemble(self):
        self.initialize_discretization()

        k = self.generate_k_matrix()
        h = self.generate_h_matrix()

        alpha = self.calculate_alpha(k)

        self.M, self.C = self.generate_matrix_form(alpha, self.qr, self.T_mod, k, h)

    def calculate_qr(self):
        self.initialize_discretization()
        self.qr = np.sum(self.phi_g * self.Sigma_f * self.kappa * self.Delta_V, axis = 1)

    def get_residuals(self, X):
        T = X.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)

        self.calculate_qr()
        self.initialize_discretization()

        k = self.generate_k_matrix(T)
        h = self.generate_h_matrix()
        alpha = self.calculate_alpha(k)

        qr = self.qr
        T_mod = self.T_mod

        N_Z = self.cfg.mesh.N_Z
        N_R = self.cfg.mesh.N_R
        N_fuel = self.cfg.mesh.N_fuel

        res = np.zeros((N_Z, N_R), dtype=float)

        # -----------------------
        # Bulk elements
        if N_Z > 2 and N_R > 2:
            kc = k[1:-1, 1:-1]
            ac = alpha[1:-1, 1:-1]

            res[1:-1, 1:-1] = (
                -kc * (ac[..., 0] + ac[..., 1] + ac[..., 2] + ac[..., 3]) * T[1:-1, 1:-1]
                + kc * ac[..., 0] * T[1:-1, 2:]
                + kc * ac[..., 1] * T[1:-1, :-2]
                + kc * ac[..., 2] * T[2:, 1:-1]
                + kc * ac[..., 3] * T[:-2, 1:-1]
            )

        # -----------------------
        # Insulated wall at z = 0, excluding corners
        if N_R > 2:
            z = 0
            kc = k[z, 1:-1]
            ac = alpha[z, 1:-1]

            res[z, 1:-1] = (
                -kc * (ac[:, 0] + ac[:, 1] + ac[:, 2]) * T[z, 1:-1]
                + kc * ac[:, 0] * T[z, 2:]
                + kc * ac[:, 1] * T[z, :-2]
                + kc * ac[:, 2] * T[z + 1, 1:-1]
            )

        # -----------------------
        # Insulated wall at z = N_Z - 1, excluding corners
        if N_R > 2:
            z = N_Z - 1
            kc = k[z, 1:-1]
            ac = alpha[z, 1:-1]

            res[z, 1:-1] = (
                -kc * (ac[:, 0] + ac[:, 1] + ac[:, 3]) * T[z, 1:-1]
                + kc * ac[:, 0] * T[z, 2:]
                + kc * ac[:, 1] * T[z, :-2]
                + kc * ac[:, 3] * T[z - 1, 1:-1]
            )

        # -----------------------
        # Cladding BC elements, excluding corners
        if N_Z > 2:
            r = N_R - 1
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]
            hc = h[1:-1, r]

            res[1:-1, r] = (
                (-kc * (ac[:, 1] + ac[:, 2] + ac[:, 3]) - hc * ac[:, 0]) * T[1:-1, r]
                + kc * ac[:, 1] * T[1:-1, r - 1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
                + hc * ac[:, 0] * T_mod[1:-1]
            )

        # -----------------------
        # Fuel inner BC elements, excluding corners
        if N_Z > 2:
            r = 0
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]

            res[1:-1, r] = (
                -kc * (ac[:, 0] + ac[:, 2] + ac[:, 3]) * T[1:-1, r]
                + kc * ac[:, 0] * T[1:-1, r + 1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
            )

        # -----------------------
        # Lowermost cladding outer corner: z = 0, r = N_R - 1
        z = 0
        r = N_R - 1
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
            + h[z, r] * alpha[z, r, 0] * T_mod[z]
        )

        # -----------------------
        # Uppermost cladding outer corner: z = N_Z - 1, r = N_R - 1
        z = N_Z - 1
        r = N_R - 1
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
            + h[z, r] * alpha[z, r, 0] * T_mod[z]
        )

        # -----------------------
        # Lowermost fuel inner corner: z = 0, r = 0
        z = 0
        r = 0
        res[z, r] = (
            -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
        )

        # -----------------------
        # Uppermost fuel inner corner: z = N_Z - 1, r = 0
        z = N_Z - 1
        r = 0
        res[z, r] = (
            -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
        )

        # -----------------------
        # Fuel heat production
        res[:, :N_fuel] += qr[:, None]

        res_norm_denom = (
            (np.sum(qr) * self.Delta_Z / self.cfg.geometry.l)
            / (np.pi * self.cfg.geometry.r**2)
        )

        return res.reshape(-1) / res_norm_denom

    def post_process(self, X):
        return self.unpack(X)

    def unpack(self, X):
        T_HP = X.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)
        return (T_HP, )
    
    def pack(self, X_tuple):
        X = X_tuple[0].reshape(self.cfg.mesh.N_Z * self.cfg.mesh.N_R)
        return X

    def linear_solve(self):
        self.calculate_qr()
        self.assemble()

        T = np.linalg.solve(self.M, self.C)

        self.T = T

    def initialize_discretization(self):     
        R         = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt(self.cfg.geometry.r**2 / (self.cfg.mesh.N_R * 2))

        for i in range(1, 2 * self.cfg.mesh.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0]
        for i in range(1, 2 * self.cfg.mesh.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_Rm = delta_R[0::2]
        delta_Rp = delta_R[1::2]

        surface_tensor = self.calculate_surfaces(delta_Rp, delta_Rm)

        self.R = R
        self.delta_Rm = delta_Rm
        self.delta_Rp = delta_Rp
        self.Delta_V = np.pi * (delta_Rp[0] + delta_Rm[0])**2 * self.Delta_Z 
        self.surface_tensor = surface_tensor


    def calculate_surfaces(self, delta_Rp, delta_Rm):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2 * np.pi
        S_rm = Rm * 2 * np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.cfg.mesh.N_Z, axis=0)
        surface_tensor[..., 0:2] *= self.Delta_Z

        return surface_tensor
    
    
    def generate_k_matrix(self, T=None):
        if T is None:
            T = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        k_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        def eval_material_prop(prop, T_slice, scale=1.0):
            value = prop(T_slice) if callable(prop) else prop
            return value * scale     # type: ignore

        fuel_slice = slice(0, self.cfg.mesh.N_fuel)
        gap_slice  = slice(self.cfg.mesh.N_fuel, self.cfg.mesh.N_fuel + self.cfg.mesh.N_gap)
        clad_slice = slice(self.cfg.mesh.N_fuel + self.cfg.mesh.N_gap, self.cfg.mesh.N_R)

        T_fuel = T[:, fuel_slice]
        T_gap  = T[:, gap_slice]
        T_clad = T[:, clad_slice]

        k_matrix[:, fuel_slice] = eval_material_prop(self.cfg.material.k_fuel, T_fuel)
        k_matrix[:, gap_slice]  = eval_material_prop(self.cfg.material.h_gap, T_gap, scale=self.cfg.geometry.delta_gap)
        k_matrix[:, clad_slice] = eval_material_prop(self.cfg.material.k_clad, T_clad)

        return k_matrix


    def generate_h_matrix(self, T=800.):
        h_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))

        h_matrix[:, -1] = self.cfg.material.h_mod

        return h_matrix


    def calculate_alpha(self, k_matrix):
        alpha = np.zeros_like(self.surface_tensor)

        alpha[:, :-1, 0] = (
            self.surface_tensor[:, :-1, 0] * k_matrix[:, 1:]
            / (k_matrix[:, :-1] * self.delta_Rm[1:] + k_matrix[:, 1:] * self.delta_Rp[:-1])
        )

        alpha[:, 1:, 1] = (
            self.surface_tensor[:, 1:, 1] * k_matrix[:, :-1]
            / (k_matrix[:, 1:] * self.delta_Rp[:-1] + k_matrix[:, :-1] * self.delta_Rm[1:])
        )

        alpha[:-1, :, 2] = (
            self.surface_tensor[:-1, :, 2] * k_matrix[1:, :]
            / (k_matrix[:-1, :] * self.Delta_Z + k_matrix[1:, :] * self.Delta_Z)
        )

        alpha[1:, :, 3] = (
            self.surface_tensor[1:, :, 3] * k_matrix[:-1, :]
            / (k_matrix[1:, :] * self.Delta_Z + k_matrix[:-1, :] * self.Delta_Z)
        )
        # Init mask
        cooling_mask = np.zeros_like(alpha, dtype=bool)
                                     
        # Configure masks
        cooling_mask[:, -1, 0] = True

        # Apply masks
        alpha[cooling_mask] = self.surface_tensor[cooling_mask]

        return alpha
    
    
    def generate_matrix_form(self, alpha, qr, T_mod, k, h):
        N = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        stride = self.cfg.mesh.N_R 

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.cfg.mesh.N_Z - 1):
            for r in range(1, self.cfg.mesh.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx][T_idx] = -k[z][r] * (
                    alpha[z][r][0] +
                    alpha[z][r][1] +
                    alpha[z][r][2] +
                    alpha[z][r][3]
                )

                M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
                M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
                M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  
                M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3]

        # -----------------------
        # Insulated wall at z = 0, no corners.
        z = 0
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride]  = k[z][r] * alpha[z][r][2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1, no corners.
        z = self.cfg.mesh.N_Z - 1
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx][T_idx] = -k[z][r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]       = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx - 1]       = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx - stride]  = k[z][r] * alpha[z][r][3]  

        # -----------------------
        # Cladding BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][1] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )
            M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Fuel inner BC elements, no corners.
        r = 0
        for z in range(1, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (
                alpha[z][r][0] + 
                alpha[z][r][2] + 
                alpha[z][r][3]
            )

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Lowermost cladding outer corner (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Uppermost cladding outer corner (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0] 

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Lowermost fuel inner corner (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        # -----------------------
        # Uppermost fuel inner corner (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Fuel heat production
        for z in range(0, self.cfg.mesh.N_Z):
            for r in range(0, self.cfg.mesh.N_fuel):
                T_idx = (stride * z) + r
                
                C[T_idx] += -qr[z]
        
        return M, C

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.solver import Solver

    N_Z = 100
    N_R = 30
    data = {
        "geometry": {
            "delta_gap": 2.5e-3,     # random
            "delta_wall": 2.5e-3,
            "r": 1e-2,
            "l": 2.,
        },

        "mesh": {
            "N_R": N_R,
            "N_Z": N_Z,
        },

        "energy": {
            "N_G": 2,
        },

        "material": {
            "k_fuel": 15.0,
            "k_clad": 16.5,
            "h_gap": 1e5,
            "h_mod": 1e4,
        },
    }

    phi_ng  = np.tile(np.array([4.16e18, 5.47e17], dtype=float), (N_Z, 1))
    Sigma_f = np.array([9.4e-2, 5.48e1], dtype=float)
    kappa   = 3.204e-11
    T_mod   = np.ones(N_Z) * 1000.0

    geom   = FuelPinGeometry(**data["geometry"])
    mesh   = FuelPinMesh(**data["mesh"])
    energy = FuelPinEnergy(**data["energy"])
    mat    = FuelPinMaterial(**data["material"])
    cfg    = FuelPinConfig(geom, mesh, energy, mat)
    cfg = cfg.resolve()

    fuel_pin = FuelPin(cfg)
    solver = Solver([fuel_pin])
    solver.newton_krylov()

    T, = solver.solution

    fuel_pin.linear_solve()

    plt.plot(fuel_pin.R[::2], T[0], label="non-linear")
    plt.plot(fuel_pin.R[::2], fuel_pin.T.reshape(N_Z, N_R)[0], ls="--", label="linear")
    
    plt.legend()
    plt.show()