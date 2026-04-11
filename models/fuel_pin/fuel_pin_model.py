import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from project_data.neutronics_dataclasses import *

class FuelPin:
    def __init__(self, config: FuelPinConfigResolved):

        self.cfg = config

        self.phi_g   = np.tile(np.array([4.16e18, 5.47e17], dtype=float), (self.cfg.mesh.N_Z, 1))
        self.Sigma_f = np.array([9.4e-2, 5.48e1], dtype=float)
        self.kappa   = 3.204e-11
        self.T_mod   = np.ones(self.cfg.mesh.N_Z) * 1000.0
        self.qr      = -1

        self.Delta_Z = self.cfg.geometry.l / self.cfg.mesh.N_Z

    def assemble(self):
        self.initialize_discretization()

        surface_tensor = self.calculate_surfaces()

        self.generate_k_matrix()
        self.generate_h_matrix()

        alpha = self.calculate_alpha(surface_tensor)

        self.M, self.C = self.generate_matrix_form(alpha, self.qr, self.T_mod)

    def calculate_qr(self):
        self.initialize_discretization()
        self.qr = np.sum(self.phi_g * self.Sigma_f * self.kappa * self.Delta_V, axis = 1)

    def solve(self):
        self.calculate_qr()
        self.assemble()

        T = np.linalg.solve(self.M, self.C)

        self.T = T

    def get_residuals(self, T_FP):
        self.assemble()
    
        res = self.M @ T_FP - self.C

        res_norm_denom = (np.sum(self.qr) * self.Delta_Z / self.cfg.geometry.l) / (np.pi * self.cfg.geometry.r**2)

        return res / res_norm_denom

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

        self.R = R

        self.delta_Rm = delta_Rm
        self.delta_Rp = delta_Rp

        self.Delta_V = np.pi * (delta_Rp[0] + delta_Rm[0])**2 * self.Delta_Z 


    def calculate_surfaces(self):
        delta_R = self.delta_Rp + self.delta_Rm
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

        self.k_matrix = k_matrix


    def generate_h_matrix(self, T=800.):
        h_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))

        h_matrix[:, -1] = self.cfg.material.h_mod

        self.h_matrix = h_matrix


    def calculate_alpha(self, surface_tensor):
        alpha_tensor = np.zeros_like(surface_tensor)

        for i in range(alpha_tensor.shape[0]):
            for j in range(alpha_tensor.shape[1]):
                if not (j == alpha_tensor.shape[1] - 1):
                    alpha_tensor[i, j, 0] = (surface_tensor[i, j, 0] * self.k_matrix[i, j+1]) / (self.k_matrix[i, j]*self.delta_Rm[j+1] + self.k_matrix[i, j+1]*self.delta_Rp[j])
                if not (j == 0):
                    alpha_tensor[i, j, 1] = (surface_tensor[i, j, 1] * self.k_matrix[i, j-1]) / (self.k_matrix[i, j]*self.delta_Rp[j-1] + self.k_matrix[i, j-1]*self.delta_Rm[j])
                if not (i == alpha_tensor.shape[0] - 1):
                    alpha_tensor[i, j, 2] = (surface_tensor[i, j, 2] * self.k_matrix[i+1, j]) / (self.k_matrix[i, j]*self.Delta_Z + self.k_matrix[i+1, j]*self.Delta_Z)
                if not (i == 0):
                    alpha_tensor[i, j, 3] = (surface_tensor[i, j, 3] * self.k_matrix[i-1, j]) / (self.k_matrix[i, j]*self.Delta_Z + self.k_matrix[i-1, j]*self.Delta_Z)

        # Init mask
        cooling_mask = np.zeros_like(alpha_tensor, dtype=bool)
                                     
        # Configure masks
        cooling_mask[:, -1, 0] = True

        # Apply masks
        alpha_tensor[cooling_mask] = surface_tensor[cooling_mask]

        return alpha_tensor
    
    
    def generate_matrix_form(self, alpha, qr, T_mod):
        N = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        stride = self.cfg.mesh.N_R 

        k = self.k_matrix
        h = self.h_matrix

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
    N_Z = 10
    N_R = 20
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

    fuelPin_conduction = FuelPin(cfg)
    fuelPin_conduction.solve()

    sol, info, ier, mesg = fsolve(fuelPin_conduction.get_residuals, fuelPin_conduction.T, full_output=True)

    plt.plot(fuelPin_conduction.R[::2], sol.reshape(N_Z, N_R)[0], label="non-linear")
    plt.plot(fuelPin_conduction.R[::2], fuelPin_conduction.T.reshape(N_Z, N_R)[0], ls="--", label="linear")
    
    plt.legend()
    plt.show()