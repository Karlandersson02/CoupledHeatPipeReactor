import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

class FuelPin:
    def __init__(self, data):

        # Geometry
        self.N_R = data.get("N_R")
        self.N_Z = data.get("N_Z")
        self.r = data.get("r_FP")
        self.l = data.get("l_FP")

        self.delta_gap = data.get("delta_gap")

        self.N_clad = data.get("N_clad")
        self.N_gap = data.get("N_gap")
        self.N_fuel = self.N_R - self.N_clad - self.N_gap

        self.Delta_Z = self.l / self.N_Z 

        # Neutron flux properties
        self.N_G = data.get("N_G")
        self.phi_g = data.get("phi_g")

        # Neutron material properties
        self.Sigma_f = data.get("Sigma_f")
        self.kappa = data.get("kappa")

        # Material properties
        self.k_fuel = data.get("k_fuel")
        self.k_clad = data.get("k_clad")
        self.h_gap = data.get("h_gap")
        self.k_gap = self.h_gap * self.delta_gap
        self.h_moderator = data.get("h_moderator")

        self.T_moderator = data.get("T_moderator")
        self.T_HP_c = 300.


    def solve(self):
        self.initialize_discretization()

        surface_tensor = self.calculate_surfaces()

        self.generate_k_matrix()
        self.generate_h_matrix()

        alpha = self.calculate_alpha(surface_tensor)

        qr = np.sum(self.phi_g * self.Sigma_f * self.kappa * np.pi * self.Delta_V)
        M, C = self.generate_matrix_form(alpha, qr, self.T_moderator)
    
        T = np.linalg.solve(M, C)

        self.T = T

    def get_residuals(self, T_FP, qr, T_mod):
        self.initialize_discretization()

        surface_tensor = self.calculate_surfaces()

        self.generate_k_matrix(T_FP)
        self.generate_h_matrix(T_FP)

        alpha = self.calculate_alpha(surface_tensor)
        
        M, C = self.generate_matrix_form(alpha, qr, T_mod)
    
        res = M @ T_FP - C 

        res_norm_denom = (np.sum(qr) * self.Delta_Z / self.l) / (np.pi * self.r **2)

        return res / res_norm_denom

    def initialize_discretization(self):     
        R         = np.zeros(2 * self.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.N_R, dtype=float)
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt(self.r**2 / (self.N_R * 2))

        for i in range(1, 2 * self.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0]
        for i in range(1, 2 * self.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_Rm = delta_R[0::2]
        delta_Rp = delta_R[1::2]

        self.delta_Rm = delta_Rm
        self.delta_Rp = delta_Rp

        self.Delta_V = np.pi * (delta_Rp[0] + delta_Rm[0])**2 * self.Delta_Z 


    def calculate_surfaces(self):
        delta_R = self.delta_Rp + self.delta_Rm
        Rp = np.cumsum(delta_R)
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2 * self.Delta_Z

        return surface_tensor
    
    
    def generate_k_matrix(self, T=800.):
        k_matrix = np.zeros((self.N_Z, self.N_R))

        k_matrix[:, :self.N_fuel]                           = self.k_fuel
        k_matrix[:, self.N_fuel:(self.N_fuel + self.N_gap)] = self.k_gap
        k_matrix[:, (self.N_fuel + self.N_gap):]            = self.k_clad

        self.k_matrix = k_matrix


    def generate_h_matrix(self, T=800.):
        h_matrix = np.zeros((self.N_Z, self.N_R))

        h_matrix[:, -1] = self.h_moderator

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
        N = self.N_R * self.N_Z
        stride = self.N_R 

        k = self.k_matrix
        h = self.h_matrix

        M = np.zeros((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.N_Z - 1):
            for r in range(1, self.N_R - 1):
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
        for r in range(1, self.N_R - 1):
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
        z = self.N_Z - 1
        for r in range(1, self.N_R - 1):
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
        r = self.N_R - 1
        for z in range(1, self.N_Z-1): #
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
        for z in range(1, self.N_Z-1):
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
        r = self.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])
        M[T_idx][T_idx] += -h[z][r] * alpha[z][r][0]

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -h[z][r] * alpha[z][r][0] * T_mod[z]

        # -----------------------
        # Uppermost cladding outer corner (z = N_Z - 1, r = N_R - 1).
        z = self.N_Z - 1
        r = self.N_R - 1
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
        z = self.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

        # -----------------------
        # Fuel heat production
        for z in range(0, self.N_Z):
            for r in range(0, self.N_fuel):
                T_idx = (stride * z) + r
                
                C[T_idx] += -qr[z]
        
        return M, C

if __name__ == "__main__":
    N_Z = 30
    N_R = 20
    data = {
        # ---------------------------
        # Geometry / mesh
        # ---------------------------
        "N_R": N_R,                 # radial cells (numerical choice)
        "N_Z": N_Z,                 # axial cells (numerical choice)

        # Treat r_FP as OUTER fuel-pin radius, since your model has fuel + gap + clad
        "r_FP": 1e-2,           # [m] = 1 cm outer radius
        "l_FP": 2.,             # [m] active fuel length

        "delta_gap": 1e-3,

        "N_gap": 1,                # radial cells assigned to gas gap
        "N_clad": 5,               # radial cells assigned to cladding

        # ---------------------------
        # Neutronics
        # ---------------------------
        "N_G": 2,                  # 2-group model: [fast, thermal]

        # Approximate 2-group flux [n/m^2/s]
        # fast group = collapsed from non-thermal groups
        # thermal group = lowest-energy group
        "phi_g": np.tile(
            np.array([4.16e18, 5.47e17], dtype=float),
            (N_Z, 1)
        ),

        # Approximate macroscopic fission cross section [1/m]
        "Sigma_f": np.array([
            9.4e-2,                # fast-group placeholder
            5.48e1                 # thermal-group estimate
        ], dtype=float),

        # Recoverable energy per fission
        "kappa": 3.204e-11,        # [J/fission]

        # ---------------------------
        # Thermal material properties
        # ---------------------------
        "k_fuel": 15.0,             # [W/m-K] simple UO2 operating-value placeholder
        "k_clad": 16.5,            # [W/m-K] Zircaloy near ~600 K
        "h_gap": 1e5,            # [W/m^2-K] reasonable mid-range gap conductance
        "h_moderator": 1e4,
        # ---------------------------
        # Coolant / moderator
        # ---------------------------
        "T_moderator": np.ones(N_Z) * 1000.0       # [K]
    }

    fuelPin_conduction = FuelPin(data)
    fuelPin_conduction.solve()

    sol, info, ier, mesg = fsolve(fuelPin_conduction.get_residuals, fuelPin_conduction.T, full_output=True)

    plt.plot(300. * sol.reshape(N_Z, N_R)[0], label="non-linear")
    plt.plot(fuelPin_conduction.T.reshape(N_Z, N_R)[0], ls="--", label="linear")
    
    plt.legend()
    plt.show()