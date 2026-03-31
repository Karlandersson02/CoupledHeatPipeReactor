import numpy as np
import matplotlib.pyplot as plt
import json

from scipy.optimize import newton_krylov

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from project_data.heatpipe_dataclasses import *

class HeatpipeDiscretised:
    def __init__(self, config: HeatpipeConfigResolved):

        self.cfg = config

    def assemble(self):
        delta_Rp, delta_Rm, delta_Z = self._initialize_discretization()

        surface_areas = self._generate_surfaces(delta_Rp, delta_Rm, delta_Z)

        k_matrix = self._generate_k_matrix()
        h_matrix = self._generate_h_matrix()

        alpha = self._generate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k_matrix)        
        
        if self.cfg.bc.Temperature_BC:
            M, C = self._generate_matrix_form_temperature_bc(alpha, k_matrix, h_matrix)
        else:
            M, C = self._generate_matrix_form_heat_bc(alpha, k_matrix, h_matrix)

        return M, C

    def linear_solve(self):
        M, C = self.assemble()

        T = np.array(spsolve(M, C))

        if not self.cfg.bc.Temperature_BC:
            T = np.concatenate([T, np.array([self.cfg.bc.T_op])])

        return T

    def get_residuals(self, T, M = None, C = None):
        if M is None or C is None:
            M, C = self.assemble()

        res = M @ T - C
        return res

    def _initialize_discretization(self):     
        R         = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R_m = np.zeros(self.cfg.mesh.N_R, dtype=float)
        delta_R_p = np.zeros(self.cfg.mesh.N_R, dtype=float)
        Z         = np.zeros(2 * self.cfg.mesh.N_Z, dtype=float)
        delta_Z   = np.zeros(self.cfg.mesh.N_Z, dtype=float)
        
        # Calculating the radii of the half-elements
        R[0] = np.sqrt((self.cfg.geometry.r_outer**2 - self.cfg.geometry.r_vapour**2) / (self.cfg.mesh.N_R * 2) + self.cfg.geometry.r_vapour**2)

        for i in range(1, 2 * self.cfg.mesh.N_R):
            R[i] = np.sqrt(R[i-1]**2 + R[0]**2 - self.cfg.geometry.r_vapour**2)

        # Calculating the differences in the radius of the half-elements
        delta_R[0] = R[0] - self.cfg.geometry.r_vapour
        for i in range(1, 2 * self.cfg.mesh.N_R):
            delta_R[i] = R[i] - R[i - 1]

        delta_R_m = delta_R[0::2]
        delta_R_p = delta_R[1::2]

        # Calculating the Z-position of the bulk and edges of the elements 
        for i in range(self.cfg.mesh.N_evap*2):
            Z[i] = (i/2 + 1/2) * self.cfg.geometry.l_evap/self.cfg.mesh.N_evap

        for i in range(self.cfg.mesh.N_adiabatic*2):
            Z[i + self.cfg.mesh.N_evap*2] = (i/2 + 1/2) * self.cfg.geometry.l_adiabatic/self.cfg.mesh.N_adiabatic + self.cfg.geometry.l_evap

        for i in range(self.cfg.mesh.N_cond*2):
            Z[i+ self.cfg.mesh.N_evap*2 + self.cfg.mesh.N_adiabatic*2] = (i/2 + 1/2) * self.cfg.geometry.l_cond/self.cfg.mesh.N_cond + (self.cfg.geometry.l_evap + self.cfg.geometry.l_adiabatic)

        for i in range(self.cfg.mesh.N_Z):
            delta_Z[i] = Z[2*i + 1] - Z[2*i]

        return delta_R_p, delta_R_m, delta_Z

    def _generate_surfaces(self, delta_Rp, delta_Rm, delta_Z):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R) + self.cfg.geometry.r_vapour
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.cfg.mesh.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2*delta_Z[:, None, None]

        return surface_tensor

    def _generate_k_matrix(self):
        k_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))
        k_matrix[:, :self.cfg.mesh.N_wick] = self.cfg.material.k_wick
        k_matrix[:, self.cfg.mesh.N_wick:] = self.cfg.material.k_wall

        return k_matrix
    
    def _generate_h_matrix(self) -> np.ndarray:

        h_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R))
        
        # Heat transfer coefficient for the vapor section.
        h_matrix[:self.cfg.mesh.N_evap, 0]                      = self.cfg.material.h_vap
        h_matrix[(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond):self.cfg.mesh.N_Z, 0] = self.cfg.material.h_vap

        # Heat transfer coefficient for the condensator section.
        h_matrix[(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond):self.cfg.mesh.N_Z, self.cfg.mesh.N_R - 1] = self.cfg.material.h_cond

        return h_matrix

    def _generate_alpha(self, surface_tensor, delta_Rm, delta_Rp, delta_Z, k_matrix):
        alpha = np.zeros_like(surface_tensor)

        alpha[:, :-1, 0] = (
            surface_tensor[:, :-1, 0] * k_matrix[:, 1:]
            / (k_matrix[:, :-1] * delta_Rm[1:] + k_matrix[:, 1:] * delta_Rp[:-1])
        )

        alpha[:, 1:, 1] = (
            surface_tensor[:, 1:, 1] * k_matrix[:, :-1]
            / (k_matrix[:, 1:] * delta_Rp[:-1] + k_matrix[:, :-1] * delta_Rm[1:])
        )

        alpha[:-1, :, 2] = (
            surface_tensor[:-1, :, 2] * k_matrix[1:, :]
            / (k_matrix[:-1, :] * delta_Z[1:, None] + k_matrix[1:, :] * delta_Z[:-1, None])
        )

        alpha[1:, :, 3] = (
            surface_tensor[1:, :, 3] * k_matrix[:-1, :]
            / (k_matrix[1:, :] * delta_Z[:-1, None] + k_matrix[:-1, :] * delta_Z[1:, None])
        )

        # Init masks
        vapour_mask = np.zeros_like(alpha, dtype=bool)
        cooling_mask = np.zeros_like(alpha, dtype=bool)
        
        # Configure masks
        vapour_mask[0:self.cfg.mesh.N_evap, 0, 1] = True
        vapour_mask[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):, 0, 1] = True
        cooling_mask[(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic):, -1, 0] = True

        # Apply masks
        alpha[vapour_mask] = surface_tensor[vapour_mask]
        alpha[cooling_mask] = surface_tensor[cooling_mask]

        # Adiabatic middle boundaries
        adiabatic_mask = np.zeros_like(alpha, dtype=bool)
        adiabatic_mask[self.cfg.mesh.N_evap:(self.cfg.mesh.N_evap + self.cfg.mesh.N_adiabatic), :, 0:2] = True
        alpha[adiabatic_mask] = 0
        return alpha

    def _generate_matrix_form_temperature_bc(self, alpha, k, h):
        # Number of physical grid nodes + extra vapor node (stored at index -1).
        N_phys = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        N = N_phys + 1
        stride = self.cfg.mesh.N_R 

        M = lil_matrix((N, N), dtype=float)
        C = np.zeros(N, dtype=float)

        # -----------------------
        # Bulk elements.
        for z in range(1, self.cfg.mesh.N_Z - 1):
            for r in range(1, self.cfg.mesh.N_R - 1):
                T_idx = (stride * z) + r

                M[T_idx, T_idx] = -k[z, r] * (
                    alpha[z, r, 0] +
                    alpha[z, r, 1] +
                    alpha[z, r, 2] +
                    alpha[z, r, 3]
                )

                M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
                M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
                M[T_idx, T_idx + stride]  = k[z, r] * alpha[z, r, 2]  
                M[T_idx, T_idx - stride]  = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Insulated wall at z = 0
        z = 0
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx, T_idx] = -k[z, r] * (
                alpha[z][r][0] +
                alpha[z][r][1] +
                alpha[z][r][2]
            )

            M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
            M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride]  = k[z, r] * alpha[z, r, 2]  

        # -----------------------
        # Insulated wall at z = N_Z - 1
        z = self.cfg.mesh.N_Z - 1
        for r in range(1, self.cfg.mesh.N_R - 1):
            T_idx = (stride * z) + r

            M[T_idx, T_idx] = -k[z, r] * (
                alpha[z, r, 0] +
                alpha[z, r, 1] +
                alpha[z, r, 3]
            )

            M[T_idx, T_idx + 1]       = k[z, r] * alpha[z, r, 0]
            M[T_idx, T_idx - 1]       = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx - stride]  = k[z, r] * alpha[z, r, 3]  


        # -----------------------
        # Wall BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1): #
            T_idx = z * stride + r 

            M[T_idx, T_idx] = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2] + alpha[z, r, 3])
            M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 0]

            M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 
            M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

            if z < self.cfg.mesh.N_evap:
                C[T_idx] = -self.cfg.bc.Q[z]
            else:
                C[T_idx] = -h[z, r] * alpha[z, r, 0] * self.cfg.bc.T_cond

        # -----------------------
        # Wick BC elements against vapor, no corners.
        r = 0
        for z in range(0, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2] + alpha[z, r, 3])
            M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1]

            M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
            M[T_idx, -1]             = h[z, r] * alpha[z, r, 1]
            M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 
            M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx, T_idx] = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2])

        M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2] 

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3])
        M[T_idx, T_idx] += -h[z, r] * alpha[z, r, 0] 

        M[T_idx, T_idx - 1]      = k[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3]  

        C[T_idx] = -h[z][r] * alpha[z][r][0] * self.cfg.bc.T_cond 

        # -----------------------
        # Corner next to evaporator vapor inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2])
        M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1] 

        M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
        M[T_idx, -1]             = h[z][r] * alpha[z, r, 1]
        M[T_idx, T_idx + stride] = k[z, r] * alpha[z, r, 2]

        # -----------------------
        # Corner next to condenser vapor outlet/inlet (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx, T_idx]  = -k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3])
        M[T_idx, T_idx] -=  h[z, r] * alpha[z, r, 1]

        M[T_idx, T_idx + 1]      = k[z, r] * alpha[z, r, 0]
        M[T_idx, -1]             = h[z, r] * alpha[z, r, 1]
        M[T_idx, T_idx - stride] = k[z, r] * alpha[z, r, 3] 

        # -----------------------
        # Vapor elements. 

        for z in range(0, self.cfg.mesh.N_evap):
            r = 0
            T_idx = z * stride + r 

            M[-1, -1   ] -= h[z][r] * alpha[z, r, 1]
            M[-1, T_idx] += h[z][r] * alpha[z, r, 1]

        for z in range(self.cfg.mesh.N_Z - self.cfg.mesh.N_cond, self.cfg.mesh.N_Z):
            r = 0
            T_idx = z * stride + r 

            M[-1, -1   ] -= h[z, r] * alpha[z, r, 1]
            M[-1, T_idx] += h[z, r] * alpha[z, r, 1]

        # -----------------------
        # Return matrix and vector

        M = M.tocsr()

        return M, C

    def _generate_matrix_form_heat_bc(self, alpha: np.ndarray, k: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Number of physical grid nodes + extra vapour node (stored at index -1).
        N_phys = self.cfg.mesh.N_R * self.cfg.mesh.N_Z
        N = N_phys
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
        # Insulated wall at z = 0
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
        # Insulated wall at z = N_Z - 1
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
        # Wall BC elements, no corners.
        r = self.cfg.mesh.N_R - 1
        for z in range(1, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2] + alpha[z][r][3])

            M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Wick BC elements against vapour, no corners.
        r = 0
        for z in range(0, self.cfg.mesh.N_Z-1):
            T_idx = z * stride + r 

            M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2] + alpha[z][r][3])
            M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

            M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
            M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 
            M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3] 

            C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op
            

        # -----------------------
        # Corner next to evaporator entrance (z = 0, r = N_R-1).
        z = 0
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx] = -k[z][r] * (alpha[z][r][1] + alpha[z][r][2])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2] 

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to condenser outlet (z = N_Z - 1, r = N_R - 1).
        z = self.cfg.mesh.N_Z - 1
        r = self.cfg.mesh.N_R - 1
        T_idx = z * stride + r 

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][1] + alpha[z][r][3])

        M[T_idx][T_idx - 1]      = k[z][r] * alpha[z][r][1]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]  

        C[T_idx] = -self.cfg.bc.Q[z]

        # -----------------------
        # Corner next to evaporator vapour inlet (z = 0, r = 0).
        z = 0
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][2])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1] 

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx + stride] = k[z][r] * alpha[z][r][2]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op

        # -----------------------
        # Corner next to condenser vapour inlet (z = N_Z - 1, r = 0).
        z = self.cfg.mesh.N_Z - 1
        r = 0
        T_idx = z * stride + r

        M[T_idx][T_idx]  = -k[z][r] * (alpha[z][r][0] + alpha[z][r][3])
        M[T_idx][T_idx] -=  h[z][r] * alpha[z][r][1]

        M[T_idx][T_idx + 1]      = k[z][r] * alpha[z][r][0]
        M[T_idx][T_idx - stride] = k[z][r] * alpha[z][r][3]

        C[T_idx] = -h[z][r] * alpha[z][r][1] * self.cfg.bc.T_op

        return M, C

if __name__ == "__main__":
    with open("./project_data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]
    
    geom = HeatpipeGeometry(**data["geometry"])
    mesh = HeatpipeMesh(N_R=40, N_Z=100)
    mat = HeatpipeMaterial(**data["material"])
    bc = HeatpipeBC(**data["bc"])
    cfg = HeatpipeConfig(geom, mesh, mat, bc)
    cfg = cfg.resolve()

    heatpipe = HeatpipeDiscretised(cfg)

    T_initial = np.full((cfg.mesh.N_Z * cfg.mesh.N_R + 1, ), 1)
    M, C = heatpipe.assemble()

    sol = newton_krylov(
        lambda T: heatpipe.get_residuals(T, M, C), 
        T_initial,
        verbose=True,
        line_search='armijo',
        rdiff=1e-6
    )

    N_Z = cfg.mesh.N_Z
    N_R = cfg.mesh.N_R

    T_linear = heatpipe.linear_solve()

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize = (16, 9))

    ax = fig.add_subplot(111)
    ax.plot(sol[:-1].reshape(N_Z, N_R)[0], lw=4, label="non-linear")
    ax.plot(T_linear[:-1].reshape(N_Z, N_R)[0], ls="--", lw=4, label="linear")
    
    plt.legend()
    plt.show()