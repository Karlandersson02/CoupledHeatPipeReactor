import numpy as np

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from data.dataclass import *
from models.component import Component

class HeatpipeDiscretisedVapour(Component):
    def __init__(self, config: HeatpipeConfigResolved):

        self.cfg = config

        self._initialize_discretization()

    def set_T_v(self, T_v):
        self.T_v = T_v

    def initial_guess(self):
        X_initial = np.full(self.cfg.mesh.N_Z * self.cfg.mesh.N_R, 800)
        return X_initial

    def assemble(self):
        surface_areas, delta_Rp, delta_Rm, delta_Z = self._initialize_discretization()

        self.k_matrix = self._generate_k_matrix()
        self.h_matrix = self._generate_h_matrix()

        self.alpha = self._generate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, self.k_matrix)

    def get_residuals(self, X):
        T, = self.unpack(X)
        T = T.reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)
        T_v = self.T_v

        surface_areas, delta_Rp, delta_Rm, delta_Z = self._initialize_discretization()
        k = self._generate_k_matrix(T)
        h = self._generate_h_matrix()
        alpha = self._generate_alpha(surface_areas, delta_Rm, delta_Rp, delta_Z, k)

        N_Z = self.cfg.mesh.N_Z
        N_R = self.cfg.mesh.N_R
        N_evap = self.cfg.mesh.N_evap
        N_cond = self.cfg.mesh.N_cond

        res = np.zeros((N_Z, N_R), dtype=float)

        # Bulk
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

        # Top insulated boundary, z = 0, excluding corners
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

        # Bottom insulated boundary, z = N_Z - 1, excluding corners
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

        # Outer wall boundary, r = N_R - 1, excluding corners
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
                + hc * ac[:, 0] * self.cfg.bc.T_cond
            )

            if N_evap > 1:
                res[1:N_evap, r] += self.cfg.bc.Q[1:N_evap]

        # Inner wick-vapour boundary, r = 0, excluding corners
        if N_Z > 2:
            r = 0
            kc = k[1:-1, r]
            ac = alpha[1:-1, r]
            hc = h[1:-1, r]

            res[1:-1, r] = (
                (-kc * (ac[:, 0] + ac[:, 2] + ac[:, 3]) - hc * ac[:, 1]) * T[1:-1, r]
                + kc * ac[:, 0] * T[1:-1, r + 1]
                + hc * ac[:, 1] * T_v[1:-1]
                + kc * ac[:, 2] * T[2:, r]
                + kc * ac[:, 3] * T[:-2, r]
            )

        # Corner: z = 0, r = N_R - 1
        z = 0
        r = N_R - 1
        res[z, r] = (
            -k[z, r] * (alpha[z, r, 1] + alpha[z, r, 2]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
            + self.cfg.bc.Q[z]
        )

        # Corner: z = N_Z - 1, r = N_R - 1
        z = N_Z - 1
        r = N_R - 1
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 1] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 0]) * T[z, r]
            + k[z, r] * alpha[z, r, 1] * T[z, r - 1]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
            + h[z, r] * alpha[z, r, 0] * self.cfg.bc.T_cond
        )

        # Corner: z = 0, r = 0
        z = 0
        r = 0
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 0] + alpha[z, r, 2]) - h[z, r] * alpha[z, r, 1]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + h[z, r] * alpha[z, r, 1] * T_v[z]
            + k[z, r] * alpha[z, r, 2] * T[z + 1, r]
        )

        # Corner: z = N_Z - 1, r = 0
        z = N_Z - 1
        r = 0
        res[z, r] = (
            (-k[z, r] * (alpha[z, r, 0] + alpha[z, r, 3]) - h[z, r] * alpha[z, r, 1]) * T[z, r]
            + k[z, r] * alpha[z, r, 0] * T[z, r + 1]
            + h[z, r] * alpha[z, r, 1] * T_v[z]
            + k[z, r] * alpha[z, r, 3] * T[z - 1, r]
        )

        # # Vapour residual contribution at each axial cell
        # beta = h[:, 0] * alpha[:, 0, 1]
        # res_v = beta * (T[:, 0] - T_v)

        return res.reshape(-1)

    def post_process(self, X):
        return X[0].reshape(self.cfg.mesh.N_Z, self.cfg.mesh.N_R)
    
    def unpack(self, X):
        return (X, )
    
    def pack(self, X_tuple):
        return X_tuple[0]

    def _initialize_discretization(self):     
        R         = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R   = np.zeros(2 * self.cfg.mesh.N_R, dtype=float)
        delta_R_m = np.zeros(self.cfg.mesh.N_R, dtype=float)
        delta_R_p = np.zeros(self.cfg.mesh.N_R, dtype=float)
        Z         = np.zeros(2 * self.cfg.mesh.N_Z, dtype=float)
        delta_Z   = np.zeros(self.cfg.mesh.N_Z, dtype=float)

        self.Z = Z[0::2]
        self.R = R[0::2]
        self.delta_R = delta_R
        
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

        # Generate surface tensors
        surface_areas = self._generate_surfaces(delta_R_p, delta_R_m, delta_Z)

        return surface_areas, delta_R_p, delta_R_m, delta_Z

    def _generate_surfaces(self, delta_Rp, delta_Rm, delta_Z):
        delta_R = delta_Rp + delta_Rm
        Rp = np.cumsum(delta_R) + self.cfg.geometry.r_vapour
        Rm = Rp - delta_R

        S_rp = Rp * 2*np.pi
        S_rm = Rm * 2*np.pi
        S_z = (Rp**2 - Rm**2) * np.pi

        surface_tensor = np.concatenate([S_rp[:, None], S_rm[:, None], S_z[:, None], S_z[:, None]], axis=1)
        surface_tensor = np.repeat(surface_tensor[None], self.cfg.mesh.N_Z, axis=0)
        surface_tensor[..., 0:2] *= 2 * delta_Z[:, None, None] # This *2 should be there due to Deltz_Z is half the distance between cell centers!!!

        return surface_tensor

    def _generate_k_matrix(self, T=None):
        if T is None:
            T = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        k_matrix = np.zeros((self.cfg.mesh.N_Z, self.cfg.mesh.N_R), dtype=float)

        def eval_material_prop(prop, T_slice, scale=1.0):
            value = prop(T_slice) if callable(prop) else prop
            return value * scale    # type: ignore

        wick_slice = slice(0, self.cfg.mesh.N_wick)
        gap_slice  = slice(self.cfg.mesh.N_wick, self.cfg.mesh.N_wick + self.cfg.mesh.N_gap)
        wall_slice = slice(self.cfg.mesh.N_R - self.cfg.mesh.N_wall, self.cfg.mesh.N_R)

        T_wick = T[:, wick_slice]
        T_gap  = T[:, gap_slice]
        T_wall = T[:, wall_slice]

        k_matrix[:, wick_slice] = eval_material_prop(self.cfg.material.k_wick, T_wick)
        k_matrix[:, gap_slice]  = eval_material_prop(self.cfg.material.k_gap, T_gap)
        k_matrix[:, wall_slice] = eval_material_prop(self.cfg.material.k_wall, T_wall)

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

    def linear_solve(self):
        self.assemble()
        
        self.M, self.C = self._generate_matrix_form_temperature_bc(self.alpha, self.k_matrix, self.h_matrix)
        
        T = np.array(spsolve(self.M, self.C))

        if not self.cfg.bc.Temperature_BC:
            T = np.concatenate([T, np.array([self.cfg.bc.T_op])])

        return T
    
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
        for z in range(1, self.cfg.mesh.N_Z-1):
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
        for z in range(1, self.cfg.mesh.N_Z - 1):
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import json
    from typing import Sequence
    from utils.solver import Solver

    with open("./data/vapour_data.json", "r") as f:
        data_guoju = json.load(f)
        data = data_guoju["data_guoju_560"]
    
    geom = HeatpipeGeometry(**data["geometry"])
    geom.delta_wick = 0.00025
    geom.delta_gap = 0.00025
    mesh = HeatpipeMesh(**data["mesh"])
    mesh.N_R = 30
    mesh.N_Z = 30
    mat = HeatpipeMaterial(**data["material"])
    wick = HeatpipeWick(**data["wick"])
    bc = HeatpipeBC(**data["bc"])
    cfg = HeatpipeConfig(geom, mesh, mat, wick, bc)
    cfgs: Sequence[HeatpipeConfigResolved] = make_mesh_sequence(cfg, 1)
    
    heatpipes = [HeatpipeDiscretised(cfg) for cfg in cfgs]
    for heatpipe in heatpipes:
        print(f"N_Z = {heatpipe.cfg.mesh.N_Z}, N_R = {heatpipe.cfg.mesh.N_R}")
    solver = Solver(heatpipes)
    solver.newton_krylov()

    T_solid, T_vap = solver.solution

    N_Z = cfgs[-1].mesh.N_Z
    N_R = cfgs[-1].mesh.N_R

    heatpipe = heatpipes[-1]
    T_linear = heatpipe.linear_solve()

    plt.rcParams["font.size"] = 22
    plt.rcParams["font.family"] = "Computer modern"
    plt.rcParams["text.usetex"] = True

    fig = plt.figure(figsize = (16, 9))

    ax = fig.add_subplot(111)
    r_centers = heatpipe.R[0::2]

    delta_R = heatpipe.delta_R[0::2] + heatpipe.delta_R[1::2]
    r_faces = np.r_[heatpipe.cfg.geometry.r_vapour, heatpipe.cfg.geometry.r_vapour + np.cumsum(delta_R)]

    r_wick_disc = r_faces[heatpipe.cfg.mesh.N_wick]
    r_gap_disc  = r_faces[heatpipe.cfg.mesh.N_wick + heatpipe.cfg.mesh.N_gap]

    ax.plot(r_centers, T_solid[0], lw=4, label="non-linear")
    ax.plot(r_centers, T_linear[:-1].reshape(N_Z, N_R)[0], ls="--", lw=4, label="linear")
    ax.vlines([r_wick_disc, r_gap_disc], np.min(T_solid[0]), np.max(T_solid[0]), colors="black")
    
    plt.legend()
    plt.show()