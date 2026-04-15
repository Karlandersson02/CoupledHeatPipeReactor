from unittest import skip

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import meshio # type: ignore
import pyvista as pv # type: ignore

from numba import njit # type: ignore

from scipy.optimize import fsolve

from models.moderator.triangle_mesh import UnstructuredMesh, Surface

from utils.material_variables import moderator_k

class ModeratorDiscretisedMeshFast:
    def __init__(self, data, mesh):
        self.mesh = mesh
        n_triangles = mesh.triangles.shape[0]

        self.Q_in = data.get("Q_in")

        self.HP_centers       = data.get("HP_centers")
        self.fuel_pin_centers = data.get("fuel_pin_centers")

        self.HP_radius       = data.get("HP_radius")
        self.fuel_pin_radius = data.get("fuel_pin_radius")

        self.T_HP = data.get("T_HP")
        self.T_FP = data.get("T_FP")

        self.HP_BC = data.get("HP_BC")
        self.FP_BC = data.get("FP_BC")

        self.cell_k = (
            np.ones(mesh.n_triangles, dtype=float)
            if data.get("cell_k") is None
            else np.asarray(data.get("cell_k"), dtype=float)
        )
        self.cell_T = (
            np.zeros(mesh.n_triangles, dtype=float)
            if data.get("cell_T") is None
            else np.asarray(data.get("cell_T"), dtype=float)
        )

        if self.cell_k.shape != (n_triangles,):
            raise ValueError("cell_k must have shape (n_triangles,)")
        if self.cell_T.shape != (n_triangles,):
            raise ValueError("cell_T must have shape (n_triangles,)")
        

    def solve_linearly(self, iterations: int):
        
        self.calculate_von_Neumann_boundary_conditions()
        self.calculate_Dirichlet_boundary_conditions()

        # Iterations to allow for cross diffusion correction to converge.
        for i in range(iterations):
            n_triangles = self.mesh.triangles.shape[0]

            M = np.zeros((n_triangles, n_triangles))
            C = np.zeros(n_triangles)

            self.calculate_cross_diffusion_correction()

            for triangle_idx in range(n_triangles):
                triangle_center = self.mesh._centers[triangle_idx]
                neighbours, surfaces = self.mesh.get_faces_and_neighbours(triangle_idx)

                # Setting the FluxF and FluxC terms, in addition to the boundary FluxV terms (BC). 
                for neighbour_idx, surface in zip(neighbours, surfaces):    
                    if neighbour_idx == -1: # Indicating a boundary triangle. 
                        FluxV_f_BC_vN = self.vN_boundary_conditions[triangle_idx]
                        FluxV_f_BC_Di = self.di_boundary_conditions[triangle_idx][1]

                        FluxC_f_BC_Di = self.di_boundary_conditions[triangle_idx][0]

                        M[triangle_idx][triangle_idx ] += FluxC_f_BC_Di

                        C[triangle_idx] += -FluxV_f_BC_vN -FluxV_f_BC_Di 
                    
                    else:
                        neighbour_center = self.mesh._centers[neighbour_idx]
                        surface_center = surface.center

                        d_CF = np.linalg.norm(triangle_center - neighbour_center)
                        d_Cs = np.linalg.norm(triangle_center - surface_center)
                        d_sF = np.linalg.norm(surface_center - neighbour_center)

                        # Utilizing orthogonal correction approach.
                        gDiff_f = surface.length / d_CF

                        g_f = d_Cs / (d_Cs + d_sF)
                        k_C = self.cell_k[triangle_idx]
                        k_F = self.cell_k[neighbour_idx]
                        
                        k_s = k_C * k_F / ( (1 - g_f) * k_C + g_f * k_F )

                        FluxF_f = -k_s * gDiff_f
                        FluxC_f =  k_s * gDiff_f

                        M[triangle_idx][neighbour_idx] =  FluxF_f
                        M[triangle_idx][triangle_idx ] += FluxC_f
                
                # Setting the FluxV terms that corresponds to the cross diffusion (CD) correction.
                FluxV_f_CD = self.cross_diffusion_correction[triangle_idx]
                C[triangle_idx] += -FluxV_f_CD

            T = np.linalg.solve(M, C)
            self.cell_T = T

        return T
    

    def assemble_mesh_data(
        self,
        mesh,
        HP_centers,
        fuel_pin_centers,
        HP_radius,
        fuel_pin_radius,
        epsilon=1e-2,
    ):
        """
        Assemble mesh/topology/geometry arrays for fast numerical kernels.

        Returns
        -------
        centers : ndarray, shape (n_triangles, 2)
        areas : ndarray, shape (n_triangles)
        neighbours : ndarray, shape (n_triangles, 3)
            Neighbour triangle index, or -1 for boundary face.
        edge_lengths : ndarray, shape (n_triangles, 3)
        edge_centers : ndarray, shape (n_triangles, 3, 2)
        surface_normals : ndarray, shape (n_triangles, 3, 2)
            Outward unit normals for each triangle face.
        is_boundary : ndarray, shape (n_triangles, 3)
            True where neighbour == -1.
        boundary_kind : ndarray, shape (n_triangles, 3)
            0 = internal
            1 = HP boundary
            2 = fuel pin boundary
            3 = other boundary
        """

        n_triangles = mesh.n_triangles
        n_edges = 3  # triangle mesh

        centers = np.zeros((n_triangles, 2), dtype=np.float64)
        areas = np.zeros(n_triangles, dtype=np.float64)
        neighbours = np.full((n_triangles, n_edges), -1, dtype=np.int64)
        edge_lengths = np.zeros((n_triangles, n_edges), dtype=np.float64)
        edge_centers = np.zeros((n_triangles, n_edges, 2), dtype=np.float64)
        edge_normals = np.zeros((n_triangles, n_edges, 2), dtype=np.float64)
        is_boundary = np.zeros((n_triangles, n_edges), dtype=np.bool_)
        boundary_kind = np.zeros((n_triangles, n_edges), dtype=np.int64)

        for triangle_idx in range(n_triangles):
            centers[triangle_idx] = mesh._centers[triangle_idx]
            areas[triangle_idx] = mesh.triangle_area(triangle_idx)
 
            tri_center = mesh._centers[triangle_idx]
            tri_edges = mesh._triangle_to_edges[triangle_idx]

            for edge_idx, edge in enumerate(tri_edges):
                p0 = mesh.points[edge[0]]
                p1 = mesh.points[edge[1]]

                edge_center = 0.5 * (p0 + p1)
                edge_vec = p1 - p0
                edge_length = np.linalg.norm(edge_vec)

                edge_centers[triangle_idx, edge_idx] = edge_center
                edge_lengths[triangle_idx, edge_idx] = edge_length

                # --- outward unit normal ---
                n1 = np.array([edge_vec[1], -edge_vec[0]], dtype=np.float64)
                n2 = -n1

                to_edge = edge_center - tri_center
                n = n1 if np.dot(n1, to_edge) > 0.0 else n2

                norm_n = np.linalg.norm(n)
                if norm_n == 0.0:
                    raise ValueError(f"Degenerate edge found in triangle {triangle_idx}")

                edge_normals[triangle_idx, edge_idx] = n / norm_n

                # --- topology ---
                attached = self.mesh._edge_to_triangles[edge]
                if len(attached) == 2:
                    neighbour = attached[0] if attached[1] == triangle_idx else attached[1]
                    neighbours[triangle_idx, edge_idx] = neighbour
                    is_boundary[triangle_idx, edge_idx] = False
                    boundary_kind[triangle_idx, edge_idx] = 0
                elif len(attached) == 1:
                    neighbours[triangle_idx, edge_idx] = -1
                    is_boundary[triangle_idx, edge_idx] = True

                    if self.is_edge_on_circumference(edge, HP_centers, HP_radius):
                        boundary_kind[triangle_idx, edge_idx] = 1
                    elif self.is_edge_on_circumference(edge, fuel_pin_centers, fuel_pin_radius):
                        boundary_kind[triangle_idx, edge_idx] = 2
                    else:
                        boundary_kind[triangle_idx, edge_idx] = 3
                else:
                    raise ValueError(
                        f"Edge {edge} is attached to {len(attached)} triangles. "
                        f"Expected 1 or 2."
                    )

        return (
            centers,
            neighbours,
            edge_lengths,
            edge_centers,
            surface_normals,
            is_boundary,
            boundary_kind,
        )
    

    def solve_nonlinear(
        self,
        T0: np.ndarray | None = None,
        xtol: float = 1e-8,
        maxfev: int = 0,
        verbose: bool = True,
        use_linear_guess: bool = True,
    ):
        """
        Solve the nonlinear system get_residuals(T) = 0 using scipy.optimize.fsolve.

        Parameters
        ----------
        T0 : np.ndarray | None
            Initial guess for the temperature field.
        xtol : float
            Relative error tolerance between iterates for fsolve.
        maxfev : int
            Maximum number of residual evaluations. If 0, SciPy uses its default.
        verbose : bool
            Print convergence information.
        use_linear_guess : bool
            If True and T0 is None, use solve_linearly(iterations=1) as initial guess.

        Returns
        -------
        np.ndarray
            Converged temperature field.
        """
        n_triangles = self.mesh.triangles.shape[0]

        # Needed because get_residuals() uses self.vN_boundary_conditions
        self.calculate_von_Neumann_boundary_conditions()

        # Initial guess
        if T0 is None:
            if use_linear_guess:
                try:
                    T0 = self.solve_linearly(iterations=1).copy()
                except Exception:
                    T0 = self.cell_T.copy()
            else:
                T0 = self.cell_T.copy()

        T0 = np.asarray(T0, dtype=float).reshape(-1)

        if T0.shape != (n_triangles,):
            raise ValueError(f"T0 must have shape ({n_triangles},), got {T0.shape}")

        call_count = {"n": 0}

        def residual_wrapper(T):
            call_count["n"] += 1
            R = 1. #self.get_residuals(T)

            if verbose:
                res_norm = np.linalg.norm(R, ord=2)
                print(f"fsolve call {call_count['n']:3d}: ||R||_2 = {res_norm:.6e}")

            return R

        T_sol, infodict, ier, mesg = fsolve(
            residual_wrapper,
            T0,
            xtol=xtol,
            full_output=True,
        )

        # Store final consistent state
        self.cell_T = np.asarray(T_sol, dtype=float).copy()
        self.cell_k = moderator_k(self.cell_T)
        self.calculate_Dirichlet_boundary_conditions()
        self.calculate_cross_diffusion_correction()

        # final_residual = self.get_residuals(self.cell_T)
        final_norm = 1. #np.linalg.norm(final_residual, ord=2)

        if verbose:
            print(f"fsolve ier   = {ier}")
            print(f"fsolve mesg  = {mesg}")
            print(f"Final ||R||_2 = {final_norm:.6e}")

        if ier != 1:
            raise RuntimeError(f"Nonlinear solve did not converge: {mesg}")

        return self.cell_T

    @staticmethod
    @njit(parallel=True)
    def get_residuals(
            T,
            cell_k,
            centers,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            is_boundary,
            boundary_kind,
            vN_bc,
            di_bc,
            cross_diffusion_corr
    ):
        
        # Calculate Dirichlet BC.

        n_triangles = centers.shape[0]

        res = np.zeros(n_triangles, dtype=float)

        for triangle_idx in range(n_triangles):
            triangle_center = centers[triangle_idx]
            T_C = T[triangle_idx]

            # iterating over all triangle sides
            for edge_idx in range(2):
                neighbour_idx = neighbours[triangle_idx][edge_idx]

                if neighbour_idx == -1:
                    # Boundary face
                    FluxV_f_BC_vN = vN_bc[triangle_idx]
                    FluxC_f_BC_Di = di_bc[0]
                    FluxV_f_BC_Di = di_bc[1]

                    # Residual = M(T) @ T - C(T)
                    res[triangle_idx] += FluxC_f_BC_Di * T_C
                    res[triangle_idx] += FluxV_f_BC_vN + FluxV_f_BC_Di

                else:
                    # Internal face
                    neighbour_center = centers[neighbour_idx][edge_idx]
                    edge_center = edge_centers[triangle_idx][edge_idx]
                    edge_length = edge_lengths[triangle_idx][edge_idx]

                    d_CF = np.linalg.norm(triangle_center - neighbour_center)
                    d_Cs = np.linalg.norm(triangle_center - edge_center)
                    d_sF = np.linalg.norm(edge_center - neighbour_center)

                    gDiff_f = edge_length / d_CF
                    g_f = d_Cs / (d_Cs + d_sF)

                    k_C = cell_k[triangle_idx]
                    k_F = cell_k[neighbour_idx]

                    k_s = k_C * k_F / ((1.0 - g_f) * k_C + g_f * k_F)

                    FluxF_f = -k_s * gDiff_f
                    FluxC_f =  k_s * gDiff_f

                    T_F = T[neighbour_idx]
                    res[triangle_idx] += FluxC_f * T_C + FluxF_f * T_F

            # Cross-diffusion correction
            FluxV_f_CD = cross_diffusion_corr[triangle_idx]
            res[triangle_idx] += FluxV_f_CD

        return res


    def calculate_von_Neumann_boundary_conditions(self):

        total_length_HP = 0
        total_length_fuel_pin = 0
        HP_triangle_indices = []
        fuel_pin_triangle_indices = []

        self.vN_boundary_conditions = np.zeros(self.mesh.triangles.shape[0])

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        for boundary_edge in self.mesh.get_boundary_edges():
            triangle_idx = self.mesh._edge_to_triangles[boundary_edge]

            if ((self.is_edge_on_circumference(boundary_edge, self.HP_centers, self.HP_radius)) and
                (self.HP_BC == "vonNeumann")):

                HP_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Positive due to "q dot S > 0" since both S and q points outward. 
                self.vN_boundary_conditions[triangle_idx] += self.Q_in * edge_length

                total_length_HP += edge_length
            
            if ((self.is_edge_on_circumference(boundary_edge, self.fuel_pin_centers, self.fuel_pin_radius)) and
                (self.FP_BC == "vonNeumann")):

                fuel_pin_triangle_indices.append(triangle_idx)

                edge_length = np.linalg.norm(
                    self.mesh.points[boundary_edge[0]] - self.mesh.points[boundary_edge[1]])

                # Negative due to "q dot S < 0" since both S points outwards and q point inward.
                self.vN_boundary_conditions[triangle_idx] += -self.Q_in * edge_length

                total_length_fuel_pin += edge_length
                
        # Using the found triangle indecies and totalt length exposed to HPs, and fuel pins, to 
        # normalize the flux BC.        
        for triangle_idx in HP_triangle_indices:
            # Positive due to "q dot S > 0" since both S and q points outward. 
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_HP  #Need to scale Q_in with edge length.
                
        for triangle_idx in fuel_pin_triangle_indices:
            # Negative due to "q dot S < 0" since both S points outwards and q point inward.
            self.vN_boundary_conditions[triangle_idx] *= 1 / total_length_fuel_pin


    @staticmethod
    @njit(parallel=True)
    def calculate_Dirichlet_boundary_conditions(
            cell_k,
            centers,
            edge_lengths,
            edge_centers,
            boundary_kind,
            T_HP,
            T_FP,
            HP_bc,
            FP_bc
        ):
        n_triangles = centers.shape[0]

        # First term in the matrix represents FluxC_b, the second represents FluxV_b.
        di_bc = np.zeros((n_triangles, 2))

        # Iterating over all the boundary edges to see which are in contact with HP and fuel pins. 
        for triangle_idx in range(n_triangles):
            triangle_center = centers[triangle_idx]

            for edge_idx in range(2):
                if boundary_kind[triangle_idx][edge_idx] == 1 and HP_bc == "Dirichlet":

                    edge_length = edge_lengths[triangle_idx][edge_idx]
                    edge_center = edge_centers[triangle_idx][edge_idx]
                    
                    d_Cb = np.linalg.norm(triangle_center - edge_center)

                    Diff_b = edge_length / d_Cb
                    Flux_C_b = Diff_b * cell_k[triangle_idx]

                    # Positive due to "q dot S > 0" since both S and q points outward. 
                    di_bc[triangle_idx][0] += Flux_C_b
                    di_bc[triangle_idx][1] += -Flux_C_b * T_HP

                if boundary_kind[triangle_idx][edge_idx] == 2 and FP_bc == "Dirichlet":
                    
                    edge_length = edge_lengths[triangle_idx][edge_idx]
                    edge_center = edge_centers[triangle_idx][edge_idx]
                    
                    d_Cb = np.linalg.norm(triangle_center - edge_center)

                    Diff_b = edge_length / d_Cb
                    Flux_C_b = Diff_b * cell_k[triangle_idx]

                    # Positive due to "q dot S > 0" since both S and q points outward. 
                    di_bc[triangle_idx][0] += Flux_C_b
                    di_bc[triangle_idx][1] += -Flux_C_b * T_FP

        return di_bc
    

    @staticmethod
    @njit(parallel=True)
    def calculate_cross_diffusion_correction(
            T,
            cell_k,
            centers,
            areas,
            neighbours,
            edge_lengths,
            edge_centers,
            edge_normals,
            boundary_kind,
            T_HP,
            T_FP,
            HP_bc,
            FP_bc,
            vN_bc
    ):
        def calculate_cell_flux_gradient(
                triangle_idx,
                T,
                cell_k,
                centers,
                areas,
                neighbours,
                edge_lengths,
                edge_centers,
                edge_normals,
                boundary_kind,
                T_HP,
                T_FP,
                HP_bc,
                FP_bc,
                vN_bc
            ):
            # Utilizing the Green-Gauss gradient theorem.

            V_C = areas[triangle_idx]
            C_center = centers[triangle_idx]

            flux_gradient_C = np.array([0., 0.])
            for edge_idx in range(2):
                neighbour_idx = neighbours[triangle_idx][edge_idx]

                if neighbour_idx == -1:
                    if boundary_kind[triangle_idx][edge_idx] == 1 and HP_bc == "Dirichlet":
                        T_surface = T_HP
                    elif boundary_kind[triangle_idx][edge_idx] == 2 and FP_bc == "Dirichlet":
                        T_surface = T_FP
                    else:
                        # If the neighbouring triangle is an boundary then apply note 3., eq. (8.42) on page 222 in F.Moukalled et al.
                        edge_length = edge_lengths[triangle_idx][edge_idx]
                        edge_center = edge_centers[triangle_idx][edge_idx]

                        q_surface = vN_bc[triangle_idx] / edge_length
                        
                        # Not multiplying gDiff with S due to q_surface is equal to "S * q_b" 
                        T_C = T[triangle_idx]
                        d_Cf = np.linalg.norm(edge_center - C_center)
                        k_C = cell_k[triangle_idx]

                        T_surface = T_C - q_surface * d_Cf / k_C
                        
                    n = edge_normals[triangle_idx][edge_idx]
                    S_surface = edge_length * n

                    flux_gradient_C += S_surface * T_surface

                else:
                    F_center = centers[neighbour_idx]

                    edge_length = edge_lengths[triangle_idx][edge_idx]
                    edge_center = edge_centers[triangle_idx][edge_idx]

                    d_Cs = np.linalg.norm(C_center - edge_center)
                    d_sF = np.linalg.norm(edge_center - F_center)

                    g_F = d_Cs / (d_Cs + d_sF)
                    g_C = 1. - g_F

                    T_surface = g_C * T[triangle_idx] + g_F * T[neighbour_idx]

                    n = edge_normals[triangle_idx][edge_idx]
                    S_surface = edge_length * n

                    flux_gradient_C += S_surface * T_surface

            return flux_gradient_C / V_C
        
        n_triangles = centers.shape[0]
        cell_gradients = np.zeros((n_triangles, 2))

        # Compute each cell gradient once
        for triangle_idx in range(n_triangles):
            cell_gradients[triangle_idx] = calculate_cell_flux_gradient(                
                triangle_idx,
                T,
                cell_k,
                centers,
                areas,
                neighbours,
                edge_lengths,
                edge_centers,
                edge_normals,
                boundary_kind,
                T_HP,
                T_FP,
                HP_bc,
                FP_bc,
                vN_bc
                )

        # Assemble cross-diffusion correction
        cross_diffusion_correction = np.zeros(n_triangles)
        for triangle_idx in range(n_triangles):
            C_center = centers[triangle_idx]

            for edge_idx in range(2):
                neighbour_idx = neighbours[triangle_idx][edge_idx]

                if neighbour_idx == -1:
                    continue

                grad_flux_C = cell_gradients[triangle_idx]
                grad_flux_F = cell_gradients[neighbour_idx]

                F_center = centers[neighbour_idx]

                edge_length = edge_lengths[triangle_idx][edge_idx]
                edge_center = edge_centers[triangle_idx][edge_idx]

                d_CF = np.linalg.norm(C_center - F_center)
                d_Cs = np.linalg.norm(C_center - edge_center)
                d_sF = np.linalg.norm(edge_center - F_center)

                g_F = d_Cs / (d_Cs + d_sF)
                g_C = 1.0 - g_F

                grad_flux_surface = g_C * grad_flux_C + g_F * grad_flux_F

                k_C = cell_k[triangle_idx]
                k_F = cell_k[neighbour_idx]
                k_surface = k_C * k_F / (g_C * k_C + g_F * k_F)

                n_vec = edge_normals[triangle_idx][edge_idx]
                e = (F_center - C_center) / d_CF
                T_vector_surface = (n_vec - e) * edge_length

                cross_diffusion_correction[triangle_idx] += (
                    k_surface * np.dot(grad_flux_surface, T_vector_surface)
                )


    def is_edge_on_circumference(self, edge, centers, radius):
        p1 = self.mesh.points[edge[0]]
        p2 = self.mesh.points[edge[1]]
        epsilon = 1e-2

        for center in centers:
            if (
                abs(np.linalg.norm(p1 - center) - radius) < epsilon
                and abs(np.linalg.norm(p2 - center) - radius) < epsilon
            ):
                return True
            
        return False
    
def plot_mesh_with_temp_profile(points, triangles, T):
    # Convert to 3D
    points_3d = np.column_stack([points, np.zeros(len(points))])

    # PyVista cell format: [3, n1, n2, n3, 3, n1, n2, n3, ...]
    cells = np.hstack([
        np.hstack([[3, *tri] for tri in triangles])
    ])

    # Cell types: 5 = triangle
    cell_types = np.full(len(triangles), pv.CellType.TRIANGLE)

    mesh_visualization = pv.UnstructuredGrid(cells, cell_types, points_3d)
    mesh_visualization.cell_data["u"] = T

    mesh_visualization.plot(
        scalars="u",
        show_edges=True,
        cmap="viridis"
    )
    

if __name__ == "__main__":
    mesh = meshio.read("models/moderator/hex_mesh.msh")

    points = mesh.points[:, :2]                
    triangles = mesh.cells_dict["triangle"]     
    mesh = UnstructuredMesh(points, triangles)

    print(f"Number of elements: {triangles.shape}, Number of points: {points.shape}")

    cell_k = np.ones(triangles.shape[0])
    cell_T = np.ones(triangles.shape[0])
    l_pitch = 10.
    r_HP = 3.
    r_f = 1.5
    theta_hex = (np.pi / 6)

    data = {
        "cell_k": 20 * np.ones(triangles.shape[0]),
        "cell_T": np.ones(triangles.shape[0]),

        "Q_in": 1000,

        "HP_centers":       [[0., 0.], [np.tan(theta_hex) * 2. * l_pitch, 2. * l_pitch]],
        "fuel_pin_centers": [[0, l_pitch * 3./2.], [0, l_pitch * 5./2.], [np.tan(theta_hex) * 2 * l_pitch, l_pitch * 7./2.]],
        "HP_radius": r_HP,
        "fuel_pin_radius": r_f,

        "T_HP": 850,
        "T_FP": 885,

        "HP_BC": "Dirichlet", #"Dirichlet"
        "FP_BC": "vonNeumann"   #"vonNeumann"
    }

    mod_mesh = ModeratorDiscretisedMeshFast(mesh=mesh, data=data)

    (
    centers,
    neighbours,
    face_lengths,
    face_centers,
    surface_normals,
    is_boundary,
    boundary_kind,
    ) = mod_mesh.assemble_mesh_data(
    mesh=mesh,
    HP_centers=data["HP_centers"],
    fuel_pin_centers=data["fuel_pin_centers"],
    HP_radius=data["HP_radius"],
    fuel_pin_radius=data["fuel_pin_radius"],
    )

    print("centers shape:", centers.shape)
    print("neighbours shape:", neighbours.shape)
    print("face_lengths shape:", face_lengths.shape)
    print("face_centers shape:", face_centers.shape)
    print("surface_normals shape:", surface_normals.shape)
    print("is_boundary shape:", is_boundary.shape)
    print("boundary_kind shape:", boundary_kind.shape)

    # T = mod_mesh.solve_nonlinear(
    #     xtol=1e-8,
    #     maxfev=500,
    #     verbose=True,
    #     use_linear_guess=True,)

    # T = mod_mesh.solve_linearly(4)

    # plot_mesh_with_temp_profile(mod_mesh.mesh.points, mod_mesh.mesh.triangles, T)